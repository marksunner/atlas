// SPDX-License-Identifier: AGPL-3.0-only

//! Step 3.7 Flash weight loader.
//!
//! Hybrid of MiniMax M2 and Qwen 3.5 patterns:
//!   * Sigmoid MoE routing + correction bias (MiniMax M2 pattern)
//!   * Shared expert per MoE layer (Qwen 3.5 pattern)
//!   * Per-HEAD attention gate g_proj [nq, hidden] — sigmoid scalar per
//!     head broadcast over head_dim (NOT Qwen 3.5's interleaved Q+G)
//!   * Per-layer RoPE: theta=5e6/prf=0.5 on full-attention layers,
//!     theta=1e4/prf=1.0 on sliding layers (rope_theta_per_layer /
//!     partial_rotary_factors from config)
//!   * Heterogeneous Q heads: 64 on full-attention layers, 96 on sliding
//!     (detected from q_proj shape, applied via dimension overrides)
//!   * Per-head q_norm / k_norm
//!   * Mixed dense FFN (layers 0-2) + MoE (layers 3-44)
//!   * 3 MTP modules at layers 45-47 (different prefix: `model.layers.`)
//!
//! Weight prefix: `model.language_model.layers.{i}` for main layers.
//! MTP prefix: `model.layers.{45|46|47}` (different namespace!).
//!
//! KEY ARCHITECTURAL DIFFERENCE: Step 3.7 stores expert weights as FUSED
//! tensors — one tensor per projection type containing ALL 288 experts
//! concatenated. Atlas needs per-expert QuantizedWeight entries, so we
//! slice by computing byte offsets into the fused GPU allocations.
//!
//! NVFP4 format: ModelOpt style with `weight`, `weight_scale`, `weight_scale_2`,
//! `input_scale` per projection. Shared expert is BF16.

mod fp8;
mod load_layers;

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight, dense};

pub struct Step3p7WeightLoader;

/// Resolve the text-stack weight prefix by probing the store.
///
/// The NVFP4 release nests the language model under
/// `model.language_model.layers.*`; the FP8 release stores it directly
/// under `model.layers.*`. Probe for layer 0's input norm to pick the
/// right namespace instead of trusting the config constant.
fn resolve_weight_prefix(store: &WeightStore, config: &atlas_core::config::ModelConfig) -> String {
    let configured = if config.weight_prefix.is_empty() {
        "model.language_model"
    } else {
        &config.weight_prefix
    };
    if store.contains(&format!("{configured}.layers.0.input_layernorm.weight")) {
        return configured.to_string();
    }
    if store.contains("model.layers.0.input_layernorm.weight") {
        tracing::info!("step3p7: text stack found under `model.layers.*` (FP8-release layout)");
        return "model".to_string();
    }
    configured.to_string()
}

/// Step 3.7 uses shifted RMSNorm: `output = (x / rms) * (weight + 1)`.
/// The checkpoint stores norm weights centered around 0, not 1.
/// This function adds 1.0 to each element so the standard RMSNorm kernel
/// `output = (x / rms) * weight` produces the correct result.
fn offset_norm_weights_plus_one(
    weight: &DenseWeight,
    size: usize,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let byte_len = size * 2; // BF16 = 2 bytes per element
    let mut buf = vec![0u8; byte_len];
    gpu.copy_d2h(weight.weight, &mut buf)?;

    for i in 0..size {
        let bits = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        let f32_val = f32::from_bits((bits as u32) << 16);
        let new_val = f32_val + 1.0;
        // Round to BF16: add 0x7FFF + bit 16 for round-to-nearest-even
        let f32_bits = new_val.to_bits();
        let new_bits = ((f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16) as u16;
        buf[i * 2] = new_bits as u8;
        buf[i * 2 + 1] = (new_bits >> 8) as u8;
    }

    gpu.copy_h2d(&buf, weight.weight)?;
    Ok(())
}

/// Slice a fused NVFP4 tensor into per-expert QuantizedWeight entries.
///
/// Step 3.7's original checkpoint stores all experts in one contiguous
/// tensor per projection:
///   weight: [num_experts * n, k] packed NVFP4 (0.5 bytes/element)
///   weight_scale: [num_experts * n, k/group_size] FP8 per-group scales
///   input_scale: [num_experts * n] (optional, activation quantization)
///
/// This function creates `num_experts` QuantizedWeight entries, each
/// pointing to a different offset within the fused allocations.
#[allow(clippy::too_many_arguments)]
fn slice_fused_experts(
    fused_weight: DevicePtr,
    fused_scale: DevicePtr,
    fused_input_scale: DevicePtr,
    global_scale_2: f32,
    num_experts: usize,
    local_start: usize,
    local_end: usize,
    n: usize,
    k: usize,
) -> Vec<QuantizedWeight> {
    let group_size = 16usize;
    let packed_bytes_per_expert = n * k / 2;
    let scale_bytes_per_expert = n * k.div_ceil(group_size);
    let input_scale_bytes_per_expert = n * 4;

    (0..num_experts)
        .map(|e| {
            if e < local_start || e >= local_end {
                return QuantizedWeight::null();
            }
            let local_expert = e - local_start;
            QuantizedWeight {
                weight: fused_weight.offset(local_expert * packed_bytes_per_expert),
                weight_scale: fused_scale.offset(local_expert * scale_bytes_per_expert),
                weight_scale_2: global_scale_2,
                input_scale: if fused_input_scale == DevicePtr::NULL {
                    DevicePtr::NULL
                } else {
                    fused_input_scale.offset(local_expert * input_scale_bytes_per_expert)
                },
            }
        })
        .collect()
}

/// Detect whether this checkpoint uses per-expert tensor format.
fn has_per_expert_tensors(store: &WeightStore, layer_prefix: &str) -> bool {
    let pattern = format!("{layer_prefix}.moe.experts.");
    let found = store.names().any(|k| k.starts_with(&pattern));
    tracing::debug!("has_per_expert_tensors('{layer_prefix}'): pattern='{pattern}', found={found}");
    found
}

/// Load a fused NVFP4 tensor from the store (Standard ModelOpt format).
fn load_fused_nvfp4(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, f32)> {
    let weight = store.get(&format!("{prefix}.weight"))?.ptr;
    let weight_scale = store.get(&format!("{prefix}.weight_scale"))?.ptr;

    let ws2_key = format!("{prefix}.weight_scale_2");
    let ws2_ptr = store.get(&ws2_key)?.ptr;
    let mut ws2_buf = [0u8; 4];
    gpu.copy_d2h(ws2_ptr, &mut ws2_buf)?;
    let weight_scale_2 = f32::from_le_bytes(ws2_buf);

    let is_key = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&is_key) {
        store.get(&is_key)?.ptr
    } else {
        DevicePtr::NULL
    };

    Ok((weight, weight_scale, input_scale, weight_scale_2))
}

impl ModelWeightLoader for Step3p7WeightLoader {
    fn supports_tp(&self) -> bool {
        false // Single-GPU initial bring-up
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        let prefix = resolve_weight_prefix(store, config);
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = resolve_weight_prefix(store, config);
        let w = dense(store, &format!("{prefix}.norm.weight"))?;
        offset_norm_weights_plus_one(&w, config.hidden_size, gpu)?;
        Ok(w)
    }

    fn load_lm_head(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None) // Multi-module MTP — use load_mtp_weights_multi
    }

    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        let first_mtp_idx = config.num_hidden_layers;
        let probe = format!("model.layers.{first_mtp_idx}.input_layernorm.weight");
        if !store.contains(&probe) {
            tracing::info!(
                "step3p7: no MTP module weights found \
                 (expected at layer {first_mtp_idx}); MTP disabled"
            );
            return Ok(Vec::new());
        }

        tracing::info!(
            "step3p7: MTP module weights detected at layers {}-{} but MTP loader \
             not yet implemented. Run with --speculative 0 for non-MTP decode.",
            first_mtp_idx,
            first_mtp_idx + config.mtp_num_hidden_layers - 1,
        );
        Ok(Vec::new())
    }
}
