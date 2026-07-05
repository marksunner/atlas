// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, Fp8ExpertWeight, MoeWeights, QuantizedWeight,
    dense, dense_auto, detect_nvfp4_variant, load_kv_scales, quantize_to_nvfp4,
};

use super::fp8::{has_fused_fp8_experts, load_fused_fp8_experts, quantize_dense_to_fp8_blockscaled};
use super::{
    has_per_expert_tensors, load_fused_nvfp4, offset_norm_weights_plus_one, resolve_weight_prefix,
    slice_fused_experts,
};

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let h = config.hidden_size;
    let variant = detect_nvfp4_variant(store, config);
    let inter = config.moe_intermediate_size;
    let shared_inter = config.shared_expert_intermediate_size;

    tracing::info!(
        "step3p7: loading {} layers, variant={:?}, hidden_size={h}, \
         experts={}, moe_inter={inter}, shared_inter={shared_inter}",
        config.num_hidden_layers,
        variant,
        config.num_experts,
    );

    let prefix = resolve_weight_prefix(store, config);

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_layer_idx = 0usize;

    for i in 0..config.num_hidden_layers {
        let lp = format!("{prefix}.layers.{i}");
        tracing::debug!("step3p7: layer {i}");

        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

        // Step 3.7 shifted RMSNorm: add 1.0 to norm weights so standard
        // kernel computes (x/rms) * (weight+1) correctly.
        offset_norm_weights_plus_one(&input_norm, h, gpu)?;
        offset_norm_weights_plus_one(&post_attn_norm, h, gpu)?;

        // ── FFN: detect MoE vs dense by probing for gate weight ─────
        let moe_gate_key = format!("{lp}.moe.gate.weight");
        let is_moe = store.contains(&moe_gate_key);

        let ffn = if is_moe {
            load_moe_ffn(
                store,
                config,
                gpu,
                &lp,
                &moe_gate_key,
                h,
                inter,
                shared_inter,
                absmax_k,
                quantize_k,
                stream,
                i,
            )?
        } else {
            load_dense_ffn(
                store,
                gpu,
                &lp,
                config.intermediate_size,
                h,
                absmax_k,
                quantize_k,
                stream,
                i,
            )?
        };

        // ── Attention ───────────────────────────────────────────────
        let layer = load_attention_layer(
            store,
            config,
            gpu,
            &lp,
            input_norm,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            h,
            layer_kv_dtypes,
            absmax_k,
            quantize_k,
            stream,
            i,
        )?;

        layers.push(Box::new(layer));
        attn_layer_idx += 1;
    }

    tracing::info!(
        "step3p7: built {} layers ({} attention)",
        layers.len(),
        attn_layer_idx
    );
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
fn load_moe_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    moe_gate_key: &str,
    h: usize,
    inter: usize,
    shared_inter: usize,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    i: usize,
) -> Result<FfnComponent> {
    let gate = dense(store, moe_gate_key)?;

    // Router bias (sigmoid routing)
    let bias_key = format!("{lp}.moe.router_bias");
    let correction_bias = if store.contains(&bias_key) {
        Some(dense(store, &bias_key)?)
    } else {
        None
    };

    let moe_p = format!("{lp}.moe");
    // FP8 release: fused FP8 block-scaled routed experts. The NVFP4
    // QuantizedWeight expert table stays NULL and dispatch goes through
    // the native FP8 pointer tables set below (same mechanism as qwen35
    // native_fp8). Everything outside the routed experts is BF16 on disk
    // in BOTH releases, so all other paths are byte-identical.
    let fp8_fused = has_fused_fp8_experts(store, &moe_p);
    let use_per_expert = has_per_expert_tensors(store, lp);

    let experts: Vec<ExpertWeight> = if fp8_fused {
        vec![ExpertWeight::null(); config.num_experts]
    } else if use_per_expert {
        tracing::debug!("step3p7: layer {lp} using per-expert tensor format");
        (0..config.num_experts)
            .map(|e| {
                let ep = format!("{moe_p}.experts.{e}");
                let gp_key = format!("{ep}.gate_proj.weight");
                if !store.contains(&gp_key) {
                    return Ok(ExpertWeight::null());
                }
                let load_expert_proj = |proj: &str| -> Result<QuantizedWeight> {
                    let pp = format!("{ep}.{proj}");
                    let weight = store.get(&format!("{pp}.weight"))?.ptr;
                    let weight_scale = store.get(&format!("{pp}.weight_scale"))?.ptr;
                    let ws2_key = format!("{pp}.weight_scale_2");
                    let global_ws2_key = format!("{moe_p}.{proj}.weight_scale_2");
                    let ws2_ptr = if store.contains(&ws2_key) {
                        store.get(&ws2_key)?.ptr
                    } else if store.contains(&global_ws2_key) {
                        store.get(&global_ws2_key)?.ptr
                    } else {
                        anyhow::bail!(
                            "weight_scale_2 not found for {pp} \
                             (tried per-expert and global)"
                        );
                    };
                    let mut ws2_buf = [0u8; 4];
                    gpu.copy_d2h(ws2_ptr, &mut ws2_buf).ok();
                    let weight_scale_2 = f32::from_le_bytes(ws2_buf);
                    let is_key = format!("{pp}.input_scale");
                    let input_scale = if store.contains(&is_key) {
                        store
                            .get(&is_key)
                            .ok()
                            .map(|t| t.ptr)
                            .unwrap_or(DevicePtr::NULL)
                    } else {
                        DevicePtr::NULL
                    };
                    Ok(QuantizedWeight {
                        weight,
                        weight_scale,
                        weight_scale_2,
                        input_scale,
                    })
                };
                let gate_proj = load_expert_proj("gate_proj")?;
                let up_proj = load_expert_proj("up_proj")?;
                let down_proj = load_expert_proj("down_proj")?;
                Ok(ExpertWeight {
                    gate_proj,
                    up_proj,
                    down_proj,
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        tracing::debug!("step3p7: layer {lp} using fused tensor format");
        let (gp_w, gp_s, gp_is, gp_s2) =
            load_fused_nvfp4(store, &format!("{moe_p}.gate_proj"), gpu)?;
        let (up_w, up_s, up_is, up_s2) = load_fused_nvfp4(store, &format!("{moe_p}.up_proj"), gpu)?;
        let (dp_w, dp_s, dp_is, dp_s2) =
            load_fused_nvfp4(store, &format!("{moe_p}.down_proj"), gpu)?;

        let (local_start, local_end) = config.local_expert_range();
        let gate_projs = slice_fused_experts(
            gp_w,
            gp_s,
            gp_is,
            gp_s2,
            config.num_experts,
            local_start,
            local_end,
            inter,
            h,
        );
        let up_projs = slice_fused_experts(
            up_w,
            up_s,
            up_is,
            up_s2,
            config.num_experts,
            local_start,
            local_end,
            inter,
            h,
        );
        let down_projs = slice_fused_experts(
            dp_w,
            dp_s,
            dp_is,
            dp_s2,
            config.num_experts,
            local_start,
            local_end,
            h,
            inter,
        );

        (0..config.num_experts)
            .map(|e| ExpertWeight {
                gate_proj: gate_projs[e],
                up_proj: up_projs[e],
                down_proj: down_projs[e],
            })
            .collect()
    };

    // Shared expert (BF16 on disk). NVFP4 path: runtime-quantize to NVFP4.
    // FP8 path: the fused FP8 MoE kernels compute routed + shared in one
    // launch and need FP8 block-scaled shared weights, so it is quantized
    // BF16 -> FP8 below instead; the NVFP4 slot stays NULL.
    let se_p = format!("{lp}.share_expert");
    let shared_expert = if fp8_fused {
        ExpertWeight::null()
    } else {
        let se_gate = dense_auto(store, &format!("{se_p}.gate_proj.weight"), gpu)?;
        let se_up = dense_auto(store, &format!("{se_p}.up_proj.weight"), gpu)?;
        let se_down = dense_auto(store, &format!("{se_p}.down_proj.weight"), gpu)?;
        ExpertWeight {
            gate_proj: quantize_to_nvfp4(
                &se_gate,
                shared_inter,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?,
            up_proj: quantize_to_nvfp4(&se_up, shared_inter, h, gpu, absmax_k, quantize_k, stream)?,
            down_proj: quantize_to_nvfp4(
                &se_down,
                h,
                shared_inter,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?,
        }
    };

    let shared_expert_gate = DenseWeight {
        weight: DevicePtr::NULL,
    };

    let moe_weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias,
    };

    // Router gate stays BF16 (gate_nvfp4 = None → dense_gemv/dense_gemm
    // router path). The checkpoint ships the router in BF16 and vLLM
    // routes in full precision; runtime-quantizing it to NVFP4 perturbs
    // the sigmoid+bias top-8-of-288 selection on every token of every
    // MoE layer for a negligible memory win (~2.4 MB/layer). Same lever
    // as qwen3's ATLAS_BF16_ROUTER, on by default for Step 3.7.
    let mut moe_layer = MoeLayer::new(moe_weights, config.num_experts, None, gpu, config)?;
    // Per-layer SwiGLU clamp (config swiglu_limits / swiglu_limits_shared;
    // Step 3.7: 7.0 routed / 16.0 shared on layers 43-44). vLLM equivalent:
    // swiglustep activation. Skipping this served the last two layers
    // before the LM head unclamped.
    let routed_limit = config.swiglu_limits.get(i).copied().unwrap_or(0.0);
    let shared_limit = config.swiglu_limits_shared.get(i).copied().unwrap_or(0.0);
    if routed_limit > 0.0 || shared_limit > 0.0 {
        tracing::info!(
            "step3p7: layer {i} SwiGLU clamp enabled (routed limit {routed_limit}, \
             shared limit {shared_limit})"
        );
        moe_layer.set_swiglu_limits(routed_limit, shared_limit);
    }
    if fp8_fused {
        let fp8_experts = load_fused_fp8_experts(store, &moe_p, config)?;
        let shared_fp8 = Fp8ExpertWeight {
            gate_proj: quantize_dense_to_fp8_blockscaled(
                store,
                &format!("{se_p}.gate_proj.weight"),
                gpu,
            )?,
            up_proj: quantize_dense_to_fp8_blockscaled(
                store,
                &format!("{se_p}.up_proj.weight"),
                gpu,
            )?,
            down_proj: quantize_dense_to_fp8_blockscaled(
                store,
                &format!("{se_p}.down_proj.weight"),
                gpu,
            )?,
        };
        moe_layer.set_fp8_experts(&fp8_experts, shared_fp8, gpu)?;
        tracing::info!(
            "step3p7: layer {i} MoE on native FP8 block-scaled path \
             ({} routed experts local, shared expert BF16->FP8)",
            config.local_expert_range().1 - config.local_expert_range().0,
        );
    } else {
        moe_layer.predequant_for_prefill(gpu, config, stream)?;
    }
    Ok(FfnComponent::Moe(moe_layer))
}

#[allow(clippy::too_many_arguments)]
fn load_dense_ffn(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    lp: &str,
    intermediate_size: usize,
    h: usize,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    i: usize,
) -> Result<FfnComponent> {
    tracing::info!("step3p7: layer {i} is dense FFN");
    let gate_w = dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?;
    let up_w = dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?;
    let down_w = dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?;

    let gate_q = quantize_to_nvfp4(
        &gate_w,
        intermediate_size,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let up_q = quantize_to_nvfp4(
        &up_w,
        intermediate_size,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let down_q = quantize_to_nvfp4(
        &down_w,
        h,
        intermediate_size,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    let dense_weights = DenseFfnWeights {
        gate_proj: gate_q,
        up_proj: up_q,
        down_proj: down_q,
    };
    Ok(FfnComponent::Dense(DenseFfnLayer::new(dense_weights, gpu)?))
}

#[allow(clippy::too_many_arguments)]
fn load_attention_layer(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    attn_layer_idx: usize,
    h: usize,
    layer_kv_dtypes: &[KvCacheDtype],
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    i: usize,
) -> Result<Qwen3AttentionLayer> {
    let p = format!("{lp}.self_attn");
    let q_proj = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
    let k_proj = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
    let v_proj = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
    let o_proj_w = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;

    let q_proj_shape = store.get(&format!("{p}.q_proj.weight"))?.shape.clone();
    let q_proj_n = q_proj_shape[0];
    let kv_proj_n = config.num_key_value_heads * config.head_dim;
    let actual_q_heads = q_proj_n / config.head_dim;
    tracing::info!(
        "step3p7: layer {i} attention: q_proj_n={q_proj_n} \
         ({actual_q_heads} Q heads), kv_proj_n={kv_proj_n}"
    );
    let q_nvfp4 = quantize_to_nvfp4(&q_proj, q_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
    let k_nvfp4 = quantize_to_nvfp4(&k_proj, kv_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
    let v_nvfp4 = quantize_to_nvfp4(&v_proj, kv_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
    let o_nvfp4 = quantize_to_nvfp4(&o_proj_w, h, q_proj_n, gpu, absmax_k, quantize_k, stream)?;

    // Per-head attention gate (g_proj)
    let g_proj_key = format!("{p}.g_proj.weight");
    let g_proj_weight = if store.contains(&g_proj_key) {
        let w = dense_auto(store, &g_proj_key, gpu)?;
        tracing::info!("step3p7: layer {i} loaded g_proj gate [{actual_q_heads}, {h}]");
        Some(w)
    } else {
        None
    };

    // Per-head q_norm / k_norm (also shifted RMSNorm)
    let q_norm = dense(store, &format!("{p}.q_norm.weight"))?;
    let k_norm = dense(store, &format!("{p}.k_norm.weight"))?;
    offset_norm_weights_plus_one(&q_norm, config.head_dim, gpu)?;
    offset_norm_weights_plus_one(&k_norm, config.head_dim, gpu)?;

    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

    let attn = AttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj: o_nvfp4,
        q_norm,
        k_norm,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };

    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_layer_idx,
        Some(q_nvfp4),
        Some(k_nvfp4),
        Some(v_nvfp4),
        gpu,
        layer_kv_dtypes[attn_layer_idx],
        config.fp8_kv_calibration_tokens,
        config,
    )?;

    layer.set_dimension_overrides(config.head_dim, actual_q_heads, config.num_key_value_heads);

    let is_sliding = if !config.layer_types.is_empty() {
        config.layer_types.get(i).copied() == Some(atlas_core::config::LayerType::SlidingAttention)
    } else {
        !i.is_multiple_of(4)
    };

    if is_sliding && config.sliding_window > 0 {
        layer.set_sliding_window(Some(config.sliding_window));
    } else {
        layer.set_sliding_window(None);
    }

    // Per-layer RoPE, following vLLM step3p5.py `Step3p5Attention.__init__`:
    // `rope_theta = rope_theta[self.layer_idx]` and `partial_rotary_factor =
    // partial_rotary_factors[layer_idx]`. Step 3.7 full-attention layers:
    // theta=5e6, prf=0.5 → rotary_dim 64; sliding layers: theta=1e4,
    // prf=1.0 → rotary_dim 128 (head_dim). Falls back to the layer-type
    // defaults when the config didn't ship per-layer arrays.
    let rope_theta = config
        .rope_theta_per_layer
        .get(i)
        .copied()
        .unwrap_or(if is_sliding { 10000.0 } else { config.rope_theta });
    let prf = config
        .partial_rotary_factors
        .get(i)
        .copied()
        .unwrap_or(if is_sliding {
            1.0
        } else {
            config.partial_rotary_factor
        });
    let rotary_dim = (config.head_dim as f64 * prf) as u32;
    layer.set_rope_overrides(rope_theta as f32, rotary_dim);

    // llama3 (NTK-by-parts) RoPE frequency scaling. Step 3.7 applies
    // `rope_scaling = {"rope_type":"llama3","factor":2.0,...}` to
    // full-attention layers only (`yarn_only_types: ["full_attention"]`).
    // Precompute the per-layer llama3-scaled inv_freq table and attach it so
    // the RoPE dispatch routes this layer through the table-based kernel.
    // Without this, full-attention (long-range retrieval) layers rotate too
    // fast past original_max_position_embeddings → content generation
    // degenerates beyond ~8K context while the sliding/thinking path stays
    // coherent. Reference: HF `_compute_llama3_parameters`; vLLM does the
    // same via `yarn_only_types`.
    if config.rope_llama3_factor > 0.0
        && (!config.rope_llama3_full_attention_only || !is_sliding)
        && rotary_dim >= 2
    {
        let table = compute_llama3_inv_freq(
            rope_theta as f32,
            rotary_dim,
            config.rope_llama3_factor,
            config.rope_llama3_low_freq_factor,
            config.rope_llama3_high_freq_factor,
            config.rope_llama3_original_max_position,
            gpu,
        )?;
        layer.set_rope_inv_freq_table(table);
        tracing::info!(
            "step3p7: layer {i} llama3 RoPE scaling \
             (factor={}, rotary_dim={rotary_dim}, theta={rope_theta}, \
             low={}, high={}, orig_max={})",
            config.rope_llama3_factor,
            config.rope_llama3_low_freq_factor,
            config.rope_llama3_high_freq_factor,
            config.rope_llama3_original_max_position,
        );
    }

    if let Some(gw) = g_proj_weight {
        layer.set_head_gate_weight(gw);
    }

    let qt = q_nvfp4.transpose_for_gemm(gpu, q_proj_n, h)?;
    let kt = k_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
    let vt = v_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
    let ot = layer.attn.o_proj.transpose_for_gemm(gpu, h, q_proj_n)?;
    layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));

    Ok(layer)
}

/// Precompute the Llama-3.1 "llama3" (NTK-by-parts) RoPE inv_freq table on
/// GPU: `[rotary_dim/2]` FP32 frequencies. Mirrors HF
/// `_compute_llama3_parameters` (`modeling_rope_utils.py`):
///   base_i     = 1 / theta^(2i/rotary_dim)
///   wavelen_i  = 2π / base_i
///   if wavelen > old_ctx/low_freq_factor:  base / factor       (low freq)
///   elif wavelen < old_ctx/high_freq_factor: base              (high freq)
///   else: smooth interpolate between the two              (medium freq)
/// The medium-frequency `inv_freq_llama` in HF is the un-scaled `base` (the
/// first `where` only divides low-freq pairs), so
///   smoothed = (1-s)·base/factor + s·base,
///   s = (old_ctx/wavelen - low_freq_factor)/(high_freq_factor - low_freq_factor).
/// `attention_factor` for llama3 is 1.0 (no output mscale), so only the
/// frequencies change. Rotation pairs/convention are identical to the plain
/// `rope_forward` kernel — only the frequency source differs — so routing a
/// full-attention layer through `rope_forward_yarn` with this table is exact.
fn compute_llama3_inv_freq(
    theta: f32,
    rotary_dim: u32,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_pos: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let table = llama3_inv_freq_table(
        theta,
        rotary_dim,
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_pos,
    );
    let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}

/// Pure CPU computation of the llama3 inv_freq table (see
/// `compute_llama3_inv_freq`). Split out so the frequency transform can be
/// unit-tested without a GPU.
fn llama3_inv_freq_table(
    theta: f32,
    rotary_dim: u32,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_pos: usize,
) -> Vec<f32> {
    let n_pairs = (rotary_dim / 2) as usize;
    let dim = rotary_dim as f64;
    let theta = theta as f64;
    let old_ctx = original_max_pos.max(1) as f64;
    let factor = if factor > 0.0 { factor } else { 1.0 };
    let low_freq_wavelen = old_ctx / low_freq_factor.max(f64::MIN_POSITIVE);
    let high_freq_wavelen = old_ctx / high_freq_factor.max(f64::MIN_POSITIVE);
    let two_pi = 2.0 * std::f64::consts::PI;
    let denom = (high_freq_factor - low_freq_factor).abs().max(1e-9);

    let mut table = vec![0.0f32; n_pairs];
    for (j, slot) in table.iter_mut().enumerate() {
        // base inv_freq computed in f64 to match the plain kernel's FP64
        // frequency path (stored as f32 — identical rounding for the
        // un-scaled high-frequency pairs).
        let base = 1.0 / theta.powf((2 * j) as f64 / dim);
        let wavelen = two_pi / base;
        let scaled = if wavelen > low_freq_wavelen {
            base / factor
        } else if wavelen < high_freq_wavelen {
            base
        } else {
            let smooth = (old_ctx / wavelen - low_freq_factor) / denom;
            (1.0 - smooth) * (base / factor) + smooth * base
        };
        *slot = scaled as f32;
    }
    table
}

#[cfg(test)]
mod llama3_rope_tests {
    use super::llama3_inv_freq_table;

    /// Plain (unscaled) inv_freq for comparison.
    fn plain_inv_freq(theta: f64, rotary_dim: u32) -> Vec<f32> {
        let dim = rotary_dim as f64;
        (0..(rotary_dim / 2) as usize)
            .map(|j| (1.0 / theta.powf((2 * j) as f64 / dim)) as f32)
            .collect()
    }

    /// The llama3 transform must (a) leave the highest-frequency pairs
    /// unchanged, (b) divide the lowest-frequency pairs by `factor`, and
    /// (c) therefore differ from plain RoPE precisely on the long-range
    /// (low-frequency) dimensions the full-attention layers use for
    /// long-context (over-8K) retrieval. This is the Step 3.7 context-length
    /// cliff regression guard: if the scaling is ever dropped again, the
    /// table collapses to `plain` and this test fails.
    #[test]
    fn llama3_scales_low_freq_and_preserves_high_freq() {
        // Step 3.7 full-attention layer: theta=5e6, rotary_dim=64, factor=2.
        let theta = 5_000_000.0f32;
        let rotary_dim = 64u32;
        let factor = 2.0;
        let (low_ff, high_ff, orig) = (1.0, 4.0, 8192);
        let scaled =
            llama3_inv_freq_table(theta, rotary_dim, factor, low_ff, high_ff, orig);
        let plain = plain_inv_freq(theta as f64, rotary_dim);
        assert_eq!(scaled.len(), plain.len());
        assert_eq!(scaled.len(), 32);

        // Highest-frequency pair (j=0, wavelen=2π ≪ high_freq_wavelen):
        // unchanged.
        assert!(
            (scaled[0] - plain[0]).abs() <= plain[0] * 1e-6,
            "high-freq pair 0 must be unchanged: {} vs {}",
            scaled[0],
            plain[0]
        );

        // Lowest-frequency pair (j=31): wavelen huge → divided by factor.
        let last = scaled.len() - 1;
        assert!(
            (scaled[last] - plain[last] / factor as f32).abs() <= plain[last] * 1e-6,
            "low-freq pair {last} must be plain/factor: {} vs {}",
            scaled[last],
            plain[last] / factor as f32
        );

        // The scaling MUST actually change some frequencies — a dropped
        // transform (table == plain) would silently reintroduce the cliff.
        let changed = scaled
            .iter()
            .zip(&plain)
            .filter(|(s, p)| (**s - **p).abs() > **p * 1e-4)
            .count();
        assert!(
            changed >= 8,
            "llama3 must alter the low-frequency band (changed={changed})"
        );

        // Scaled low-frequency inv_freq must be <= plain everywhere
        // (long-range dims rotate slower, extending effective context).
        for (s, p) in scaled.iter().zip(&plain) {
            assert!(*s <= *p * (1.0 + 1e-5), "scaled freq must not exceed plain");
        }
    }

    /// The mechanism behind the >8K content-degeneration cliff: on the
    /// full-attention layers, the RoPE angle a query/key acquires diverges
    /// from the correctly-scaled value by an amount that GROWS with absolute
    /// position. Below ~8K the max divergence is a small fraction of a
    /// radian (tolerable); past ~10K it exceeds a full radian on the
    /// long-range frequency band, corrupting retrieval logits — while the
    /// sliding (local, window=512) layers, and hence the thinking phase,
    /// stay coherent because their positions never span that range.
    #[test]
    fn missing_scaling_causes_large_angle_error_past_8k() {
        let theta = 5_000_000.0f32;
        let rotary_dim = 64u32;
        let scaled = llama3_inv_freq_table(theta, rotary_dim, 2.0, 1.0, 4.0, 8192);
        let plain = plain_inv_freq(theta as f64, rotary_dim);

        // Max per-pair angle divergence |pos·(plain − scaled)| over the band.
        let max_div = |pos: f32| -> f32 {
            scaled
                .iter()
                .zip(&plain)
                .map(|(s, p)| (pos * (p - s)).abs())
                .fold(0.0f32, f32::max)
        };
        let at_2k = max_div(2_000.0);
        let at_10k = max_div(10_000.0);

        // Grows with position (linear in pos) — this is the cliff.
        assert!(at_10k > at_2k * 4.0, "divergence must scale with position");
        // Past 10K it is more than a full radian — enough to scramble the
        // long-range attention dot products the content phase depends on.
        assert!(
            at_10k > 1.0,
            "expected >1 rad max angle divergence at 10K, got {at_10k}"
        );
        // Within the original 8K training window the worst-case divergence
        // stays modest, matching the observed sub-3K "PASS" tests.
        assert!(
            max_div(2_000.0) < 1.0,
            "sub-window divergence should stay bounded, got {at_2k}"
        );
    }
}
