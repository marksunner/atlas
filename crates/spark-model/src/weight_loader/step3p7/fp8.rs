// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 block-scaled expert loading for the Step 3.7 Flash FP8 release.
//!
//! The FP8 checkpoint (`Step-3.7-Flash-FP8`) differs from the NVFP4 one:
//!   * Routed experts are fused 3-D FP8 tensors per projection:
//!     `{lp}.moe.{gate,up,down}_proj.weight`        F8_E4M3 [E, N, K]
//!     `{lp}.moe.{gate,up,down}_proj.weight_scale_inv` F32   [E, N/128, K/128]
//!     (DeepSeek-style 128x128 block scales, dequant = fp8 * scale_inv).
//!   * Attention, dense FFN, router gate, shared expert, norms, embeddings
//!     and lm_head are all BF16 — identical to the NVFP4 checkpoint, so
//!     those loader paths are untouched. Only routed-expert precision
//!     changes, which is exactly the isolation the NVFP4-quality experiment
//!     needs.
//!   * Text-stack tensors live under `model.layers.*` (no
//!     `model.language_model.` nesting) — handled by prefix probing in the
//!     main loader, not here.
//!
//! The shared expert ships BF16 but the fused FP8 MoE kernels compute
//! routed + shared in one launch and expect FP8 block-scaled shared
//! weights, so we quantize it BF16 -> FP8-blockscaled on the CPU at load
//! (~0.7 GB total across 42 layers; FP8 is strictly higher precision than
//! the NVFP4 the shared expert gets on the NVFP4 path).

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_map::{Fp8ExpertWeight, Fp8Weight, WeightQuantFormat};

const BLOCK: usize = 128;
/// Largest finite FP8 E4M3FN magnitude.
const E4M3_MAX: f32 = 448.0;

/// Null placeholder for remote experts under EP. Kernels check for NULL
/// weight pointers and write zero output; the tag is conventional.
fn null_fp8() -> Fp8Weight {
    Fp8Weight {
        weight: DevicePtr::NULL,
        row_scale: DevicePtr::NULL,
        n: 0,
        k: 0,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// Whether this layer's MoE ships fused FP8 block-scaled experts.
pub(super) fn has_fused_fp8_experts(store: &WeightStore, moe_prefix: &str) -> bool {
    store.contains(&format!("{moe_prefix}.gate_proj.weight_scale_inv"))
}

/// Load the fused FP8 expert tensors for one MoE layer and slice them into
/// per-expert [`Fp8ExpertWeight`] entries (NULL for remote experts under EP).
///
/// The WeightStore has already EP-sliced both `.weight` and
/// `.weight_scale_inv` on the expert (leading) dimension, so offsets here
/// are LOCAL expert indices into the loaded slices.
pub(super) fn load_fused_fp8_experts(
    store: &WeightStore,
    moe_prefix: &str,
    config: &ModelConfig,
) -> Result<Vec<Fp8ExpertWeight>> {
    let (local_start, local_end) = config.local_expert_range();
    let local_count = local_end - local_start;

    let load_proj = |proj: &str| -> Result<Vec<Fp8Weight>> {
        let w_key = format!("{moe_prefix}.{proj}.weight");
        let s_key = format!("{moe_prefix}.{proj}.weight_scale_inv");
        let w = store.get(&w_key)?;
        let s = store.get(&s_key)?;
        ensure!(
            w.dtype == WeightDtype::FP8E4M3,
            "Expected FP8E4M3 for {w_key}, got {:?}",
            w.dtype
        );
        ensure!(
            s.dtype == WeightDtype::FP32,
            "Expected FP32 for {s_key}, got {:?}",
            s.dtype
        );
        ensure!(
            w.shape.len() == 3 && s.shape.len() == 3,
            "Expected 3D fused expert tensors for {w_key}: weight {:?}, scale {:?}",
            w.shape,
            s.shape
        );
        let (e_w, n, k) = (w.shape[0], w.shape[1], w.shape[2]);
        ensure!(
            e_w == local_count && s.shape[0] == local_count,
            "{w_key}: expert dim mismatch (weight {e_w}, scale {}, expected local {local_count})",
            s.shape[0]
        );
        let (sn, sk) = (s.shape[1], s.shape[2]);
        ensure!(
            sn == n.div_ceil(BLOCK) && sk == k.div_ceil(BLOCK),
            "{s_key}: scale shape [{sn},{sk}] doesn't match weight [{n},{k}] at 128x128 blocks"
        );
        let weight_bytes_per_expert = n * k; // 1 byte per FP8 element
        let scale_bytes_per_expert = sn * sk * 4; // F32
        Ok((0..local_count)
            .map(|le| Fp8Weight {
                weight: w.ptr.offset(le * weight_bytes_per_expert),
                row_scale: s.ptr.offset(le * scale_bytes_per_expert),
                n: n as u32,
                k: k as u32,
                scale_format: WeightQuantFormat::Fp8BlockScaled,
            })
            .collect())
    };

    let gate = load_proj("gate_proj")?;
    let up = load_proj("up_proj")?;
    let down = load_proj("down_proj")?;

    Ok((0..config.num_experts)
        .map(|e| {
            if e >= local_start && e < local_end {
                let le = e - local_start;
                Fp8ExpertWeight {
                    gate_proj: gate[le],
                    up_proj: up[le],
                    down_proj: down[le],
                }
            } else {
                Fp8ExpertWeight {
                    gate_proj: null_fp8(),
                    up_proj: null_fp8(),
                    down_proj: null_fp8(),
                }
            }
        })
        .collect())
}

/// Quantize a BF16 checkpoint tensor `[N, K]` to FP8 E4M3 with 128x128
/// FP32 block scales (scale = block_absmax / 448, dequant = fp8 * scale)
/// and upload weight + scales to the GPU as an [`Fp8Weight`].
///
/// CPU round-trip is load-time only (shared experts: ~16 MB per matrix).
pub(super) fn quantize_dense_to_fp8_blockscaled(
    store: &WeightStore,
    key: &str,
    gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let t = store.get(key)?;
    ensure!(
        t.dtype == WeightDtype::BF16,
        "Expected BF16 for {key}, got {:?}",
        t.dtype
    );
    ensure!(t.shape.len() == 2, "Expected 2D tensor for {key}, got {:?}", t.shape);
    let (n, k) = (t.shape[0], t.shape[1]);

    let mut bf16 = vec![0u8; n * k * 2];
    gpu.copy_d2h(t.ptr, &mut bf16)
        .with_context(|| format!("d2h for {key}"))?;

    let (fp8, scales) = quantize_bf16_slice_blockscaled(&bf16, n, k);

    let weight = gpu.alloc(fp8.len())?;
    gpu.copy_h2d(&fp8, weight)?;
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
    let row_scale = gpu.alloc(scale_bytes.len())?;
    gpu.copy_h2d(&scale_bytes, row_scale)?;

    Ok(Fp8Weight {
        weight,
        row_scale,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    })
}

/// Pure-CPU BF16 -> (FP8 bytes [N*K], FP32 block scales [ceil(N/128)*ceil(K/128)]).
/// Split out for unit testing without a GPU.
fn quantize_bf16_slice_blockscaled(bf16: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let sn = n.div_ceil(BLOCK);
    let sk = k.div_ceil(BLOCK);
    let mut fp8 = vec![0u8; n * k];
    let mut scales = vec![1.0f32; sn * sk];

    let at = |i: usize, j: usize| -> f32 {
        let idx = (i * k + j) * 2;
        let bits = u16::from_le_bytes([bf16[idx], bf16[idx + 1]]);
        f32::from_bits((bits as u32) << 16)
    };

    for bi in 0..sn {
        for bj in 0..sk {
            let i_end = ((bi + 1) * BLOCK).min(n);
            let j_end = ((bj + 1) * BLOCK).min(k);
            let mut absmax = 0.0f32;
            for i in (bi * BLOCK)..i_end {
                for j in (bj * BLOCK)..j_end {
                    absmax = absmax.max(at(i, j).abs());
                }
            }
            let scale = if absmax > 0.0 { absmax / E4M3_MAX } else { 1.0 };
            scales[bi * sk + bj] = scale;
            let inv = 1.0 / scale;
            for i in (bi * BLOCK)..i_end {
                for j in (bj * BLOCK)..j_end {
                    fp8[i * k + j] = f32_to_e4m3(at(i, j) * inv);
                }
            }
        }
    }
    (fp8, scales)
}

/// Convert f32 to FP8 E4M3FN with round-to-nearest-even, saturating to
/// +-448. NaN maps to 0 (matches the Atlas decode LUT, which maps the
/// E4M3 NaN encodings to 0.0).
fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a >= 464.0 {
        // Beyond the 448 / 480 midpoint (480 = would-be next step): saturate.
        return sign | 0x7E;
    }

    // Exponent of the E4M3 binade containing `a`, clamped to the denormal
    // floor at 2^-6. f32 exponent extraction is exact for the range here.
    let mut e = {
        let bits = a.to_bits();
        (((bits >> 23) & 0xFF) as i32) - 127
    };
    if e < -6 {
        e = -6;
    }
    // Quantum for this binade: 2^(e-3) (3 mantissa bits).
    let step = (2.0f32).powi(e - 3);
    // f32 has far more mantissa precision than needed here, so the divide
    // is exact enough that ties survive; round ties-to-even explicitly.
    let q = round_ties_even(a / step);
    let mut q = q as i32; // normals: 8..=16; denormals (e==-6): 0..=16
    let mut e_out = e;
    if q == 16 {
        // Mantissa overflow -> next binade.
        q = 8;
        e_out += 1;
    }
    if e_out > 8 || (e_out == 8 && q > 14) {
        return sign | 0x7E; // saturate (448 = 1.75 * 2^8 -> q=14)
    }
    if q == 0 {
        return sign; // rounded to zero
    }
    let byte = if e_out == -6 && q < 8 {
        // Denormal: exp field 0, mantissa = q (value q/8 * 2^-6).
        q as u8
    } else {
        let exp_field = (e_out + 7) as u8; // 1..=15
        (exp_field << 3) | ((q - 8) as u8)
    };
    sign | byte
}

fn round_ties_even(x: f32) -> f32 {
    let r = x.round(); // half away from zero
    if (x - x.trunc()).abs() == 0.5 && r.rem_euclid(2.0) == 1.0 {
        r - x.signum()
    } else {
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference E4M3FN decode (mirrors atlas-quant's LUT semantics:
    /// NaN encodings -> 0).
    fn e4m3_to_f32(bits: u8) -> f32 {
        let sign = if bits & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = (bits >> 3) & 0x0F;
        let mant = bits & 0x07;
        if exp == 0x0F && mant == 0x07 {
            return 0.0; // NaN encoding
        }
        if exp == 0 {
            sign * (mant as f32 / 8.0) * (2.0f32).powi(-6)
        } else {
            sign * (1.0 + mant as f32 / 8.0) * (2.0f32).powi(exp as i32 - 7)
        }
    }

    /// Brute-force nearest E4M3 value (ties to even mantissa code).
    fn nearest_ref(x: f32) -> f32 {
        let mut best = 0.0f32;
        let mut best_d = f32::INFINITY;
        for b in 0u16..=255 {
            let b = b as u8;
            if (b & 0x7F) == 0x7F {
                continue; // NaN encodings
            }
            let v = e4m3_to_f32(b);
            let d = (v - x).abs();
            if d < best_d {
                best_d = d;
                best = v;
            }
        }
        best
    }

    #[test]
    fn encode_round_trips_every_e4m3_value() {
        for b in 0u16..=255 {
            let b = b as u8;
            if (b & 0x7F) == 0x7F {
                continue; // NaN encodings unreachable from finite input
            }
            let v = e4m3_to_f32(b);
            let enc = f32_to_e4m3(v);
            assert_eq!(
                e4m3_to_f32(enc),
                v,
                "round-trip failed for {b:#04x} (value {v}): got {enc:#04x}"
            );
        }
    }

    #[test]
    fn encode_matches_bruteforce_nearest() {
        // Deterministic pseudo-random sweep over the E4M3 dynamic range.
        let mut state = 0x12345678u32;
        for _ in 0..20_000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
            let mag = 500.0 * u * u * u; // bias toward small magnitudes
            let x = if state & 1 == 0 { mag } else { -mag };
            let got = e4m3_to_f32(f32_to_e4m3(x));
            let want = nearest_ref(x);
            // Ties may legitimately differ in which neighbor is chosen only
            // if both are equidistant; accept either equidistant neighbor.
            let d_got = (got - x).abs();
            let d_want = (want - x).abs();
            assert!(
                d_got <= d_want + d_want.abs() * 1e-6,
                "not nearest for x={x}: got {got} (d={d_got}), want {want} (d={d_want})"
            );
        }
    }

    #[test]
    fn encode_saturates_and_handles_edges() {
        assert_eq!(e4m3_to_f32(f32_to_e4m3(1000.0)), 448.0);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(-1000.0)), -448.0);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(448.0)), 448.0);
        assert_eq!(f32_to_e4m3(0.0), 0x00);
        assert_eq!(f32_to_e4m3(-0.0), 0x80);
        assert_eq!(f32_to_e4m3(f32::NAN), 0x00);
        // Denormal floor: 2^-9 is the smallest nonzero E4M3 step / 2... the
        // smallest denormal is 2^-6/8 = 2^-9.
        let tiny = (2.0f32).powi(-9);
        assert_eq!(e4m3_to_f32(f32_to_e4m3(tiny)), tiny);
        // Below half the smallest denormal -> rounds to zero.
        assert_eq!(f32_to_e4m3(tiny * 0.49), 0x00);
    }

    #[test]
    fn block_quantize_reconstructs_within_e4m3_error() {
        // 256x256 synthetic matrix spanning several block scales.
        let (n, k) = (256usize, 256usize);
        let mut bf16 = vec![0u8; n * k * 2];
        let mut vals = vec![0.0f32; n * k];
        let mut state = 0xdeadbeefu32;
        for i in 0..n {
            for j in 0..k {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let u = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                // Per-block magnitude varies 100x to exercise scale logic.
                let block_mag = 0.01 * (1.0 + 99.0 * ((i / 128) * 2 + (j / 128)) as f32 / 3.0);
                let v = u * block_mag;
                // Store as BF16 (truncate like the checkpoint would).
                let vb = ((v.to_bits() + 0x8000) >> 16) as u16;
                let v_bf16 = f32::from_bits((vb as u32) << 16);
                vals[i * k + j] = v_bf16;
                bf16[(i * k + j) * 2..(i * k + j) * 2 + 2].copy_from_slice(&vb.to_le_bytes());
            }
        }
        let (fp8, scales) = quantize_bf16_slice_blockscaled(&bf16, n, k);
        assert_eq!(scales.len(), 4);
        for i in 0..n {
            for j in 0..k {
                let scale = scales[(i / 128) * 2 + (j / 128)];
                let deq = e4m3_to_f32(fp8[i * k + j]) * scale;
                let v = vals[i * k + j];
                // E4M3 error bound per element: half the local quantum.
                // Normals: <= |v|/16 relative; below the denormal floor the
                // absolute quantum is scale * 2^-9 (smallest denormal).
                let bound = f32::max(scale * (2.0f32).powi(-10), v.abs() / 16.0) * 1.01;
                assert!(
                    (deq - v).abs() <= bound,
                    "dequant error too large at [{i},{j}]: v={v}, deq={deq}, bound={bound}"
                );
            }
        }
    }
}
