// SPDX-License-Identifier: AGPL-3.0-only

//! FlashAttention, gates, and O projection for cache-skip prefill.

use std::time::Instant;

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

pub(super) struct CacheSkipFlashArgs {
    pub(super) normed: DevicePtr,
    pub(super) qg_out: DevicePtr,
    pub(super) q_contiguous: DevicePtr,
    pub(super) k_contiguous: DevicePtr,
    pub(super) v_contiguous: DevicePtr,
    pub(super) num_tokens: usize,
    pub(super) kv_write_start: usize,
    pub(super) n: u32,
    pub(super) h: u32,
    pub(super) nq: u32,
    pub(super) nkv: u32,
    pub(super) hd: u32,
    pub(super) q_dim: usize,
    pub(super) q_proj_dim: usize,
    pub(super) bf16: usize,
    pub(super) stream: u64,
}

impl Qwen3AttentionLayer {
    pub(super) fn prefill_attention_cache_skip_flash(
        &self,
        ctx: &ForwardContext,
        args: &CacheSkipFlashArgs,
        t0: Option<Instant>,
    ) -> Result<DevicePtr> {
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(args.hd);

        // TurboQuant WHT bookends (mirrors prefill/paged.rs). For turbo
        // dtypes, write_kv_cache WHT-rotated the written range of K/V in
        // place before quantizing it into the cache. Bring prefix-cache hits,
        // Q, and the attention output into the matching basis.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let wht_runtime_active =
            !weight_pre_rotated && (args.hd == 128 || args.hd == 256 || args.hd == 512);
        if wht_runtime_active && args.kv_write_start > 0 && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            let prefix_heads = args.kv_write_start as u32 * args.nkv;
            if k_is_turbo {
                KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                    .grid([prefix_heads, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(args.k_contiguous)
                    .arg_u32(args.hd)
                    .launch(args.stream)?;
            }
            if v_is_turbo {
                KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                    .grid([prefix_heads, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(args.v_contiguous)
                    .arg_u32(args.hd)
                    .launch(args.stream)?;
            }
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([args.n * args.nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(args.q_contiguous)
                .arg_u32(args.hd)
                .launch(args.stream)?;
        }
        if args.hd > 256 && self.prefill_attn_512_k.0 != 0 {
            ops::prefill_attention(
                ctx.gpu,
                self.prefill_attn_512_k,
                args.q_contiguous,
                args.k_contiguous,
                args.v_contiguous,
                attn_out,
                args.n,
                1,
                args.nq,
                args.nkv,
                args.hd,
                inv_sqrt_d,
                true,
                0,
                args.stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "prefill_512 failed: n={} nq={} nkv={} hd={}: {e}",
                    args.n,
                    args.nq,
                    args.nkv,
                    args.hd
                )
            })?;
        } else {
            ops::prefill_attention_64(
                ctx.gpu,
                self.prefill_attn_64_k,
                args.q_contiguous,
                args.k_contiguous,
                args.v_contiguous,
                attn_out,
                args.n,
                1,
                args.nq,
                args.nkv,
                args.hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                args.stream,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "flash_attn_64 failed: n={} nq={} nkv={} hd={}: {e}",
                    args.n,
                    args.nq,
                    args.nkv,
                    args.hd
                )
            })?;
        }

        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k_inv)
                .grid([args.n * args.nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(args.hd)
                .launch(args.stream)?;
        }
        let mut t0 = profile_next(ctx, args.stream, "flash_attn_64", args.n, t0)?;

        if args.num_tokens > 0 {
            let nq_hd = (args.nq * args.hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (args.num_tokens - 1) * nq_hd * args.bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_pre_gate",
                args.stream,
            )?;
        }

        if self.gated {
            let gate_base = args.qg_out.offset(args.q_dim * args.bf16);
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                gate_base,
                attn_out,
                args.nq * args.hd,
                args.q_proj_dim as u32,
                args.n,
                args.stream,
            )?;
        }

        if let Some(ref g_proj) = self.head_gate_weight {
            let gate_buf = args.q_contiguous;
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                args.normed,
                g_proj,
                gate_buf,
                args.n,
                args.nq,
                args.h,
                args.stream,
            )?;
            ops::sigmoid_gate_mul_head_broadcast(
                ctx.gpu,
                self.sigmoid_gate_head_broadcast_k,
                attn_out,
                gate_buf,
                attn_out,
                args.nq,
                args.hd,
                args.n,
                args.stream,
            )?;
        }
        t0 = profile_next(ctx, args.stream, "sigmoid_gate", args.n, t0)?;

        if args.num_tokens > 0 {
            let nq_hd = (args.nq * args.hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (args.num_tokens - 1) * nq_hd * args.bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_post_gate",
                args.stream,
            )?;
        }

        let o_out = self.prefill_attention_paged_oproj(
            attn_out,
            args.n,
            args.h,
            args.nq,
            args.hd,
            ctx,
            args.stream,
        )?;
        profile_log(ctx, args.stream, "o_proj", args.n, t0)?;
        Ok(o_out)
    }
}

fn profile_next(
    ctx: &ForwardContext,
    stream: u64,
    label: &str,
    n: u32,
    t0: Option<Instant>,
) -> Result<Option<Instant>> {
    if ctx.profile {
        if let Some(t0) = t0 {
            ctx.gpu.synchronize(stream)?;
            let elapsed = t0.elapsed().as_micros();
            tracing::info!("  ATTN prefill [{}] N={}: {}µs", label, n, elapsed);
        }
        ctx.gpu.synchronize(stream)?;
        Ok(Some(Instant::now()))
    } else {
        Ok(None)
    }
}

fn profile_log(
    ctx: &ForwardContext,
    stream: u64,
    label: &str,
    n: u32,
    t0: Option<Instant>,
) -> Result<()> {
    if ctx.profile
        && let Some(t0) = t0
    {
        ctx.gpu.synchronize(stream)?;
        let elapsed = t0.elapsed().as_micros();
        tracing::info!("  ATTN prefill [{}] N={}: {}µs", label, n, elapsed);
    }
    Ok(())
}
