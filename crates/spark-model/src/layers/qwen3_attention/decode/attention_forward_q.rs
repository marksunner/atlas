// SPDX-License-Identifier: AGPL-3.0-only

//! Q projection cluster for single-token decode attention.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_forward_q(
        &self,
        normed: DevicePtr,
        q_out: DevicePtr,
        q_dim: u32,
        q_proj_dim: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            // Q+Gate projection with inline deinterleave (output is [Q_all | Gate_all])
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                // FP8 native: w8a16_gemv + separate deinterleave (no fused QG variant yet)
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    q_out,
                    q_proj_dim,
                    h,
                    stream,
                )?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv_qg(
                    ctx.gpu,
                    self.w4a16_gemv_qg_k,
                    normed,
                    nvfp4,
                    q_out,
                    q_proj_dim,
                    h,
                    nq,
                    hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.attn.q_proj,
                    q_out,
                    q_proj_dim,
                    h,
                    stream,
                )?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        } else {
            // Ungated: Q projection only (no gate)
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    nvfp4,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.attn.q_proj,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            }
        }

        // DIAG: dump normed input and Q output for L0
        if self.attn_layer_idx == 0 && ctx.profile {
            ctx.gpu.synchronize(stream)?;
            let mut input_buf = vec![0u8; 16]; // first 8 BF16 values
            ctx.gpu.copy_d2h(normed, &mut input_buf)?;
            let input_vals: Vec<f32> = input_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let mut q_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(q_out, &mut q_buf)?;
            let q_vals: Vec<f32> = q_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!(
                "GEMV_DIAG L0: input[0:8]={:.4?} q_out[0:8]={:.4?} nq={nq} hd={hd} h={h}",
                input_vals,
                q_vals
            );
        }

        Ok(())
    }
}
