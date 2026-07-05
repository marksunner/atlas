// SPDX-License-Identifier: AGPL-3.0-only
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;

impl TransformerModel {
    /// Read back first `n` BF16 values from device and return as f32 + L2 norm.
    pub(super) fn readback_bf16(&self, ptr: DevicePtr, n: usize) -> Result<(Vec<f32>, f32)> {
        let bytes = n * 2;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        let vals: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect();
        let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok((vals, norm))
    }

    /// Read FP32 values from GPU memory (for FP32 residual stream diagnostics).
    pub(super) fn readback_f32(&self, ptr: DevicePtr, n: usize) -> Result<(Vec<f32>, f32)> {
        let bytes = n * 4;
        let mut buf = vec![0u8; bytes];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        let vals: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let norm = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        Ok((vals, norm))
    }
}
