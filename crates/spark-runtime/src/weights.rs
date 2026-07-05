// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loading from safetensors files (SBIO IORouter for filesystem I/O).

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::Path;

/// Advise the OS to evict a file's pages from the page cache.
///
/// On GB10 (unified memory), mmap'd safetensors share the GPU memory pool.
/// After copying tensors to GPU, the mmap pages linger in the page cache,
/// consuming memory that should be available for KV cache and inference buffers.
/// This function tells the kernel those pages are no longer needed.
#[cfg(target_os = "linux")]
pub(crate) fn evict_page_cache(file: &std::fs::File) {
    use std::os::unix::io::AsRawFd;
    // POSIX_FADV_DONTNEED = 4 on Linux (POSIX standard).
    // macOS lacks posix_fadvise — see the non-linux branch below.
    const POSIX_FADV_DONTNEED: libc::c_int = 4;
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn evict_page_cache(_file: &std::fs::File) {
    // No-op: macOS/BSD have no posix_fadvise. Apple Silicon UMA already
    // shares page cache with the GPU pool, so eviction is unnecessary.
}

/// Data type of a weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    BF16,
    FP32,
    FP8E4M3,
    UInt8,
}

impl WeightDtype {
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 => 2,
            Self::FP32 => 4,
            Self::FP8E4M3 => 1,
            Self::UInt8 => 1,
        }
    }

    fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self> {
        match dtype {
            safetensors::Dtype::BF16 => Ok(Self::BF16),
            safetensors::Dtype::F32 => Ok(Self::FP32),
            safetensors::Dtype::U8 => Ok(Self::UInt8),
            safetensors::Dtype::F8_E4M3 => Ok(Self::FP8E4M3),
            other => bail!("Unsupported safetensors dtype: {other:?}"),
        }
    }
}

/// A weight tensor on the GPU.
pub struct WeightTensor {
    pub ptr: DevicePtr,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl WeightTensor {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.byte_size()
    }
}

/// All model weights loaded onto the GPU, keyed by HuggingFace name.
pub struct WeightStore {
    weights: HashMap<String, WeightTensor>,
}

impl WeightStore {
    /// Create an empty weight store (for testing).
    pub fn empty() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Crate-internal: wrap a pre-built map. Used by alternate loaders
    /// (e.g. `fast_weights::FastSafetensorsLoader`).
    pub(crate) fn from_map(weights: HashMap<String, WeightTensor>) -> Self {
        Self { weights }
    }

    /// Get a weight tensor by name. Fails fast if not found.
    pub fn get(&self, name: &str) -> Result<&WeightTensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight '{name}' not found in store"))
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Number of loaded weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True if no weights are loaded.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Iterator over all weight names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(|s| s.as_str())
    }

    /// Total bytes across all weight tensors on the GPU.
    pub fn total_bytes(&self) -> usize {
        self.weights.values().map(|w| w.byte_size()).sum()
    }

    /// Check if any tensor has FP8 dtype.
    pub fn has_fp8_weights(&self) -> bool {
        self.weights
            .values()
            .any(|w| matches!(w.dtype, WeightDtype::FP8E4M3))
    }
}

/// SBIO IORouter trait for weight loading.
pub trait WeightLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore>;
}

/// Loads weights from safetensors files using mmap.
pub struct SafetensorsLoader {
    /// EP rank (0-based). Only used when ep_world_size > 1.
    pub ep_rank: usize,
    /// EP world size. When > 1, remote expert tensors are skipped.
    pub ep_world_size: usize,
    /// Total number of MoE experts in the model (for EP partitioning).
    pub num_experts: usize,
    /// Override for the peak memory multiplier in the pre-flight OOM check.
    /// Set from QuantFormat::peak_memory_multiplier() in the caller.
    /// When None, the pre-flight uses its own heuristic (1.3x NVFP4 / 1.5x FP8).
    pub peak_memory_multiplier: Option<f64>,
}

impl Default for SafetensorsLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetensorsLoader {
    /// Create a loader with no expert parallelism (loads all tensors).
    pub fn new() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
            peak_memory_multiplier: None,
        }
    }

    /// Create a loader with EP-aware filtering.
    pub fn with_ep(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
            peak_memory_multiplier: None,
        }
    }

    /// Check if a tensor should be skipped under EP.
    /// Skips `*.experts.{E}.*` tensors where E is not in local range.
    /// MTP head experts are never skipped (small, fully replicated).
    fn should_skip_tensor(&self, name: &str) -> bool {
        // Operator-directed skip list: `ATLAS_SKIP_TENSOR_PREFIXES` is a
        // comma-separated list of tensor-name prefixes to drop at load
        // (e.g. unused MTP modules / vision tower when memory is tight).
        // Applies regardless of EP mode.
        if skip_tensor_prefixes()
            .iter()
            .any(|p| name.starts_with(p.as_str()))
        {
            return true;
        }
        if self.ep_world_size <= 1 {
            return false;
        }
        // MTP head experts are small — always replicate, never shard.
        if name.starts_with("mtp.") {
            return false;
        }
        // Parse expert index from patterns like "*.experts.42.gate_proj*"
        if let Some(idx) = parse_expert_index(name) {
            let per_rank = self.num_experts / self.ep_world_size;
            let local_start = self.ep_rank * per_rank;
            let local_end = if self.ep_rank == self.ep_world_size - 1 {
                self.num_experts
            } else {
                local_start + per_rank
            };
            idx < local_start || idx >= local_end
        } else {
            false // Non-expert tensors are always loaded (replicated)
        }
    }
}

/// Prefixes from `ATLAS_SKIP_TENSOR_PREFIXES`, parsed once per process.
fn skip_tensor_prefixes() -> &'static [String] {
    static PREFIXES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    PREFIXES.get_or_init(|| {
        std::env::var("ATLAS_SKIP_TENSOR_PREFIXES")
            .map(|v| {
                v.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Parse expert index from tensor name (e.g. "model.layers.3.mlp.experts.42.gate_proj.weight" → 42).
pub(crate) fn parse_expert_index(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if *part == "experts" && i + 1 < parts.len() {
            return parts[i + 1].parse().ok();
        }
    }
    None
}

fn is_fused_moe_proj(name: &str) -> bool {
    const PROJS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
    // `weight_scale` = NVFP4 fused group scales; `weight_scale_inv` = FP8
    // block scales ([E, N/128, K/128] — expert-major, so slicing dim 0 is
    // correct for it too).
    const SUFFIXES: [&str; 4] = ["weight", "weight_scale", "weight_scale_inv", "input_scale"];
    for suffix in SUFFIXES {
        let Some(rest) = name.strip_suffix(suffix).and_then(|r| r.strip_suffix('.')) else {
            continue;
        };
        for proj in PROJS {
            if rest
                .strip_suffix(proj)
                .is_some_and(|head| head.ends_with(".moe."))
            {
                return true;
            }
        }
    }
    false
}

#[derive(Clone, Copy)]
pub(crate) struct EpShard {
    pub ep_rank: usize,
    pub ep_world_size: usize,
    pub num_experts: usize,
}

impl EpShard {
    pub fn inactive() -> Self {
        Self {
            ep_rank: 0,
            ep_world_size: 1,
            num_experts: 0,
        }
    }

    pub fn from_parts(ep_rank: usize, ep_world_size: usize, num_experts: usize) -> Self {
        Self {
            ep_rank,
            ep_world_size,
            num_experts,
        }
    }

    fn active(self) -> bool {
        self.ep_world_size > 1 && self.num_experts > 0
    }

    fn local_range(self) -> (usize, usize) {
        let per_rank = self.num_experts / self.ep_world_size;
        let start = self.ep_rank * per_rank;
        let end = if self.ep_rank + 1 == self.ep_world_size {
            self.num_experts
        } else {
            start + per_rank
        };
        (start, end)
    }

    pub fn fused_slice(
        self,
        name: &str,
        shape: &[usize],
        full_len: usize,
    ) -> Option<(usize, usize, Vec<usize>)> {
        if !self.active() || !is_fused_moe_proj(name) || name.starts_with("mtp.") {
            return None;
        }
        let rows = *shape.first()?;
        if rows == 0 || !rows.is_multiple_of(self.num_experts) || !full_len.is_multiple_of(rows) {
            return None;
        }
        let (start, end) = self.local_range();
        let rows_per_expert = rows / self.num_experts;
        let bytes_per_row = full_len / rows;
        let local_rows = end.saturating_sub(start) * rows_per_expert;
        let sub_offset = start * rows_per_expert * bytes_per_row;
        let sub_len = local_rows * bytes_per_row;
        let mut local_shape = shape.to_vec();
        local_shape[0] = local_rows;
        Some((sub_offset, sub_len, local_shape))
    }

    pub fn local_bytes(self, name: &str, shape: &[usize], full_len: usize) -> usize {
        match self.fused_slice(name, shape, full_len) {
            Some((_, sub_len, _)) => sub_len,
            None => full_len,
        }
    }
}

mod loader;
pub mod mlx_int8;
pub(crate) use loader::{check_oom_guard, estimate_has_fp8, estimate_load_bytes};
