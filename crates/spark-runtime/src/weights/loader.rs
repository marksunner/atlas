// SPDX-License-Identifier: AGPL-3.0-only

//! `impl WeightLoader for SafetensorsLoader` + sharded/single loader helpers.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;

use super::{EpShard, SafetensorsLoader, WeightLoader, WeightStore};
use crate::gpu::GpuBackend;

impl WeightLoader for SafetensorsLoader {
    fn load(
        &self,
        model_dir: &Path,
        gpu: &dyn GpuBackend,
        oom_reserve_bytes: usize,
    ) -> Result<WeightStore> {
        let skip_fn = |name: &str| self.should_skip_tensor(name);

        // Collect all safetensor files (indexed, single, or unindexed shards).
        // Supports both HuggingFace standard (model.safetensors*) and Mistral
        // consolidated format (consolidated.safetensors*).
        let index_path = model_dir.join("model.safetensors.index.json");
        let consolidated_index = model_dir.join("consolidated.safetensors.index.json");
        let shard_files: Vec<std::path::PathBuf>;
        let use_index;
        let actual_index_path;

        if index_path.exists() {
            use_index = true;
            actual_index_path = index_path;
            shard_files = vec![];
        } else if consolidated_index.exists() {
            use_index = true;
            actual_index_path = consolidated_index;
            shard_files = vec![];
        } else {
            use_index = false;
            actual_index_path = index_path; // unused
            let single = model_dir.join("model.safetensors");
            if single.exists() {
                shard_files = vec![single];
            } else {
                // Try both model.safetensors-* and consolidated-* shard patterns
                let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                            (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                                && n.ends_with(".safetensors")
                        })
                    })
                    .collect();
                shards.sort();
                if shards.is_empty() {
                    bail!(
                        "No safetensor files found in {}. Expected model.safetensors*, \
                         consolidated.safetensors*, or consolidated-*-of-*.safetensors",
                        model_dir.display()
                    );
                }
                shard_files = shards;
            }
        }

        // Pre-flight OOM estimate: scan safetensor headers (no data) to compute
        // total bytes this rank will load, then apply a model-building overhead
        // multiplier and abort early if the model won't fit.
        //
        // Model building creates additional GPU allocations on top of the raw
        // weight store: transposed weight copies for prefill GEMM, predequanted
        // FP8 copies, NVFP4 quantized copies (for FP8 checkpoints), and transient
        // BF16 intermediates during FP8→NVFP4 conversion.
        //
        // Empirical overhead multipliers (peak memory / on-disk weight bytes):
        //   NVFP4 (Sehyo): ~2.0x  (store aliased + transposed/predequant copies)
        //   FP8 native:    ~1.5x  (store stays FP8, only attention prefill gets NVFP4 copies)
        let ep = EpShard::from_parts(self.ep_rank, self.ep_world_size, self.num_experts);
        {
            let estimated = estimate_load_bytes(&shard_files, &skip_fn, ep)?;
            let has_fp8 = estimate_has_fp8(&shard_files, &skip_fn, ep)?;
            let overhead_multiplier: f64 =
                self.peak_memory_multiplier
                    .unwrap_or(if has_fp8 { 1.5 } else { 1.3 });
            let peak_estimated = (estimated as f64 * overhead_multiplier) as usize;
            let free = gpu.free_memory()?;
            let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
            let est_gb = estimated as f64 / (1024.0 * 1024.0 * 1024.0);
            let peak_gb = peak_estimated as f64 / (1024.0 * 1024.0 * 1024.0);
            let reserve_gb = oom_reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!(
                "Pre-flight estimate: {:.2} GB on-disk weights, {:.1}x overhead = {:.2} GB peak, \
                 {:.2} GB free, {:.1} GB reserve (FP8: {})",
                est_gb,
                overhead_multiplier,
                peak_gb,
                free_gb,
                reserve_gb,
                has_fp8,
            );
            if peak_estimated + oom_reserve_bytes > free {
                bail!(
                    "OOM pre-flight: model peak memory ({:.2} GB = {:.2} GB weights × {:.1}x \
                     model-building overhead) + {:.1} GB reserve = {:.2} GB, \
                     but only {:.2} GB GPU memory is available. \
                     This model is too large. Use a smaller quantization (NVFP4 instead of FP8) \
                     or add more GPUs for expert parallelism.",
                    peak_gb,
                    est_gb,
                    overhead_multiplier,
                    reserve_gb,
                    peak_gb + reserve_gb,
                    free_gb,
                );
            }
        }

        let mut weight_map = if use_index {
            load_sharded(
                model_dir,
                &actual_index_path,
                gpu,
                oom_reserve_bytes,
                &skip_fn,
                self.peak_memory_multiplier,
                ep,
            )?
        } else if shard_files.len() == 1 {
            load_single(&shard_files[0], gpu, oom_reserve_bytes, &skip_fn, ep)?
        } else {
            tracing::info!("Loading {} unindexed safetensor shards", shard_files.len());
            let initial_free = gpu.free_memory()?;
            let mut combined = HashMap::new();
            for (i, shard) in shard_files.iter().enumerate() {
                let map = load_single(shard, gpu, oom_reserve_bytes, &skip_fn, ep)?;
                let free_now = gpu.free_memory().unwrap_or(0);
                let used = initial_free.saturating_sub(free_now);
                tracing::info!(
                    "  Shard {}/{} done — GPU memory: {:.2} GB used, {:.2} GB free",
                    i + 1,
                    shard_files.len(),
                    used as f64 / (1024.0 * 1024.0 * 1024.0),
                    free_now as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                check_oom_guard(
                    gpu,
                    oom_reserve_bytes,
                    &format!("weight loading (shard {}/{})", i + 1, shard_files.len()),
                )?;
                combined.extend(map);
            }
            combined
        };

        // Load extra weight files (e.g. MTP weights grafted from another quantization).
        // Extra weights (MTP) are always fully loaded — they have their own expert lists.
        let no_skip = |_: &str| false;
        let extra = model_dir.join("extra_weights.safetensors");
        if extra.exists() {
            let extra_weights = load_single(
                &extra,
                gpu,
                oom_reserve_bytes,
                &no_skip,
                EpShard::inactive(),
            )?;
            tracing::info!(
                "Loaded {} extra weight tensors from extra_weights.safetensors",
                extra_weights.len()
            );
            weight_map.extend(extra_weights);
        }

        Ok(WeightStore {
            weights: weight_map,
        })
    }
}

/// Index file format: { "weight_map": { "tensor_name": "shard_filename" } }
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Read only the safetensor header from a file (no mmap, no GPU memory impact).
/// The header is typically a few KB of JSON — safe to read on GB10 unified memory
/// without consuming GPU pages.
pub(crate) fn read_safetensor_header(
    path: &Path,
) -> Result<Vec<(String, Vec<usize>, safetensors::Dtype)>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Pre-flight: failed to open {}", path.display()))?;

    // Safetensors format: 8-byte LE header size, then JSON header, then data.
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;

    // Sanity check: header shouldn't exceed 64 MB.
    if header_size > 64 * 1024 * 1024 {
        bail!(
            "Safetensor header too large ({} bytes) in {}",
            header_size,
            path.display()
        );
    }

    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;

    // Parse the JSON header manually to extract tensor metadata.
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;
    let obj = header.as_object().context("Invalid safetensor header")?;

    let mut tensors = Vec::new();
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
        let dtype = match dtype_str {
            "F32" => safetensors::Dtype::F32,
            "F16" => safetensors::Dtype::F16,
            "BF16" => safetensors::Dtype::BF16,
            "I32" => safetensors::Dtype::I32,
            "I16" => safetensors::Dtype::I16,
            "I8" => safetensors::Dtype::I8,
            "U8" => safetensors::Dtype::U8,
            "F8_E4M3" => safetensors::Dtype::F8_E4M3,
            "F8_E5M2" => safetensors::Dtype::F8_E5M2,
            _ => safetensors::Dtype::BF16,
        };
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        tensors.push((name.clone(), shape, dtype));
    }
    Ok(tensors)
}

/// Scan safetensor file headers (metadata only, no data loaded) to estimate
/// total GPU bytes this rank will load. Reads only the JSON header from each
/// file — does NOT mmap, so it's safe on GB10 unified memory.
pub(crate) fn estimate_load_bytes(
    files: &[std::path::PathBuf],
    skip_fn: &dyn Fn(&str) -> bool,
    ep: EpShard,
) -> Result<usize> {
    let mut total = 0usize;
    for path in files {
        for (name, shape, dtype) in read_safetensor_header(path)? {
            if skip_fn(&name) {
                continue;
            }
            total += estimate_tensor_bytes(&name, &shape, dtype, ep);
        }
    }
    Ok(total)
}

fn estimate_tensor_bytes(
    name: &str,
    shape: &[usize],
    dtype: safetensors::Dtype,
    ep: EpShard,
) -> usize {
    let numel: usize = shape.iter().product();
    let elem_bytes = dtype_byte_size(dtype);
    let bytes = numel * elem_bytes;
    ep.local_bytes(name, shape, bytes)
}

fn dtype_byte_size(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F16
        | safetensors::Dtype::BF16
        | safetensors::Dtype::I16
        | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::I8
        | safetensors::Dtype::U8
        | safetensors::Dtype::F8_E4M3
        | safetensors::Dtype::F8_E5M2 => 1,
        _ => 2,
    }
}

/// Check if the model is predominantly FP8 (>50% of weight bytes are FP8).
/// Sehyo NVFP4 models have a few FP8 scale tensors but the bulk is uint8 (NVFP4 packed).
/// True FP8 checkpoints (e.g. Qwen/Qwen3.5-122B-A10B-FP8) have most bytes as FP8.
pub(crate) fn estimate_has_fp8(
    files: &[std::path::PathBuf],
    skip_fn: &dyn Fn(&str) -> bool,
    ep: EpShard,
) -> Result<bool> {
    let mut fp8_bytes = 0usize;
    let mut total_bytes = 0usize;
    for path in files {
        for (name, shape, dtype) in read_safetensor_header(path)? {
            if skip_fn(&name) {
                continue;
            }
            let numel: usize = shape.iter().product();
            let elem_bytes = match dtype {
                safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
                safetensors::Dtype::F16
                | safetensors::Dtype::BF16
                | safetensors::Dtype::I16
                | safetensors::Dtype::U16 => 2,
                safetensors::Dtype::I8
                | safetensors::Dtype::U8
                | safetensors::Dtype::F8_E4M3
                | safetensors::Dtype::F8_E5M2 => 1,
                _ => 2,
            };
            let bytes = ep.local_bytes(&name, &shape, numel * elem_bytes);
            total_bytes += bytes;
            if matches!(
                dtype,
                safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2
            ) {
                fp8_bytes += bytes;
            }
        }
    }
    let fp8_frac = if total_bytes > 0 {
        fp8_bytes as f64 / total_bytes as f64
    } else {
        0.0
    };
    tracing::debug!(
        "FP8 fraction: {:.1}% ({} / {} bytes)",
        fp8_frac * 100.0,
        fp8_bytes,
        total_bytes
    );
    Ok(fp8_frac > 0.5)
}

pub(crate) fn check_oom_guard(
    gpu: &dyn GpuBackend,
    reserve_bytes: usize,
    phase: &str,
) -> Result<()> {
    let free = gpu.free_memory()?;
    if free < reserve_bytes {
        let free_gb = free as f64 / (1024.0 * 1024.0 * 1024.0);
        let reserve_gb = reserve_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        bail!(
            "OOM guard: aborting during {phase}. \
             Free GPU memory ({free_gb:.2} GB) is below the {reserve_gb:.1} GB safety reserve. \
             This model is too large for available GPU memory. \
             Reduce --max-seq-len, increase --oom-guard-mb, or use a smaller model."
        );
    }
    Ok(())
}
mod load_fns;
use load_fns::{load_sharded, load_single};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_header_only_safetensor(tensors: &[(&str, &str, &[usize])]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "atlas-estimate-test-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape) in tensors {
            let bytes = shape.iter().product::<usize>() * dtype_bytes_for_test(dtype);
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let header = serde_json::Value::Object(header).to_string();
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        path
    }

    fn dtype_bytes_for_test(dtype: &str) -> usize {
        match dtype {
            "F32" => 4,
            "BF16" => 2,
            "U8" | "F8_E4M3" => 1,
            other => panic!("unsupported test dtype {other}"),
        }
    }

    #[test]
    fn ep_estimate_shards_step3p7_fused_expert_tensors() {
        let file = write_header_only_safetensor(&[
            ("model.embed_tokens.weight", "BF16", &[10]),
            ("model.layers.3.moe.gate_proj.weight", "BF16", &[10, 4]),
        ]);
        let files = [file.clone()];
        let skip = |_: &str| false;

        let rank1 = estimate_load_bytes(&files, &skip, EpShard::from_parts(1, 4, 10)).unwrap();
        let rank3 = estimate_load_bytes(&files, &skip, EpShard::from_parts(3, 4, 10)).unwrap();

        std::fs::remove_file(file).unwrap();
        assert_eq!(rank1, 20 + 16);
        assert_eq!(rank3, 20 + 32);
    }

    #[test]
    fn ep_estimate_shards_experts_fused_tensors_for_any_ep_count() {
        let file = write_header_only_safetensor(&[
            ("model.layers.3.moe.up_proj.weight", "U8", &[9, 4]),
            ("mtp.layers.0.mlp.experts.gate_up_proj", "U8", &[9, 4]),
        ]);
        let files = [file.clone()];
        let skip = |_: &str| false;

        let rank0 = estimate_load_bytes(&files, &skip, EpShard::from_parts(0, 3, 9)).unwrap();
        let rank2 = estimate_load_bytes(&files, &skip, EpShard::from_parts(2, 3, 9)).unwrap();

        std::fs::remove_file(file).unwrap();
        assert_eq!(rank0, 12 + 36);
        assert_eq!(rank2, 12 + 36);
    }
}
