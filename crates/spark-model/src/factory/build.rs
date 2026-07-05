// SPDX-License-Identifier: AGPL-3.0-only

//! `build_model` — entry point that wires up the configured loader,
//! buffers, KV cache, and (optional) DFlash drafter into a `TransformerModel`.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::weights::WeightStore;

use super::DflashBuildArgs;
use super::loader_for_config;
use super::m2_setup::maybe_run_minimax_m2_moe_transpose;
use crate::layers::MtpQuantization;
use crate::model::TransformerModel;
use crate::traits::Model;
use crate::weight_loader::load_dflash_weights;

pub fn build_model(
    mut config: ModelConfig,
    store: &WeightStore,
    gpu: Box<dyn GpuBackend>,
    max_batch_tokens: usize,
    kv_block_size: usize,
    max_seq_len: usize,
    max_batch_size: usize,
    mtp_quant: MtpQuantization,
    use_speculative: bool,
    prefix_cache: Box<dyn PrefixCache>,
    mtp_vocab_size: u32,
    comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    self_speculative: bool,
    num_drafts: usize,
    kv_dtype: KvCacheDtype,
    inference_reserve: usize,
    gpu_memory_utilization: f64,
    ssm_cache_slots: usize,
    layer_dtypes: Vec<KvCacheDtype>,
    ssm_checkpoint_interval: usize,
    // Phase 6.1.f: per-sequence HBM cache cap. `Some(N)` enables
    // `--high-speed-swap` HBM-shrink behavior. `None` preserves the
    // pre-Phase-6 unbounded behavior.
    hss_cache_blocks_per_seq: Option<u32>,
    // DFlash speculative-decoding pairing. `None` = no DFlash; existing
    // MTP / no-spec paths unchanged.
    dflash_args: Option<DflashBuildArgs<'_>>,
) -> Result<Box<dyn Model>> {
    // ── Step 1: Select weight loader (only model-specific dispatch) ──
    let loader = loader_for_config(&config)?;

    // Pre-construction: when DFlash is active, populate the target's
    // capture-layer indices from the drafter's `dflash_config.target_layer_ids`
    // so `TransformerModel::new` allocates the 5×hidden_size capture buffer.
    //
    // HF `output_hidden_states[i]` semantics: index 0 = post-embedding,
    // index k>=1 = post-layer-(k-1). The drafter's `target_layer_ids`
    // are interpreted as HF `output_hidden_states` indices (so layer_id=1
    // means post-layer-0). Atlas captures AFTER `layer.decode()` for the
    // listed `dflash_capture_layers` index — to match HF semantics we
    // subtract 1 from each id (clamped at 0). Set
    // ATLAS_DFLASH_CAPTURE_LAYER_OFFSET=0 to disable this adjustment for
    // a back-to-back A/B test.
    if let Some(ref args) = dflash_args
        && let Some(ref sub) = args.drafter_config.dflash_config
    {
        let offset: i64 = std::env::var("ATLAS_DFLASH_CAPTURE_LAYER_OFFSET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1);
        config.dflash_capture_layers = sub
            .target_layer_ids
            .iter()
            .map(|&id| (id as i64 + offset).max(0) as usize)
            .collect();
        tracing::info!(
            "DFlash: target layer capture indices = {:?} (offset={offset} from raw {:?})",
            config.dflash_capture_layers,
            sub.target_layer_ids,
        );
    }

    // ── Step 2: Load weights (model-agnostic from here) ──
    let attn_layer_dtypes: Vec<KvCacheDtype> = if layer_dtypes.is_empty() {
        vec![kv_dtype; config.num_attention_layers()]
    } else {
        layer_dtypes.clone()
    };

    // Populate per-layer KV dims for heterogeneous-attention models (Gemma-4).
    // Homogeneous models return an empty Vec which the KV cache treats as
    // "use global num_kv_heads/head_dim for all layers" (backward compatible).
    config.kv_layer_dims = loader.kv_layer_dims(&config);

    let mut layers = loader.load_layers(store, &config, gpu.as_ref(), &attn_layer_dtypes)?;
    let embed = loader.load_embedding(store, &config)?;
    let final_norm = loader.load_final_norm(store, &config, gpu.as_ref())?;
    let lm_head = loader.load_lm_head(store, &config)?;
    let mtp_weights = loader.load_mtp_weights_multi(store, &config, gpu.as_ref())?;
    // Capability warning: user asked for `--speculative` but the model has no
    // MTP head bundled, so speculative decoding will silently no-op. Surface
    // this loudly so the user knows the flag was inert.
    if use_speculative && mtp_weights.is_empty() {
        tracing::warn!(
            "`--speculative` was requested but no MTP weights were loaded for this \
             model — speculative decoding will be disabled. Either drop `--speculative` \
             or use a checkpoint that ships an MTP head (e.g. `mtp.safetensors`)."
        );
    }
    let vision_encoder = loader.load_vision_encoder(store, &config, gpu.as_ref())?;

    // If the checkpoint's `quantization_config.ignore_modules` lists MTP
    // (e.g. Sehyo/Qwen3.5-35B-A3B-NVFP4 ignores `mtp.*`), the MTP weights
    // were stored as BF16 on disk. Runtime-quantizing them to NVFP4
    // anyway — which is what `mtp_quant` would otherwise do — produces
    // garbage drafts (vllm PR #38832). Force BF16 in that case.
    let effective_mtp_quant = if !mtp_weights.is_empty() {
        let quant_fmt = crate::quant_format::detect_quant_format(&config, store);
        if quant_fmt.is_ignored("mtp.fc.weight")
            || quant_fmt.is_ignored("mtp.layers.0.self_attn.q_proj.weight")
        {
            if mtp_quant != MtpQuantization::Bf16 {
                tracing::info!(
                    "MTP head listed in checkpoint ignore_modules — overriding \
                     --mtp-quantization {:?} → Bf16 to preserve precision",
                    mtp_quant,
                );
            }
            MtpQuantization::Bf16
        } else {
            mtp_quant
        }
    } else {
        mtp_quant
    };

    // ── Step 3: LM-head quantization (NVFP4 / FP8 / BF16-skip) + the
    // draft-only NVFP4 head for MTP — extracted to lm_head_setup.rs
    // (file-size cap; pure code move).
    let (lm_head_nvfp4, lm_head_fp8, mtp_lm_head_nvfp4) = super::lm_head_setup::setup_lm_heads(
        store,
        &lm_head,
        &config,
        gpu.as_ref(),
        use_speculative,
        !mtp_weights.is_empty(),
    )?;

    // ── Step 3b: Post-load MoE prefill transpose (MiniMax EP=2 TTFT fix) ──
    //
    // MiniMax M2.7-NVFP4 EP=2 has ~46 GB free at layer-0 load time but
    // ~65 GB free here (the BF16 lm_head just freed ~22 GB during NVFP4
    // quantization). The transpose costs ~59 GB — fits in the post-load
    // window but not the pre-load one. Other loaders (qwen35, qwen3,
    // gemma4) still call `transpose_for_prefill` inline during layer
    // construction; this default-no-op hook doesn't perturb them.
    maybe_run_minimax_m2_moe_transpose(&config, gpu.as_ref(), &mut layers)?;
    // ── Step 4: Create buffer arena ──
    let buffers = BufferArena::new(
        &config,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        gpu.as_ref(),
    )?;

    // ── Step 5: Size KV cache from actual free memory ──
    // MLA absorbed: cache compressed latent [kv_lora + rope] instead of expanded [nkv * hd]
    // This gives 12.8x smaller KV cache AND better precision (no expand→cache→read roundtrip)
    let (kv_num_heads, kv_head_dim) = if config.kv_lora_rank > 0 {
        let mla_cache_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        tracing::info!(
            "MLA absorbed KV cache: 1 head × {} dims ({}+{}) per token (vs {} heads × {})",
            mla_cache_dim,
            config.kv_lora_rank,
            config.qk_rope_head_dim,
            config.num_key_value_heads,
            config.head_dim,
        );
        (1, mla_cache_dim)
    } else {
        (config.num_key_value_heads, config.head_dim)
    };
    let mut kv_config = KvCacheConfig {
        block_size: kv_block_size,
        num_kv_heads: kv_num_heads,
        head_dim: kv_head_dim,
        num_layers: config.num_attention_layers(),
        dtype: kv_dtype,
        layer_dtypes: layer_dtypes.clone(),
        layer_dims: config.kv_layer_dims.clone(),
        cache_blocks_per_seq: hss_cache_blocks_per_seq,
        layer_sliding: Vec::new(),
        num_sliding_blocks: 0,
        sliding_ring_blocks: 0,
    };

    // ── Sliding-window split KV pool (Step 3.7 / Gemma-4 hybrid attention) ──
    //
    // Hybrid models waste enormous KV memory when every layer is allocated
    // `max_seq_len` worth of blocks: Step 3.7's 33 sliding layers
    // (window=512) only ever attend to the last 512 positions. The split
    // pool gives sliding layers their own small block-ID space and a
    // per-sequence RING of R blocks; logical block `i` maps to
    // `ring[i % R]`. Correctness of the ring size: the KV-write for a
    // forward pass of M tokens lands BEFORE that pass's paged-attention
    // reads, so every position in `(chunk_start − window, chunk_end]`
    // must be alias-free simultaneously → R×block_size ≥ M_max + window,
    // with M_max = max_batch_tokens (the arena bound on any single pass).
    // Two positions p < p' collide iff p' − p is a multiple of R×bs; with
    // that bound, at most one of them is ever in the live span, and the
    // per-(slot, offset) overwrite retires positions exactly as they
    // fall out of every sliding layer's window.
    let sliding_flags = config.attention_layer_sliding_flags();
    let n_sliding = sliding_flags.iter().filter(|s| **s).count();
    let n_full_attn = sliding_flags.len() - n_sliding;
    // STEP37-QUALITY Round 6 reported the split pool's per-sequence ring
    // intermittently corrupting sliding-layer KV during LONG generation
    // (~50 % catastrophic repetition at ≥~10 k). Round 7 re-ran that exact
    // repro on the same dual-Spark EP-2 rig with `ATLAS_SLIDING_KV_SPLIT=1`
    // and could NOT reproduce it: ~20 requests at 10 k–27.6 k (needle,
    // creative ×N identical, and a 2 494-token loop-prone enumeration) were
    // all deterministic (bit-identical MD5 across runs) and loop-free. The
    // Round-6 catastrophe was almost certainly measured on an interim
    // pre-fix binary; the ring staging in this tree is correct, and the
    // size algebra above holds with margin (`R·bs − (M_max+window) ≥ 1`
    // block; verified at runtime under `ATLAS_SLIDING_KV_VERIFY=1`, which
    // asserts the live read window is alias-free every step). The split
    // stays OPT-IN per the Round-7 brief: `ATLAS_SLIDING_KV_SPLIT=1` enables
    // it (needed for >~18 k context — Hermes 27 k); `ATLAS_NO_SLIDING_KV_SPLIT=1`
    // still force-disables. Default OFF keeps vLLM-parity KV allocation.
    let split_opt_in = std::env::var("ATLAS_SLIDING_KV_SPLIT").as_deref() == Ok("1");
    let split_kill_switch = std::env::var("ATLAS_NO_SLIDING_KV_SPLIT").as_deref() == Ok("1");
    let spec_requested = use_speculative || self_speculative || dflash_args.is_some();
    let mut split_sliding = split_opt_in
        && config.sliding_window > 0
        && config.kv_lora_rank == 0            // MLA paths are not split-aware
        && hss_cache_blocks_per_seq.is_none()  // HSS has its own windowing
        && n_sliding > 0
        && n_full_attn > 0
        && !split_kill_switch;
    if split_sliding && spec_requested {
        tracing::warn!(
            "sliding-window KV split pool DISABLED: speculative decoding (MTP/self-spec/\
             DFlash) verify paths have no sliding metadata twin yet. Drop the speculative \
             flags to unlock ~{}× less KV memory on the {} sliding layers.",
            (max_seq_len.div_ceil(kv_block_size))
                .div_ceil((max_batch_tokens + config.sliding_window as usize).div_ceil(kv_block_size))
                .max(1),
            n_sliding,
        );
        split_sliding = false;
    }
    if split_sliding {
        // R·bs must exceed the widest live span `M_max + window` so every
        // simultaneously-live position owns a distinct (ring slot, offset).
        // `div_ceil` alone already satisfies `R·bs ≥ M_max + window`; the
        // `+ 1` block buys a full extra block of headroom so the guarantee
        // is not razor-thin at pathological chunk/offset alignments (costs
        // one 16-token block per sliding layer — negligible).
        let ring =
            (max_batch_tokens + config.sliding_window as usize).div_ceil(kv_block_size) + 1;
        kv_config.layer_sliding = sliding_flags;
        kv_config.sliding_ring_blocks = ring;
        // One ring per concurrent sequence + 1 permanent dummy block
        // (OOB-safe padding target for sliding block tables).
        kv_config.num_sliding_blocks = max_batch_size * ring + 1;
        tracing::info!(
            "Sliding-window KV split pool: {} sliding layers (window={}) get {} blocks each \
             (ring {} blocks/seq × {} seqs + dummy) instead of scaling with max_seq_len={}; \
             {} full-attention layers keep budget-driven sizing. \
             Prefix caching + speculative decoding are unavailable in this mode \
             (ATLAS_NO_SLIDING_KV_SPLIT=1 restores the legacy allocator).",
            n_sliding,
            config.sliding_window,
            kv_config.num_sliding_blocks,
            ring,
            max_batch_size,
            max_seq_len,
            n_full_attn,
        );
    }

    // Phase 6.2.c — KV-dtype gating for `--high-speed-swap`.
    //
    // All quantization variants are now supported via host-side dequant before
    // disk-write (the orchestrator's tiled-attention kernel reads BF16):
    //   - BF16    : direct stream; predictor anchor (K_lr) computed natively.
    //   - FP8     : E4M3 → BF16 (per-tensor calibration scale). Predictor
    //               degrades to LRU (BF16-only kernel can't read FP8 layout).
    //   - NVFP4   : E2M1 nibble + per-group FP8 scale → BF16. Predictor LRU.
    //   - Turbo4  : Lloyd-Max 16-level + per-group FP8 scale + WHT(K/V) on
    //               disk. Decode flow's WHT(Q)/iWHT(out) bookends handle the
    //               Walsh-Hadamard round-trip transparently. Predictor LRU.
    //   - Turbo3  : 3-bit packed (8 vals per 3 bytes), 8-level codebook,
    //               per-group FP8 scales, WHT bookended. Predictor LRU.
    //   - Turbo8  : FP8 E4M3 + per-group FP8 scales + WHT bookended.
    //               Predictor LRU.
    fn dtype_label(dt: KvCacheDtype) -> &'static str {
        match dt {
            KvCacheDtype::Bf16
            | KvCacheDtype::Bf16KTurbo4V
            | KvCacheDtype::Bf16KTurbo3V
            | KvCacheDtype::Bf16KTurbo2V => "BF16",
            KvCacheDtype::Fp8
            | KvCacheDtype::Fp8KTurbo4V
            | KvCacheDtype::Fp8KTurbo3V
            | KvCacheDtype::Fp8KTurbo2V => "FP8",
            KvCacheDtype::Nvfp4 => "NVFP4",
            KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V | KvCacheDtype::Turbo2 => "Turbo3",
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo4KTurbo3V | KvCacheDtype::Turbo4KTurbo8V => {
                "Turbo4"
            }
            KvCacheDtype::Turbo8 => "Turbo8",
        }
    }
    if hss_cache_blocks_per_seq.is_some() {
        let mut counts: std::collections::BTreeMap<&'static str, usize> =
            std::collections::BTreeMap::new();
        if kv_config.layer_dtypes.is_empty() {
            *counts.entry(dtype_label(kv_config.dtype)).or_default() += kv_config.num_layers;
        } else {
            for dt in &kv_config.layer_dtypes {
                *counts.entry(dtype_label(*dt)).or_default() += 1;
            }
        }
        let total: usize = counts.values().sum();
        let summary: Vec<String> = counts
            .iter()
            .map(|(name, n)| format!("{n} {name}"))
            .collect();
        tracing::info!(
            "--high-speed-swap KV: {} attn layers ({}); HBM-shrink applies to all \
             (Phase 6.2.c proper — host dequant for FP8/NVFP4/Turbo3/Turbo4/Turbo8; \
             predictor scoring uses LRU for non-BF16 layers)",
            total,
            summary.join(" + ")
        );
    }
    let actual_free = gpu.free_memory()?;
    let allocatable = actual_free.saturating_sub(inference_reserve);
    let kv_budget = (allocatable as f64 * gpu_memory_utilization) as usize;
    // Phase 6.1.f: when HBM-shrink is active, size the production cache to
    // `max_batch_size × cache_blocks_per_seq` rather than the unbounded
    // budget-driven sum. This is the *whole point* of the HBM-shrink
    // feature — the production cache becomes write staging only; older
    // blocks live on disk under the orchestrator's control.
    let num_kv_blocks = match hss_cache_blocks_per_seq {
        Some(cap) => {
            // Phase 6.3 (original): pool = max_batch × cap + 1 dummy + 1 spare per seq.
            // Issue #31 (2026-05-08): the cap×bs sizing assumed prefill would
            // fit in cap blocks AND the slide-during-prefill path would handle
            // any overflow. Live-tested: slides during prefill produce silently
            // wrong attention output (the orchestrator-fed disk-read path is
            // wired up for DECODE attention only — Phase 6.2.a — not for
            // prefill — Phase 6.2.b deferred). The companion change in
            // `block_mgmt::ensure_blocks_through_prefill` removes the broken
            // slide; this change resizes the pool so prefill can grow up to
            // `max_seq_len` blocks without hitting "no free blocks". HBM-shrink
            // remains in effect post-prefill: the FIRST decode step finds
            // bt_len > cap and slides down via the orchestrator-aware path
            // (which IS correct).
            //
            // Sizing rationale:
            //   * Per-seq blocks: `max(cap + 1, ceil(max_seq_len / block_size))`
            //     so prefill of any prompt up to max_seq_len fits in HBM.
            //   * +1 dummy slot for OOB-safe paged-kernel reads.
            //
            // For multi-seq HSS where the user wanted strict HBM-shrink, this
            // increases pool size by `(max_seq_len_blocks - cap) × max_batch`
            // bytes per block. The existing post-load OOM check (line 304+)
            // catches infeasible configs at startup with a clear message.
            let max_seq_blocks = max_seq_len.div_ceil(kv_block_size);
            let per_seq = (cap as usize + 1).max(max_seq_blocks);
            let n = max_batch_size * per_seq + 1;
            tracing::info!(
                "--high-speed-swap: HBM cache sized to {n} blocks ({} batch × max(cap={cap}+1, max_seq_len_blocks={max_seq_blocks}) + 1 dummy); \
                 prefill grows monotonically, decode shrinks to cap × bs and streams older blocks from disk via the orchestrator",
                max_batch_size
            );
            n
        }
        None => {
            let n = PagedKvCache::compute_num_blocks(&kv_config, kv_budget)?;
            let max_kv_tokens = n * kv_block_size;
            tracing::info!(
                "KV cache (post-construction): {:.1} GB free, {:.1} GB allocatable, \
                 {} blocks × {} tok/block = {} max tokens",
                actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
                allocatable as f64 / (1024.0 * 1024.0 * 1024.0),
                n,
                kv_block_size,
                max_kv_tokens,
            );
            n
        }
    };
    let _max_kv_tokens = num_kv_blocks * kv_block_size;
    // Phase 6.1.f / 6.2.c — when --high-speed-swap is on with HBM-shrink, the
    // production KV cache only has to fit the per-seq HBM window, not the full
    // sequence (older blocks live on disk). Compare against `cache_blocks_per_seq`
    // in that mode; the legacy "blocks per max_seq_len" check is invalid for
    // HBM-shrunk pools by design.
    let blocks_per_seq = match hss_cache_blocks_per_seq {
        Some(cap) => cap as usize,
        None => max_seq_len.div_ceil(kv_block_size),
    };
    let max_concurrent = num_kv_blocks / blocks_per_seq.max(1);
    if max_concurrent < max_batch_size {
        // Suggest a max_seq_len that lets the requested batch size fit.
        let suggested_max_seq_len = (num_kv_blocks / max_batch_size.max(1)) * kv_block_size;
        anyhow::bail!(
            "KV cache can hold at most {} concurrent sequence(s) at --max-seq-len={}, \
             but --max-batch-size={} was requested. \
             KV pool has {} block(s) of {} tokens each; each sequence needs {} block(s). \
             Try --max-seq-len {} (keeps max_batch_size={}) or reduce --max-batch-size.",
            max_concurrent,
            max_seq_len,
            max_batch_size,
            num_kv_blocks,
            kv_block_size,
            blocks_per_seq,
            suggested_max_seq_len.max(kv_block_size),
            max_batch_size,
        );
    }
    let kv_cache = PagedKvCache::new(kv_config, num_kv_blocks, gpu.as_ref())?;

    // ── Step 6: Assemble model ──
    // Capture pointers for any post-construction sharing (DFlash drafter
    // shares embed_tokens + lm_head with the target). DenseWeight is Copy
    // so this clones the device pointer cheaply.
    let target_embed_for_dflash = embed.weight;
    let target_lm_head_for_dflash = lm_head.weight;
    let target_hidden_for_dflash = config.hidden_size;

    let mut model = TransformerModel::new(
        config,
        embed,
        final_norm,
        lm_head,
        lm_head_nvfp4,
        lm_head_fp8,
        mtp_lm_head_nvfp4,
        layers,
        buffers,
        kv_cache,
        mtp_weights,
        gpu,
        max_seq_len,
        max_batch_size,
        effective_mtp_quant,
        use_speculative,
        prefix_cache,
        mtp_vocab_size,
        comm,
        self_speculative,
        num_drafts,
        vision_encoder,
        ssm_cache_slots,
        ssm_checkpoint_interval,
    )?;

    // ── Step 7: DFlash drafter (optional, post-construction) ──
    //
    // Loaded last because it depends on the target's `embed_tokens` and
    // `lm_head` pointers (the drafter checkpoint omits these — they're
    // shared at runtime, mirroring vLLM PR #40898's `skip_substrs` flow).
    if let Some(args) = dflash_args {
        let weights = load_dflash_weights(
            args.drafter_store,
            &args.drafter_config,
            model.gpu_backend(),
            1, // tp_size for the drafter side: replicated, so always 1
        )?;
        if let Some(weights) = weights {
            let head = crate::layers::BlockDiffusionDraftHead::from_weights(
                weights,
                target_embed_for_dflash,
                target_lm_head_for_dflash,
                target_hidden_for_dflash,
                args.gamma,
                args.window_size,
                model.gpu_backend(),
                max_seq_len,
            )?;
            model.set_dflash_proposer(std::sync::Arc::new(head));
            tracing::info!("DFlash drafter installed as the active proposer");
        } else {
            tracing::warn!(
                "DFlash drafter store had no fc.weight — proposer not installed; \
                 falling back to whatever proposer (if any) the target's MTP path built"
            );
        }
    }

    Ok(Box::new(model))
}
