// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;

#[test]
fn test_buffer_sizes_qwen3() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16);

    // hidden_states: 1 * 2048 * 2 = 4096 (BF16, 2 bytes/elem).
    // (Was FP32 = 8192 in earlier prototypes; NVFP4 path keeps the
    // residual stream in BF16, halving the buffer size.)
    assert_eq!(sizes.hidden_states, 4096);
    // qkv: 1 * (16*2 + 2*2) * 256 * 2 = 1 * 36 * 256 * 2 = 18432
    // Q+gate: 16*2*256, K: 2*256, V: 2*256
    assert_eq!(sizes.qkv_output, 18432);
    // attn: 1 * 16 * 256 * 2 = 8192
    assert_eq!(sizes.attn_output, 8192);
    // gate: 1 * 512 * 2 = 1024
    assert_eq!(sizes.gate_logits, 1024);
    // logits: 1 * 151936 * 2 = 303872
    assert_eq!(sizes.logits, 303872);
    // ssm_qkvz: 1 * 12288 * 2 = 24576
    // Q(16*128) + K(16*128) + V(32*128) + Z(32*128) = 12288
    assert_eq!(sizes.ssm_qkvz, 24576);
    // ssm_ba: max(1 * 64 * 2, 256) = 256 (minimum allocation)
    assert_eq!(sizes.ssm_ba, 256);
    // ssm_deinterleaved: same as ssm_qkvz = 24576
    assert_eq!(sizes.ssm_deinterleaved, 24576);
    // ssm_gates: 1 * 32 * 2 * 4 = 256 (FP32 gate + beta, scaled by M)
    assert_eq!(sizes.ssm_gates, 256);
}

#[test]
fn test_buffer_arena_alloc() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let arena = BufferArena::new(&cfg, 128, 4096, 16, &gpu).unwrap();

    assert!(!arena.hidden_states().is_null());
    assert!(!arena.logits().is_null());
    assert_eq!(arena.max_batch_tokens(), 128);
    // 19 allocations for 19 buffers (12 data + 1 scratch + 3 expert + 2 splitk
    // + 1 gdn_fla_scratch). Bump from 18 reflects the GDN FLA chunked-prefill
    // scratch buffer added when ATLAS_GDN_FLA was wired into the arena.
    assert_eq!(gpu.alloc_count(), 19);
}

#[test]
fn test_expert_buffers_account_for_hybrid_dense_ffn() {
    // Step 3.7 Flash shape: 288-expert top-8 MoE (moe_inter=1280) with
    // layers 0-2 dense FFN. `DenseFfnLayer::forward_prefill` writes
    // [M, intermediate_size] into expert_gate_out/expert_up_out, so a
    // hybrid model whose dense intermediate_size exceeds
    // top_k × moe_intermediate_size overflows MoE-only sizing — the
    // CUDA_ERROR_ILLEGAL_ADDRESS (700) at "Prefill chunk layer 2" on
    // full-size chunks of 26K-token prompts.
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.num_experts = 288;
    cfg.num_experts_per_tok = 8;
    cfg.moe_intermediate_size = 1280;
    cfg.intermediate_size = 12288; // dense FFN wider than 8 × 1280 = 10240
    let m = 2049; // max_batch_tokens = prefill_budget 2048 + max_batch_size 1

    // No dense layers marked → MoE-only sizing (unchanged for pure-MoE models).
    cfg.num_dense_ffn_layers = 0;
    cfg.decoder_sparse_step = 1;
    let moe_only = BufferSizes::from_config(&cfg, m, 28000, 16);
    assert_eq!(moe_only.expert_gate_out, m * 8 * 1280 * 2);

    // Dense layers present → buffers must also fit [M, intermediate_size].
    cfg.num_dense_ffn_layers = 3;
    let hybrid = BufferSizes::from_config(&cfg, m, 28000, 16);
    assert!(hybrid.expert_gate_out >= m * cfg.intermediate_size * 2);
    assert!(hybrid.expert_up_out >= m * cfg.intermediate_size * 2);

    // decoder_sparse_step staggering (DeepSeek/Mistral-style) also counts
    // as having dense layers.
    cfg.num_dense_ffn_layers = 0;
    cfg.decoder_sparse_step = 2;
    let staggered = BufferSizes::from_config(&cfg, m, 28000, 16);
    assert!(staggered.expert_gate_out >= m * cfg.intermediate_size * 2);
}

#[test]
fn test_buffer_sizes_scale_with_batch() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s1 = BufferSizes::from_config(&cfg, 1, 4096, 16);
    let s128 = BufferSizes::from_config(&cfg, 128, 4096, 16);
    assert_eq!(s128.hidden_states, s1.hidden_states * 128);
    // logits is capped at 16 tokens; FP32 sampling buffer (4 bytes/elem),
    // so s128.logits = 16 * vocab * 4 (not 128× the unbatched value).
    assert_eq!(s128.logits, 16 * cfg.vocab_size * 4);
}

#[test]
fn test_sliding_meta_sized_only_for_hybrid_attention() {
    use atlas_core::config::LayerType;
    // Qwen3-Next has no sliding layers → no sliding metadata staging.
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1024, 4096, 16);
    assert_eq!(sizes.sliding_meta, 0);

    // Hybrid full+sliding (Step 3.7 shape): staging = 256 B slot region +
    // 64B-aligned 32-row table region + m×8 prefill slots.
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.sliding_window = 512;
    cfg.layer_types = (0..48)
        .map(|i| {
            if i % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::SlidingAttention
            }
        })
        .collect();
    let m = 1024usize;
    let sizes = BufferSizes::from_config(&cfg, m, 4096, 16);
    let max_blocks = 4096 / 16 + 1;
    let tables = 32 * max_blocks * 4;
    assert_eq!(sizes.sliding_meta, 256 + ((tables + 63) & !63) + m * 8);
    // Total accounting includes the new buffer.
    assert!(sizes.total_bytes() > sizes.sliding_meta);
}
