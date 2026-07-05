// SPDX-License-Identifier: AGPL-3.0-only

//! Eligibility checks for Q12 Path B kernel-batched prefill.

use super::super::super::types::TransformerModel;
use crate::traits::PrefillSlice;

impl TransformerModel {
    /// Returns true when the batched-kernel path is viable for these
    /// streams. Cheap upfront check — caller (dispatch) falls back to
    /// per-stream when false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        // Sliding split pool: the Q12 batched metadata (stage_batched.rs)
        // has no sliding slot/table twin yet — fall back to the per-stream
        // path, which is fully split-pool aware.
        if self.sliding_ring_len() > 0 {
            return false;
        }
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            &self.config.model_type,
            self.config.head_dim,
            self.buffers.scratch_bytes(),
            self.config.num_experts_per_tok,
            self.config.mrope_interleaved,
        )
    }
}

/// Pure-data predicate extracted from [`TransformerModel::kernel_batched_eligible`]
/// so the gating rules are unit-testable without a real `TransformerModel`.
/// Caller materialises per-stream tuples `(chunk_len, chunk_start, is_last_chunk)`.
#[allow(clippy::too_many_arguments)]
pub(in crate::model) fn check_kernel_batched_eligible<I>(
    streams: I,
    n: usize,
    arena_cap: usize,
    model_type: &str,
    head_dim: usize,
    scratch_cap: usize,
    top_k: usize,
    mrope: bool,
) -> bool
where
    I: IntoIterator<Item = (usize, usize, bool)>,
{
    if n < 2 {
        return false;
    }
    // No MLA layers in stack (batched attention doesn't support MLA).
    // Conservatively check via model_type — mistral is the only MLA
    // model in Atlas today.
    if model_type == "mistral" {
        return false;
    }
    // No HDIM=512 layers (Gemma-4 long-attention).
    if head_dim > 256 {
        return false;
    }
    let mut first: Option<(usize, usize, bool)> = None;
    let mut total = 0usize;
    for (chunk_len, chunk_start, is_last) in streams {
        // `chunk_len`, `chunk_start`, and `is_last_chunk` must all
        // match across streams. Different `chunk_start` produces
        // different `effective_seq_len_start` post-Marconi (which the
        // batched attention kernel cannot handle); mixing
        // `is_last_chunk` can't dispatch one finalize_last and one
        // save_checkpoint in a single batched call.
        match first {
            None => first = Some((chunk_len, chunk_start, is_last)),
            Some((cl, cs, il)) => {
                if chunk_len != cl || chunk_start != cs || is_last != il {
                    return false;
                }
            }
        }
        total += chunk_len;
    }
    // Total stacked tokens fit in the token arena (hidden_states buffer).
    if total > arena_cap {
        return false;
    }
    // #110: the kernel-batched staging footprint must fit in scratch. PURE
    // pre-flight — runs before any stream mutation, so a false routes to the
    // per-stream path from a clean state (a mid-dispatch overrun would leave
    // streams dirty and the fallback would re-run setup → corruption).
    let chunk_len = first.map(|(cl, _, _)| cl).unwrap_or(0);
    spark_runtime::buffers::q12_batched_scratch_bytes(n, chunk_len, top_k, mrope) <= scratch_cap
}
