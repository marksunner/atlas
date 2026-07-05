// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};

use super::chat::chat_completions_inner;
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::inference_types::*;
use super::sanitizer::*;

impl InferenceRequest {
    /// Number of prompt tokens in this request.
    pub fn prompt_len(&self) -> usize {
        match self {
            InferenceRequest::Blocking { prompt_tokens, .. } => prompt_tokens.len(),
            InferenceRequest::Streaming { prompt_tokens, .. } => prompt_tokens.len(),
        }
    }

    /// Preprocessed image data, consumed by the scheduler before prefill.
    pub fn take_image_pixels(&mut self) -> Vec<(Vec<f32>, usize, usize)> {
        match self {
            InferenceRequest::Blocking { image_pixels, .. } => std::mem::take(image_pixels),
            InferenceRequest::Streaming { image_pixels, .. } => std::mem::take(image_pixels),
        }
    }

    /// Per-request stop tokens, consumed by the scheduler.
    pub fn take_stop_tokens(&mut self) -> Vec<u32> {
        match self {
            InferenceRequest::Blocking { stop_tokens, .. } => std::mem::take(stop_tokens),
            InferenceRequest::Streaming { stop_tokens, .. } => std::mem::take(stop_tokens),
        }
    }

    /// Top-k sampling parameter.
    pub fn top_k(&self) -> u32 {
        match self {
            InferenceRequest::Blocking { top_k, .. } => *top_k,
            InferenceRequest::Streaming { top_k, .. } => *top_k,
        }
    }

    /// Top-p sampling parameter.
    pub fn top_p(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { top_p, .. } => *top_p,
            InferenceRequest::Streaming { top_p, .. } => *top_p,
        }
    }

    /// Top-n-sigma sampling parameter.
    pub fn top_n_sigma(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { top_n_sigma, .. } => *top_n_sigma,
            InferenceRequest::Streaming { top_n_sigma, .. } => *top_n_sigma,
        }
    }

    /// Min-p sampling parameter.
    pub fn min_p(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { min_p, .. } => *min_p,
            InferenceRequest::Streaming { min_p, .. } => *min_p,
        }
    }

    /// Repetition penalty parameter.
    pub fn repetition_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                repetition_penalty, ..
            } => *repetition_penalty,
            InferenceRequest::Streaming {
                repetition_penalty, ..
            } => *repetition_penalty,
        }
    }

    /// Presence penalty (OpenAI-style additive).
    pub fn presence_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                presence_penalty, ..
            } => *presence_penalty,
            InferenceRequest::Streaming {
                presence_penalty, ..
            } => *presence_penalty,
        }
    }

    /// Frequency penalty (OpenAI-style additive).
    pub fn frequency_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking {
                frequency_penalty, ..
            } => *frequency_penalty,
            InferenceRequest::Streaming {
                frequency_penalty, ..
            } => *frequency_penalty,
        }
    }

    /// DRY (Don't-Repeat-Yourself) penalty multiplier. 0.0 = disabled.
    pub fn dry_multiplier(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { dry_multiplier, .. } => *dry_multiplier,
            InferenceRequest::Streaming { dry_multiplier, .. } => *dry_multiplier,
        }
    }

    /// LZ penalty (A.1, arXiv:2504.20131). 0.0 = disabled.
    pub fn lz_penalty(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { lz_penalty, .. } => *lz_penalty,
            InferenceRequest::Streaming { lz_penalty, .. } => *lz_penalty,
        }
    }

    /// DRY penalty exponential base.
    pub fn dry_base(&self) -> f32 {
        match self {
            InferenceRequest::Blocking { dry_base, .. } => *dry_base,
            InferenceRequest::Streaming { dry_base, .. } => *dry_base,
        }
    }

    /// DRY minimum match length before penalty applies.
    pub fn dry_allowed_length(&self) -> u32 {
        match self {
            InferenceRequest::Blocking {
                dry_allowed_length, ..
            } => *dry_allowed_length,
            InferenceRequest::Streaming {
                dry_allowed_length, ..
            } => *dry_allowed_length,
        }
    }

    /// Per-token logit bias.
    pub fn logit_bias(&self) -> &[(u32, f32)] {
        match self {
            InferenceRequest::Blocking { logit_bias, .. } => logit_bias,
            InferenceRequest::Streaming { logit_bias, .. } => logit_bias,
        }
    }

    /// Session hash for SSM snapshot isolation.
    pub fn session_hash(&self) -> u64 {
        match self {
            InferenceRequest::Blocking { session_hash, .. } => *session_hash,
            InferenceRequest::Streaming { session_hash, .. } => *session_hash,
        }
    }

    /// Whether thinking mode is enabled for this request.
    pub fn enable_thinking(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                enable_thinking, ..
            } => *enable_thinking,
            InferenceRequest::Streaming {
                enable_thinking, ..
            } => *enable_thinking,
        }
    }

    /// Thinking token budget (None = unlimited).
    pub fn thinking_budget(&self) -> Option<u32> {
        match self {
            InferenceRequest::Blocking {
                thinking_budget, ..
            } => *thinking_budget,
            InferenceRequest::Streaming {
                thinking_budget, ..
            } => *thinking_budget,
        }
    }

    /// Per-request override for the vLLM-anchored token-loop detector.
    /// `None` = use the boot-global watchdog parameters.
    pub fn repetition_detection(&self) -> Option<crate::openai::RepetitionDetectionParams> {
        match self {
            InferenceRequest::Blocking {
                repetition_detection,
                ..
            } => *repetition_detection,
            InferenceRequest::Streaming {
                repetition_detection,
                ..
            } => *repetition_detection,
        }
    }

    /// Whether a tool call is required for this request.
    pub fn require_tool_call(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                require_tool_call, ..
            } => *require_tool_call,
            InferenceRequest::Streaming {
                require_tool_call, ..
            } => *require_tool_call,
        }
    }

    /// Whether this turn has tools available (any `tool_choice`, including
    /// `"auto"`). The sticky "this is a tool turn" signal — used to set
    /// `ActiveSeq::tool_request` even when the tool-call grammar is disabled
    /// (`[behavior].disable_tool_grammar=true`) and `tool_choice="auto"`, the
    /// case where both `require_tool_call` and `grammar_spec` are absent.
    pub fn tools_active(&self) -> bool {
        match self {
            InferenceRequest::Blocking { tools_active, .. } => *tools_active,
            InferenceRequest::Streaming { tools_active, .. } => *tools_active,
        }
    }

    /// Whether `<tool_call>` should be suppressed (loop detected).
    pub fn suppress_tool_call(&self) -> bool {
        match self {
            InferenceRequest::Blocking {
                suppress_tool_call, ..
            } => *suppress_tool_call,
            InferenceRequest::Streaming {
                suppress_tool_call, ..
            } => *suppress_tool_call,
        }
    }

    /// F60 (2026-04-27): whether MTP speculative decoding should be
    /// disabled for this request (set when tools are active and the
    /// env gate is on).
    pub fn disable_mtp(&self) -> bool {
        match self {
            InferenceRequest::Blocking { disable_mtp, .. } => *disable_mtp,
            InferenceRequest::Streaming { disable_mtp, .. } => *disable_mtp,
        }
    }

    /// Seed for deterministic sampling (None = non-deterministic).
    pub fn seed(&self) -> Option<u64> {
        match self {
            InferenceRequest::Blocking { seed, .. } => *seed,
            InferenceRequest::Streaming { seed, .. } => *seed,
        }
    }

    /// Take the grammar specification for constrained decoding.
    pub fn take_grammar_spec(&mut self) -> Option<GrammarSpec> {
        match self {
            InferenceRequest::Blocking { grammar_spec, .. } => grammar_spec.take(),
            InferenceRequest::Streaming { grammar_spec, .. } => grammar_spec.take(),
        }
    }

    /// Minimum tokens before allowing EOS/stop (0 = no minimum).
    pub fn min_tokens(&self) -> usize {
        match self {
            InferenceRequest::Blocking { min_tokens, .. } => *min_tokens,
            InferenceRequest::Streaming { min_tokens, .. } => *min_tokens,
        }
    }

    /// Number of top logprobs to return per token. None = disabled.
    pub fn top_logprobs(&self) -> Option<u8> {
        match self {
            InferenceRequest::Blocking { top_logprobs, .. } => *top_logprobs,
            InferenceRequest::Streaming { top_logprobs, .. } => *top_logprobs,
        }
    }

    /// Request timeout deadline. None = no timeout.
    pub fn timeout_at(&self) -> Option<std::time::Instant> {
        match self {
            InferenceRequest::Blocking { timeout_at, .. } => *timeout_at,
            InferenceRequest::Streaming { timeout_at, .. } => *timeout_at,
        }
    }
}

/// Tokenize stop sequence strings into single-token stop IDs.
/// Multi-token stop sequences are logged but excluded (require string matching).
pub(crate) fn tokenize_stop_sequences(
    tokenizer: &crate::tokenizer::ChatTokenizer,
    stops: &[String],
) -> Vec<u32> {
    let mut tokens = Vec::new();
    for s in stops {
        match tokenizer.encode(s) {
            Ok(ids) if ids.len() == 1 => tokens.push(ids[0]),
            Ok(ids) if ids.len() > 1 => {
                tracing::info!(
                    "Multi-token stop '{}' ({} tokens) — use string matching",
                    s,
                    ids.len()
                );
            }
            Ok(_) => {} // Empty encoding
            Err(e) => tracing::warn!("Failed to tokenize stop '{}': {e}", s),
        }
    }
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// Truncate `text` at the first occurrence of any stop sequence.
///
/// Per the OpenAI spec, when a `stop` string appears the returned text must be
/// cut at the stop sequence — the stop string itself AND everything after it are
/// removed — regardless of where in the text the stop string lands.
///
/// Issue #100: the previous implementation used `strip_suffix`, i.e. it only
/// removed a stop string that happened to be the *trailing* suffix of the
/// output. A stop string appearing mid-text was silently ignored. This made
/// user-supplied `stop: ["\nuser", "\nassistant", "<|im_end|>"]` a no-op against
/// the role-marker / prompt-echo run-on: the requested substrings appeared
/// verbatim in `content` (reporter observation #1) because they were never at
/// the very end. We now truncate at the earliest match of any stop string,
/// mirroring the streaming path's `apply_stop_string_holdback`
/// (`chat_stream/handle_token.rs`), which already truncates at `find`.
///
/// Empty stop strings are skipped — `"".find` matches at position 0 and would
/// otherwise erase the entire response (the old `strip_suffix("")` was a no-op,
/// so empty stops must stay harmless).
///
/// #100 Finding 5 (acknowledged limitation, out of scope): this is a POST-HOC
/// text truncation only — generation does not actually halt early at a
/// multi-token stop string (e.g. `"\nuser"`, which the tokenizer may split
/// across several tokens). The scheduler stops early only on single stop TOKENS
/// (`eos_tokens`, incl. the force-added `<|im_end|>`; see `is_eos_stop`). So for
/// a multi-token stop string the model runs to a token-level stop (or
/// `max_tokens`) and this function then trims the text — the content is clean
/// but `finish_reason` may be `"length"` rather than `"stop"`. Making a
/// multi-token stop string halt decoding would require detokenized-suffix
/// matching in the decode loop (scheduler-level change) and is deliberately not
/// attempted here.
pub(crate) fn strip_stop_sequences(mut text: String, stops: &[String]) -> String {
    let earliest = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min();
    if let Some(pos) = earliest {
        text.truncate(pos);
    }
    text
}

/// Strip `<think>...</think>` reasoning content from model output.
///
/// Qwen3.5 models generate internal reasoning between `<think>` and `</think>` tags.
/// This must be removed from the API response so that:
/// 1. Clients don't see internal reasoning in the content field
/// 2. Multi-turn conversations aren't corrupted when clients echo assistant content back
///
/// Returns only the text after the final `</think>` tag (the actual response),
/// trimmed of leading whitespace.
/// Extract `<think>...</think>` reasoning from model output.
///
/// Returns `(reasoning_content, response_content)`.
/// - `enable_thinking=true`: reasoning extracted into first element, response in second.
/// - `enable_thinking=false`: reasoning discarded (None), only response returned.
pub(crate) fn extract_thinking(
    text: &str,
    enable_thinking: bool,
    parser: Option<&dyn crate::reasoning_parser::ReasoningParser>,
) -> (Option<String>, String) {
    if let Some(p) = parser {
        p.extract_thinking(text, enable_thinking)
    } else {
        (None, text.to_string())
    }
}

#[cfg(test)]
mod strip_stop_sequences_tests {
    //! Issue #100: `strip_stop_sequences` must truncate at the FIRST occurrence
    //! of any stop string anywhere in the text — not only when the stop string
    //! is a trailing suffix. The pre-fix `strip_suffix` logic ignored mid-text
    //! stop strings, so `stop: ["\nuser", ...]` was a no-op against the
    //! role-marker / prompt-echo run-on.
    use super::strip_stop_sequences;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn truncates_at_midtext_stop_string() {
        // The exact #100 run-on shape: a real answer followed by ChatML role
        // markers + a prompt echo. With `stop: ["\nuser"]` the output must be
        // cut at the first "\nuser".
        let out = strip_stop_sequences(
            "Hallo!\nuser\nSag in einem Satz hallo.\nassistant\nHallo!".to_string(),
            &s(&["\nuser", "\nassistant"]),
        );
        assert_eq!(out, "Hallo!");
    }

    #[test]
    fn picks_earliest_of_multiple_stops() {
        // "\nassistant" occurs later than "\nuser"; truncate at the earliest.
        let out = strip_stop_sequences(
            "answer\nassistant before\nuser after".to_string(),
            &s(&["\nuser", "\nassistant"]),
        );
        assert_eq!(out, "answer");
    }

    #[test]
    fn still_strips_trailing_suffix() {
        // Backwards-compatible with the old suffix behavior.
        let out = strip_stop_sequences("done<|im_end|>".to_string(), &s(&["<|im_end|>"]));
        assert_eq!(out, "done");
    }

    #[test]
    fn no_match_is_unchanged() {
        let out = strip_stop_sequences("clean output".to_string(), &s(&["\nuser"]));
        assert_eq!(out, "clean output");
    }

    #[test]
    fn empty_stops_list_is_unchanged() {
        let out = strip_stop_sequences("clean output".to_string(), &[]);
        assert_eq!(out, "clean output");
    }

    #[test]
    fn empty_stop_string_does_not_erase_text() {
        // An empty stop string must NOT truncate at position 0 (regression
        // guard: the old `strip_suffix("")` was a no-op).
        let out = strip_stop_sequences("keep me".to_string(), &s(&[""]));
        assert_eq!(out, "keep me");
    }

    #[test]
    fn truncation_respects_utf8_boundaries() {
        // The stop string lands right after multi-byte content; truncating at
        // the byte offset returned by `find` is always char-aligned because
        // `find` returns the start byte of the match.
        let out = strip_stop_sequences("Grüße\nuserX".to_string(), &s(&["\nuser"]));
        assert_eq!(out, "Grüße");
    }

    #[test]
    fn truncates_at_eos_marker_strings() {
        // Basic EOS-as-stop-string: the ChatML end-of-turn marker and the
        // fallback EOS text are cut wherever they appear. Mirrors the
        // token-level stop for models/clients that surface them as text.
        let out = strip_stop_sequences(
            "answer<|im_end|>\n<|im_start|>user".to_string(),
            &s(&["<|im_end|>", "<|endoftext|>"]),
        );
        assert_eq!(out, "answer");
    }

    #[test]
    fn overlapping_prefix_stops_truncate_at_shared_offset() {
        // Overlapping prefixes (`</answer` vs `</answer>`): both start at the
        // same offset, so truncating at the earliest `find` position yields the
        // identical result no matter which one the `min` tie picks. (The old
        // suffix logic needed longest-first sorting to get this right; the
        // first-occurrence rule is order-independent.)
        let out = strip_stop_sequences(
            "text</answer>tail".to_string(),
            &s(&["</answer>", "</answer"]),
        );
        assert_eq!(out, "text");
    }

    #[test]
    fn truncates_at_multibyte_cjk_stop_string() {
        // A fully multi-byte (3-bytes-per-char) stop string truncates cleanly
        // on a char boundary — no reliance on the stop or the prefix being ASCII.
        let out = strip_stop_sequences("結果です。終了ここから先".to_string(), &s(&["終了"]));
        assert_eq!(out, "結果です。");
    }

    /// #100 Finding 3: `/v1/completions` must strip hidden reasoning BEFORE
    /// applying user stop sequences. This test pins the operation ORDER used in
    /// `completions.rs`: a stop string that also appears inside
    /// `<think>...</think>` must NOT truncate the visible answer.
    ///
    /// Correct order (thinking-first): the reasoning — including its own
    /// `\nuser` — is removed, leaving `"The answer is 42."`; the stop then finds
    /// no match and the answer survives.
    ///
    /// Buggy order (stops-first, pre-fix): the raw text `find("\nuser")` hits
    /// inside the reasoning at an early offset, truncating there and dropping the
    /// real answer entirely — the regression this reorder prevents.
    #[test]
    fn completions_strips_thinking_before_stops() {
        use crate::api::strip::strip_thinking_tags;
        let raw = "<think>Consider the\nuser question carefully.</think>The answer is 42.";
        let stops = s(&["\nuser"]);

        // Correct order (as shipped in completions.rs after the fix).
        let thinking_first = strip_stop_sequences(strip_thinking_tags(raw), &stops);
        assert_eq!(thinking_first, "The answer is 42.");

        // Buggy order truncates inside the hidden reasoning — asserted here so a
        // future refactor that reintroduces it is caught.
        let stops_first = strip_thinking_tags(&strip_stop_sequences(raw.to_string(), &stops));
        assert_ne!(
            stops_first, "The answer is 42.",
            "stops-before-thinking must lose the answer (regression guard)"
        );
    }
}
