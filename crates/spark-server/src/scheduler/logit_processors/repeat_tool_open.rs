// SPDX-License-Identifier: AGPL-3.0-only

//! Suppress a *repeated* `<tool_call>` open immediately after one was
//! just emitted.
//!
//! STEP37-QUALITY Round 13. Step-3.7-Flash emits the tool-call opener
//! `<tool_call>` as a single special token, then continues with the
//! qwen3_coder body `\n<function=NAME>\n<parameter=…>…`. At temperature 0
//! the token *after* the first `<tool_call>` is, for some tool schemas
//! (e.g. `get_weather`), an EXACT greedy tie between `\n` (correct — leads
//! to `<function=NAME>`) and a second `<tool_call>` special token (lp
//! −0.846 vs −0.846 on hardware). The argmax tie-break picks the second
//! `<tool_call>`, which the model then "closes" with a stray `>` — the
//! `<function=NAME>` opener never appears, so the post-hoc tool parser
//! sees `<tool_call><tool_call>>\n<parameter=city>…</function>` with no
//! function name and drops the call (`finish=stop`, `content=">"`), or the
//! `param_as_function_salvage` pass mis-reads the first parameter as the
//! function name. The Round 11B BOS-context change perturbed the
//! distribution enough to expose this tie for `get_weather`; the Round 10
//! binary happened to tie-break the other way.
//!
//! A tool-call opener NEVER validly follows another opener in the
//! qwen3_coder / hermes XML dialect — every call is
//! `<tool_call>…</tool_call>` with a `</tool_call>` before the next open,
//! and Atlas serves one call per response (`</tool_call>` is a stop
//! token), so a second opener never legitimately appears at all. Hard-
//! masking `<tool_call>` while a body is already open is therefore always
//! safe: it removes only a malformed continuation and lets greedy fall to
//! the equally-probable `\n`/`</tool_call>` that puts the model back on the
//! well-formed path (or stops it cleanly).
//!
//! STEP37-QUALITY Round 16 broadens the trigger from "the *immediately*
//! preceding token was the opener" to "we are anywhere inside a tool body
//! (`inside_tool_body`)". The Round 13 case was a tight `<tool_call>
//! <tool_call>` double; the Hermes-format channel loop instead re-opens
//! non-consecutively — `<tool_call>\nweb_search>` ×hundreds, with
//! `\nweb_search>` between the openers — so `last_token == opener` was
//! false and the mask never fired. Because the model never emits the
//! closing `</tool_call>`, `inside_tool_body` stays `true` for the whole
//! loop, so gating on it catches every reopen. `inside_tool_body` is set
//! on the opener and cleared on `</tool_call>` (see
//! `update_tool_param_state`), so the healthy first call and legal
//! parallel calls (separated by `</tool_call>`, which also stops
//! generation) are unaffected.
//!
//! Scoped to `tool_request` (a real tools-present turn) and to the
//! content phase (`!inside_thinking`); `ToolCallDuringThinkingMask`
//! already handles the thinking phase.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

/// Pure decision: should `<tool_call>` be hard-masked this step?
///
/// Returns `Some(opener_id)` iff this is a tools-present turn, we are in
/// the content phase, the opener resolves to a single token, and either
/// (a) we are already inside an unclosed tool body, or (b) the
/// immediately-preceding emitted token was that same opener. Both are
/// always-malformed reopens (Round 16 / Round 13 respectively). Factored
/// out of [`SuppressRepeatedToolOpen::apply`] so the gating logic is
/// unit-testable without constructing an [`ActiveSeq`].
fn should_suppress(
    tool_request: bool,
    inside_thinking: bool,
    inside_tool_body: bool,
    tool_call_start_token: Option<u32>,
    last_token: Option<u32>,
) -> Option<u32> {
    if tool_request
        && !inside_thinking
        && let Some(tc_start) = tool_call_start_token
        && (inside_tool_body || last_token == Some(tc_start))
    {
        return Some(tc_start);
    }
    None
}

pub struct SuppressRepeatedToolOpen;

impl LogitsProcessor for SuppressRepeatedToolOpen {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if let Some(tc_start) = should_suppress(
            a.tool_request,
            a.inside_thinking,
            a.inside_tool_body,
            ctx.tool_call_start_token,
            a.output_tokens.last().copied(),
        ) {
            let idx = tc_start as usize;
            if idx < logits.len() {
                logits[idx] = f32::NEG_INFINITY;
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "suppress_repeated_tool_open"
    }
}

#[cfg(test)]
mod tests {
    use super::should_suppress;

    const TC: Option<u32> = Some(42);

    #[test]
    fn fires_on_doubled_open_in_content_phase() {
        // Round 13: previous emitted token was `<tool_call>` (42), not yet
        // inside a body → mask it.
        assert_eq!(should_suppress(true, false, false, TC, Some(42)), Some(42));
    }

    #[test]
    fn fires_on_reopen_inside_tool_body() {
        // Round 16: Hermes-format loop — inside an unclosed body, the last
        // token was `>` (7), not the opener, but a reopen is still malformed.
        assert_eq!(should_suppress(true, false, true, TC, Some(7)), Some(42));
    }

    #[test]
    fn no_fire_when_previous_token_differs_and_not_in_body() {
        // Previous token was `\n` (7), not the opener, and no body open →
        // leave logits alone (the well-formed first-open path).
        assert_eq!(should_suppress(true, false, false, TC, Some(7)), None);
    }

    #[test]
    fn no_fire_during_thinking() {
        // ToolCallDuringThinkingMask owns the thinking phase.
        assert_eq!(should_suppress(true, true, true, TC, Some(42)), None);
    }

    #[test]
    fn no_fire_without_tool_request() {
        // Plain chat: never touch `<tool_call>` on a repeat.
        assert_eq!(should_suppress(false, false, true, TC, Some(42)), None);
    }

    #[test]
    fn no_fire_when_opener_unresolved() {
        // Tokenizer never resolved `<tool_call>` to a single id.
        assert_eq!(should_suppress(true, false, true, None, None), None);
    }
}
