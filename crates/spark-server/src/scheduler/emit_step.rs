// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    // Cooperative cancellation from the streaming pipeline. The
    // stream-side loop guards (Bug-2 name-run cap, F11 within-dedup,
    // F44 perm-fail, loop-watchdog) flip this flag when they decide
    // the response should end. Treat it like an EOS: finalise now so
    // `handle_done` runs with the proper `tool_loop_capped` /
    // `finish_reason="length"` machinery, instead of letting the
    // model keep emitting tokens that just get suppressed.
    if let Some(ref f) = a.cancel_flag
        && f.load(std::sync::atomic::Ordering::Acquire)
    {
        a.finished = true;
        return;
    }

    // ChatML role-boundary HARD stop (`<|im_start|>`).
    //
    // Handled BEFORE grammar advance / EOS suppression: if the model
    // hallucinated a `<|im_start|>` mid-turn, we must end the turn regardless
    // of grammar / require_tool_call / min_tokens. The regular EOS path at
    // line ~3020 honors `suppress_eos`, which is true while a tool-call
    // grammar is active — so if we fell through to it, the tokenizer would
    // strip `<|im_start|>` (special-token) but the following role literal
    // (`user` / `assistant` — regular tokens) would stream to the client,
    // poisoning its context and causing the observed multi-turn drift /
    // "file was corrupted" hallucinations in opencode.
    if let Some(ims) = im_start_hard_stop()
        && tok == ims
    {
        // Push the hard-stop token to output_tokens so lifecycle.rs reports
        // `finish_reason="stop"` (because `<|im_start|>` is registered in
        // `eos_tokens` at startup — see tokenizer_runtime.rs::im_start_id).
        // Without this push, `last_tok = output_tokens.last()` is the prior
        // content token, lifecycle's `is_eos` check fails, and the response
        // is mis-reported as `finish_reason="length"` (Bug 3 from OpenClaw
        // 2026-05-08 session: "Done: 13 tokens (length) despite max_tokens=
        // 8192" — clients then misinterpret the truncation as a real
        // length-limit hit and either retry or surface a wrong error).
        // The streamed-text path strips stop tokens server-side, so the
        // client never sees the literal `<|im_start|>` bytes.
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!(
            "<|im_start|> hard-stop fired (id={ims}); ending turn before grammar/suppress_eos"
        );
        return;
    }

    // Fix B (2026-06-05, kill-switch): <tool_response> hard stop — the model must
    // never generate this control token; if it does (post-tool-call runaway), end
    // the turn. Mirrors the <|im_start|> hard stop above.
    if tool_response_stop_enabled()
        && let Some(trs) = tool_response_hard_stop()
        && tok == trs
    {
        a.output_tokens.push(tok);
        a.finished = true;
        tracing::debug!("<tool_response> hard-stop fired (id={trs}); ending turn");
        return;
    }

    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        a.thinking_budget = Some(a.spontaneous_think_budget);
        tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        return; // don't emit <think> as content
    }

    // Silently skip </think> tokens outside thinking mode (same as process_decode_logits).
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
        }
        return;
    }

    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }

    // Tool-body / parameter-body state machine.
    //
    // SM1 (2026-05-26): extracted from inline-in-emit_token to the free
    // function `update_tool_param_state` so the regular non-MTP decode
    // path (`decode_logits_step.rs`) can call it too. Previously the
    // state was ONLY updated when `emit_token` ran — which happens from
    // spec/verify paths but NOT from `process_decode_logits`. With
    // mtp=false (Qwen3.6 baseline), the state machine never ran and
    // every dependent gate (A1 rep-penalty toggle, B1 margin detector)
    // was silently dead code. (The pos-0 close-tag/AM1 logit-bias that
    // also depended on this state was removed 2026-06-03; the state is
    // still required for A1/B1 and the adadec_diag dump.)
    update_tool_param_state(a, tok);

    // Fix A (2026-06-05): mark a tool call complete on `</tool_call>` (outside
    // thinking) so the EOS-escape gate can lift suppression. Inert unless
    // `tool_eos_escape_enabled()` (default OFF).
    if a.tool_call_end_token == Some(tok) && !a.inside_thinking {
        a.tool_call_completed = true;
    }

    // F2 mirror (Iter 46, 2026-06-02): reset the inter-tool prose budget when
    // a tool call opens on the MTP/emit path — parity with the non-MTP reset
    // in `decode_logits_step.rs` (the `tool_call_start_token` branch). Without
    // this the budget would accrue across the whole response and the MTP-path
    // budget watchdog (added below) would false-fire after the first
    // `max_inter_tool_prose` content tokens even across legitimate multi-tool
    // turns. Keyed identically: tool-call open, not inside `<think>`.
    if a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
    }

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    let mut disengage_grammar = false;
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        let advanced = gs.accept_token(tok);
        if !advanced {
            // Grammar/model disagreement (BUG#2 class: e.g. a merged BPE token
            // like `><` or a `</X` content run the qwen3_coder value rule
            // forbids, often surfaced via an under-masked MTP draft). The token
            // is already a legitimate model emission; the matcher is now
            // desynced. Previously we set `a.finished = true` here — a
            // CATASTROPHIC cliff that lost the ENTIRE agentic turn on a single
            // refused token (root cause of the opencode webserver_ok gap).
            // Instead, DISENGAGE the grammar for the remainder of this response
            // and continue decoding UNCONSTRAINED — exactly what vLLM (the 10/10
            // reference) does by parsing tools post-hoc. Atlas's server-side
            // tool parser still extracts tool calls from the emitted text, so
            // the structural guarantee is gracefully traded for turn survival.
            tracing::warn!(
                tok,
                output_len = a.output_tokens.len(),
                "gs.accept_token returned false — grammar/model disagreement; disengaging grammar for the remainder of this response (free decode + post-hoc tool parse) instead of aborting the turn."
            );
            disengage_grammar = true;
        }
    }
    if disengage_grammar {
        // Drop the matcher: subsequent decode steps see `grammar_state == None`
        // and decode unconstrained. Set after the `ref mut gs` borrow ends.
        a.grammar_state = None;
    }

    // Thinking / content bookkeeping — run FIRST, BEFORE the EOS decision, to
    // mirror `decode_logits_step::process_decode_logits` (#100 v3 Finding 2).
    //
    // The non-MTP path advances its thinking-token counter (or runs
    // `handle_content_token`: `consume_generation_budget`, content-loop / prose
    // watchdogs) in its per-token step, AHEAD of its EOS gate — so a SUPPRESSED
    // EOS there still advances `thinking_tokens`, the thinking budget, the
    // content watchdogs, and the generation budget before being discarded. v2
    // ran this block AFTER the push and returned early on a suppressed EOS,
    // freezing all of that state on the MTP/verify path (the v2 evaluator's
    // parity finding). Running it first restores exact parity: every token —
    // normal, stopping EOS, or suppressed EOS — advances the same state as
    // non-MTP. Like non-MTP this runs PRE-push (`output_tokens` does not yet
    // contain `tok`), so the loop detectors see the identical slice.
    //
    // Detect the `</think>` transition. Thinking tokens are "free" (don't
    // decrement remaining) and are stripped from the API output.
    if a.inside_thinking {
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            a.sentence_defer_count = 0;
            a.think_ended = true;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            if let Some(budget) = a.thinking_budget
                && a.thinking_tokens >= budget
                && !a.force_end_thinking
            {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Thinking budget exhausted ({budget} tokens), arming </think>; \
                     deferring to next sentence boundary"
                );
            }
        }
    } else {
        a.consume_generation_budget();
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
        // Content-phase loop watchdog. Mirrored from
        // `handle_content_token` (decode_logits_content.rs) because
        // that handler is only invoked on the non-MTP decode path
        // (`process_decode_logits`). MTP speculative decode
        // (`verify_k2_step`) reaches every token through this
        // `emit_token` instead — without this mirror, the
        // content-loop watchdog never fires while MTP is enabled, and
        // the model can burn the full `max_tokens` budget on a
        // period-N attractor. Observed live 2026-05-24 on
        // opencode-hotfix2b.jsonl seq=13: 8193 content tokens of
        // `[29, 198, 510, 15704, …]` period-4 loop (the
        // `parameter>\n` attractor) with no watchdog fire,
        // finish=length.
        //
        // 2026-05-24 sweep #3: Re-introduced the `!a.inside_tool_body`
        // gate (mirrors the handle_content_token policy). The previous
        // inside-body false-positives turned out to be triggered by a
        // separate MTP-pipeline gap (see bench/hotfix3-debug/
        // SYNTHESIS.md). With the pipeline correctly applied to MTP
        // verify, JSON structural repetition is bounded by the
        // grammar's terminal state. The `parameter>\n` real-loop case
        // is still caught one tick AFTER the model exits the tool
        // body — its emission outside the body forms a tight period-N
        // tail.
        //
        // Skip rollback here — `emit_token` doesn't take `&dyn Model`
        // (the SSM rewind requires it) and plumbing it through every
        // call site would balloon the diff. Instead set `a.finished`
        // and let the lifecycle close the response. The non-MTP path
        // retains rollback via `handle_content_token`.
        use crate::scheduler::helpers::{
            CONTENT_LOOP_CHECK_STRIDE, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_PERIOD_MIN, detect_content_token_loop_normalized_with,
            detect_content_token_loop_with, disable_watchdogs, enable_loop_watchdog,
            numeric_token_mask,
        };
        a.content_tokens = a.content_tokens.saturating_add(1);
        // F1 (2026-06-02): unconditional per-generation post-think content
        // cap. Fires regardless of `inside_tool_body` so it bounds the
        // runaway no matter which heuristic state machine desynced (RC1/
        // RC2/RC3). Gated on `grammar_state.is_some()` ⇒ only tool-active
        // requests are ever capped (plain chat attaches no grammar and is
        // never truncated). Default 100_000 (`MAX_POST_THINK_CONTENT_TOKENS`)
        // = no-op; Qwen3.6-35B-A3B-FP8 sets 1536 in MODEL.toml.
        if !disable_watchdogs()
            && a.grammar_state.is_some()
            && a.content_tokens > watchdog_params().max_post_think_content_tokens
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                max = watchdog_params().max_post_think_content_tokens,
                "post-think content cap exceeded in MTP/emit path; ending response (tool-active request would otherwise burn to max_tokens)"
            );
            a.finished = true;
        }
        if !disable_watchdogs()
            && enable_loop_watchdog()
            && !a.inside_tool_body
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && (detect_content_token_loop_with(&a.output_tokens, a.repetition_detection)
                || numeric_token_mask().as_deref().is_some_and(|m| {
                    detect_content_token_loop_normalized_with(
                        &a.output_tokens,
                        m,
                        a.repetition_detection,
                    )
                }))
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                "Content-loop watchdog fired in MTP/emit path (period-{}…{} repeat); ending response",
                CONTENT_LOOP_PERIOD_MIN,
                CONTENT_LOOP_PERIOD_MAX,
            );
            // Push an EOS so finish_sequence reports "stop", not "length" —
            // see decode_logits_content.rs content-loop fallback for rationale.
            if let Some(&eos) = a.eos_tokens.first() {
                a.output_tokens.push(eos);
            }
            a.finished = true;
        }

        // F2 mirror (Iter 46, 2026-06-02): inter-tool PROSE-BUDGET watchdog on
        // the MTP/emit path. The 2026-05-24 mirror above copied only the
        // content-LOOP guard; the prose-budget guard (decode_logits_content.rs)
        // stayed non-MTP-only — so with `--num-drafts ≥ 1` (MTP/verify path),
        // a turn that wanders WITHOUT producing a parseable tool call had NO
        // bound and burned the whole `max_tokens` budget (~270s at 30 tok/s),
        // starving the agent of turns. This was the dominant opencode
        // `webserver_ok` 360s-timeout cause: at deep context the model flips
        // its tool opener to Anthropic-XML `<invoke name=…>`, which never
        // matches the qwen3_coder trigger, so the trigger-gated grammar stays
        // dormant and the wander is not a tight period-≤64 loop the content
        // watchdog catches. Same gates as the non-MTP block: free-text only
        // (`!inside_tool_body`) and grammar attached (`grammar_state.is_some()`
        // ⇒ a tool request, never plain chat — so a long chat answer is not
        // truncated). No rollback here: `emit_token` has no `&dyn Model` (the
        // SSM rewind needs it), so we hard-stop exactly like the content-loop
        // mirror; the sanitizer + post-hoc tool parser salvage what was emitted.
        // F4 (2026-06-02): gate on the sticky `tool_request` flag (set at
        // prefill, survives a graceful grammar disengage) instead of
        // `grammar_state.is_some()` — otherwise a disengaged tool turn on
        // the MTP path wanders to `max_tokens` with the budget inert.
        if !disable_watchdogs() && !a.inside_tool_body && a.tool_request {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = watchdog_params().max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    output_len = a.output_tokens.len(),
                    "Inter-tool prose budget exhausted in MTP/emit path; ending response \
                     (no tool call after budget — would otherwise burn to max_tokens)."
                );
                a.finished = true;
            }
        }
    }

    // EOS handling — mirror `decode_logits_step::process_decode_logits` EXACTLY
    // so the MTP/verify path and the non-MTP decode path share one stop policy
    // (#100 Finding 1). Decided AFTER the thinking/content bookkeeping above (so
    // a suppressed EOS advances the same counters/budget/watchdogs as non-MTP —
    // #100 v3 Finding 2) but BEFORE `output_tokens.push(tok)` below, so a
    // SUPPRESSED EOS is never counted in `output_tokens` — identical to the
    // non-MTP empty suppressed branch, which discards without pushing.
    //
    // Fix A (2026-06-05, kill-switch): in tool_choice="auto" the grammar's
    // is_terminated() never becomes true after a tool call, so EOS is suppressed
    // forever — trapping the model into a hallucinated-transcript runaway. When
    // enabled and a tool call has completed (and we're not inside a tool body /
    // thinking), lift the grammar suppression so the model's natural EOS ends the
    // turn. Inert unless ATLAS_TOOL_EOS_ESCAPE=1.
    let eos_escape = tool_eos_escape_enabled()
        && a.tool_call_completed
        && !a.inside_tool_body
        && !a.inside_thinking;
    let grammar_suppresses_eos = a
        .grammar_state
        .as_ref()
        .is_some_and(|gs| !gs.is_terminated())
        && !eos_escape;
    let legacy_suppresses_eos = a.require_tool_call;
    let min_tokens_suppresses = a.output_tokens.len() < a.min_tokens;
    // #100 Finding 1: thinking / post-think EOS suppression — MISSING on this
    // path pre-fix, so an `<|im_end|>` sampled inside `<think>` (spurious) or in
    // the first few post-`</think>` content tokens could prematurely end the
    // turn. Only `</think>` (think_end_token) ends the thinking phase. Kept
    // byte-for-byte in lockstep with `decode_logits_step.rs`. NB: the bookkeeping
    // above already ran, so if `tok` was `</think>` we are now `!inside_thinking`
    // with `think_ended`, exactly as non-MTP computes these at the same point.
    let thinking_suppresses_eos = a.inside_thinking;
    // Keep in lockstep with decode_logits_step's POST_THINK_MIN_CONTENT.
    const POST_THINK_MIN_CONTENT: u32 = 16;
    let post_think_content_tokens =
        (a.output_tokens.len() as u32).saturating_sub(a.thinking_tokens);
    let post_think_suppresses_eos =
        a.think_ended && post_think_content_tokens < POST_THINK_MIN_CONTENT;
    let suppress_eos = grammar_suppresses_eos
        || legacy_suppresses_eos
        || min_tokens_suppresses
        || thinking_suppresses_eos
        || post_think_suppresses_eos;

    match retain_token_after_eos_decision(
        tok,
        logprobs,
        &a.eos_tokens,
        suppress_eos,
        &mut a.output_tokens,
        &mut a.logprobs_data,
    ) {
        EmitRetention::StoppingEos => {
            // Stopping EOS: count it (correct token count + finish_reason="stop") but
            // do NOT stream it (OpenAI spec: returned text must exclude the stop
            // sequence). Bookkeeping already ran above — matching the non-MTP path,
            // which runs `handle_content_token` before its `is_eos_stop` push.
            a.finished = true;
            return;
        }
        EmitRetention::SuppressedEos => {
            // EOS suppressed (grammar mid tool-call, unmet min_tokens, legacy
            // require_tool_call, inside `<think>`, or the post-think content floor).
            // Discard it: do NOT push to `output_tokens` (#100 Finding 2) and do NOT
            // stream it — the model keeps generating. The thinking/content counters,
            // budget, and watchdogs were ALREADY advanced by the bookkeeping above,
            // so this is byte-for-byte the non-MTP suppressed branch: bookkeeping
            // ran, token discarded, no finish.
            return;
        }
        EmitRetention::Retained => {}
    }

    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if let ResponseSink::Streaming(ref tx) = a.sink
        && !suppress_stream
    {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        // Discriminate transient backpressure (channel full) from a real
        // consumer-drop (channel closed). The previous `try_send().is_err()`
        // collapsed the two and silently terminated the seq with
        // `finish_reason="length"` whenever the SSE consumer momentarily
        // stalled — surfaced as "request stops half-way" in Open WebUI.
        match tx.try_send(event) {
            Ok(()) => {}
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!("Streaming receiver dropped, finishing seq");
                a.finished = true;
                return;
            }
            Err(tokio::sync::mpsc::error::TrySendError::Full(event)) => {
                if let Err(e) = tx.blocking_send(event) {
                    tracing::error!("Streaming send failed during backpressure: {e}");
                    a.finished = true;
                    return;
                }
            }
        }
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Some(GrammarState)` if compilation succeeds, `None` otherwise
/// (logging a warning on failure so the request falls back to legacy tool_call
/// suppression). Called once per request during prefill.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
    eos_tokens: &[u32],
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return None;
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            match GrammarState::new(&grammar, vocab_size) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    // Exempt the model's stop/EOS tokens from grammar refusal
                    // so a legitimate end-of-turn token cannot desync the NPDA
                    // and truncate the response (see GrammarState::accept_token).
                    Some(state.with_stop_tokens(eos_tokens))
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}

// Tool-body / parameter-body state machine, hoisted out of
// `emit_token` (SM1, 2026-05-26).
//
// Both speculative-decoding paths (`verify_k2_step`, `verify_k4_step`,
// `verify_dflash_step`, `spec_step`) and the regular non-spec decode
// path (`decode_logits_step::process_decode_logits`) call this on
// every emitted token so the state machine stays in sync with
// `a.output_tokens`. The previous inline version was unreachable
// from the non-spec path, leaving the close-tag mask, AM1 attractor
// suppression, B1 margin detector, and A1 penalty toggle all silently
// dead.
//
// **Slice semantics**: this function does NOT assume `tok` has been
// pushed onto `a.output_tokens` or that it has not. It auto-detects
// from `a.output_tokens.last()` and slices accordingly:
//  - `emit_token` calls this BEFORE pushing → `last()` is the prior
//    token, lookback uses the full slice.
//  - `decode_logits_step::process_decode_logits` calls this AFTER
//    pushing → `last()` is `tok`, lookback excludes the trailing
//    entry so the search for `<parameter=KEY>` ending at the current
//    `>` is correct in both cases.
//
// State mutations:
//  - `a.inside_tool_body`         set on `<tool_call>`, cleared on `</tool_call>`
//  - `a.tool_body_streak_tokens`  ++ per body token, reset on enter/exit
//  - `a.inside_parameter_body`    set on `<parameter=KEY>` close `>`, cleared on `</`
//  - `a.param_body_chars_emitted` ++ per non-close body token
//  - `a.finished`                 forced when stuck >MAX_TOOL_BODY_TOKENS
//
// Token IDs are Qwen3.6 byte-level BPE (verified via /tokenize 2026-05-25):
//   27 = `<`, 28 = `=`, 29 = `>`, 510 = `</`, 15704 = `parameter`.

/// Cap on tool-call ENVELOPE tokens (everything inside `<tool_call>…</tool_call>`
/// that is NOT a parameter-value body). Catches a model that opens `<tool_call>`
/// and never reaches `</tool_call>` — it would otherwise burn to max_tokens.
const MAX_TOOL_BODY_TOKENS: u32 = 1024;

/// Pure decision core for the envelope-stuck guard (CC6, 2026-06-07).
/// Tokens of a parameter VALUE (`inside_parameter_body`) are exempt — a
/// legitimately large single-file Write must stream without tripping the cap.
/// Only envelope tokens (`<parameter=KEY>` openers, inter-parameter junk, any
/// non-value token) advance the streak. Pure over scalars so it is unit-tested
/// directly, mirroring `rollback_tests.rs`'s pure-core approach (no `ActiveSeq`
/// fixture needed). Returns `(new_streak, exceeded_cap)`.
fn advance_envelope_streak(inside_parameter_body: bool, streak: u32) -> (u32, bool) {
    if inside_parameter_body {
        (streak, false)
    } else {
        let s = streak.saturating_add(1);
        (s, s > MAX_TOOL_BODY_TOKENS)
    }
}

/// Pure decision core (#100): does sampling `tok` end the sequence *now*?
///
/// A token stops generation iff it is one of the sequence's EOS / stop-token
/// ids (`eos_tokens`) AND EOS is not currently suppressed. Each decode path
/// composes its own `suppress_eos` (grammar mid tool-call, unmet `min_tokens`
/// floor, legacy `require_tool_call`, thinking / post-think gates); this core
/// is the single "should we stop?" decision both paths share:
///   * non-MTP decode — `decode_logits_step::process_decode_logits`
///   * MTP / speculative verify — `emit_token`, invoked per accepted draft by
///     `verify_dflash_step` / `verify_k4_step` / `verify_k2_step`, each of
///     which returns the moment `a.finished` flips. So an EOS anywhere inside a
///     speculated run flips `finished` here, the verify loop breaks, and every
///     later draft in that run is dropped — the "verify path catches EOS in
///     speculated tokens" guarantee.
///
/// Model-agnostic: `eos_tokens` is whatever the tokenizer/config declared plus
/// the force-added ChatML `<|im_end|>` (see `tokenizer_runtime.rs`), never a
/// hardcoded per-model id. Pure over slices/scalars so it is unit-tested
/// directly, mirroring `advance_envelope_streak` (an `ActiveSeq` fixture needs
/// a GPU-backed `SequenceState` and is deliberately never built in unit tests).
pub(crate) fn is_eos_stop(tok: u32, eos_tokens: &[u32], suppress_eos: bool) -> bool {
    eos_tokens.contains(&tok) && !suppress_eos
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmitRetention {
    Retained,
    StoppingEos,
    SuppressedEos,
}

pub(crate) fn retain_token_after_eos_decision(
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    eos_tokens: &[u32],
    suppress_eos: bool,
    output_tokens: &mut Vec<u32>,
    logprobs_data: &mut Vec<crate::api::TokenLogprobs>,
) -> EmitRetention {
    if is_eos_stop(tok, eos_tokens, suppress_eos) {
        output_tokens.push(tok);
        if let Some(lp) = logprobs {
            logprobs_data.push(lp);
        }
        return EmitRetention::StoppingEos;
    }

    if eos_tokens.contains(&tok) && suppress_eos {
        return EmitRetention::SuppressedEos;
    }

    output_tokens.push(tok);
    if let Some(lp) = logprobs {
        logprobs_data.push(lp);
    }
    EmitRetention::Retained
}

pub fn update_tool_param_state(a: &mut ActiveSeq, tok: u32) {
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        // STEP37-QUALITY Round 16: a `<tool_call>` opener that arrives while
        // we are ALREADY inside a tool body (no intervening `</tool_call>`) is
        // always malformed — a second call never validly nests inside the
        // first, and single-call-per-response means `</tool_call>` would have
        // stopped generation. Hermes-format tool loops re-emit the opener
        // (`<tool_call>\nweb_search>` ×hundreds) every ~4 tokens; the old
        // unconditional `= 0` reset pinned the streak at zero forever, so the
        // `MAX_TOOL_BODY_TOKENS` envelope guard never tripped and the runaway
        // ran to the tool `max_tokens` cap (~8192 tok ≈ 270s of garbage). Count
        // the malformed reopen as envelope junk instead of resetting, so the
        // guard bounds the opener loop. The special-token id only appears when
        // the model deliberately emits it (arbitrary parameter-value text
        // tokenizes to byte-BPE, not this single id), so counting it is safe
        // even mid-`<parameter=…>` value.
        if a.inside_tool_body {
            a.tool_body_streak_tokens = a.tool_body_streak_tokens.saturating_add(1);
            if a.tool_body_streak_tokens > MAX_TOOL_BODY_TOKENS {
                tracing::warn!(
                    streak = a.tool_body_streak_tokens,
                    "Repeated <tool_call> opener with no </tool_call> for \
                     {MAX_TOOL_BODY_TOKENS}+ tokens; ending response (opener loop \
                     — would otherwise burn to the tool max_tokens cap)."
                );
                a.finished = true;
            }
        } else {
            a.inside_tool_body = true;
            a.tool_body_streak_tokens = 0;
        }
        return;
    }
    if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
        a.tool_body_streak_tokens = 0;
        a.inside_parameter_body = false;
        a.param_body_chars_emitted = 0;
        return;
    }
    if !a.inside_tool_body {
        return;
    }
    // CC6 (2026-06-07): count ONLY envelope tokens — tokens of a
    // `<parameter=…>` VALUE (the file content of a `write` call) are exempt,
    // so a legitimately large single-file Write streams without tripping the
    // cap. The never-closing-envelope runaway still accumulates here (openers,
    // inter-parameter junk, any token emitted while `inside_parameter_body ==
    // false` still counts). Resets on tool open/close (above) and `</parameter>`
    // exit (below) are unchanged.
    let (streak, exceeded) =
        advance_envelope_streak(a.inside_parameter_body, a.tool_body_streak_tokens);
    a.tool_body_streak_tokens = streak;
    if exceeded {
        tracing::warn!(
            streak = a.tool_body_streak_tokens,
            "Stuck in tool-call ENVELOPE for {MAX_TOOL_BODY_TOKENS}+ tokens with no </tool_call> (excludes parameter-value content); ending response (model never closed the envelope — would otherwise burn to max_tokens). Sanitizer will salvage what it can."
        );
        a.finished = true;
    }

    const TOK_LT: u32 = 27;
    const TOK_PARAMETER: u32 = 15704;
    const TOK_EQ: u32 = 28;
    const TOK_GT: u32 = 29;
    const TOK_LT_SLASH: u32 = 510;

    if a.inside_parameter_body {
        if tok == TOK_LT_SLASH {
            // Start of `</parameter>` close-tag — exit body.
            a.inside_parameter_body = false;
            a.param_body_chars_emitted = 0;
        } else {
            // Any non-close body token advances the counter. The
            // position-0 mask in `decode_logits_seq.rs` (close-tag +
            // AM1 attractor) fires only while this counter is 0, so it
            // deactivates after the first emitted body token.
            a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(1);
        }
        return;
    }

    // Not yet inside_parameter_body: scan for `<parameter=KEY>` opener
    // ending at this `>` (29). Lookback 8 tokens for `[27, 15704, 28]`
    // signature without an intervening close.
    if tok != TOK_GT {
        return;
    }
    // Auto-detect whether `tok` is already in output_tokens (caller
    // pushed) or not (caller has not yet pushed). The signature search
    // must NOT include `tok` itself — the lookback is "what came
    // BEFORE this `>`".
    let n = a.output_tokens.len();
    let n_for_lookback = if n > 0 && a.output_tokens[n - 1] == tok {
        n - 1
    } else {
        n
    };
    if n_for_lookback < 3 {
        return;
    }
    let start = n_for_lookback.saturating_sub(8);
    let window = &a.output_tokens[start..n_for_lookback];
    let mut sig_idx: Option<usize> = None;
    for i in 0..window.len().saturating_sub(2) {
        if window[i] == TOK_LT && window[i + 1] == TOK_PARAMETER && window[i + 2] == TOK_EQ {
            sig_idx = Some(i + 3);
        }
    }
    let Some(after_eq) = sig_idx else { return };
    // Check no intervening `</` or `>` in the KEY span between
    // `<parameter=` and the current `>`.
    let body_segment = &window[after_eq..];
    let intervening_close = body_segment
        .iter()
        .any(|&t| t == TOK_LT_SLASH || t == TOK_GT);
    if !intervening_close {
        a.inside_parameter_body = true;
        a.param_body_chars_emitted = 0;
    }
}

// SM1 unit tests deferred: ActiveSeq has 60+ fields and no public
// constructor; building a test instance requires more boilerplate
// than the state machine itself. Live-verification post-deploy is via
// the A1 rep-penalty toggle / B1 margin-detector behaviour.

#[cfg(test)]
mod cc6_envelope_streak_tests {
    //! CC6 (2026-06-07): the envelope-stuck guard must NOT truncate a large
    //! legitimate file write (parameter-value content), while STILL catching a
    //! `<tool_call>` that never closes. Tested on the pure `advance_envelope_streak`
    //! core (mirrors `rollback_tests.rs` — no `ActiveSeq` fixture required).
    use super::{MAX_TOOL_BODY_TOKENS, advance_envelope_streak};

    #[test]
    fn parameter_value_content_is_exempt_at_any_size() {
        // Simulate a ~6000-token file content streaming inside <parameter=content>.
        let mut streak = 0u32;
        for _ in 0..6000 {
            let (s, exceeded) = advance_envelope_streak(true, streak);
            streak = s;
            assert!(
                !exceeded,
                "parameter-value content must never trip the envelope cap"
            );
        }
        assert_eq!(
            streak, 0,
            "value content must not advance the envelope streak"
        );
    }

    #[test]
    fn never_closing_envelope_still_trips_cap() {
        // True runaway: envelope tokens (NOT inside a parameter value) past the cap.
        let mut streak = 0u32;
        let mut tripped = false;
        for _ in 0..(MAX_TOOL_BODY_TOKENS + 5) {
            let (s, exceeded) = advance_envelope_streak(false, streak);
            streak = s;
            if exceeded {
                tripped = true;
                break;
            }
        }
        assert!(
            tripped,
            "a never-closing envelope emitting >cap non-value tokens must trip"
        );
        assert_eq!(
            streak,
            MAX_TOOL_BODY_TOKENS + 1,
            "fires exactly one token past the cap"
        );
    }

    #[test]
    fn exact_cap_boundary() {
        assert_eq!(
            advance_envelope_streak(false, MAX_TOOL_BODY_TOKENS - 1),
            (MAX_TOOL_BODY_TOKENS, false)
        );
        assert_eq!(
            advance_envelope_streak(false, MAX_TOOL_BODY_TOKENS),
            (MAX_TOOL_BODY_TOKENS + 1, true)
        );
    }

    #[test]
    fn saturates_without_panic() {
        // Envelope streak at u32::MAX must not panic (saturating_add) and stays tripped.
        let (s, exceeded) = advance_envelope_streak(false, u32::MAX);
        assert_eq!(s, u32::MAX);
        assert!(exceeded);
    }
}

#[cfg(test)]
mod eos_stop_tests {
    //! Issue #100: the shared EOS-stop decision (`is_eos_stop`) governs when a
    //! sampled/speculated token ends the sequence, for BOTH the non-MTP decode
    //! path and the MTP speculative-verify path. Tested on the pure core (no
    //! `ActiveSeq` fixture — it needs a GPU `SequenceState`; see
    //! `cc6_envelope_streak_tests` / `rollback_tests.rs` for the same pattern).
    use super::{EmitRetention, is_eos_stop, retain_token_after_eos_decision};

    // ── basic EOS ───────────────────────────────────────────────────────
    #[test]
    fn basic_eos_token_stops() {
        // 151645 = Qwen ChatML <|im_end|>; 151643 = <|endoftext|>. Referenced
        // by id only — no hardcoded per-model literal in production.
        let eos = [151645u32, 151643];
        assert!(is_eos_stop(151645, &eos, false), "<|im_end|> must stop");
        assert!(is_eos_stop(151643, &eos, false), "<|endoftext|> must stop");
    }

    #[test]
    fn non_eos_token_does_not_stop() {
        let eos = [151645u32];
        assert!(
            !is_eos_stop(42, &eos, false),
            "a content token must not stop"
        );
    }

    // ── multiple stop tokens in the set ─────────────────────────────────
    #[test]
    fn any_of_multiple_eos_ids_stops() {
        // A model / request can declare several stop-token ids; matching ANY
        // one ends the turn.
        let eos = [2u32, 7, 151645];
        for &t in &eos {
            assert!(is_eos_stop(t, &eos, false), "id {t} in the set must stop");
        }
        assert!(!is_eos_stop(3, &eos, false));
    }

    // ── suppression (thinking / grammar / min_tokens / require_tool_call) ─
    #[test]
    fn suppressed_eos_does_not_stop() {
        // When EOS is suppressed (grammar mid tool-call, unmet min_tokens,
        // legacy require_tool_call, inside <think>), the very same EOS id must
        // NOT terminate — the model keeps generating. This is what lets a
        // spurious <|im_end|> inside <think> flow through unchanged.
        let eos = [151645u32];
        assert!(
            !is_eos_stop(151645, &eos, true),
            "suppressed EOS must not stop"
        );
    }

    // ── MTP / speculative decode: verify path catches EOS ───────────────
    #[test]
    fn mtp_speculated_run_stops_at_first_eos_and_drops_tail() {
        // Models the real verify loop (`verify_dflash_step` / `verify_k4_step`):
        //   for d in drafts { emit_token(d); if a.finished { return } }
        // `emit_token` pushes the token to output_tokens (so the EOS is counted
        // and finish_reason="stop") then finishes exactly when `is_eos_stop` is
        // true. Every draft AFTER the EOS must be dropped.
        let eos = [151645u32];
        let drafts = [10u32, 11, 151645, 12, 13]; // EOS speculated at index 2
        let mut emitted = Vec::new();
        let mut logprobs = Vec::new();
        let mut finished = false;
        for &d in &drafts {
            let retained =
                retain_token_after_eos_decision(d, None, &eos, false, &mut emitted, &mut logprobs);
            if retained == EmitRetention::StoppingEos {
                finished = true;
                break;
            }
        }
        assert!(finished, "EOS in a speculated run must finish the sequence");
        assert_eq!(
            emitted,
            vec![10, 11, 151645],
            "accepted run includes the EOS; later drafts are dropped"
        );
        assert_eq!(drafts.len() - emitted.len(), 2, "2 trailing drafts dropped");
    }

    #[test]
    fn mtp_suppressed_eos_in_run_does_not_break() {
        // If EOS is suppressed (e.g. speculated <|im_end|> while a tool-call
        // grammar is still open), the speculated run must NOT terminate on it.
        let eos = [151645u32];
        let drafts = [10u32, 151645, 11];
        let mut emitted = Vec::new();
        let mut logprobs = Vec::new();
        let mut finished = false;
        for &d in &drafts {
            let retained = retain_token_after_eos_decision(
                d,
                None,
                &eos,
                /* suppress_eos */ true,
                &mut emitted,
                &mut logprobs,
            );
            if retained == EmitRetention::StoppingEos {
                finished = true;
                break;
            }
        }
        assert!(!finished, "suppressed EOS must not end the speculated run");
        assert_eq!(
            emitted,
            vec![10, 11],
            "suppressed EOS is discarded while later drafts can be retained"
        );
    }
}

#[cfg(test)]
mod emit_retention_tests {
    //! Issue #100 v4: unit tests cover the production retention helper that
    //! `emit_token` and the non-MTP decode path call after bookkeeping. A full
    //! `ActiveSeq` fixture is not constructible here because `SequenceState`
    //! owns model-private GPU/SSM state, so this is the narrowest real emit-path
    //! surface available to unit tests.

    use super::{EmitRetention, retain_token_after_eos_decision};

    fn lp(tok: u32) -> crate::api::TokenLogprobs {
        crate::api::TokenLogprobs {
            token_id: tok,
            logprob: -0.25,
            top: vec![(tok, -0.25)],
        }
    }

    #[test]
    fn suppressed_eos_discards_token_and_logprobs() {
        let eos = [151645u32];
        let mut output_tokens = vec![10];
        let mut logprobs_data = vec![lp(10)];

        let retained = retain_token_after_eos_decision(
            151645,
            Some(lp(151645)),
            &eos,
            /* suppress_eos */ true,
            &mut output_tokens,
            &mut logprobs_data,
        );

        assert_eq!(retained, EmitRetention::SuppressedEos);
        assert_eq!(output_tokens, vec![10]);
        assert_eq!(logprobs_data.len(), 1);
        assert_eq!(logprobs_data[0].token_id, 10);
    }

    #[test]
    fn stopping_eos_retains_token_and_logprobs() {
        let eos = [151645u32];
        let mut output_tokens = vec![10];
        let mut logprobs_data = vec![lp(10)];

        let retained = retain_token_after_eos_decision(
            151645,
            Some(lp(151645)),
            &eos,
            /* suppress_eos */ false,
            &mut output_tokens,
            &mut logprobs_data,
        );

        assert_eq!(retained, EmitRetention::StoppingEos);
        assert_eq!(output_tokens, vec![10, 151645]);
        assert_eq!(logprobs_data.len(), 2);
        assert_eq!(logprobs_data[1].token_id, 151645);
    }

    #[test]
    fn normal_token_retains_token_and_logprobs() {
        let eos = [151645u32];
        let mut output_tokens = Vec::new();
        let mut logprobs_data = Vec::new();

        let retained = retain_token_after_eos_decision(
            42,
            Some(lp(42)),
            &eos,
            /* suppress_eos */ false,
            &mut output_tokens,
            &mut logprobs_data,
        );

        assert_eq!(retained, EmitRetention::Retained);
        assert_eq!(output_tokens, vec![42]);
        assert_eq!(logprobs_data.len(), 1);
        assert_eq!(logprobs_data[0].token_id, 42);
    }
}
