// SPDX-License-Identifier: AGPL-3.0-only

//! Budget-aware grammar graceful close.

use std::sync::OnceLock;

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

const DEFAULT_FORCED_CLOSE_THRESHOLD: usize = 32;

pub fn forced_close_threshold() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_FORCED_CLOSE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_FORCED_CLOSE_THRESHOLD)
    })
}

pub struct ForcedClose;

impl LogitsProcessor for ForcedClose {
    fn apply(
        &self,
        _logits: &mut [f32],
        a: &mut ActiveSeq,
        _ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.inside_thinking || a.grammar_state.is_none() || a.remaining >= forced_close_threshold()
        {
            return ProcessorOutcome::Continue;
        }

        let Some(gs) = a.grammar_state.as_mut() else {
            return ProcessorOutcome::Continue;
        };
        let Some(tok) = gs.forced_close_token() else {
            return ProcessorOutcome::Continue;
        };

        a.forced_close_activated = true;
        ProcessorOutcome::EmitToken(tok)
    }

    fn name(&self) -> &'static str {
        "forced_close"
    }
}
