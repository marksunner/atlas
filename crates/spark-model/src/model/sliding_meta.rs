// SPDX-License-Identifier: AGPL-3.0-only

//! Sliding-window split-pool metadata helpers.
//!
//! Sliding-window layers under the split pool have their own (smaller)
//! block-ID space, so every path that uploads slot mappings / block tables
//! for the FULL pool also uploads a sliding twin built from the sequence's
//! ring (`SequenceState::sliding_block_table`). These helpers are the
//! single source of truth for the staging layout inside the arena's
//! `sliding_meta` buffer and for the ring → table/slot expansion, so the
//! decode / prefill / mixed builders can't drift apart.
//!
//! Layout of `buffers.sliding_meta()` (sized in `BufferSizes::from_config`):
//!   [0 .. 256)                      decode/verify sliding slots (≤32 × i64)
//!   [256 .. 256 + 32*max_blocks*4)  decode/verify ring-expanded block
//!                                   tables (32 rows × max_blocks × i32,
//!                                   row stride = max_blocks entries)
//!   [tables_end(64B-aligned) .. )   prefill sliding slots (m × i64)

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use crate::traits::SequenceState;

/// Runtime ring-invariant verification (STEP37-QUALITY Round 7). When
/// `ATLAS_SLIDING_KV_VERIFY=1`, every prefill chunk and decode step checks
/// that the live read window maps to alias-free `(ring slot, offset)` pairs
/// — the exact property the ring's size algebra guarantees
/// (`R·bs ≥ M_max + window`). This turns a silent, non-deterministic
/// KV-corruption (the failure mode Round 6 attributed to the split) into a
/// loud, deterministic error at the first offending step, so the ring can be
/// trusted "correct when opted in" rather than merely assumed correct on
/// paper. Cheap when off (one atomic load); ~O(window) when on.
pub(crate) static SLIDING_KV_VERIFY: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("ATLAS_SLIDING_KV_VERIFY").as_deref() == Ok("1"));

/// Rows provisioned for decode/verify sliding tables — matches the
/// `bt_rows` headroom in `BufferSizes::from_config`.
pub(crate) const SLIDING_BT_ROWS: usize = 32;
/// Byte offset of the decode/verify ring-expanded tables region.
pub(crate) const SLIDING_TABLES_OFF: usize = 256;

impl TransformerModel {
    /// Ring length R when the split pool is active for this model, else 0.
    /// Cached at construction so metadata builders don't need the KV lock.
    #[inline]
    pub(crate) fn sliding_ring_len(&self) -> usize {
        self.sliding_split_ring
    }

    /// Base of the decode/verify sliding-slot region (row `i` = seq `i`).
    #[inline]
    pub(crate) fn sliding_decode_slots_base(&self) -> DevicePtr {
        self.buffers.sliding_meta()
    }

    /// Base of the decode/verify ring-expanded table region.
    #[inline]
    pub(crate) fn sliding_decode_tables_base(&self) -> DevicePtr {
        self.buffers.sliding_meta().offset(SLIDING_TABLES_OFF)
    }

    /// Base of the prefill sliding-slot region (m × i64).
    #[inline]
    pub(crate) fn sliding_prefill_slots_base(&self) -> DevicePtr {
        let tables = SLIDING_BT_ROWS * self.max_blocks_per_seq as usize * 4;
        self.buffers
            .sliding_meta()
            .offset(SLIDING_TABLES_OFF + ((tables + 63) & !63))
    }

    /// Verify (env `ATLAS_SLIDING_KV_VERIFY=1`) that the live read window
    /// `[lo_pos, hi_pos)` of a single forward pass maps to distinct
    /// `(physical ring slot, offset)` pairs. Two live positions sharing a
    /// pair means the KV-write for one silently overwrites the other before
    /// it is read — the split's only correctness hazard, and the signature
    /// of an under-sized ring. Returns `Err` (caller bails the forward pass)
    /// naming the two colliding positions. No-op when the split is off or
    /// the flag is unset. Pure host-side check against
    /// `SequenceState::sliding_block_table` — mirrors exactly what the KV
    /// write/read kernels index, so a pass here proves the on-device slots
    /// are alias-free for this step.
    pub(crate) fn verify_sliding_no_alias(
        &self,
        seq: &SequenceState,
        lo_pos: usize,
        hi_pos: usize,
        bs: usize,
        ring_len: usize,
    ) -> Result<()> {
        if ring_len == 0 || !*SLIDING_KV_VERIFY {
            return Ok(());
        }
        // Only the trailing `R·bs` positions can possibly collide (anything
        // older is >= R·bs away from `hi_pos` and thus a different ring
        // wrap); bounding the scan keeps long-context decode cheap.
        let span = ring_len * bs;
        let lo = lo_pos.max(hi_pos.saturating_sub(span));
        let mut seen: std::collections::HashMap<(u32, usize), usize> =
            std::collections::HashMap::with_capacity(hi_pos - lo);
        for pos in lo..hi_pos {
            let blk = seq
                .sliding_physical_block_for(pos / bs, ring_len)
                .unwrap_or(self.dummy_sliding_block);
            let key = (blk, pos % bs);
            if let Some(prev) = seen.insert(key, pos) {
                anyhow::bail!(
                    "ATLAS_SLIDING_KV_VERIFY: ring alias — positions {prev} and {pos} both map \
                     to sliding (block {blk}, offset {}); live window [{lo_pos},{hi_pos}), \
                     R={ring_len}, bs={bs}. The ring is under-sized for M_max+window; \
                     the KV write for {prev} was clobbered before its read.",
                    pos % bs,
                );
            }
        }
        Ok(())
    }

    /// Sliding slot mapping (i64) for absolute token position `pos`:
    /// `ring[(pos / bs) % R] * bs + pos % bs`. Falls back to the dummy
    /// sliding block if the ring slot is missing (defensive — the ensure
    /// helpers allocate the ring before any slot is consumed).
    #[inline]
    pub(crate) fn sliding_slot_for(
        &self,
        seq: &SequenceState,
        pos: usize,
        bs: usize,
        ring_len: usize,
    ) -> i64 {
        let blk = seq
            .sliding_physical_block_for(pos / bs, ring_len)
            .unwrap_or(self.dummy_sliding_block);
        (blk as i64) * (bs as i64) + ((pos % bs) as i64)
    }

    /// Ring-expanded block-table entries for logical blocks
    /// `[0, num_entries)`: entry `i` = `ring[i % R]`. Missing ring slots
    /// (sequence shorter than the ring) pad with the dummy sliding block —
    /// those logical blocks are beyond the current seq_len, so kernels
    /// only ever mask/skip them.
    pub(crate) fn build_sliding_table_i32(
        &self,
        seq: &SequenceState,
        num_entries: usize,
        ring_len: usize,
    ) -> Vec<i32> {
        (0..num_entries)
            .map(|i| {
                seq.sliding_physical_block_for(i, ring_len)
                    .unwrap_or(self.dummy_sliding_block) as i32
            })
            .collect()
    }

    /// Batched twin of the decode metadata upload: sliding slots
    /// `[padded_n] i64` + ring-expanded tables `[padded_n × row_stride]`
    /// i32 (dummy-sliding-block padded, same sentinel pattern as the
    /// full-pool tables). Returns the (slots, tables) device pointers, or
    /// NULLs when the split pool is off. `row_stride_entries` MUST equal
    /// the `max_blocks_per_seq` the kernels use for full-pool row
    /// indexing — both tables are indexed with the same stride.
    pub(crate) fn upload_sliding_batch(
        &self,
        seqs: &[&mut SequenceState],
        padded_n: usize,
        row_stride_entries: usize,
        bs: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let ring_len = self.sliding_ring_len();
        if ring_len == 0 {
            return Ok((DevicePtr(0), DevicePtr(0)));
        }
        anyhow::ensure!(
            padded_n <= SLIDING_BT_ROWS && row_stride_entries <= self.max_blocks_per_seq as usize,
            "sliding metadata staging overflow: padded_n={padded_n} (max {SLIDING_BT_ROWS}), \
             row_stride={row_stride_entries} (max {})",
            self.max_blocks_per_seq,
        );
        let dummy_slot = (self.dummy_sliding_block as i64) * (bs as i64);
        let mut slots: Vec<i64> = Vec::with_capacity(padded_n);
        let mut tables: Vec<i32> =
            vec![self.dummy_sliding_block as i32; padded_n * row_stride_entries];
        for (i, seq) in seqs.iter().enumerate() {
            slots.push(self.sliding_slot_for(seq, seq.seq_len, bs, ring_len));
            let entries = seq.block_table.len().min(row_stride_entries);
            for (j, v) in self
                .build_sliding_table_i32(seq, entries, ring_len)
                .into_iter()
                .enumerate()
            {
                tables[i * row_stride_entries + j] = v;
            }
        }
        slots.resize(padded_n, dummy_slot);

        let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
        let table_bytes: Vec<u8> = tables.iter().flat_map(|b| b.to_le_bytes()).collect();
        self.gpu
            .copy_h2d_async(&slot_bytes, self.sliding_decode_slots_base(), stream)?;
        self.gpu
            .copy_h2d_async(&table_bytes, self.sliding_decode_tables_base(), stream)?;
        Ok((
            self.sliding_decode_slots_base(),
            self.sliding_decode_tables_base(),
        ))
    }

    /// Upload one sequence's ring-expanded table into decode/verify table
    /// row `row` and its sliding slot for position `pos` into slot row
    /// `row`. Used by the single-seq decode and (per row) by batched
    /// decode/mixed builders. No-op when the split pool is off.
    pub(crate) fn upload_sliding_decode_row(
        &self,
        seq: &SequenceState,
        bs: usize,
        row: usize,
        pos: usize,
        num_entries: usize,
        row_stride_entries: usize,
        stream: u64,
    ) -> Result<()> {
        let ring_len = self.sliding_ring_len();
        if ring_len == 0 {
            return Ok(());
        }
        debug_assert!(row < SLIDING_BT_ROWS);
        debug_assert!(row_stride_entries <= self.max_blocks_per_seq as usize);
        let slot = self.sliding_slot_for(seq, pos, bs, ring_len);
        self.gpu.copy_h2d_async(
            &slot.to_le_bytes(),
            self.sliding_decode_slots_base().offset(row * 8),
            stream,
        )?;
        let table = self.build_sliding_table_i32(seq, num_entries, ring_len);
        if !table.is_empty() {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(table.as_ptr() as *const u8, table.len() * 4)
            };
            self.gpu.copy_h2d_async(
                bytes,
                self.sliding_decode_tables_base()
                    .offset(row * row_stride_entries * 4),
                stream,
            )?;
        }
        Ok(())
    }
}
