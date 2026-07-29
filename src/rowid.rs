//! The row-id map.
//!
//! The row-id map is the authoritative "where does this row live" structure:
//! `row_id -> (page_id, slot_index)`. Every point lookup, every index-driven
//! access, and every `update`/`delete` by id resolves a row through this map.
//!
//! The map is built at decode time from the decoded data pages and is kept in
//! sync by the mutation layer — *except* that compaction's in-place repack,
//! which renumbers slot indices, is responsible for updating the slot indices
//! stored here. That propagation is exactly the kind of cross-subsystem
//! bookkeeping the surrounding code assumes is handled elsewhere.

use crate::pager::PageId;
use std::collections::BTreeMap;

/// `row_id -> (page_id, slot_index)`.
#[derive(Debug, Clone, Default)]
pub struct RowIdMap {
    map: BTreeMap<u64, (PageId, usize)>,
}

impl RowIdMap {
    pub fn new() -> Self {
        RowIdMap::default()
    }

    pub fn insert(&mut self, row_id: u64, page: PageId, slot: usize) {
        self.map.insert(row_id, (page, slot));
    }

    pub fn get(&self, row_id: u64) -> Option<(PageId, usize)> {
        self.map.get(&row_id).copied()
    }

    pub fn remove(&mut self, row_id: u64) -> Option<(PageId, usize)> {
        self.map.remove(&row_id)
    }

    pub fn contains(&self, row_id: u64) -> bool {
        self.map.contains_key(&row_id)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// All row ids in ascending order.
    pub fn row_ids(&self) -> Vec<u64> {
        self.map.keys().copied().collect()
    }

    /// All rows homed in `page`, as `(row_id, slot_index)` pairs, sorted by
    /// row id.
    pub fn rows_in_page(&self, page: PageId) -> Vec<(u64, usize)> {
        self.map
            .iter()
            .filter(|(_, (p, _))| *p == page)
            .map(|(&r, &(_, s))| (r, s))
            .collect()
    }

    /// Update the slot index of a row (used when a row moves within its page).
    pub fn set_slot(&mut self, row_id: u64, slot: usize) -> bool {
        if let Some(entry) = self.map.get_mut(&row_id) {
            entry.1 = slot;
            return true;
        }
        false
    }

    /// Move a row to a different page and slot.
    pub fn relocate(&mut self, row_id: u64, page: PageId, slot: usize) -> bool {
        if let Some(entry) = self.map.get_mut(&row_id) {
            *entry = (page, slot);
            return true;
        }
        false
    }

    /// Encode for the `.sht` container.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.map.len() * 16);
        out.extend_from_slice(&(self.map.len() as u32).to_le_bytes());
        for (&r, &(p, s)) in &self.map {
            out.extend_from_slice(&r.to_le_bytes());
            out.extend_from_slice(&p.to_le_bytes());
            out.extend_from_slice(&(s as u32).to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> RowIdMap {
        let mut m = RowIdMap::new();
        if buf.len() < 4 {
            return m;
        }
        let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        for _ in 0..n {
            if pos + 16 > buf.len() {
                break;
            }
            let r = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            let p = u32::from_le_bytes(buf[pos + 8..pos + 12].try_into().unwrap());
            let s = u32::from_le_bytes(buf[pos + 12..pos + 16].try_into().unwrap()) as usize;
            pos += 16;
            m.insert(r, p, s);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_round_trip() {
        let mut m = RowIdMap::new();
        m.insert(10, 1, 3);
        m.insert(20, 2, 0);
        assert_eq!(m.get(10), Some((1, 3)));
        assert_eq!(m.rows_in_page(1), vec![(10, 3)]);
        m.set_slot(10, 7);
        assert_eq!(m.get(10), Some((1, 7)));
        let bytes = m.encode();
        let m2 = RowIdMap::decode(&bytes);
        assert_eq!(m2.get(10), Some((1, 7)));
        assert_eq!(m2.get(20), Some((2, 0)));
    }
}
