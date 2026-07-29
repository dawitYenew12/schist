//! The free-space map.
//!
//! The FSM tracks two things:
//!
//! - which *page ids* are currently free (the free-page list), and
//! - how much free space each live data page has (so the insert path can pick a
//!   page with room, or decide to allocate a new one).
//!
//! The FSM is an in-memory structure serialized into an `Fsm` page on
//! checkpoint. The compaction layer returns freed page ids to the FSM and
//! updates per-page free-space counters when pages are split or merged.

use std::collections::BTreeSet;

/// The free-space map.
#[derive(Debug, Clone, Default)]
pub struct Fsm {
    /// Free page ids, kept sorted for deterministic allocation.
    free_pages: BTreeSet<u32>,
    /// `page_id -> bytes of free space remaining`.
    space: std::collections::BTreeMap<u32, u32>,
}

impl Fsm {
    pub fn new() -> Self {
        Fsm::default()
    }

    /// Record a free page id (returned by compaction or delete of an entire
    /// page).
    pub fn return_page(&mut self, id: u32) {
        self.free_pages.insert(id);
        self.space.remove(&id);
    }

    /// Take a free page id, if any. Returns the smallest, for determinism.
    pub fn take_free_page(&mut self) -> Option<u32> {
        let id = *self.free_pages.iter().next()?;
        self.free_pages.remove(&id);
        Some(id)
    }

    pub fn has_free_page(&self) -> bool {
        !self.free_pages.is_empty()
    }

    pub fn free_page_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Declare a page's current free space.
    pub fn set_space(&mut self, id: u32, free: u32) {
        self.space.insert(id, free);
    }

    /// Adjust a page's free space by a delta.
    pub fn adjust_space(&mut self, id: u32, delta: i64) {
        let cur = self.space.get(&id).copied().unwrap_or(0) as i64;
        let next = (cur + delta).max(0) as u32;
        self.space.insert(id, next);
    }

    pub fn space_of(&self, id: u32) -> u32 {
        self.space.get(&id).copied().unwrap_or(0)
    }

    /// Find a data page with at least `need` bytes free. Returns the id of the
    /// page with the most free space among the candidates, to delay splits.
    pub fn page_with_space(&self, need: u32) -> Option<u32> {
        let mut best: Option<(u32, u32)> = None;
        for (&id, &free) in self.space.iter() {
            if free >= need {
                match best {
                    Some((_, bf)) if bf >= free => {}
                    _ => best = Some((id, free)),
                }
            }
        }
        best.map(|(id, _)| id)
    }

    pub fn forget(&mut self, id: u32) {
        self.free_pages.remove(&id);
        self.space.remove(&id);
    }

    pub fn total_free_space(&self) -> u64 {
        self.space.values().map(|&v| v as u64).sum()
    }

    pub fn iter_space(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.space.iter().map(|(&id, &free)| (id, free))
    }

    /// Encode the FSM for the `.sht` container: a u32 count of free pages,
    /// then the free page ids, then a u32 count of space entries, then
    /// `(page_id, free)` pairs.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.free_pages.len() as u32).to_le_bytes());
        for id in &self.free_pages {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&(self.space.len() as u32).to_le_bytes());
        for (&id, &free) in &self.space {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&free.to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Fsm {
        let mut fsm = Fsm::new();
        if buf.len() < 4 {
            return fsm;
        }
        let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        for _ in 0..n {
            if pos + 4 > buf.len() {
                break;
            }
            let id = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            fsm.free_pages.insert(id);
        }
        if pos + 4 > buf.len() {
            return fsm;
        }
        let m = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        for _ in 0..m {
            if pos + 8 > buf.len() {
                break;
            }
            let id = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let free = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;
            fsm.space.insert(id, free);
        }
        fsm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_page_round_trip() {
        let mut f = Fsm::new();
        f.return_page(3);
        f.return_page(1);
        assert_eq!(f.take_free_page(), Some(1));
        assert_eq!(f.take_free_page(), Some(3));
        assert_eq!(f.take_free_page(), None);
    }

    #[test]
    fn page_with_space_picks_most() {
        let mut f = Fsm::new();
        f.set_space(0, 10);
        f.set_space(1, 50);
        f.set_space(2, 30);
        assert_eq!(f.page_with_space(20), Some(1));
    }

    #[test]
    fn encode_decode_round_trips() {
        let mut f = Fsm::new();
        f.return_page(7);
        f.set_space(1, 100);
        f.set_space(2, 40);
        let bytes = f.encode();
        let g = Fsm::decode(&bytes);
        assert_eq!(g.free_page_count(), 1);
        assert_eq!(g.space_of(1), 100);
        assert_eq!(g.space_of(2), 40);
    }
}
