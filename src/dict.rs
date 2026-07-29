//! Dictionary pages.
//!
//! A dictionary-encoded (or RLE text) column does not store its text values
//! inline. Instead it stores integer *dictionary ids*, and the actual byte
//! strings live in a *dictionary page*. A column points at its dictionary page
//! via `Column::dict_page`.
//!
//! A dictionary page is a contiguous byte buffer with this layout:
//!
//! ```text
//!   [u32 entry_count]
//!   for each entry: [u32 id][u32 len][len bytes of UTF-8]
//! ```
//!
//! The page's [`Page::gen`](crate::pager::Page::gen) is its *generation*. Every
//! time the page's buffer is replaced — because the dictionary grew past the
//! page's capacity and had to be rewritten into a larger buffer — the
//! generation is bumped and the previous buffer is freed.

use crate::error::Result;
use crate::pager::{PageId, PageKind, Pager};

/// The initial capacity, in bytes, of a freshly allocated dictionary page.
/// Small on purpose: a few distinct values fill it, so growing the dictionary
/// (and therefore rewriting the page) is a routine, organic operation rather
/// than something that only happens under extreme load.
pub const DICT_PAGE_CAPACITY: usize = 64;

/// In-memory mirror of a column's dictionary: the page id and, for each entry,
/// its id and byte offset/length within the page buffer. The authoritative
/// bytes live in the pager page; this struct only records where to find them.
#[derive(Debug, Clone)]
pub struct Dict {
    pub page_id: PageId,
    pub gen: u64,
    /// `id -> (offset, len)` within the page buffer.
    pub entries: std::collections::BTreeMap<u32, (u32, u32)>,
    /// `bytes -> id`, for interning.
    pub by_bytes: std::collections::BTreeMap<Vec<u8>, u32>,
    pub next_id: u32,
    pub capacity: usize,
}

impl Dict {
    pub fn new(page_id: PageId) -> Self {
        Dict {
            page_id,
            gen: 0,
            entries: std::collections::BTreeMap::new(),
            by_bytes: std::collections::BTreeMap::new(),
            next_id: 0,
            capacity: DICT_PAGE_CAPACITY,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the id for a byte string, if interned.
    pub fn id_of(&self, bytes: &[u8]) -> Option<u32> {
        self.by_bytes.get(bytes).copied()
    }

    /// Look up the byte string for an id, reading from the pager page.
    pub fn bytes_of<'a>(&self, pager: &'a Pager, id: u32) -> Option<&'a [u8]> {
        let &(off, len) = self.entries.get(&id)?;
        let page = pager.get(self.page_id)?;
        let off = off as usize;
        let len = len as usize;
        page.buf.get(off..off + len)
    }

    /// The offset/length recorded for an id (without touching the pager).
    pub fn location_of(&self, id: u32) -> Option<(u32, u32)> {
        self.entries.get(&id).copied()
    }

    /// Intern a byte string, returning its id. If the string is new and the
    /// page slab cannot hold it, the page is rewritten into a larger slab
    /// (which bumps the generation). A routine intern that fits writes into
    /// the existing slab in place.
    pub fn intern(&mut self, pager: &mut Pager, bytes: &[u8]) -> Result<u32> {
        if let Some(id) = self.by_bytes.get(bytes).copied() {
            return Ok(id);
        }
        let needed = 8 + bytes.len();
        let used = self.used_bytes();
        if used + needed > self.capacity {
            self.grow(pager)?;
        }
        let id = self.next_id;
        self.next_id += 1;
        let page = pager
            .get_mut(self.page_id)
            .expect("dict page must exist");
        // Write the entry into the slab at the current end, in place.
        let off = used;
        page.buf[off..off + 4].copy_from_slice(&id.to_le_bytes());
        page.buf[off + 4..off + 8].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        page.buf[off + 8..off + 8 + bytes.len()].copy_from_slice(bytes);
        // Patch the entry count at offset 0.
        let count = self.entries.len() as u32 + 1;
        page.buf[0..4].copy_from_slice(&count.to_le_bytes());
        self.entries.insert(id, ((off + 8) as u32, bytes.len() as u32));
        self.by_bytes.insert(bytes.to_vec(), id);
        Ok(id)
    }

    /// Bytes currently used in the page slab (count header + all entries). The
    /// remainder of the slab up to `capacity` is zero padding.
    pub fn used_bytes(&self) -> usize {
        let mut total = 4; // count header
        for &(_, len) in self.entries.values() {
            total += 8 + len as usize;
        }
        total
    }

    /// Grow the dictionary page: allocate a new slab of doubled capacity, copy
    /// every entry into it packed contiguously in id order, and replace the
    /// page's buffer.
    pub fn grow(&mut self, pager: &mut Pager) -> Result<()> {
        self.capacity = self.capacity.saturating_mul(2);
        let mut buf = vec![0u8; self.capacity];
        buf[0..4].copy_from_slice(&(self.entries.len() as u32).to_le_bytes());
        let mut new_entries = std::collections::BTreeMap::new();
        let mut off = 4usize;
        for (&id, &(_, len)) in &self.entries {
            let bytes = self.bytes_of(pager, id).unwrap_or(&[]).to_vec();
            buf[off..off + 4].copy_from_slice(&id.to_le_bytes());
            buf[off + 4..off + 8].copy_from_slice(&len.to_le_bytes());
            buf[off + 8..off + 8 + bytes.len()].copy_from_slice(&bytes);
            new_entries.insert(id, ((off + 8) as u32, len));
            off += 8 + bytes.len();
        }
        let page = pager
            .get_mut(self.page_id)
            .expect("dict page must exist");
        page.replace_buf(buf.into_boxed_slice());
        self.gen = page.gen;
        self.entries = new_entries;
        // by_bytes stays valid (keys unchanged).
        Ok(())
    }

    /// Build a `Dict` mirror from a decoded dict page buffer.
    pub fn from_page(page_id: PageId, buf: &[u8], gen: u64) -> Result<Dict> {
        let mut dict = Dict::new(page_id);
        dict.gen = gen;
        dict.capacity = buf.len().max(DICT_PAGE_CAPACITY);
        if buf.len() < 4 {
            return Ok(dict);
        }
        let count = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        for _ in 0..count {
            if pos + 8 > buf.len() {
                break;
            }
            let id = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let len = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;
            if pos + len as usize > buf.len() {
                break;
            }
            let bytes = buf[pos..pos + len as usize].to_vec();
            pos += len as usize;
            dict.entries.insert(id, ((pos - len as usize) as u32, len));
            // store offset of the byte data (after the 8-byte header)
            dict.by_bytes.insert(bytes, id);
            if id >= dict.next_id {
                dict.next_id = id + 1;
            }
        }
        // Recompute offsets to point at the byte data (after each 8-byte header).
        // The loop above stored the offset of the byte data already; but we
        // stored `pos - len` which is the start of bytes. Good.
        Ok(dict)
    }

    /// Allocate a fresh dict page in the pager and return its id.
    pub fn allocate(pager: &mut Pager) -> PageId {
        let mut buf = vec![0u8; DICT_PAGE_CAPACITY];
        buf[0..4].copy_from_slice(&0u32.to_le_bytes());
        pager.alloc(PageKind::Dict, buf.into_boxed_slice())
    }
}

/// Read the entry count from a dict page buffer.
pub fn entry_count(buf: &[u8]) -> u32 {
    if buf.len() < 4 {
        return 0;
    }
    u32::from_le_bytes(buf[0..4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_pager_and_dict() -> (Pager, PageId, Dict) {
        let mut pager = Pager::new();
        let pid = Dict::allocate(&mut pager);
        let dict = Dict::new(pid);
        (pager, pid, dict)
    }

    #[test]
    fn intern_and_lookup() {
        let (mut pager, pid, mut dict) = fresh_pager_and_dict();
        let id = dict.intern(&mut pager, b"hello").unwrap();
        assert_eq!(dict.bytes_of(&pager, id), Some(b"hello" as &[u8]));
        assert_eq!(dict.id_of(b"hello"), Some(id));
        assert_eq!(pid, dict.page_id);
    }

    #[test]
    fn grow_bumps_gen_and_preserves_entries() {
        let (mut pager, _pid, mut dict) = fresh_pager_and_dict();
        // Fill past the initial capacity with many distinct values.
        let mut ids = Vec::new();
        for i in 0..30u32 {
            let s = format!("value-{i}");
            ids.push(dict.intern(&mut pager, s.as_bytes()).unwrap());
        }
        assert!(dict.capacity > DICT_PAGE_CAPACITY);
        // Every entry is still readable.
        for (i, &id) in ids.iter().enumerate() {
            let s = format!("value-{i}");
            assert_eq!(dict.bytes_of(&pager, id), Some(s.as_bytes()));
        }
        // Generation must have advanced at least once.
        assert!(dict.gen > 0);
    }
}
