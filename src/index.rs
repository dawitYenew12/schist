//! Secondary indexes.
//!
//! A secondary index over a column maps each distinct value to the list of row
//! ids that carry it. To answer `col = V` without first resolving `V` to a
//! dictionary id and then back to bytes, the index caches, for each distinct
//! value, a raw pointer into the column's dictionary page buffer at that
//! value's byte string. Probing the index then reads the cached bytes directly
//! and compares them to the query value — a fast path that avoids two extra
//! indirections per probe.

use crate::pager::{Page, PageId, Pager};

/// One indexed distinct value.
pub struct IndexEntry {
    /// The dictionary id of this value.
    pub value_id: u32,
    /// A raw pointer into the dictionary page buffer at this value's byte
    /// string. Captured when the entry was built.
    pub ptr: *const u8,
    /// The dictionary page generation observed when `ptr` was captured.
    pub gen: u64,
    /// The row ids carrying this value, in ascending order.
    pub row_ids: Vec<u64>,
}

impl std::fmt::Debug for IndexEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexEntry")
            .field("value_id", &self.value_id)
            .field("gen", &self.gen)
            .field("rows", &self.row_ids.len())
            .finish()
    }
}

/// The index cache for one column.
pub struct IndexCache {
    pub column: String,
    pub dict_page: PageId,
    pub entries: Vec<IndexEntry>,
}

impl std::fmt::Debug for IndexCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexCache")
            .field("column", &self.column)
            .field("dict_page", &self.dict_page)
            .field("entries", &self.entries.len())
            .finish()
    }
}

// The raw pointer inside `IndexEntry` makes it !Send/!Sync by default; the
// database is single-threaded, so opt back in to keep the type ergonomic.
unsafe impl Send for IndexEntry {}
unsafe impl Sync for IndexEntry {}

impl IndexCache {
    pub fn empty(column: String) -> Self {
        IndexCache {
            column,
            dict_page: 0,
            entries: Vec::new(),
        }
    }

    pub fn with_dict(column: String, dict_page: PageId) -> Self {
        IndexCache {
            column,
            dict_page,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a row to an existing entry for `value_id`, or create a new entry
    /// capturing `ptr`/`gen` from the current dictionary page.
    pub fn add_or_append(&mut self, value_id: u32, ptr: *const u8, gen: u64, row_id: u64) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.value_id == value_id) {
            if !e.row_ids.contains(&row_id) {
                e.row_ids.push(row_id);
            }
            return;
        }
        self.entries.push(IndexEntry {
            value_id,
            ptr,
            gen,
            row_ids: vec![row_id],
        });
    }

    /// Read the cached byte string for an entry.
    ///
    /// # Safety
    /// `entry.ptr` must point to readable memory and the four bytes preceding
    /// it must encode a valid length.
    unsafe fn entry_bytes(&self, entry: &IndexEntry) -> &[u8] {
        // The dictionary entry layout is [id:u32][len:u32][bytes]; `ptr` points
        // at the byte data, so the length lives four bytes before it.
        let len_ptr = entry.ptr.offset(-4) as *const u32;
        let len = std::ptr::read(len_ptr) as usize;
        std::slice::from_raw_parts(entry.ptr, len)
    }

    /// Probe the index for rows whose indexed value equals `query` (compared as
    /// raw bytes). Returns the matching row ids. This is the fast path used by
    /// index scans and by equality predicates on indexed columns.
    pub fn probe(&self, _pager: &Pager, query: &[u8]) -> Vec<u64> {
        let mut out = Vec::new();
        for entry in &self.entries {
            let bytes = unsafe { self.entry_bytes(entry) };
            if bytes == query {
                out.extend_from_slice(&entry.row_ids);
            }
        }
        out
    }

    /// All row ids in the index, concatenated.
    pub fn all_rows(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for e in &self.entries {
            out.extend_from_slice(&e.row_ids);
        }
        out
    }

    /// Re-bind every cached pointer to the current dictionary page buffer,
    /// using the byte-data offsets recorded in the dictionary mirror.
    pub fn rebind(&mut self, pager: &Pager, offsets: &[(u32, u32)], gen: u64) {
        for entry in self.entries.iter_mut() {
            if let Some(&(off, _len)) = offsets.get(entry.value_id as usize) {
                if let Some(page) = pager.get(self.dict_page) {
                    // Safety: `off` is a byte offset within the page buffer
                    // recorded by the dictionary mirror, and the page is live.
                    entry.ptr = unsafe { page.raw_ptr().add(off as usize) };
                    entry.gen = gen;
                }
            }
        }
    }

    /// The generation the cache was last bound at (the max across entries, or
    /// 0 if empty).
    pub fn bound_gen(&self) -> u64 {
        self.entries.iter().map(|e| e.gen).max().unwrap_or(0)
    }

    /// Capture a pointer into a dict page at a byte-data offset.
    pub fn capture_ptr(page: &Page, byte_off: usize) -> *const u8 {
        // Safety: `byte_off` is a byte offset within the page buffer supplied
        // by the dictionary mirror; the caller guarantees it is in range.
        unsafe { page.raw_ptr().add(byte_off) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::Dict;
    use crate::pager::PageKind;

    #[test]
    fn probe_finds_matching_rows() {
        let mut pager = Pager::new();
        let pid = Dict::allocate(&mut pager);
        let mut dict = Dict::new(pid);
        let id_hello = dict.intern(&mut pager, b"hello").unwrap();
        let id_world = dict.intern(&mut pager, b"world").unwrap();

        let page = pager.get(pid).unwrap();
        let (off_hello, _) = dict.location_of(id_hello).unwrap();
        let (off_world, _) = dict.location_of(id_world).unwrap();
        let gen = page.gen;
        let mut idx = IndexCache::with_dict("t".to_string(), pid);
        idx.add_or_append(id_hello, IndexCache::capture_ptr(page, off_hello as usize), gen, 1);
        idx.add_or_append(id_hello, IndexCache::capture_ptr(page, off_hello as usize), gen, 2);
        idx.add_or_append(id_world, IndexCache::capture_ptr(page, off_world as usize), gen, 3);

        let rows = idx.probe(&pager, b"hello");
        assert_eq!(rows, vec![1, 2]);
        let rows = idx.probe(&pager, b"world");
        assert_eq!(rows, vec![3]);
        let rows = idx.probe(&pager, b"missing");
        assert!(rows.is_empty());
    }

    #[test]
    fn rebind_after_grow_keeps_probe_working() {
        let mut pager = Pager::new();
        let pid = Dict::allocate(&mut pager);
        let mut dict = Dict::new(pid);
        let mut idx = IndexCache::with_dict("t".to_string(), pid);
        // Intern one value, capture ptr.
        let id = dict.intern(&mut pager, b"abc").unwrap();
        let (off, _) = dict.location_of(id).unwrap();
        let gen = pager.get(pid).unwrap().gen;
        idx.add_or_append(id, IndexCache::capture_ptr(pager.get(pid).unwrap(), off as usize), gen, 7);
        // Force several grows.
        for i in 0..20u32 {
            dict.intern(&mut pager, format!("v{i}").as_bytes()).unwrap();
        }
        // Re-bind using the dictionary's current offsets.
        let mut offsets = vec![(0u32, 0u32); dict.next_id as usize];
        for (&vid, &loc) in &dict.entries {
            offsets[vid as usize] = loc;
        }
        idx.rebind(&pager, &offsets, dict.gen);
        assert_eq!(idx.probe(&pager, b"abc"), vec![7]);
    }

    #[test]
    fn page_kind_for_dict() {
        // sanity: dict pages are tagged Dict
        let mut pager = Pager::new();
        let pid = Dict::allocate(&mut pager);
        assert_eq!(pager.get(pid).unwrap().kind, PageKind::Dict);
    }
}
