//! The page buffer manager and the `Database` aggregate.
//!
//! The pager is the only subsystem that holds raw page memory. Every page is a
//! [`Page`]: a typed blob (`buf`) plus, for data pages, a separate heap-allocated
//! slot directory (`slot_dir`) that maps slot indices to rows. The pager hands
//! out [`PageId`]s, recycles freed ids, and owns the pages for the lifetime of
//! the database.
//!
//! The unsafe accessors here are deliberately narrow: [`Page::raw_ptr`] hands
//! out a pointer into a page's buffer, and [`Page::slot_at_unchecked`] reads a
//! slot directory entry by index without a bounds check. Both are `unsafe`
//! because their correctness is the caller's responsibility.

use crate::dict::Dict;
use crate::fsm::Fsm;
use crate::index::IndexCache;
use crate::rowid::RowIdMap;
use crate::schema::Schema;
use crate::zonemap::ZoneMap;
use std::collections::HashMap;

/// A page identifier. Ids are recycled: a freed id may be handed out again by a
/// later allocation, but a live id is unique at any instant.
pub type PageId = u32;

/// The kind of a page. The pager is type-agnostic (it just stores bytes), but
/// every page carries its kind so the verifier and the serializer can walk the
/// page set without an external directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    Schema,
    Data,
    Dict,
    Index,
    Fsm,
    RowId,
    ZoneMap,
    Free,
}

impl PageKind {
    pub fn as_u8(self) -> u8 {
        match self {
            PageKind::Schema => 0,
            PageKind::Data => 1,
            PageKind::Dict => 2,
            PageKind::Index => 3,
            PageKind::Fsm => 4,
            PageKind::RowId => 5,
            PageKind::ZoneMap => 6,
            PageKind::Free => 7,
        }
    }

    pub fn from_u8(b: u8) -> Option<PageKind> {
        match b {
            0 => Some(PageKind::Schema),
            1 => Some(PageKind::Data),
            2 => Some(PageKind::Dict),
            3 => Some(PageKind::Index),
            4 => Some(PageKind::Fsm),
            5 => Some(PageKind::RowId),
            6 => Some(PageKind::ZoneMap),
            7 => Some(PageKind::Free),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PageKind::Schema => "schema",
            PageKind::Data => "data",
            PageKind::Dict => "dict",
            PageKind::Index => "index",
            PageKind::Fsm => "fsm",
            PageKind::RowId => "rowid",
            PageKind::ZoneMap => "zonemap",
            PageKind::Free => "free",
        }
    }
}

/// One entry in a data page's slot directory.
///
/// A data page stores its encoded columnar bytes in `buf` and a slot directory
/// in `slot_dir`. Slot `i` corresponds to the row whose encoded bytes live at
/// `buf[off..off+len]`. `live` is `false` for tombstoned (deleted) slots; a
/// tombstoned slot is retained until compaction reclaims it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotEntry {
    pub row_id: u64,
    pub off: u32,
    pub len: u32,
    pub live: bool,
}

impl SlotEntry {
    pub fn new(row_id: u64, off: u32, len: u32) -> Self {
        SlotEntry {
            row_id,
            off,
            len,
            live: true,
        }
    }
}

/// A single page.
pub struct Page {
    pub id: PageId,
    pub kind: PageKind,
    /// Monotonically increasing generation. Bumped every time the page's
    /// buffer is replaced (a dictionary page that is rewritten, for example).
    pub gen: u64,
    pub buf: Box<[u8]>,
    /// Present only on data pages: the slot directory, heap-allocated to exactly
    /// the number of slots the page currently homes.
    pub slot_dir: Option<Box<[SlotEntry]>>,
}

impl Page {
    pub fn new(id: PageId, kind: PageKind, buf: Box<[u8]>) -> Self {
        Page {
            id,
            kind,
            gen: 0,
            buf,
            slot_dir: None,
        }
    }

    pub fn with_slots(mut self, slots: Vec<SlotEntry>) -> Self {
        self.slot_dir = Some(slots.into_boxed_slice());
        self
    }

    /// The number of slots in this page's directory (0 for non-data pages).
    pub fn slot_count(&self) -> usize {
        self.slot_dir.as_ref().map_or(0, |d| d.len())
    }

    /// The number of *live* (non-tombstoned) slots.
    pub fn live_count(&self) -> usize {
        self.slot_dir
            .as_ref()
            .map_or(0, |d| d.iter().filter(|s| s.live).count())
    }

    /// A pointer into this page's buffer. Used by the index cache to read a
    /// dictionary value's bytes without re-fetching the page.
    pub fn raw_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    /// A checked 4-byte little-endian read at `off`.
    pub fn read_u32_at(&self, off: usize) -> Option<u32> {
        if off + 4 > self.buf.len() {
            return None;
        }
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.buf[off..off + 4]);
        Some(u32::from_le_bytes(b))
    }

    /// A checked 8-byte little-endian read at `off`.
    pub fn read_u64_at(&self, off: usize) -> Option<u64> {
        if off + 8 > self.buf.len() {
            return None;
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.buf[off..off + 8]);
        Some(u64::from_le_bytes(b))
    }

    /// A checked 1-byte read at `off`.
    pub fn read_u8_at(&self, off: usize) -> Option<u8> {
        self.buf.get(off).copied()
    }

    /// A checked slot lookup.
    pub fn slot_at(&self, idx: usize) -> Option<SlotEntry> {
        self.slot_dir.as_ref()?.get(idx).copied()
    }

    /// # Safety
    /// `idx` must be less than the slot directory length.
    pub unsafe fn slot_at_unchecked(&self, idx: usize) -> SlotEntry {
        let dir = self
            .slot_dir
            .as_ref()
            .expect("slot access on non-data page");
        std::ptr::read(dir.as_ptr().add(idx))
    }

    /// Replace this page's buffer and bump its generation. The previous buffer
    /// is dropped (and therefore freed). Used when a dictionary page is
    /// rewritten in place.
    pub fn replace_buf(&mut self, new_buf: Box<[u8]>) {
        self.buf = new_buf;
        self.gen = self.gen.wrapping_add(1);
    }

    /// Replace this page's slot directory. The previous directory is dropped.
    pub fn replace_slots(&mut self, slots: Vec<SlotEntry>) {
        self.slot_dir = Some(slots.into_boxed_slice());
        self.gen = self.gen.wrapping_add(1);
    }
}

impl std::fmt::Debug for Page {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Page")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("gen", &self.gen)
            .field("buf_len", &self.buf.len())
            .field("slots", &self.slot_count())
            .finish()
    }
}

/// The page buffer manager. Pages are stored in a slot vector indexed by id;
/// freed ids are recycled.
pub struct Pager {
    slots: Vec<Option<Page>>,
    free_ids: Vec<PageId>,
    high_water: PageId,
}

impl Default for Pager {
    fn default() -> Self {
        Pager::new()
    }
}

impl Pager {
    pub fn new() -> Self {
        Pager {
            slots: Vec::new(),
            free_ids: Vec::new(),
            high_water: 0,
        }
    }

    /// Allocate a new page and return its id.
    pub fn alloc(&mut self, kind: PageKind, buf: Box<[u8]>) -> PageId {
        if let Some(id) = self.free_ids.pop() {
            let page = Page::new(id, kind, buf);
            self.slots[id as usize] = Some(page);
            return id;
        }
        let id = self.high_water;
        self.high_water += 1;
        self.slots.push(Some(Page::new(id, kind, buf)));
        id
    }

    /// Look up a page by id.
    pub fn get(&self, id: PageId) -> Option<&Page> {
        self.slots.get(id as usize).and_then(|o| o.as_ref())
    }

    pub fn get_mut(&mut self, id: PageId) -> Option<&mut Page> {
        self.slots.get_mut(id as usize).and_then(|o| o.as_mut())
    }

    /// Free a page. Its buffer and slot directory are dropped immediately, and
    /// its id is recycled.
    pub fn free(&mut self, id: PageId) {
        if let Some(slot) = self.slots.get_mut(id as usize) {
            if slot.take().is_some() {
                self.free_ids.push(id);
            }
        }
    }

    pub fn count(&self) -> usize {
        self.slots.iter().filter(|o| o.is_some()).count()
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// All live page ids, in ascending order.
    pub fn ids(&self) -> Vec<PageId> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_some())
            .map(|(i, _)| i as PageId)
            .collect()
    }

    /// Iterate over all live pages.
    pub fn iter(&self) -> impl Iterator<Item = &Page> {
        self.slots.iter().filter_map(|o| o.as_ref())
    }

    /// Iterate over data pages only.
    pub fn data_pages(&self) -> impl Iterator<Item = &Page> {
        self.iter().filter(|p| p.kind == PageKind::Data)
    }

    /// Iterate over dict pages only.
    pub fn dict_pages(&self) -> impl Iterator<Item = &Page> {
        self.iter().filter(|p| p.kind == PageKind::Dict)
    }
}

/// Lightweight statistics maintained alongside the database.
#[derive(Debug, Clone, Default)]
pub struct DbStats {
    pub inserts: u64,
    pub updates: u64,
    pub deletes: u64,
    pub compactions: u64,
    pub checkpoints: u64,
    pub scans: u64,
    pub index_scans: u64,
}

/// The aggregate database. It owns the pager plus the in-memory bookkeeping
/// structures (free-space map, row-id map, secondary index caches, zone maps)
/// that the format layer serializes into pages on checkpoint.
pub struct Database {
    pub schema: Schema,
    pub pager: Pager,
    pub fsm: Fsm,
    pub rowid_map: RowIdMap,
    /// One index cache per indexed column, in schema column order. Unindexed
    /// columns have `None` here; the entry is present iff the column declares
    /// an `index_page` in the schema.
    pub indexes: Vec<Option<IndexCache>>,
    /// One dictionary mirror per text column with a `dict_page`, in column
    /// order. `None` for non-text or `Plain` columns.
    pub dicts: Vec<Option<Dict>>,
    pub zone_maps: Vec<ZoneMap>,
    pub next_row_id: u64,
    pub stats: DbStats,
    /// Per data page, the row-id column values present in that page, kept so
    /// the query layer can re-emit rows without re-decoding the id column. This
    /// is a convenience mirror; the authoritative location of a row is the
    /// row-id map.
    pub page_rows: HashMap<PageId, Vec<u64>>,
}

impl Database {
    pub fn new(schema: Schema) -> Self {
        let n = schema.arity();
        let mut indexes = Vec::with_capacity(n);
        let mut dicts = Vec::with_capacity(n);
        let mut zone_maps = Vec::with_capacity(n);
        for col in &schema.columns {
            indexes.push(col.index_page.map(|_| IndexCache::empty(col.name.clone())));
            dicts.push(col.dict_page.map(Dict::new));
            zone_maps.push(ZoneMap::default());
        }
        Database {
            schema,
            pager: Pager::new(),
            fsm: Fsm::new(),
            rowid_map: RowIdMap::new(),
            indexes,
            dicts,
            zone_maps,
            next_row_id: 1,
            stats: DbStats::default(),
            page_rows: HashMap::new(),
        }
    }

    /// The page id of a column's index, if any.
    pub fn index_of(&self, col: usize) -> Option<&IndexCache> {
        self.indexes.get(col).and_then(|o| o.as_ref())
    }

    pub fn index_of_mut(&mut self, col: usize) -> Option<&mut IndexCache> {
        self.indexes.get_mut(col).and_then(|o| o.as_mut())
    }

    /// Resolve a row id to its home page and slot via the row-id map.
    pub fn locate(&self, row_id: u64) -> Option<(PageId, usize)> {
        self.rowid_map.get(row_id)
    }
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database")
            .field("pages", &self.pager.count())
            .field("rows", &self.rowid_map.len())
            .field("next_row_id", &self.next_row_id)
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free_recycles_ids() {
        let mut p = Pager::new();
        let a = p.alloc(PageKind::Data, vec![0u8; 16].into_boxed_slice());
        let b = p.alloc(PageKind::Data, vec![1u8; 16].into_boxed_slice());
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(p.count(), 2);
        p.free(a);
        assert_eq!(p.count(), 1);
        let c = p.alloc(PageKind::Data, vec![2u8; 16].into_boxed_slice());
        assert_eq!(c, 0, "freed id should be recycled");
        assert_eq!(p.count(), 2);
    }

    #[test]
    fn slot_dir_round_trips() {
        let page = Page::new(0, PageKind::Data, vec![0u8; 64].into_boxed_slice())
            .with_slots(vec![
                SlotEntry::new(1, 0, 8),
                SlotEntry::new(2, 8, 8),
            ]);
        assert_eq!(page.slot_count(), 2);
        assert_eq!(page.slot_at(0).unwrap().row_id, 1);
        // Safety: idx in range.
        assert_eq!(unsafe { page.slot_at_unchecked(1) }.row_id, 2);
    }

    #[test]
    fn replace_buf_bumps_gen() {
        let mut page = Page::new(0, PageKind::Dict, vec![0u8; 8].into_boxed_slice());
        let g0 = page.gen;
        page.replace_buf(vec![9u8; 8].into_boxed_slice());
        assert_eq!(page.gen, g0 + 1);
        assert_eq!(page.buf[0], 9);
    }
}
