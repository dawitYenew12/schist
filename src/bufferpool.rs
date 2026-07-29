//! A fixed-capacity buffer pool with clock (second-chance) eviction.
//!
//! The [`crate::pager`] owns page memory for a decoded database; this buffer
//! pool is the caching layer that sits in front of a hypothetical backing
//! store, tracking which frames are resident, pinned, dirty, and referenced.
//! It is a standard textbook design: a frame table, a page table mapping page
//! ids to frames, a free list, and a clock hand that sweeps reference bits to
//! choose a victim. Pinned frames are never evicted; dirty victims are reported
//! to the caller so they can be flushed.

use std::collections::HashMap;

/// Identifies a logical page in the backing store.
pub type PageId = u32;

/// A frame holds one page's bytes plus its bookkeeping bits.
#[derive(Debug, Clone)]
struct Frame {
    page_id: Option<PageId>,
    data: Vec<u8>,
    pin_count: u32,
    dirty: bool,
    referenced: bool,
}

impl Frame {
    fn empty(page_size: usize) -> Frame {
        Frame {
            page_id: None,
            data: vec![0u8; page_size],
            pin_count: 0,
            dirty: false,
            referenced: false,
        }
    }
}

/// Statistics tracked by the pool over its lifetime.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub flushes: u64,
}

impl PoolStats {
    /// Hit ratio in `[0, 1]`.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// The error returned when the pool cannot satisfy a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// Every frame is pinned; no victim can be chosen.
    AllFramesPinned,
    /// The requested page is not resident and cannot be faulted in without a
    /// loader.
    NotResident(PageId),
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::AllFramesPinned => write!(f, "all buffer frames are pinned"),
            PoolError::NotResident(p) => write!(f, "page {p} is not resident"),
        }
    }
}

impl std::error::Error for PoolError {}

/// A page that was evicted and needs to be written back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Writeback {
    pub page_id: PageId,
    pub data: Vec<u8>,
}

/// The buffer pool.
pub struct BufferPool {
    frames: Vec<Frame>,
    page_table: HashMap<PageId, usize>,
    free_list: Vec<usize>,
    clock_hand: usize,
    page_size: usize,
    stats: PoolStats,
}

impl BufferPool {
    /// A pool of `capacity` frames, each `page_size` bytes.
    pub fn new(capacity: usize, page_size: usize) -> BufferPool {
        let capacity = capacity.max(1);
        let frames = (0..capacity).map(|_| Frame::empty(page_size)).collect();
        let free_list = (0..capacity).rev().collect();
        BufferPool {
            frames,
            page_table: HashMap::new(),
            free_list,
            clock_hand: 0,
            page_size,
            stats: PoolStats::default(),
        }
    }

    /// Number of frames.
    pub fn capacity(&self) -> usize {
        self.frames.len()
    }

    /// Number of resident pages.
    pub fn resident(&self) -> usize {
        self.page_table.len()
    }

    /// Cumulative statistics.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// `true` if `page_id` currently occupies a frame.
    pub fn is_resident(&self, page_id: PageId) -> bool {
        self.page_table.contains_key(&page_id)
    }

    /// Install a fresh page into the pool, evicting if necessary. Returns any
    /// dirty page displaced so the caller can flush it. The new page starts
    /// pinned with a reference bit set.
    pub fn install(&mut self, page_id: PageId, data: Vec<u8>) -> Result<Option<Writeback>, PoolError> {
        if let Some(&frame) = self.page_table.get(&page_id) {
            self.frames[frame].data = data;
            self.frames[frame].dirty = true;
            self.frames[frame].referenced = true;
            return Ok(None);
        }
        let (frame_idx, writeback) = self.acquire_frame()?;
        let mut data = data;
        data.resize(self.page_size, 0);
        let frame = &mut self.frames[frame_idx];
        frame.page_id = Some(page_id);
        frame.data = data;
        frame.pin_count = 1;
        frame.dirty = true;
        frame.referenced = true;
        self.page_table.insert(page_id, frame_idx);
        Ok(writeback)
    }

    /// Pin a resident page and return a shared view of its bytes. A miss is
    /// reported as [`PoolError::NotResident`] — this pool has no loader.
    pub fn pin(&mut self, page_id: PageId) -> Result<&[u8], PoolError> {
        match self.page_table.get(&page_id) {
            Some(&frame) => {
                self.frames[frame].pin_count += 1;
                self.frames[frame].referenced = true;
                self.stats.hits += 1;
                Ok(&self.frames[frame].data)
            }
            None => {
                self.stats.misses += 1;
                Err(PoolError::NotResident(page_id))
            }
        }
    }

    /// Get a mutable view of a resident, pinned page and mark it dirty.
    pub fn pin_mut(&mut self, page_id: PageId) -> Result<&mut [u8], PoolError> {
        match self.page_table.get(&page_id) {
            Some(&frame) => {
                self.frames[frame].pin_count += 1;
                self.frames[frame].referenced = true;
                self.frames[frame].dirty = true;
                self.stats.hits += 1;
                Ok(&mut self.frames[frame].data)
            }
            None => {
                self.stats.misses += 1;
                Err(PoolError::NotResident(page_id))
            }
        }
    }

    /// Release one pin on a page. Unpinning an unpinned page is a no-op.
    pub fn unpin(&mut self, page_id: PageId) {
        if let Some(&frame) = self.page_table.get(&page_id) {
            if self.frames[frame].pin_count > 0 {
                self.frames[frame].pin_count -= 1;
            }
        }
    }

    /// Mark a resident page dirty.
    pub fn mark_dirty(&mut self, page_id: PageId) {
        if let Some(&frame) = self.page_table.get(&page_id) {
            self.frames[frame].dirty = true;
        }
    }

    /// The pin count of a resident page (0 if not resident).
    pub fn pin_count(&self, page_id: PageId) -> u32 {
        self.page_table
            .get(&page_id)
            .map(|&f| self.frames[f].pin_count)
            .unwrap_or(0)
    }

    /// Flush a single dirty page, returning its bytes and clearing the dirty
    /// bit. Returns `None` if the page is clean or not resident.
    pub fn flush(&mut self, page_id: PageId) -> Option<Writeback> {
        let frame = *self.page_table.get(&page_id)?;
        if !self.frames[frame].dirty {
            return None;
        }
        self.frames[frame].dirty = false;
        self.stats.flushes += 1;
        Some(Writeback {
            page_id,
            data: self.frames[frame].data.clone(),
        })
    }

    /// Flush every dirty page, returning the writebacks in frame order.
    pub fn flush_all(&mut self) -> Vec<Writeback> {
        let ids: Vec<PageId> = self.page_table.keys().copied().collect();
        let mut out = Vec::new();
        for id in ids {
            if let Some(wb) = self.flush(id) {
                out.push(wb);
            }
        }
        out.sort_by_key(|w| w.page_id);
        out
    }

    /// Acquire a free frame, evicting a clock victim if the free list is empty.
    fn acquire_frame(&mut self) -> Result<(usize, Option<Writeback>), PoolError> {
        if let Some(idx) = self.free_list.pop() {
            return Ok((idx, None));
        }
        let victim = self.choose_victim()?;
        let mut writeback = None;
        if let Some(pid) = self.frames[victim].page_id {
            if self.frames[victim].dirty {
                writeback = Some(Writeback {
                    page_id: pid,
                    data: self.frames[victim].data.clone(),
                });
                self.stats.flushes += 1;
            }
            self.page_table.remove(&pid);
        }
        self.frames[victim] = Frame::empty(self.page_size);
        self.stats.evictions += 1;
        Ok((victim, writeback))
    }

    /// Run the clock hand until it finds an unpinned frame with a clear
    /// reference bit, clearing reference bits as it sweeps.
    fn choose_victim(&mut self) -> Result<usize, PoolError> {
        let n = self.frames.len();
        let mut swept = 0;
        // At most two full sweeps: one to clear reference bits, one to pick.
        while swept < 2 * n {
            let idx = self.clock_hand;
            self.clock_hand = (self.clock_hand + 1) % n;
            swept += 1;
            if self.frames[idx].pin_count > 0 {
                continue;
            }
            if self.frames[idx].referenced {
                self.frames[idx].referenced = false;
                continue;
            }
            return Ok(idx);
        }
        Err(PoolError::AllFramesPinned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_and_pin() {
        let mut pool = BufferPool::new(2, 8);
        pool.install(1, vec![1, 2, 3]).unwrap();
        pool.unpin(1);
        let bytes = pool.pin(1).unwrap();
        assert_eq!(&bytes[..3], &[1, 2, 3]);
        assert_eq!(pool.stats().hits, 1);
    }

    #[test]
    fn miss_reports_not_resident() {
        let mut pool = BufferPool::new(2, 8);
        assert_eq!(pool.pin(99), Err(PoolError::NotResident(99)));
        assert_eq!(pool.stats().misses, 1);
    }

    #[test]
    fn second_chance_eviction_order() {
        let mut pool = BufferPool::new(2, 4);
        pool.install(1, vec![0; 4]).unwrap();
        pool.unpin(1);
        pool.install(2, vec![0; 4]).unwrap();
        pool.unpin(2);
        // Both frames have their reference bit set; the clock sweep clears both
        // and evicts the first one it revisits (page 1).
        pool.install(3, vec![0; 4]).unwrap();
        pool.unpin(3);
        assert!(!pool.is_resident(1));
        assert!(pool.is_resident(2));
        assert!(pool.is_resident(3));
        assert_eq!(pool.stats().evictions, 1);
        // Page 2's reference bit was cleared during that sweep and never reset,
        // so it is the next victim while the freshly-referenced page 3 survives.
        pool.install(4, vec![0; 4]).unwrap();
        pool.unpin(4);
        assert!(!pool.is_resident(2));
        assert!(pool.is_resident(3));
        assert!(pool.is_resident(4));
    }

    #[test]
    fn dirty_eviction_writes_back() {
        let mut pool = BufferPool::new(1, 4);
        pool.install(1, vec![9; 4]).unwrap();
        pool.unpin(1);
        let wb = pool.install(2, vec![7; 4]).unwrap();
        assert_eq!(wb, Some(Writeback { page_id: 1, data: vec![9; 4] }));
    }

    #[test]
    fn all_pinned_errors() {
        let mut pool = BufferPool::new(1, 4);
        pool.install(1, vec![0; 4]).unwrap(); // pinned
        assert_eq!(pool.install(2, vec![0; 4]), Err(PoolError::AllFramesPinned));
    }

    #[test]
    fn flush_clears_dirty() {
        let mut pool = BufferPool::new(2, 4);
        pool.install(1, vec![5; 4]).unwrap();
        let wb = pool.flush(1).unwrap();
        assert_eq!(wb.page_id, 1);
        assert!(pool.flush(1).is_none());
    }
}
