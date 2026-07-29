//! Copy-on-write page snapshots for time-travel reads.
//!
//! Snapshot isolation lets a reader see a consistent version of the database as
//! of the moment its snapshot was taken, even while writers continue. A
//! lightweight way to support this over a page store is copy-on-write versioning:
//! each page has a version chain, a snapshot records the current version stamp,
//! and a read through a snapshot resolves to the newest page version at or
//! before that stamp. Writes append a new version rather than mutating in place,
//! and a garbage-collection pass reclaims versions older than the oldest live
//! snapshot.

use std::collections::HashMap;

/// A monotonically increasing version stamp.
pub type Version = u64;

/// Identifies a page.
pub type PageId = u32;

#[derive(Debug, Clone)]
struct PageVersion {
    version: Version,
    data: Vec<u8>,
}

/// A copy-on-write page store with versioned pages.
#[derive(Debug, Default)]
pub struct SnapshotStore {
    /// For each page, its versions in ascending version order.
    pages: HashMap<PageId, Vec<PageVersion>>,
    current: Version,
    /// Active snapshot version stamps and their reference counts.
    snapshots: HashMap<Version, u32>,
}

/// A handle to a consistent read view as of a version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Snapshot {
    pub version: Version,
}

impl SnapshotStore {
    /// A fresh empty store.
    pub fn new() -> SnapshotStore {
        SnapshotStore::default()
    }

    /// The current (latest) version stamp.
    pub fn current_version(&self) -> Version {
        self.current
    }

    /// Number of distinct pages.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Total stored versions across all pages.
    pub fn total_versions(&self) -> usize {
        self.pages.values().map(|v| v.len()).sum()
    }

    /// Write a page, creating a new version and advancing the clock.
    pub fn write(&mut self, page: PageId, data: Vec<u8>) -> Version {
        self.current += 1;
        let v = self.current;
        self.pages.entry(page).or_default().push(PageVersion {
            version: v,
            data,
        });
        v
    }

    /// Take a snapshot at the current version.
    pub fn snapshot(&mut self) -> Snapshot {
        *self.snapshots.entry(self.current).or_insert(0) += 1;
        Snapshot {
            version: self.current,
        }
    }

    /// Release a snapshot (decrements its reference count).
    pub fn release(&mut self, snap: Snapshot) {
        if let Some(rc) = self.snapshots.get_mut(&snap.version) {
            *rc -= 1;
            if *rc == 0 {
                self.snapshots.remove(&snap.version);
            }
        }
    }

    /// Read a page as of `snap`: the newest version whose stamp is `<=` the
    /// snapshot version.
    pub fn read(&self, snap: Snapshot, page: PageId) -> Option<&[u8]> {
        let versions = self.pages.get(&page)?;
        // Binary search for the last version <= snap.version.
        let mut lo = 0usize;
        let mut hi = versions.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if versions[mid].version <= snap.version {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            None
        } else {
            Some(&versions[lo - 1].data)
        }
    }

    /// Read the latest version of a page.
    pub fn read_latest(&self, page: PageId) -> Option<&[u8]> {
        self.pages.get(&page).and_then(|v| v.last()).map(|pv| pv.data.as_slice())
    }

    /// The oldest live snapshot version, or the current version if none.
    pub fn oldest_snapshot(&self) -> Version {
        self.snapshots.keys().copied().min().unwrap_or(self.current)
    }

    /// Garbage-collect page versions that no live snapshot can see. For each
    /// page it keeps the newest version at or before the oldest live snapshot,
    /// plus everything after it. Returns the number of versions reclaimed.
    pub fn gc(&mut self) -> usize {
        let watermark = self.oldest_snapshot();
        let mut reclaimed = 0;
        for versions in self.pages.values_mut() {
            // Find the newest version <= watermark; everything strictly older is
            // dead.
            let keep_from = versions
                .iter()
                .rposition(|pv| pv.version <= watermark)
                .unwrap_or(0);
            if keep_from > 0 {
                reclaimed += keep_from;
                versions.drain(0..keep_from);
            }
        }
        reclaimed
    }

    /// Number of live snapshots.
    pub fn live_snapshots(&self) -> usize {
        self.snapshots.values().map(|&rc| rc as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_sees_snapshot_version() {
        let mut store = SnapshotStore::new();
        store.write(1, vec![1]);
        let snap = store.snapshot();
        store.write(1, vec![2]); // newer write after snapshot
        assert_eq!(store.read(snap, 1), Some(&[1][..]));
        assert_eq!(store.read_latest(1), Some(&[2][..]));
    }

    #[test]
    fn read_before_first_write_is_none() {
        let mut store = SnapshotStore::new();
        let snap = store.snapshot();
        store.write(1, vec![9]);
        assert_eq!(store.read(snap, 1), None);
    }

    #[test]
    fn multiple_snapshots_isolated() {
        let mut store = SnapshotStore::new();
        store.write(1, vec![10]);
        let s1 = store.snapshot();
        store.write(1, vec![20]);
        let s2 = store.snapshot();
        store.write(1, vec![30]);
        assert_eq!(store.read(s1, 1), Some(&[10][..]));
        assert_eq!(store.read(s2, 1), Some(&[20][..]));
    }

    #[test]
    fn gc_reclaims_dead_versions() {
        let mut store = SnapshotStore::new();
        store.write(1, vec![1]);
        store.write(1, vec![2]);
        store.write(1, vec![3]);
        // No snapshots: watermark is current, older versions are dead.
        let reclaimed = store.gc();
        assert!(reclaimed >= 2);
        assert_eq!(store.read_latest(1), Some(&[3][..]));
    }

    #[test]
    fn gc_respects_live_snapshot() {
        let mut store = SnapshotStore::new();
        store.write(1, vec![1]);
        let snap = store.snapshot();
        store.write(1, vec![2]);
        store.gc();
        // The snapshot still resolves to version 1.
        assert_eq!(store.read(snap, 1), Some(&[1][..]));
        store.release(snap);
        assert_eq!(store.live_snapshots(), 0);
    }

    #[test]
    fn snapshot_refcount() {
        let mut store = SnapshotStore::new();
        store.write(1, vec![1]);
        let a = store.snapshot();
        let b = store.snapshot();
        assert_eq!(store.live_snapshots(), 2);
        store.release(a);
        assert_eq!(store.live_snapshots(), 1);
        store.release(b);
        assert_eq!(store.live_snapshots(), 0);
    }
}
