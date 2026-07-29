//! A roaring-style compressed bitmap over 32-bit row ids.
//!
//! Bitmap indexes and selection vectors over large row-id spaces are wasteful
//! as dense bit arrays when the set is sparse, and wasteful as sorted id lists
//! when it is dense. The roaring layout splits the 32-bit space into 16-bit
//! high chunks; each populated chunk holds either an *array container* (a sorted
//! list of low 16-bit values, good when sparse) or a *bitmap container* (a
//! 65536-bit dense map, good when dense), converting between the two as the
//! cardinality crosses a threshold. This gives compact storage and fast set
//! algebra (union / intersection / difference).

use std::collections::BTreeMap;

const ARRAY_MAX: usize = 4096;
const BITMAP_WORDS: usize = 1024; // 65536 bits / 64

#[derive(Debug, Clone, PartialEq, Eq)]
enum Container {
    Array(Vec<u16>),
    Bitmap(Box<[u64; BITMAP_WORDS]>),
}

impl Container {
    fn cardinality(&self) -> usize {
        match self {
            Container::Array(v) => v.len(),
            Container::Bitmap(b) => b.iter().map(|w| w.count_ones() as usize).sum(),
        }
    }

    fn contains(&self, low: u16) -> bool {
        match self {
            Container::Array(v) => v.binary_search(&low).is_ok(),
            Container::Bitmap(b) => {
                let i = low as usize;
                b[i >> 6] & (1u64 << (i & 63)) != 0
            }
        }
    }

    fn insert(&mut self, low: u16) -> bool {
        match self {
            Container::Array(v) => match v.binary_search(&low) {
                Ok(_) => false,
                Err(pos) => {
                    v.insert(pos, low);
                    if v.len() > ARRAY_MAX {
                        self.to_bitmap();
                    }
                    true
                }
            },
            Container::Bitmap(b) => {
                let i = low as usize;
                let mask = 1u64 << (i & 63);
                if b[i >> 6] & mask == 0 {
                    b[i >> 6] |= mask;
                    true
                } else {
                    false
                }
            }
        }
    }

    fn remove(&mut self, low: u16) -> bool {
        match self {
            Container::Array(v) => match v.binary_search(&low) {
                Ok(pos) => {
                    v.remove(pos);
                    true
                }
                Err(_) => false,
            },
            Container::Bitmap(b) => {
                let i = low as usize;
                let mask = 1u64 << (i & 63);
                if b[i >> 6] & mask != 0 {
                    b[i >> 6] &= !mask;
                    true
                } else {
                    false
                }
            }
        }
    }

    fn to_bitmap(&mut self) {
        if let Container::Array(v) = self {
            let mut words = Box::new([0u64; BITMAP_WORDS]);
            for &low in v.iter() {
                let i = low as usize;
                words[i >> 6] |= 1u64 << (i & 63);
            }
            *self = Container::Bitmap(words);
        }
    }

    fn maybe_shrink(&mut self) {
        if let Container::Bitmap(b) = self {
            let card = b.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            if card <= ARRAY_MAX {
                let mut v = Vec::with_capacity(card);
                for (wi, &w) in b.iter().enumerate() {
                    let mut bits = w;
                    while bits != 0 {
                        let t = bits.trailing_zeros() as usize;
                        v.push((wi * 64 + t) as u16);
                        bits &= bits - 1;
                    }
                }
                *self = Container::Array(v);
            }
        }
    }

    fn values(&self) -> Vec<u16> {
        match self {
            Container::Array(v) => v.clone(),
            Container::Bitmap(b) => {
                let mut out = Vec::new();
                for (wi, &w) in b.iter().enumerate() {
                    let mut bits = w;
                    while bits != 0 {
                        let t = bits.trailing_zeros() as usize;
                        out.push((wi * 64 + t) as u16);
                        bits &= bits - 1;
                    }
                }
                out
            }
        }
    }
}

/// A compressed set of `u32` values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoaringBitmap {
    containers: BTreeMap<u16, Container>,
}

impl RoaringBitmap {
    /// An empty bitmap.
    pub fn new() -> RoaringBitmap {
        RoaringBitmap::default()
    }

    /// Build from an iterator of values.
    pub fn from_iter_vals<I: IntoIterator<Item = u32>>(iter: I) -> RoaringBitmap {
        let mut b = RoaringBitmap::new();
        for v in iter {
            b.insert(v);
        }
        b
    }

    fn split(v: u32) -> (u16, u16) {
        ((v >> 16) as u16, (v & 0xFFFF) as u16)
    }

    /// Insert a value; returns `true` if newly added.
    pub fn insert(&mut self, v: u32) -> bool {
        let (hi, lo) = Self::split(v);
        self.containers
            .entry(hi)
            .or_insert_with(|| Container::Array(Vec::new()))
            .insert(lo)
    }

    /// Remove a value; returns `true` if it was present.
    pub fn remove(&mut self, v: u32) -> bool {
        let (hi, lo) = Self::split(v);
        let removed = if let Some(c) = self.containers.get_mut(&hi) {
            let r = c.remove(lo);
            if c.cardinality() == 0 {
                self.containers.remove(&hi);
            }
            r
        } else {
            false
        };
        removed
    }

    /// Membership test.
    pub fn contains(&self, v: u32) -> bool {
        let (hi, lo) = Self::split(v);
        self.containers.get(&hi).map(|c| c.contains(lo)).unwrap_or(false)
    }

    /// Total number of values.
    pub fn cardinality(&self) -> usize {
        self.containers.values().map(|c| c.cardinality()).sum()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// The minimum value, if any.
    pub fn min(&self) -> Option<u32> {
        let (&hi, c) = self.containers.iter().next()?;
        let lo = c.values().into_iter().min()?;
        Some(((hi as u32) << 16) | lo as u32)
    }

    /// The maximum value, if any.
    pub fn max(&self) -> Option<u32> {
        let (&hi, c) = self.containers.iter().next_back()?;
        let lo = c.values().into_iter().max()?;
        Some(((hi as u32) << 16) | lo as u32)
    }

    /// Collect all values in ascending order.
    pub fn to_vec(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.cardinality());
        for (&hi, c) in &self.containers {
            let base = (hi as u32) << 16;
            for lo in c.values() {
                out.push(base | lo as u32);
            }
        }
        out
    }

    /// Set union.
    pub fn union(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut out = self.clone();
        for v in other.to_vec() {
            out.insert(v);
        }
        for c in out.containers.values_mut() {
            c.maybe_shrink();
        }
        out
    }

    /// Set intersection.
    pub fn intersect(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut out = RoaringBitmap::new();
        for (&hi, c) in &self.containers {
            if let Some(oc) = other.containers.get(&hi) {
                for lo in c.values() {
                    if oc.contains(lo) {
                        out.insert(((hi as u32) << 16) | lo as u32);
                    }
                }
            }
        }
        out
    }

    /// Set difference (self minus other).
    pub fn difference(&self, other: &RoaringBitmap) -> RoaringBitmap {
        let mut out = RoaringBitmap::new();
        for v in self.to_vec() {
            if !other.contains(v) {
                out.insert(v);
            }
        }
        out
    }

    /// Number of distinct 16-bit chunks in use (a rough storage measure).
    pub fn container_count(&self) -> usize {
        self.containers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_remove() {
        let mut b = RoaringBitmap::new();
        assert!(b.insert(5));
        assert!(!b.insert(5));
        assert!(b.contains(5));
        assert!(b.remove(5));
        assert!(!b.contains(5));
        assert!(b.is_empty());
    }

    #[test]
    fn spans_multiple_chunks() {
        let vals = [1u32, 70000, 70001, 200000, 0xFFFF_FFFF];
        let b = RoaringBitmap::from_iter_vals(vals.iter().copied());
        assert_eq!(b.cardinality(), 5);
        assert_eq!(b.min(), Some(1));
        assert_eq!(b.max(), Some(0xFFFF_FFFF));
        assert!(b.container_count() >= 3);
    }

    #[test]
    fn dense_becomes_bitmap_and_shrinks() {
        let mut b = RoaringBitmap::new();
        for i in 0..5000u32 {
            b.insert(i);
        }
        assert_eq!(b.cardinality(), 5000);
        // Remove most to force a shrink back to array on union.
        for i in 0..4990u32 {
            b.remove(i);
        }
        let u = b.union(&RoaringBitmap::new());
        assert_eq!(u.cardinality(), 10);
    }

    #[test]
    fn set_algebra() {
        let a = RoaringBitmap::from_iter_vals([1, 2, 3, 100000].iter().copied());
        let b = RoaringBitmap::from_iter_vals([2, 3, 4, 100000].iter().copied());
        assert_eq!(a.union(&b).cardinality(), 5);
        assert_eq!(a.intersect(&b).to_vec(), vec![2, 3, 100000]);
        assert_eq!(a.difference(&b).to_vec(), vec![1]);
    }

    #[test]
    fn to_vec_is_sorted() {
        let b = RoaringBitmap::from_iter_vals([9, 3, 70000, 1].iter().copied());
        assert_eq!(b.to_vec(), vec![1, 3, 9, 70000]);
    }
}
