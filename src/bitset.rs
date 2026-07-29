//! A growable dense bit set.
//!
//! Selection vectors, visited masks in graph traversals, and column-liveness
//! tracking all want a compact set of small non-negative integers with fast
//! membership, iteration, and set algebra. This is a word-packed bit set that
//! grows on demand, with population counts, rank/select, and the usual boolean
//! combinators.

/// A set of non-negative integers backed by a bit vector.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    /// An empty set.
    pub fn new() -> BitSet {
        BitSet { words: Vec::new() }
    }

    /// An empty set pre-sized to hold values up to `capacity - 1`.
    pub fn with_capacity(capacity: usize) -> BitSet {
        BitSet {
            words: vec![0u64; capacity.div_ceil(64)],
        }
    }

    /// A set containing `0..n`.
    pub fn full(n: usize) -> BitSet {
        let mut s = BitSet::with_capacity(n);
        for i in 0..n {
            s.insert(i);
        }
        s
    }

    fn ensure(&mut self, word: usize) {
        if self.words.len() <= word {
            self.words.resize(word + 1, 0);
        }
    }

    /// Insert `bit`; returns `true` if newly added.
    pub fn insert(&mut self, bit: usize) -> bool {
        let w = bit >> 6;
        self.ensure(w);
        let mask = 1u64 << (bit & 63);
        let was = self.words[w] & mask != 0;
        self.words[w] |= mask;
        !was
    }

    /// Remove `bit`; returns `true` if it was present.
    pub fn remove(&mut self, bit: usize) -> bool {
        let w = bit >> 6;
        if w >= self.words.len() {
            return false;
        }
        let mask = 1u64 << (bit & 63);
        let was = self.words[w] & mask != 0;
        self.words[w] &= !mask;
        was
    }

    /// Membership test.
    pub fn contains(&self, bit: usize) -> bool {
        let w = bit >> 6;
        w < self.words.len() && self.words[w] & (1u64 << (bit & 63)) != 0
    }

    /// Number of set bits.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// `true` if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// The largest set bit, if any.
    pub fn max(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate().rev() {
            if w != 0 {
                return Some(wi * 64 + (63 - w.leading_zeros() as usize));
            }
        }
        None
    }

    /// The smallest set bit, if any.
    pub fn min(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return Some(wi * 64 + w.trailing_zeros() as usize);
            }
        }
        None
    }

    /// Number of set bits strictly below `bit`.
    pub fn rank(&self, bit: usize) -> usize {
        let w = bit >> 6;
        let mut total = 0;
        for i in 0..w.min(self.words.len()) {
            total += self.words[i].count_ones() as usize;
        }
        if w < self.words.len() {
            let partial = self.words[w] & ((1u64 << (bit & 63)) - 1);
            total += partial.count_ones() as usize;
        }
        total
    }

    /// The position of the `n`-th set bit (0-indexed), if it exists.
    pub fn select(&self, mut n: usize) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate() {
            let c = w.count_ones() as usize;
            if n < c {
                let mut bits = w;
                for _ in 0..n {
                    bits &= bits - 1;
                }
                return Some(wi * 64 + bits.trailing_zeros() as usize);
            }
            n -= c;
        }
        None
    }

    /// Collect all set bits in ascending order.
    pub fn to_vec(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count());
        for (wi, &w) in self.words.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                out.push(wi * 64 + t);
                bits &= bits - 1;
            }
        }
        out
    }

    /// In-place union.
    pub fn union_with(&mut self, other: &BitSet) {
        self.ensure(other.words.len().saturating_sub(1));
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }

    /// In-place intersection.
    pub fn intersect_with(&mut self, other: &BitSet) {
        for (i, a) in self.words.iter_mut().enumerate() {
            *a &= other.words.get(i).copied().unwrap_or(0);
        }
    }

    /// In-place difference (self minus other).
    pub fn difference_with(&mut self, other: &BitSet) {
        for (i, a) in self.words.iter_mut().enumerate() {
            *a &= !other.words.get(i).copied().unwrap_or(0);
        }
    }

    /// `true` if every bit of `self` is also in `other`.
    pub fn is_subset_of(&self, other: &BitSet) -> bool {
        for (i, &a) in self.words.iter().enumerate() {
            let b = other.words.get(i).copied().unwrap_or(0);
            if a & !b != 0 {
                return false;
            }
        }
        true
    }

    /// `true` if the two sets share no bits.
    pub fn is_disjoint(&self, other: &BitSet) -> bool {
        let n = self.words.len().min(other.words.len());
        (0..n).all(|i| self.words[i] & other.words[i] == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_remove() {
        let mut s = BitSet::new();
        assert!(s.insert(5));
        assert!(!s.insert(5));
        assert!(s.contains(5));
        assert!(!s.contains(6));
        assert!(s.remove(5));
        assert!(!s.contains(5));
    }

    #[test]
    fn count_min_max() {
        let mut s = BitSet::new();
        for b in [3, 64, 65, 200] {
            s.insert(b);
        }
        assert_eq!(s.count(), 4);
        assert_eq!(s.min(), Some(3));
        assert_eq!(s.max(), Some(200));
    }

    #[test]
    fn rank_and_select() {
        let mut s = BitSet::new();
        for b in [1, 4, 9, 70] {
            s.insert(b);
        }
        assert_eq!(s.rank(9), 2); // bits below 9: {1,4}
        assert_eq!(s.rank(10), 3);
        assert_eq!(s.select(0), Some(1));
        assert_eq!(s.select(2), Some(9));
        assert_eq!(s.select(3), Some(70));
        assert_eq!(s.select(4), None);
    }

    #[test]
    fn set_algebra() {
        let mut a = BitSet::new();
        for b in [1, 2, 3] {
            a.insert(b);
        }
        let mut b = BitSet::new();
        for x in [2, 3, 4] {
            b.insert(x);
        }
        let mut u = a.clone();
        u.union_with(&b);
        assert_eq!(u.to_vec(), vec![1, 2, 3, 4]);
        let mut i = a.clone();
        i.intersect_with(&b);
        assert_eq!(i.to_vec(), vec![2, 3]);
        let mut d = a.clone();
        d.difference_with(&b);
        assert_eq!(d.to_vec(), vec![1]);
    }

    #[test]
    fn subset_and_disjoint() {
        let full = BitSet::full(10);
        let mut sub = BitSet::new();
        sub.insert(2);
        sub.insert(7);
        assert!(sub.is_subset_of(&full));
        assert!(!full.is_subset_of(&sub));
        let mut other = BitSet::new();
        other.insert(20);
        assert!(sub.is_disjoint(&other));
    }
}
