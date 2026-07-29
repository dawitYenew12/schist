//! A set of disjoint integer ranges.
//!
//! Predicate analysis often reduces a column restriction to a union of closed
//! integer ranges (`x IN (1,2,3) OR x BETWEEN 10 AND 20`), and the zone-map
//! pruner intersects such a set against each page's `[min, max]`. A `RangeSet`
//! keeps a sorted, coalesced list of non-overlapping `[lo, hi]` ranges and
//! supports union, intersection, membership, and complement within a universe,
//! so these predicate domains compose cleanly.

/// A set of disjoint, sorted closed integer ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RangeSet {
    ranges: Vec<(i64, i64)>,
}

impl RangeSet {
    /// The empty set.
    pub fn new() -> RangeSet {
        RangeSet { ranges: Vec::new() }
    }

    /// A set with a single range `[lo, hi]` (empty if `lo > hi`).
    pub fn single(lo: i64, hi: i64) -> RangeSet {
        if lo > hi {
            RangeSet::new()
        } else {
            RangeSet {
                ranges: vec![(lo, hi)],
            }
        }
    }

    /// A set of individual points.
    pub fn from_points(points: &[i64]) -> RangeSet {
        let mut s = RangeSet::new();
        for &p in points {
            s.add(p, p);
        }
        s
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Number of disjoint ranges.
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Total number of integers covered (saturating).
    pub fn cardinality(&self) -> u64 {
        self.ranges
            .iter()
            .map(|&(lo, hi)| (hi as i128 - lo as i128 + 1) as u64)
            .sum()
    }

    /// The disjoint ranges.
    pub fn ranges(&self) -> &[(i64, i64)] {
        &self.ranges
    }

    /// Add a range `[lo, hi]`, coalescing with any overlaps.
    pub fn add(&mut self, lo: i64, hi: i64) {
        if lo > hi {
            return;
        }
        let mut result = Vec::with_capacity(self.ranges.len() + 1);
        let mut new_lo = lo;
        let mut new_hi = hi;
        let mut inserted = false;
        for &(rlo, rhi) in &self.ranges {
            if rhi < new_lo.saturating_sub(1) {
                // Entirely before the new range.
                result.push((rlo, rhi));
            } else if rlo > new_hi.saturating_add(1) {
                // Entirely after; insert the (coalesced) new range once.
                if !inserted {
                    result.push((new_lo, new_hi));
                    inserted = true;
                }
                result.push((rlo, rhi));
            } else {
                // Overlaps/adjacent: merge.
                new_lo = new_lo.min(rlo);
                new_hi = new_hi.max(rhi);
            }
        }
        if !inserted {
            result.push((new_lo, new_hi));
        }
        result.sort_by_key(|r| r.0);
        self.ranges = result;
    }

    /// `true` if `x` is in the set.
    pub fn contains(&self, x: i64) -> bool {
        // Binary search for a range covering x.
        let mut lo = 0usize;
        let mut hi = self.ranges.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (rlo, rhi) = self.ranges[mid];
            if x < rlo {
                hi = mid;
            } else if x > rhi {
                lo = mid + 1;
            } else {
                return true;
            }
        }
        false
    }

    /// Union with another set.
    pub fn union(&self, other: &RangeSet) -> RangeSet {
        let mut out = self.clone();
        for &(lo, hi) in &other.ranges {
            out.add(lo, hi);
        }
        out
    }

    /// Intersection with another set.
    pub fn intersect(&self, other: &RangeSet) -> RangeSet {
        let mut out = RangeSet::new();
        let (mut i, mut j) = (0, 0);
        while i < self.ranges.len() && j < other.ranges.len() {
            let (alo, ahi) = self.ranges[i];
            let (blo, bhi) = other.ranges[j];
            let lo = alo.max(blo);
            let hi = ahi.min(bhi);
            if lo <= hi {
                out.ranges.push((lo, hi));
            }
            if ahi < bhi {
                i += 1;
            } else {
                j += 1;
            }
        }
        out
    }

    /// `true` if this set overlaps `[lo, hi]` (the zone-map pruning query).
    pub fn overlaps(&self, lo: i64, hi: i64) -> bool {
        if lo > hi {
            return false;
        }
        self.ranges.iter().any(|&(rlo, rhi)| rlo <= hi && lo <= rhi)
    }

    /// The complement within `[universe_lo, universe_hi]`.
    pub fn complement(&self, universe_lo: i64, universe_hi: i64) -> RangeSet {
        let mut out = RangeSet::new();
        let mut cursor = universe_lo;
        for &(lo, hi) in &self.ranges {
            if hi < universe_lo || lo > universe_hi {
                continue;
            }
            let lo = lo.max(universe_lo);
            if lo > cursor {
                out.ranges.push((cursor, lo - 1));
            }
            cursor = cursor.max(hi.saturating_add(1));
        }
        if cursor <= universe_hi {
            out.ranges.push((cursor, universe_hi));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_coalesces() {
        let mut s = RangeSet::new();
        s.add(1, 5);
        s.add(3, 8); // overlaps 1..5
        s.add(10, 12);
        assert_eq!(s.ranges(), &[(1, 8), (10, 12)]);
        // 9 is adjacent to both 8 and 10, bridging the two ranges.
        s.add(9, 9);
        assert_eq!(s.ranges(), &[(1, 12)]);
    }

    #[test]
    fn add_keeps_gap() {
        let mut s = RangeSet::new();
        s.add(1, 5);
        s.add(8, 12); // gap at 6,7
        assert_eq!(s.ranges(), &[(1, 5), (8, 12)]);
    }

    #[test]
    fn membership() {
        let mut s = RangeSet::new();
        s.add(1, 5);
        s.add(10, 15);
        assert!(s.contains(3));
        assert!(s.contains(10));
        assert!(!s.contains(7));
        assert!(!s.contains(16));
    }

    #[test]
    fn union_and_intersect() {
        let a = RangeSet::single(1, 10);
        let b = RangeSet::single(5, 15);
        assert_eq!(a.union(&b).ranges(), &[(1, 15)]);
        assert_eq!(a.intersect(&b).ranges(), &[(5, 10)]);
        let disjoint = RangeSet::single(100, 200);
        assert!(a.intersect(&disjoint).is_empty());
    }

    #[test]
    fn from_points_merges_adjacent() {
        let s = RangeSet::from_points(&[1, 2, 3, 7, 8]);
        assert_eq!(s.ranges(), &[(1, 3), (7, 8)]);
        assert_eq!(s.cardinality(), 5);
    }

    #[test]
    fn overlaps_query() {
        let mut s = RangeSet::new();
        s.add(1, 5);
        s.add(20, 25);
        assert!(s.overlaps(4, 10));
        assert!(!s.overlaps(6, 19));
        assert!(s.overlaps(0, 100));
    }

    #[test]
    fn complement_within_universe() {
        let mut s = RangeSet::new();
        s.add(3, 5);
        s.add(8, 9);
        let c = s.complement(0, 10);
        assert_eq!(c.ranges(), &[(0, 2), (6, 7), (10, 10)]);
    }
}
