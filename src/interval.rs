//! An augmented interval tree over `i64` ranges.
//!
//! Zone-map pruning and range-predicate planning need to ask "which stored
//! ranges overlap this query range?" quickly. An interval tree — a balanced
//! binary search tree keyed by interval start, with each node augmented by the
//! maximum endpoint in its subtree — answers stum overlap queries in output-
//! sensitive time by pruning subtrees whose max endpoint is below the query.
//! This implementation keeps the tree as a flat node vector and rebuilds it
//! balanced from the sorted intervals, which suits the build-once/query-many
//! access pattern of zone maps.

/// A closed interval `[lo, hi]` carrying a payload id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
    pub id: u32,
}

impl Interval {
    /// A new interval (endpoints are swapped if given reversed).
    pub fn new(lo: i64, hi: i64, id: u32) -> Interval {
        if lo <= hi {
            Interval { lo, hi, id }
        } else {
            Interval { lo: hi, hi: lo, id }
        }
    }

    /// `true` if this interval overlaps `[qlo, qhi]`.
    pub fn overlaps(&self, qlo: i64, qhi: i64) -> bool {
        self.lo <= qhi && qlo <= self.hi
    }

    /// `true` if `point` lies within.
    pub fn contains(&self, point: i64) -> bool {
        self.lo <= point && point <= self.hi
    }
}

#[derive(Debug, Clone)]
struct TreeNode {
    interval: Interval,
    max_hi: i64,
    left: i32,
    right: i32,
}

const NONE: i32 = -1;

/// An immutable interval tree built from a set of intervals.
#[derive(Debug, Clone, Default)]
pub struct IntervalTree {
    nodes: Vec<TreeNode>,
    root: i32,
}

impl IntervalTree {
    /// Build a balanced tree from the given intervals.
    pub fn build(mut intervals: Vec<Interval>) -> IntervalTree {
        intervals.sort_by_key(|iv| (iv.lo, iv.hi));
        let mut tree = IntervalTree {
            nodes: Vec::with_capacity(intervals.len()),
            root: NONE,
        };
        tree.root = tree.build_range(&intervals, 0, intervals.len());
        tree
    }

    fn build_range(&mut self, sorted: &[Interval], lo: usize, hi: usize) -> i32 {
        if lo >= hi {
            return NONE;
        }
        let mid = lo + (hi - lo) / 2;
        let idx = self.nodes.len() as i32;
        self.nodes.push(TreeNode {
            interval: sorted[mid],
            max_hi: sorted[mid].hi,
            left: NONE,
            right: NONE,
        });
        let left = self.build_range(sorted, lo, mid);
        let right = self.build_range(sorted, mid + 1, hi);
        self.nodes[idx as usize].left = left;
        self.nodes[idx as usize].right = right;
        let mut max_hi = sorted[mid].hi;
        if left != NONE {
            max_hi = max_hi.max(self.nodes[left as usize].max_hi);
        }
        if right != NONE {
            max_hi = max_hi.max(self.nodes[right as usize].max_hi);
        }
        self.nodes[idx as usize].max_hi = max_hi;
        idx
    }

    /// Number of stored intervals.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Collect all intervals overlapping `[qlo, qhi]`.
    pub fn query_overlapping(&self, qlo: i64, qhi: i64) -> Vec<Interval> {
        let mut out = Vec::new();
        self.query_node(self.root, qlo, qhi, &mut out);
        out
    }

    fn query_node(&self, node: i32, qlo: i64, qhi: i64, out: &mut Vec<Interval>) {
        if node == NONE {
            return;
        }
        let n = &self.nodes[node as usize];
        // Prune: if the max endpoint in this subtree is below the query start,
        // nothing here can overlap.
        if n.max_hi < qlo {
            return;
        }
        // Search left first (intervals with smaller start).
        self.query_node(n.left, qlo, qhi, out);
        if n.interval.overlaps(qlo, qhi) {
            out.push(n.interval);
        }
        // If this node's start is beyond the query end, the right subtree
        // (even larger starts) cannot overlap.
        if n.interval.lo <= qhi {
            self.query_node(n.right, qlo, qhi, out);
        }
    }

    /// Collect all intervals containing `point`.
    pub fn query_point(&self, point: i64) -> Vec<Interval> {
        self.query_overlapping(point, point)
    }

    /// `true` if any stored interval overlaps `[qlo, qhi]`.
    pub fn any_overlap(&self, qlo: i64, qhi: i64) -> bool {
        self.stab(self.root, qlo, qhi)
    }

    fn stab(&self, node: i32, qlo: i64, qhi: i64) -> bool {
        if node == NONE {
            return false;
        }
        let n = &self.nodes[node as usize];
        if n.max_hi < qlo {
            return false;
        }
        if n.interval.overlaps(qlo, qhi) {
            return true;
        }
        if self.stab(n.left, qlo, qhi) {
            return true;
        }
        if n.interval.lo <= qhi {
            return self.stab(n.right, qlo, qhi);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> IntervalTree {
        IntervalTree::build(vec![
            Interval::new(1, 5, 0),
            Interval::new(10, 20, 1),
            Interval::new(15, 25, 2),
            Interval::new(30, 40, 3),
            Interval::new(3, 8, 4),
        ])
    }

    #[test]
    fn overlap_query() {
        let t = tree();
        let mut ids: Vec<u32> = t.query_overlapping(4, 12).iter().map(|iv| iv.id).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 4]);
    }

    #[test]
    fn point_query() {
        let t = tree();
        let mut ids: Vec<u32> = t.query_point(18).iter().map(|iv| iv.id).collect();
        ids.sort();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn no_overlap() {
        let t = tree();
        assert!(t.query_overlapping(26, 29).is_empty());
        assert!(!t.any_overlap(26, 29));
        assert!(t.any_overlap(35, 100));
    }

    #[test]
    fn reversed_endpoints_normalized() {
        let iv = Interval::new(20, 10, 7);
        assert_eq!(iv.lo, 10);
        assert_eq!(iv.hi, 20);
        assert!(iv.contains(15));
    }

    #[test]
    fn empty_tree() {
        let t = IntervalTree::build(vec![]);
        assert!(t.is_empty());
        assert!(t.query_point(5).is_empty());
        assert!(!t.any_overlap(0, 100));
    }

    #[test]
    fn large_build_and_query_matches_bruteforce() {
        let mut ivs = Vec::new();
        let mut state = 1u64;
        for id in 0..500u32 {
            state = state.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let lo = (state >> 40) as i64 % 1000;
            let len = (state >> 20) as i64 % 50;
            ivs.push(Interval::new(lo, lo + len, id));
        }
        let t = IntervalTree::build(ivs.clone());
        for &(qlo, qhi) in &[(100i64, 150i64), (0, 10), (900, 1000), (500, 501)] {
            let mut expected: Vec<u32> = ivs
                .iter()
                .filter(|iv| iv.overlaps(qlo, qhi))
                .map(|iv| iv.id)
                .collect();
            expected.sort();
            let mut got: Vec<u32> = t
                .query_overlapping(qlo, qhi)
                .iter()
                .map(|iv| iv.id)
                .collect();
            got.sort();
            assert_eq!(got, expected);
        }
    }
}
