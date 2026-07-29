//! A disjoint-set (union-find) structure.
//!
//! The optimizer groups columns into equivalence classes when it sees equality
//! predicates (`a = b AND b = c` implies `a`, `b`, `c` are interchangeable),
//! and the join planner uses the same idea to find connected components of a
//! join graph. Union-find answers "are these in the same class?" and "merge
//! these classes" in near-constant amortized time via path compression and
//! union by rank.

/// A union-find over `0..n` elements.
#[derive(Debug, Clone)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
    size: Vec<u32>,
    sets: usize,
}

impl DisjointSet {
    /// `n` singleton sets.
    pub fn new(n: usize) -> DisjointSet {
        DisjointSet {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            sets: n,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// `true` if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Grow to at least `n` elements, adding singletons.
    pub fn ensure(&mut self, n: usize) {
        while self.parent.len() < n {
            let i = self.parent.len();
            self.parent.push(i);
            self.rank.push(0);
            self.size.push(1);
            self.sets += 1;
        }
    }

    /// Find the representative of `x`, compressing the path.
    pub fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression.
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    /// Merge the sets containing `a` and `b`. Returns `true` if they were
    /// distinct (a merge happened).
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (big, small) = if self.rank[ra] >= self.rank[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
        if self.rank[big] == self.rank[small] {
            self.rank[big] += 1;
        }
        self.sets -= 1;
        true
    }

    /// `true` if `a` and `b` are in the same set.
    pub fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    /// The size of the set containing `x`.
    pub fn set_size(&mut self, x: usize) -> u32 {
        let r = self.find(x);
        self.size[r]
    }

    /// The number of disjoint sets.
    pub fn count_sets(&self) -> usize {
        self.sets
    }

    /// Group all elements by their representative, returning sorted classes.
    pub fn classes(&mut self) -> Vec<Vec<usize>> {
        use std::collections::HashMap;
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..self.parent.len() {
            let r = self.find(i);
            map.entry(r).or_default().push(i);
        }
        let mut out: Vec<Vec<usize>> = map.into_values().collect();
        for c in &mut out {
            c.sort_unstable();
        }
        out.sort_by_key(|c| c[0]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singletons() {
        let mut d = DisjointSet::new(5);
        assert_eq!(d.count_sets(), 5);
        assert!(!d.connected(0, 1));
    }

    #[test]
    fn union_and_connected() {
        let mut d = DisjointSet::new(6);
        d.union(0, 1);
        d.union(1, 2);
        assert!(d.connected(0, 2));
        assert!(!d.connected(0, 3));
        assert_eq!(d.set_size(0), 3);
        assert_eq!(d.count_sets(), 4);
    }

    #[test]
    fn duplicate_union_no_op() {
        let mut d = DisjointSet::new(3);
        assert!(d.union(0, 1));
        assert!(!d.union(0, 1));
        assert_eq!(d.count_sets(), 2);
    }

    #[test]
    fn classes_grouped() {
        let mut d = DisjointSet::new(6);
        d.union(0, 2);
        d.union(2, 4);
        d.union(1, 3);
        let classes = d.classes();
        assert!(classes.contains(&vec![0, 2, 4]));
        assert!(classes.contains(&vec![1, 3]));
        assert!(classes.contains(&vec![5]));
    }

    #[test]
    fn dynamic_growth() {
        let mut d = DisjointSet::new(2);
        d.ensure(5);
        assert_eq!(d.len(), 5);
        d.union(3, 4);
        assert!(d.connected(3, 4));
    }
}
