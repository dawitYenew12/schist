//! A probabilistic skip-list ordered map keyed by `i64`.
//!
//! Ordered access paths (range scans over a sorted key, merge-join inputs) want
//! an ordered map with logarithmic search and cheap in-order iteration. A skip
//! list gives that without the rebalancing bookkeeping of a tree: each node is
//! promoted to a random number of levels, and searches drop down levels as they
//! overshoot. The randomness here is a small deterministic xorshift so the
//! structure is reproducible in tests.

/// A deterministic xorshift64 PRNG used to pick node heights.
#[derive(Debug, Clone)]
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> XorShift {
        XorShift {
            state: seed | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

const MAX_LEVEL: usize = 16;

#[derive(Debug, Clone)]
struct Node {
    key: i64,
    value: i64,
    /// `forward[l]` is the index of the next node at level `l`, or `usize::MAX`.
    forward: Vec<usize>,
}

const NIL: usize = usize::MAX;

/// An ordered map from `i64` keys to `i64` values.
#[derive(Debug, Clone)]
pub struct SkipList {
    nodes: Vec<Node>,
    head: Vec<usize>,
    level: usize,
    len: usize,
    rng: XorShift,
}

impl SkipList {
    /// A new empty list seeded deterministically.
    pub fn new() -> SkipList {
        SkipList::with_seed(0x9E3779B97F4A7C15)
    }

    /// A new empty list with an explicit RNG seed.
    pub fn with_seed(seed: u64) -> SkipList {
        SkipList {
            nodes: Vec::new(),
            head: vec![NIL; MAX_LEVEL],
            level: 1,
            len: 0,
            rng: XorShift::new(seed),
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn random_level(&mut self) -> usize {
        let mut lvl = 1;
        // 1/4 promotion probability per level.
        while lvl < MAX_LEVEL && (self.rng.next_u64() & 3) == 0 {
            lvl += 1;
        }
        lvl
    }

    fn forward(&self, node: usize, level: usize) -> usize {
        if node == NIL {
            self.head[level]
        } else {
            self.nodes[node].forward[level]
        }
    }

    /// Insert or update a key; returns the previous value if present.
    pub fn insert(&mut self, key: i64, value: i64) -> Option<i64> {
        let mut update = [NIL; MAX_LEVEL];
        let mut x = NIL; // NIL denotes the head sentinel here.
        for l in (0..self.level).rev() {
            loop {
                let nxt = self.forward(x, l);
                if nxt != NIL && self.nodes[nxt].key < key {
                    x = nxt;
                } else {
                    break;
                }
            }
            update[l] = x;
        }
        let candidate = self.forward(x, 0);
        if candidate != NIL && self.nodes[candidate].key == key {
            let old = self.nodes[candidate].value;
            self.nodes[candidate].value = value;
            return Some(old);
        }
        let new_level = self.random_level();
        if new_level > self.level {
            for l in self.level..new_level {
                update[l] = NIL;
            }
            self.level = new_level;
        }
        let idx = self.nodes.len();
        let mut forward = vec![NIL; new_level];
        for l in 0..new_level {
            let prev = update[l];
            forward[l] = self.forward(prev, l);
            if prev == NIL {
                self.head[l] = idx;
            } else {
                self.nodes[prev].forward[l] = idx;
            }
        }
        self.nodes.push(Node {
            key,
            value,
            forward,
        });
        self.len += 1;
        None
    }

    /// Look up a key's value.
    pub fn get(&self, key: i64) -> Option<i64> {
        let mut x = NIL;
        for l in (0..self.level).rev() {
            loop {
                let nxt = self.forward(x, l);
                if nxt != NIL && self.nodes[nxt].key < key {
                    x = nxt;
                } else {
                    break;
                }
            }
        }
        let candidate = self.forward(x, 0);
        if candidate != NIL && self.nodes[candidate].key == key {
            Some(self.nodes[candidate].value)
        } else {
            None
        }
    }

    /// `true` if the key is present.
    pub fn contains(&self, key: i64) -> bool {
        self.get(key).is_some()
    }

    /// Collect all `(key, value)` pairs in ascending key order.
    pub fn to_sorted_vec(&self) -> Vec<(i64, i64)> {
        let mut out = Vec::with_capacity(self.len);
        let mut node = self.head[0];
        while node != NIL {
            out.push((self.nodes[node].key, self.nodes[node].value));
            node = self.nodes[node].forward[0];
        }
        out
    }

    /// Collect `(key, value)` pairs with `lo <= key < hi`, in order.
    pub fn range(&self, lo: i64, hi: i64) -> Vec<(i64, i64)> {
        let mut out = Vec::new();
        // Descend to the first node with key >= lo.
        let mut x = NIL;
        for l in (0..self.level).rev() {
            loop {
                let nxt = self.forward(x, l);
                if nxt != NIL && self.nodes[nxt].key < lo {
                    x = nxt;
                } else {
                    break;
                }
            }
        }
        let mut node = self.forward(x, 0);
        while node != NIL && self.nodes[node].key < hi {
            if self.nodes[node].key >= lo {
                out.push((self.nodes[node].key, self.nodes[node].value));
            }
            node = self.nodes[node].forward[0];
        }
        out
    }

    /// The smallest key >= `key` and its value.
    pub fn ceiling(&self, key: i64) -> Option<(i64, i64)> {
        let mut x = NIL;
        for l in (0..self.level).rev() {
            loop {
                let nxt = self.forward(x, l);
                if nxt != NIL && self.nodes[nxt].key < key {
                    x = nxt;
                } else {
                    break;
                }
            }
        }
        let node = self.forward(x, 0);
        if node != NIL {
            Some((self.nodes[node].key, self.nodes[node].value))
        } else {
            None
        }
    }
}

impl Default for SkipList {
    fn default() -> Self {
        SkipList::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_update() {
        let mut s = SkipList::new();
        assert_eq!(s.insert(5, 50), None);
        assert_eq!(s.insert(3, 30), None);
        assert_eq!(s.get(5), Some(50));
        assert_eq!(s.insert(5, 55), Some(50));
        assert_eq!(s.get(5), Some(55));
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn ordered_iteration() {
        let mut s = SkipList::new();
        for k in [7, 1, 9, 3, 5, 2, 8, 4, 6] {
            s.insert(k, k * 10);
        }
        let sorted = s.to_sorted_vec();
        let keys: Vec<i64> = sorted.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn range_scan() {
        let mut s = SkipList::new();
        for k in 0..20 {
            s.insert(k, k);
        }
        let r = s.range(5, 10);
        let keys: Vec<i64> = r.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn ceiling_lookup() {
        let mut s = SkipList::new();
        for k in [2, 4, 6, 8] {
            s.insert(k, k);
        }
        assert_eq!(s.ceiling(5), Some((6, 6)));
        assert_eq!(s.ceiling(6), Some((6, 6)));
        assert_eq!(s.ceiling(9), None);
    }

    #[test]
    fn large_random_consistency() {
        let mut s = SkipList::with_seed(12345);
        let mut expected = std::collections::BTreeMap::new();
        let mut state = 1u64;
        for _ in 0..2000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let k = (state >> 33) as i64 % 500;
            s.insert(k, k + 1);
            expected.insert(k, k + 1);
        }
        assert_eq!(s.len(), expected.len());
        for (&k, &v) in &expected {
            assert_eq!(s.get(k), Some(v));
        }
        let got: Vec<(i64, i64)> = s.to_sorted_vec();
        let want: Vec<(i64, i64)> = expected.into_iter().collect();
        assert_eq!(got, want);
    }
}
