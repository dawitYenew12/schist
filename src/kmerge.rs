//! K-way merge of sorted streams via a loser tree.
//!
//! External sort and sort-merge join both merge many already-sorted runs into
//! one sorted output. A loser tree (tournament tree) is the efficient way to do
//! this: it keeps the current head of each run in a tournament, so producing the
//! next smallest element costs one root-to-leaf replay rather than a scan of all
//! runs. This module merges runs of `i64` keys; the generic ordering is by the
//! key, and each element carries the index of the run it came from so callers
//! can recover the associated payload.

/// One merged element: its key and the source run it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merged {
    pub key: i64,
    pub run: usize,
}

/// A loser-tree merger over a fixed set of sorted runs.
pub struct LoserTree {
    runs: Vec<Vec<i64>>,
    cursors: Vec<usize>,
    /// The internal nodes store the index of the "loser" leaf at each position.
    tree: Vec<usize>,
    k: usize,
    winner: usize,
}

const SENTINEL: i64 = i64::MAX;

impl LoserTree {
    /// Build a merger over the given sorted runs.
    pub fn new(runs: Vec<Vec<i64>>) -> LoserTree {
        let k = runs.len().max(1);
        let mut lt = LoserTree {
            cursors: vec![0; k],
            tree: vec![usize::MAX; k],
            runs,
            k,
            winner: 0,
        };
        while lt.runs.len() < k {
            lt.runs.push(Vec::new());
        }
        lt.build();
        lt
    }

    fn leaf_key(&self, leaf: usize) -> i64 {
        let c = self.cursors[leaf];
        self.runs[leaf].get(c).copied().unwrap_or(SENTINEL)
    }

    fn build(&mut self) {
        if self.k == 1 {
            self.winner = 0;
            return;
        }
        // Initialize all internal slots as empty, then play each leaf in.
        for slot in self.tree.iter_mut() {
            *slot = usize::MAX;
        }
        for leaf in 0..self.k {
            self.play(leaf);
        }
    }

    /// Play leaf `leaf` up the tree, settling losers and finding the winner.
    fn play(&mut self, leaf: usize) {
        let mut parent = (leaf + self.k) / 2;
        let mut winner = leaf;
        while parent >= 1 {
            let loser_slot = self.tree[parent];
            if loser_slot == usize::MAX {
                self.tree[parent] = winner;
                return;
            }
            // Compare current winner against the stored occupant.
            if self.leaf_key(loser_slot) < self.leaf_key(winner) {
                // Stored one wins; the challenger becomes the loser here.
                self.tree[parent] = winner;
                winner = loser_slot;
            }
            if parent == 1 {
                break;
            }
            parent /= 2;
        }
        self.winner = winner;
    }

    /// `true` if every run is exhausted.
    pub fn is_empty(&self) -> bool {
        (0..self.k).all(|leaf| self.cursors[leaf] >= self.runs[leaf].len())
    }

    /// Produce the next smallest element, advancing its run.
    pub fn next(&mut self) -> Option<Merged> {
        let leaf = self.winner;
        let key = self.leaf_key(leaf);
        if key == SENTINEL {
            return None;
        }
        self.cursors[leaf] += 1;
        // Replay this leaf to find the new winner.
        self.play(leaf);
        Some(Merged { key, run: leaf })
    }

    /// Drain the entire merge into a vector.
    pub fn collect(mut self) -> Vec<Merged> {
        let mut out = Vec::new();
        while let Some(m) = self.next() {
            out.push(m);
        }
        out
    }
}

/// Merge sorted runs and return the merged keys (dropping run provenance).
pub fn merge_runs(runs: Vec<Vec<i64>>) -> Vec<i64> {
    LoserTree::new(runs).collect().into_iter().map(|m| m.key).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_two_runs() {
        let merged = merge_runs(vec![vec![1, 4, 7], vec![2, 3, 8]]);
        assert_eq!(merged, vec![1, 2, 3, 4, 7, 8]);
    }

    #[test]
    fn merges_many_runs() {
        let runs = vec![
            vec![1, 10, 20],
            vec![2, 11, 21],
            vec![3, 12, 22],
            vec![0, 100],
        ];
        let merged = merge_runs(runs);
        let mut sorted = merged.clone();
        sorted.sort();
        assert_eq!(merged, sorted);
        assert_eq!(merged.len(), 11);
    }

    #[test]
    fn tracks_run_provenance() {
        let lt = LoserTree::new(vec![vec![1, 3], vec![2, 4]]);
        let merged = lt.collect();
        assert_eq!(merged[0], Merged { key: 1, run: 0 });
        assert_eq!(merged[1], Merged { key: 2, run: 1 });
        assert_eq!(merged[3], Merged { key: 4, run: 1 });
    }

    #[test]
    fn handles_empty_runs() {
        let merged = merge_runs(vec![vec![], vec![5], vec![]]);
        assert_eq!(merged, vec![5]);
        assert!(merge_runs(vec![vec![], vec![]]).is_empty());
    }

    #[test]
    fn single_run() {
        assert_eq!(merge_runs(vec![vec![3, 1, 2]]), vec![3, 1, 2]);
    }

    #[test]
    fn large_random_merge() {
        let mut runs = Vec::new();
        let mut state = 7u64;
        let mut all = Vec::new();
        for _ in 0..8 {
            let mut run = Vec::new();
            for _ in 0..100 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                run.push((state >> 40) as i64 % 1000);
            }
            run.sort();
            all.extend(run.iter().copied());
            runs.push(run);
        }
        all.sort();
        let merged = merge_runs(runs);
        assert_eq!(merged, all);
    }
}
