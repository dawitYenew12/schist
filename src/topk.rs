//! Heavy-hitter (top-k frequent item) tracking with the Space-Saving algorithm.
//!
//! Approximate group-by and "most common values" statistics need the top-k most
//! frequent keys from a stream without keeping a counter per distinct key. The
//! Space-Saving algorithm keeps only `k` monitored counters: a new key that is
//! not monitored evicts the currently-smallest counter, inheriting its count as
//! an over-estimate. This bounds memory to `k` entries while guaranteeing that
//! any item with true frequency above `stream_len / k` is reported, and the
//! reported counts are upper bounds with a tracked error.

use std::collections::HashMap;

/// One monitored key with its estimated count and maximum over-estimate error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Counter {
    pub key: i64,
    pub count: u64,
    pub error: u64,
}

/// A Space-Saving summary tracking up to `capacity` heavy hitters.
#[derive(Debug, Clone)]
pub struct SpaceSaving {
    capacity: usize,
    counters: HashMap<i64, (u64, u64)>, // key -> (count, error)
    total: u64,
}

impl SpaceSaving {
    /// A summary with `capacity` monitored counters.
    pub fn new(capacity: usize) -> SpaceSaving {
        SpaceSaving {
            capacity: capacity.max(1),
            counters: HashMap::new(),
            total: 0,
        }
    }

    /// Total number of observations.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Number of currently monitored keys.
    pub fn monitored(&self) -> usize {
        self.counters.len()
    }

    /// Observe a key `weight` times.
    pub fn add_weighted(&mut self, key: i64, weight: u64) {
        self.total += weight;
        if let Some(entry) = self.counters.get_mut(&key) {
            entry.0 += weight;
            return;
        }
        if self.counters.len() < self.capacity {
            self.counters.insert(key, (weight, 0));
            return;
        }
        // Evict the minimum-count counter and inherit its count as the error.
        let (min_key, min_count) = self
            .counters
            .iter()
            .min_by_key(|(_, (c, _))| *c)
            .map(|(k, (c, _))| (*k, *c))
            .unwrap();
        self.counters.remove(&min_key);
        self.counters.insert(key, (min_count + weight, min_count));
    }

    /// Observe a key once.
    pub fn add(&mut self, key: i64) {
        self.add_weighted(key, 1);
    }

    /// The estimated count for a key (0 if not monitored).
    pub fn estimate(&self, key: i64) -> u64 {
        self.counters.get(&key).map(|(c, _)| *c).unwrap_or(0)
    }

    /// The top `n` monitored keys by estimated count, descending.
    pub fn top(&self, n: usize) -> Vec<Counter> {
        let mut items: Vec<Counter> = self
            .counters
            .iter()
            .map(|(&key, &(count, error))| Counter { key, count, error })
            .collect();
        items.sort_by(|a, b| b.count.cmp(&a.count).then(a.key.cmp(&b.key)));
        items.truncate(n);
        items
    }

    /// `true` if `key` is *guaranteed* to have true frequency above `threshold`
    /// (its guaranteed lower bound `count - error` exceeds the threshold).
    pub fn guaranteed_above(&self, key: i64, threshold: u64) -> bool {
        self.counters
            .get(&key)
            .map(|&(count, error)| count.saturating_sub(error) > threshold)
            .unwrap_or(false)
    }

    /// All keys whose guaranteed lower bound exceeds `frequency * total`.
    pub fn heavy_hitters(&self, frequency: f64) -> Vec<Counter> {
        let threshold = (frequency * self.total as f64) as u64;
        let mut out: Vec<Counter> = self
            .counters
            .iter()
            .filter(|(_, &(count, error))| count.saturating_sub(error) > threshold)
            .map(|(&key, &(count, error))| Counter { key, count, error })
            .collect();
        out.sort_by(|a, b| b.count.cmp(&a.count));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_frequent_items() {
        let mut ss = SpaceSaving::new(3);
        for _ in 0..100 {
            ss.add(1);
        }
        for _ in 0..50 {
            ss.add(2);
        }
        for i in 0..10 {
            ss.add(1000 + i); // many rare keys
        }
        let top = ss.top(2);
        assert_eq!(top[0].key, 1);
        assert!(top[0].count >= 100);
        assert_eq!(top[1].key, 2);
    }

    #[test]
    fn capacity_bounds_memory() {
        let mut ss = SpaceSaving::new(4);
        for i in 0..1000 {
            ss.add(i);
        }
        assert!(ss.monitored() <= 4);
        assert_eq!(ss.total(), 1000);
    }

    #[test]
    fn estimate_is_upper_bound() {
        let mut ss = SpaceSaving::new(2);
        for _ in 0..10 {
            ss.add(5);
        }
        // 5 is heavy; its estimate is at least its true count.
        assert!(ss.estimate(5) >= 10);
    }

    #[test]
    fn heavy_hitters_reported() {
        let mut ss = SpaceSaving::new(8);
        // 1 dominates.
        for _ in 0..900 {
            ss.add(1);
        }
        for i in 0..100 {
            ss.add(2000 + i);
        }
        let hh = ss.heavy_hitters(0.5);
        assert_eq!(hh.len(), 1);
        assert_eq!(hh[0].key, 1);
        assert!(ss.guaranteed_above(1, 500));
    }

    #[test]
    fn weighted_add() {
        let mut ss = SpaceSaving::new(4);
        ss.add_weighted(7, 100);
        assert_eq!(ss.estimate(7), 100);
        assert_eq!(ss.total(), 100);
    }
}
