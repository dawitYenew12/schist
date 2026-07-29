//! Hash partitioning for parallel and out-of-core operators.
//!
//! A hash join or hash aggregate whose build side does not fit in memory (or
//! that wants to spread work over workers) partitions both inputs by a hash of
//! the key so that matching keys always land in the same partition, and each
//! partition can then be processed independently. This module implements the
//! partition function and a radix-style partitioner that distributes rows of
//! keyed values into a fixed number of buckets, with a histogram pass so the
//! output offsets can be computed up front.

use crate::checksum::fxhash_u64;

/// Assign a key to one of `num_partitions` buckets (must be a power of two).
pub fn partition_of(key: i64, num_partitions: usize) -> usize {
    debug_assert!(num_partitions.is_power_of_two());
    let h = fxhash_u64(key as u64);
    (h as usize) & (num_partitions - 1)
}

/// A radix partitioner over `(key, payload)` rows.
pub struct Partitioner {
    num_partitions: usize,
    buckets: Vec<Vec<(i64, u32)>>,
}

impl Partitioner {
    /// A partitioner with `num_partitions` buckets (rounded up to a power of
    /// two).
    pub fn new(num_partitions: usize) -> Partitioner {
        let n = num_partitions.max(1).next_power_of_two();
        Partitioner {
            num_partitions: n,
            buckets: (0..n).map(|_| Vec::new()).collect(),
        }
    }

    /// The (power-of-two) partition count.
    pub fn partition_count(&self) -> usize {
        self.num_partitions
    }

    /// Route one `(key, payload)` pair to its bucket.
    pub fn push(&mut self, key: i64, payload: u32) {
        let p = partition_of(key, self.num_partitions);
        self.buckets[p].push((key, payload));
    }

    /// Route a batch of keys (payload = row index).
    pub fn push_keys(&mut self, keys: &[i64]) {
        for (i, &k) in keys.iter().enumerate() {
            self.push(k, i as u32);
        }
    }

    /// Borrow a partition's rows.
    pub fn partition(&self, i: usize) -> &[(i64, u32)] {
        &self.buckets[i]
    }

    /// The number of rows in each partition.
    pub fn histogram(&self) -> Vec<usize> {
        self.buckets.iter().map(|b| b.len()).collect()
    }

    /// Total rows partitioned.
    pub fn total(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// The maximum partition size (a skew measure).
    pub fn max_partition(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).max().unwrap_or(0)
    }

    /// A skew factor: max partition size divided by the mean (1.0 = perfectly
    /// even). Returns 0.0 for an empty partitioner.
    pub fn skew(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let mean = total as f64 / self.num_partitions as f64;
        self.max_partition() as f64 / mean
    }

    /// Consume the partitioner, yielding each bucket in order.
    pub fn into_partitions(self) -> Vec<Vec<(i64, u32)>> {
        self.buckets
    }
}

/// Co-partition two key sets so that equal keys share a partition index; returns
/// `(left_partitions, right_partitions)`.
pub fn copartition(
    left: &[i64],
    right: &[i64],
    num_partitions: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let n = num_partitions.max(1).next_power_of_two();
    let mut lp: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
    let mut rp: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
    for (i, &k) in left.iter().enumerate() {
        lp[partition_of(k, n)].push(i as u32);
    }
    for (i, &k) in right.iter().enumerate() {
        rp[partition_of(k, n)].push(i as u32);
    }
    (lp, rp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_is_stable() {
        assert_eq!(partition_of(42, 16), partition_of(42, 16));
    }

    #[test]
    fn rounds_up_to_power_of_two() {
        let p = Partitioner::new(6);
        assert_eq!(p.partition_count(), 8);
    }

    #[test]
    fn distributes_keys() {
        let mut p = Partitioner::new(8);
        let keys: Vec<i64> = (0..800).collect();
        p.push_keys(&keys);
        assert_eq!(p.total(), 800);
        let hist = p.histogram();
        assert_eq!(hist.iter().sum::<usize>(), 800);
        // Reasonably even.
        assert!(p.skew() < 1.5, "skew {}", p.skew());
    }

    #[test]
    fn equal_keys_same_partition() {
        let mut p = Partitioner::new(8);
        p.push(1234, 0);
        p.push(1234, 1);
        let counts: Vec<usize> = p.histogram().iter().filter(|&&c| c > 0).copied().collect();
        assert_eq!(counts, vec![2]);
    }

    #[test]
    fn copartition_aligns_keys() {
        let left = vec![1, 2, 3, 4];
        let right = vec![3, 4, 5, 6];
        let (lp, rp) = copartition(&left, &right, 8);
        // Key 3 appears in left[2] and right[0]; they must share a partition.
        let p3 = partition_of(3, 8);
        assert!(lp[p3].contains(&2));
        assert!(rp[p3].contains(&0));
    }

    #[test]
    fn empty_skew_is_zero() {
        let p = Partitioner::new(4);
        assert_eq!(p.skew(), 0.0);
    }
}
