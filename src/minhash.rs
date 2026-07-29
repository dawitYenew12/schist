//! MinHash Jaccard-similarity sketches.
//!
//! Detecting near-duplicate columns or estimating join-key overlap between two
//! tables benefits from a compact set-similarity sketch. MinHash keeps, for each
//! of `k` independent hash functions, the minimum hash value seen over a set;
//! the fraction of the `k` slots that two sketches agree on is an unbiased
//! estimator of the Jaccard similarity of the underlying sets. The `k` hashes
//! are synthesized from one base hash with `k` random `(a, b)` multipliers, the
//! standard universal-hashing trick.

/// A MinHash sketch with a fixed number of hash slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHash {
    mins: Vec<u64>,
    a: Vec<u64>,
    b: Vec<u64>,
}

const MERSENNE: u64 = (1 << 61) - 1;

impl MinHash {
    /// A sketch with `k` slots, seeded deterministically.
    pub fn new(k: usize) -> MinHash {
        MinHash::with_seed(k, 0xA5A5_5A5A_1234_5678)
    }

    /// A sketch with an explicit seed for the hash coefficients.
    pub fn with_seed(k: usize, seed: u64) -> MinHash {
        let k = k.max(1);
        let mut a = Vec::with_capacity(k);
        let mut b = Vec::with_capacity(k);
        let mut state = seed | 1;
        for _ in 0..k {
            state = splitmix(state);
            a.push((state % (MERSENNE - 1)) + 1);
            state = splitmix(state);
            b.push(state % MERSENNE);
        }
        MinHash {
            mins: vec![u64::MAX; k],
            a,
            b,
        }
    }

    /// Number of slots.
    pub fn slots(&self) -> usize {
        self.mins.len()
    }

    /// Add an element (a pre-hashed 64-bit value).
    pub fn add_hash(&mut self, h: u64) {
        let x = h % MERSENNE;
        for i in 0..self.mins.len() {
            // Universal hash: (a*x + b) mod Mersenne prime.
            let hv = mulmod(self.a[i], x).wrapping_add(self.b[i]) % MERSENNE;
            if hv < self.mins[i] {
                self.mins[i] = hv;
            }
        }
    }

    /// Add an integer element.
    pub fn add_i64(&mut self, v: i64) {
        self.add_hash(splitmix(v as u64));
    }

    /// Add a byte-string element.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        self.add_hash(hash_bytes(bytes));
    }

    /// Estimate the Jaccard similarity with another sketch (same k and seed).
    pub fn jaccard(&self, other: &MinHash) -> f64 {
        if self.mins.len() != other.mins.len() {
            return 0.0;
        }
        let matches = self
            .mins
            .iter()
            .zip(other.mins.iter())
            .filter(|(a, b)| a == b && **a != u64::MAX)
            .count();
        matches as f64 / self.mins.len() as f64
    }

    /// Merge another sketch (union of the underlying sets).
    pub fn merge(&mut self, other: &MinHash) -> bool {
        if self.mins.len() != other.mins.len() {
            return false;
        }
        for (a, b) in self.mins.iter_mut().zip(other.mins.iter()) {
            *a = (*a).min(*b);
        }
        true
    }

    /// The raw min-signature (for banding / LSH).
    pub fn signature(&self) -> &[u64] {
        &self.mins
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn mulmod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % MERSENNE as u128) as u64
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF29CE484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    splitmix(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch_of(range: std::ops::Range<i64>, k: usize) -> MinHash {
        let mut m = MinHash::new(k);
        for i in range {
            m.add_i64(i);
        }
        m
    }

    #[test]
    fn identical_sets_similar() {
        let a = sketch_of(0..1000, 128);
        let b = sketch_of(0..1000, 128);
        assert!((a.jaccard(&b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn disjoint_sets_dissimilar() {
        let a = sketch_of(0..1000, 128);
        let b = sketch_of(10_000..11_000, 128);
        assert!(a.jaccard(&b) < 0.05);
    }

    #[test]
    fn half_overlap_estimated() {
        // A = [0,1000), B = [500,1500): Jaccard = 500/1500 = 0.333.
        let a = sketch_of(0..1000, 256);
        let b = sketch_of(500..1500, 256);
        let est = a.jaccard(&b);
        assert!((est - 0.333).abs() < 0.1, "estimate {est}");
    }

    #[test]
    fn merge_is_union() {
        let mut a = sketch_of(0..500, 128);
        let b = sketch_of(500..1000, 128);
        assert!(a.merge(&b));
        let full = sketch_of(0..1000, 128);
        // Merged signature should match the union's signature.
        assert!((a.jaccard(&full) - 1.0).abs() < 0.001);
    }

    #[test]
    fn byte_elements() {
        let mut a = MinHash::new(64);
        let mut b = MinHash::new(64);
        for w in ["apple", "banana", "cherry"] {
            a.add_bytes(w.as_bytes());
        }
        for w in ["banana", "cherry", "date"] {
            b.add_bytes(w.as_bytes());
        }
        // Jaccard of {a,b,c} and {b,c,d} = 2/4 = 0.5.
        let est = a.jaccard(&b);
        assert!(est > 0.2 && est < 0.8, "estimate {est}");
    }
}
