//! A HyperLogLog distinct-count estimator.
//!
//! Column statistics need an approximate distinct count (NDV) to drive join
//! ordering and grouping decisions, and an exact count is too expensive to keep
//! current under mutation. HyperLogLog estimates cardinality in fixed memory:
//! the hash of each element is split into a register index and a run of leading
//! zeros; the maximum run seen per register, harmonic-averaged across
//! registers, estimates the cardinality with a standard error of about
//! `1.04 / sqrt(m)`. Small- and large-range corrections are applied.

/// A HyperLogLog sketch.
#[derive(Clone, Debug)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: u8,
    m: usize,
}

impl HyperLogLog {
    /// A sketch with `2^precision` registers (precision in 4..=16).
    pub fn new(precision: u8) -> HyperLogLog {
        let precision = precision.clamp(4, 16);
        let m = 1usize << precision;
        HyperLogLog {
            registers: vec![0u8; m],
            precision,
            m,
        }
    }

    /// A sketch sized for a target relative standard error.
    pub fn for_error(target: f64) -> HyperLogLog {
        // std_err ~= 1.04 / sqrt(m) → m ~= (1.04/target)^2.
        let m = (1.04 / target.max(0.005)).powi(2);
        let mut p = 4u8;
        while (1usize << p) < m as usize && p < 16 {
            p += 1;
        }
        HyperLogLog::new(p)
    }

    /// Register count.
    pub fn register_count(&self) -> usize {
        self.m
    }

    /// Add a pre-hashed 64-bit value.
    pub fn add_hash(&mut self, hash: u64) {
        let idx = (hash >> (64 - self.precision)) as usize;
        let remaining = (hash << self.precision) | (1u64 << (self.precision - 1));
        let rank = remaining.leading_zeros() as u8 + 1;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Add an integer element.
    pub fn add_i64(&mut self, v: i64) {
        self.add_hash(mix64(v as u64));
    }

    /// Add a byte-string element.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        self.add_hash(hash_bytes(bytes));
    }

    /// Estimate the number of distinct elements added.
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let alpha = match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let mut sum = 0.0;
        let mut zeros = 0usize;
        for &r in &self.registers {
            sum += 2f64.powi(-(r as i32));
            if r == 0 {
                zeros += 1;
            }
        }
        let raw = alpha * m * m / sum;
        // Small-range correction (linear counting).
        if raw <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            raw
        }
    }

    /// Estimate rounded to the nearest integer.
    pub fn estimate_rounded(&self) -> u64 {
        self.estimate().round() as u64
    }

    /// Merge another sketch of the same precision into this one.
    pub fn merge(&mut self, other: &HyperLogLog) -> bool {
        if other.precision != self.precision {
            return false;
        }
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = (*a).max(*b);
        }
        true
    }
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF29CE484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    mix64(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_small_exact() {
        let mut hll = HyperLogLog::new(14);
        for i in 0..100i64 {
            hll.add_i64(i);
        }
        let est = hll.estimate();
        assert!((est - 100.0).abs() < 5.0, "estimate {est}");
    }

    #[test]
    fn estimates_large_within_error() {
        let mut hll = HyperLogLog::new(14);
        let n = 100_000i64;
        for i in 0..n {
            hll.add_i64(i.wrapping_mul(2654435761));
        }
        let est = hll.estimate();
        let err = (est - n as f64).abs() / n as f64;
        assert!(err < 0.05, "relative error {err} (est {est})");
    }

    #[test]
    fn duplicates_do_not_inflate() {
        let mut hll = HyperLogLog::new(12);
        for _ in 0..1000 {
            hll.add_i64(42);
        }
        assert!(hll.estimate() < 5.0);
    }

    #[test]
    fn merge_unions_cardinality() {
        let mut a = HyperLogLog::new(12);
        let mut b = HyperLogLog::new(12);
        for i in 0..5000i64 {
            a.add_i64(i);
        }
        for i in 4000..9000i64 {
            b.add_i64(i);
        }
        assert!(a.merge(&b));
        let est = a.estimate();
        // Union has 9000 distinct.
        let err = (est - 9000.0).abs() / 9000.0;
        assert!(err < 0.06, "relative error {err}");
    }

    #[test]
    fn for_error_sizes_registers() {
        let hll = HyperLogLog::for_error(0.01);
        assert!(hll.register_count() >= 4096);
    }

    #[test]
    fn byte_elements() {
        let mut hll = HyperLogLog::new(12);
        for i in 0..500 {
            hll.add_bytes(format!("item-{i}").as_bytes());
        }
        let est = hll.estimate();
        assert!((est - 500.0).abs() < 40.0, "estimate {est}");
    }
}
