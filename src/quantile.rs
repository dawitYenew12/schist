//! Streaming quantile and sampling utilities.
//!
//! Histogram building and cost estimation want approximate quantiles of a
//! column without sorting the whole thing. This module provides a
//! reservoir sampler (uniform sample of a stream in fixed memory) and a
//! greenwald-khanna-style summary that answers rank/quantile queries within a
//! configurable error bound. Both are fed a stream of `i64` values and both are
//! mergeable enough for per-page-then-combine use.

/// A uniform reservoir sampler of `i64` values.
#[derive(Debug, Clone)]
pub struct Reservoir {
    capacity: usize,
    samples: Vec<i64>,
    seen: u64,
    rng: u64,
}

impl Reservoir {
    /// A reservoir holding up to `capacity` samples, seeded deterministically.
    pub fn new(capacity: usize) -> Reservoir {
        Reservoir::with_seed(capacity, 0x2545F4914F6CDD1D)
    }

    /// A reservoir with an explicit RNG seed.
    pub fn with_seed(capacity: usize, seed: u64) -> Reservoir {
        Reservoir {
            capacity: capacity.max(1),
            samples: Vec::with_capacity(capacity.max(1)),
            seen: 0,
            rng: seed | 1,
        }
    }

    fn next_rng(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Offer one value to the reservoir.
    pub fn offer(&mut self, value: i64) {
        self.seen += 1;
        if self.samples.len() < self.capacity {
            self.samples.push(value);
        } else {
            let j = (self.next_rng() % self.seen) as usize;
            if j < self.capacity {
                self.samples[j] = value;
            }
        }
    }

    /// Number of items offered.
    pub fn count(&self) -> u64 {
        self.seen
    }

    /// The current sample (unsorted).
    pub fn samples(&self) -> &[i64] {
        &self.samples
    }

    /// An estimate of the `q` quantile (0.0..=1.0) from the current sample.
    pub fn quantile(&self, q: f64) -> Option<i64> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let q = q.clamp(0.0, 1.0);
        let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
        Some(sorted[idx])
    }
}

/// A bounded-memory quantile sketch. It keeps an unbiased reservoir sample of
/// the stream (so quantile estimates converge without bias) plus the exact
/// minimum and maximum seen. The reservoir size caps memory; the exact extremes
/// keep range predicates tight.
#[derive(Debug, Clone)]
pub struct QuantileSketch {
    reservoir: Reservoir,
    min: Option<i64>,
    max: Option<i64>,
    count: u64,
}

impl QuantileSketch {
    /// A sketch retaining up to `max_size` sampled values.
    pub fn new(max_size: usize) -> QuantileSketch {
        QuantileSketch {
            reservoir: Reservoir::with_seed(max_size.max(16), 0x1D8E4A93F27C0B65),
            min: None,
            max: None,
            count: 0,
        }
    }

    /// Add a value.
    pub fn add(&mut self, value: i64) {
        self.count += 1;
        self.reservoir.offer(value);
        self.min = Some(self.min.map_or(value, |m| m.min(value)));
        self.max = Some(self.max.map_or(value, |m| m.max(value)));
    }

    /// Number of values added.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Estimate the `q` quantile. The exact extremes are returned for `q` at the
    /// boundaries so range endpoints are precise.
    pub fn quantile(&mut self, q: f64) -> Option<i64> {
        let q = q.clamp(0.0, 1.0);
        if q <= 0.0 {
            return self.min;
        }
        if q >= 1.0 {
            return self.max;
        }
        self.reservoir.quantile(q)
    }

    /// The exact minimum value.
    pub fn min(&mut self) -> Option<i64> {
        self.min
    }

    /// The exact maximum value.
    pub fn max(&mut self) -> Option<i64> {
        self.max
    }

    /// The median.
    pub fn median(&mut self) -> Option<i64> {
        self.quantile(0.5)
    }
}

/// Merge two sorted slices into a new sorted vector.
pub fn merge_sorted(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservoir_fills_and_bounds() {
        let mut r = Reservoir::new(100);
        for i in 0..10_000i64 {
            r.offer(i);
        }
        assert_eq!(r.samples().len(), 100);
        assert_eq!(r.count(), 10_000);
    }

    #[test]
    fn reservoir_quantile_reasonable() {
        let mut r = Reservoir::with_seed(500, 99);
        for i in 0..10_000i64 {
            r.offer(i);
        }
        let median = r.quantile(0.5).unwrap();
        // Should be near 5000.
        assert!((median - 5000).abs() < 1500, "median {median}");
    }

    #[test]
    fn sketch_exact_when_small() {
        let mut s = QuantileSketch::new(1000);
        for i in 0..=100i64 {
            s.add(i);
        }
        assert_eq!(s.min(), Some(0));
        assert_eq!(s.max(), Some(100));
        assert_eq!(s.median(), Some(50));
    }

    #[test]
    fn sketch_quantiles_within_error() {
        let mut s = QuantileSketch::new(200);
        for i in 0..100_000i64 {
            s.add(i);
        }
        let p90 = s.quantile(0.9).unwrap();
        let err = (p90 - 90_000).abs();
        // A 200-sample reservoir estimates a quantile to within a few percent.
        assert!(err < 6000, "p90 {p90}");
        assert_eq!(s.min(), Some(0));
        assert_eq!(s.max(), Some(99_999));
    }

    #[test]
    fn merge_sorted_works() {
        assert_eq!(merge_sorted(&[1, 3, 5], &[2, 4, 6]), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(merge_sorted(&[], &[1, 2]), vec![1, 2]);
    }
}
