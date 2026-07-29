//! Streaming moments via Welford's algorithm.
//!
//! The `VAR`, `STDDEV`, `COVAR`, and `CORR` aggregates need running variance and
//! covariance that are numerically stable and computable in a single pass.
//! Welford's online algorithm updates the mean and the sum of squared deviations
//! incrementally without catastrophic cancellation, and two accumulators can be
//! merged (parallel/partial aggregation) with Chan's parallel formula. This
//! module provides a univariate accumulator (mean/variance/skewness/kurtosis)
//! and a bivariate one (covariance/correlation).

/// Running univariate moments.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl Moments {
    /// An empty accumulator.
    pub fn new() -> Moments {
        Moments::default()
    }

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// The running mean (0.0 for an empty accumulator).
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Add one observation.
    pub fn add(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// Population variance.
    pub fn variance_pop(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }

    /// Sample variance (n-1 denominator).
    pub fn variance_sample(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n as f64 - 1.0)
        }
    }

    /// Sample standard deviation.
    pub fn stddev_sample(&self) -> f64 {
        self.variance_sample().sqrt()
    }

    /// Population standard deviation.
    pub fn stddev_pop(&self) -> f64 {
        self.variance_pop().sqrt()
    }

    /// Excess skewness (0 for a symmetric distribution).
    pub fn skewness(&self) -> f64 {
        if self.n == 0 || self.m2 == 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        (n).sqrt() * self.m3 / self.m2.powf(1.5)
    }

    /// Excess kurtosis (0 for a normal distribution).
    pub fn kurtosis(&self) -> f64 {
        if self.n == 0 || self.m2 == 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        n * self.m4 / (self.m2 * self.m2) - 3.0
    }

    /// Merge another accumulator (Chan's parallel algorithm).
    pub fn merge(&mut self, other: &Moments) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        let na = self.n as f64;
        let nb = other.n as f64;
        let n = na + nb;
        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let mean = self.mean + delta * nb / n;
        let m2 = self.m2 + other.m2 + delta2 * na * nb / n;
        let m3 = self.m3
            + other.m3
            + delta3 * na * nb * (na - nb) / (n * n)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n;
        let m4 = self.m4
            + other.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n;

        self.n = n as u64;
        self.mean = mean;
        self.m2 = m2;
        self.m3 = m3;
        self.m4 = m4;
    }
}

/// Running bivariate moments for covariance and correlation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoMoments {
    n: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    c2: f64,
}

impl CoMoments {
    /// An empty accumulator.
    pub fn new() -> CoMoments {
        CoMoments::default()
    }

    /// Number of pairs.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Add a paired observation.
    pub fn add(&mut self, x: f64, y: f64) {
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / n;
        self.mean_y += dy / n;
        self.m2_x += dx * (x - self.mean_x);
        self.m2_y += dy * (y - self.mean_y);
        self.c2 += dx * (y - self.mean_y);
    }

    /// Sample covariance.
    pub fn covariance_sample(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.c2 / (self.n as f64 - 1.0)
        }
    }

    /// Population covariance.
    pub fn covariance_pop(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.c2 / self.n as f64
        }
    }

    /// Pearson correlation coefficient in `[-1, 1]`.
    pub fn correlation(&self) -> f64 {
        let denom = (self.m2_x * self.m2_y).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            self.c2 / denom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn mean_and_variance() {
        let mut m = Moments::new();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            m.add(x);
        }
        assert!(approx(m.mean(), 5.0, 1e-9));
        // Population variance of that classic sample is 4.0.
        assert!(approx(m.variance_pop(), 4.0, 1e-9));
        assert!(approx(m.stddev_pop(), 2.0, 1e-9));
    }

    #[test]
    fn merge_matches_single_pass() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64) * 0.37).collect();
        let mut whole = Moments::new();
        for &x in &data {
            whole.add(x);
        }
        let mut a = Moments::new();
        let mut b = Moments::new();
        for &x in &data[..40] {
            a.add(x);
        }
        for &x in &data[40..] {
            b.add(x);
        }
        a.merge(&b);
        assert!(approx(a.mean(), whole.mean(), 1e-9));
        assert!(approx(a.variance_sample(), whole.variance_sample(), 1e-6));
        assert!(approx(a.kurtosis(), whole.kurtosis(), 1e-6));
    }

    #[test]
    fn symmetric_has_zero_skew() {
        let mut m = Moments::new();
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            m.add(x);
        }
        assert!(approx(m.skewness(), 0.0, 1e-9));
    }

    #[test]
    fn covariance_and_correlation() {
        let mut c = CoMoments::new();
        // y = 2x exactly → correlation 1.
        for x in 0..10 {
            c.add(x as f64, 2.0 * x as f64);
        }
        assert!(approx(c.correlation(), 1.0, 1e-9));
        assert!(c.covariance_sample() > 0.0);
    }

    #[test]
    fn negative_correlation() {
        let mut c = CoMoments::new();
        for x in 0..10 {
            c.add(x as f64, -3.0 * x as f64 + 1.0);
        }
        assert!(approx(c.correlation(), -1.0, 1e-9));
    }

    #[test]
    fn empty_is_safe() {
        let m = Moments::new();
        assert_eq!(m.variance_pop(), 0.0);
        assert_eq!(m.skewness(), 0.0);
        let c = CoMoments::new();
        assert_eq!(c.correlation(), 0.0);
    }
}
