//! Column statistics and histograms.
//!
//! The planner and the compactor both need summaries of a column's data: its
//! cardinality, its min/max, its null count, and a histogram of value
//! distribution. This module computes those summaries from a flat list of
//! values and supports both equi-width and equi-depth (quantile) histograms.

use crate::value::Value;
use std::cmp::Ordering;

/// Summary statistics for a column.
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    pub count: u64,
    pub null_count: u64,
    pub distinct: u64,
    pub min: Value,
    pub max: Value,
    pub sum: f64,
    pub avg: f64,
    pub variance: f64,
    pub stddev: f64,
}

impl ColumnStats {
    /// Compute stats from a flat list of values.
    pub fn compute(values: &[Value]) -> ColumnStats {
        let mut stats = ColumnStats::default();
        let mut distinct = std::collections::BTreeSet::new();
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut numeric_count = 0u64;
        let mut min = Value::Null;
        let mut max = Value::Null;
        for v in values {
            stats.count += 1;
            if v.is_null() {
                stats.null_count += 1;
                continue;
            }
            distinct.insert(*v);
            if let Some(x) = v.as_real() {
                sum += x;
                sum_sq += x * x;
                numeric_count += 1;
            }
            match min {
                Value::Null => min = *v,
                m if v.total_cmp(&m) == Ordering::Less => min = *v,
                _ => {}
            }
            match max {
                Value::Null => max = *v,
                m if v.total_cmp(&m) == Ordering::Greater => max = *v,
                _ => {}
            }
        }
        stats.distinct = distinct.len() as u64;
        stats.min = min;
        stats.max = max;
        stats.sum = sum;
        if numeric_count > 0 {
            stats.avg = sum / numeric_count as f64;
            stats.variance = (sum_sq / numeric_count as f64) - stats.avg * stats.avg;
            stats.stddev = stats.variance.max(0.0).sqrt();
        }
        stats
    }

    pub fn has_min_max(&self) -> bool {
        !self.min.is_null() && !self.max.is_null()
    }

    /// An estimate of the selectivity of an equality predicate against this
    /// column: `1 / distinct` if distinct is known, else a default.
    pub fn eq_selectivity(&self) -> f64 {
        if self.distinct > 0 {
            1.0 / self.distinct as f64
        } else {
            0.1
        }
    }

    /// An estimate of the selectivity of a range predicate `[lo, hi]`.
    pub fn range_selectivity(&self, lo: &Value, hi: &Value) -> f64 {
        if !self.has_min_max() {
            return 0.3;
        }
        let total = value_span(&self.min, &self.max);
        if total <= 0.0 {
            return 1.0;
        }
        let lo = if self.min.total_cmp(lo) == Ordering::Greater { self.min } else { *lo };
        let hi = if self.max.total_cmp(hi) == Ordering::Less { self.max } else { *hi };
        let span = value_span(&lo, &hi).max(0.0);
        (span / total).clamp(0.0, 1.0)
    }
}

fn value_span(lo: &Value, hi: &Value) -> f64 {
    match (lo.as_real(), hi.as_real()) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    }
}

/// A histogram bucket.
#[derive(Debug, Clone, Copy)]
pub struct Bucket {
    pub lower: Value,
    pub upper: Value,
    pub count: u64,
    pub distinct: u64,
}

/// An equi-width histogram: the value range is divided into `num_buckets`
/// equal-width intervals.
#[derive(Debug, Clone)]
pub struct EquiWidthHistogram {
    pub buckets: Vec<Bucket>,
}

impl EquiWidthHistogram {
    pub fn build(values: &[Value], num_buckets: usize) -> EquiWidthHistogram {
        let stats = ColumnStats::compute(values);
        if !stats.has_min_max() || num_buckets == 0 {
            return EquiWidthHistogram { buckets: Vec::new() };
        }
        let lo = stats.min.as_real().unwrap_or(0.0);
        let hi = stats.max.as_real().unwrap_or(0.0);
        let width = (hi - lo) / num_buckets as f64;
        if width <= 0.0 {
            return EquiWidthHistogram {
                buckets: vec![Bucket {
                    lower: stats.min,
                    upper: stats.max,
                    count: stats.count,
                    distinct: stats.distinct,
                }],
            };
        }
        let mut counts = vec![0u64; num_buckets];
        let mut distinct: Vec<std::collections::BTreeSet<Value>> =
            (0..num_buckets).map(|_| std::collections::BTreeSet::new()).collect();
        for v in values {
            if v.is_null() {
                continue;
            }
            if let Some(x) = v.as_real() {
                let mut idx = ((x - lo) / width).floor() as isize;
                if idx < 0 {
                    idx = 0;
                }
                if idx >= num_buckets as isize {
                    idx = num_buckets as isize - 1;
                }
                counts[idx as usize] += 1;
                distinct[idx as usize].insert(*v);
            }
        }
        let mut buckets = Vec::with_capacity(num_buckets);
        for i in 0..num_buckets {
            let lower = Value::Real(lo + width * i as f64);
            let upper = Value::Real(lo + width * (i as f64 + 1.0));
            buckets.push(Bucket {
                lower,
                upper,
                count: counts[i],
                distinct: distinct[i].len() as u64,
            });
        }
        EquiWidthHistogram { buckets }
    }

    /// Estimate the number of rows matching a range `[lo, hi]`.
    pub fn estimate_range(&self, lo: &Value, hi: &Value) -> u64 {
        let lo = lo.as_real().unwrap_or(f64::NEG_INFINITY);
        let hi = hi.as_real().unwrap_or(f64::INFINITY);
        let mut total = 0u64;
        for b in &self.buckets {
            let blow = b.lower.as_real().unwrap_or(f64::NEG_INFINITY);
            let bhigh = b.upper.as_real().unwrap_or(f64::INFINITY);
            if bhigh < lo || blow > hi {
                continue;
            }
            total += b.count;
        }
        total
    }
}

/// An equi-depth (quantile) histogram: each bucket holds roughly the same
/// number of rows, with variable-width boundaries.
#[derive(Debug, Clone)]
pub struct EquiDepthHistogram {
    pub buckets: Vec<Bucket>,
}

impl EquiDepthHistogram {
    pub fn build(values: &[Value], num_buckets: usize) -> EquiDepthHistogram {
        let mut sorted: Vec<Value> = values.iter().copied().filter(|v| !v.is_null()).collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        if n == 0 || num_buckets == 0 {
            return EquiDepthHistogram { buckets: Vec::new() };
        }
        let per = (n + num_buckets - 1) / num_buckets;
        let mut buckets = Vec::with_capacity(num_buckets);
        let mut i = 0;
        while i < n {
            let end = (i + per).min(n);
            let lower = sorted[i];
            let upper = sorted[end - 1];
            let distinct = {
                let set: std::collections::BTreeSet<Value> = sorted[i..end].iter().copied().collect();
                set.len() as u64
            };
            buckets.push(Bucket {
                lower,
                upper,
                count: (end - i) as u64,
                distinct,
            });
            i = end;
        }
        EquiDepthHistogram { buckets }
    }

    /// Estimate the number of rows whose value equals `v`.
    pub fn estimate_eq(&self, v: &Value) -> u64 {
        for b in &self.buckets {
            if v.total_cmp(&b.lower) != Ordering::Less
                && v.total_cmp(&b.upper) != Ordering::Greater
            {
                if b.distinct > 0 {
                    return b.count / b.distinct;
                }
                return b.count;
            }
        }
        0
    }
}

/// Most frequent values (the top-K by frequency).
pub fn most_frequent(values: &[Value], k: usize) -> Vec<(Value, u64)> {
    let mut counts: std::collections::BTreeMap<Value, u64> = std::collections::BTreeMap::new();
    for v in values {
        if v.is_null() {
            continue;
        }
        *counts.entry(*v).or_insert(0) += 1;
    }
    let mut vec: Vec<(Value, u64)> = counts.into_iter().collect();
    vec.sort_by(|a, b| b.1.cmp(&a.1));
    vec.truncate(k);
    vec
}

/// Build a full statistics summary for a column.
#[derive(Debug, Clone)]
pub struct ColumnSummary {
    pub stats: ColumnStats,
    pub equi_width: EquiWidthHistogram,
    pub equi_depth: EquiDepthHistogram,
    pub mfv: Vec<(Value, u64)>,
}

impl ColumnSummary {
    pub fn build(values: &[Value], num_buckets: usize, mfv_k: usize) -> ColumnSummary {
        ColumnSummary {
            stats: ColumnStats::compute(values),
            equi_width: EquiWidthHistogram::build(values, num_buckets),
            equi_depth: EquiDepthHistogram::build(values, num_buckets),
            mfv: most_frequent(values, mfv_k),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_stats_basic() {
        let vals = vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Null, Value::Int(3)];
        let s = ColumnStats::compute(&vals);
        assert_eq!(s.count, 5);
        assert_eq!(s.null_count, 1);
        assert_eq!(s.distinct, 3);
        assert_eq!(s.min, Value::Int(1));
        assert_eq!(s.max, Value::Int(3));
        assert_eq!(s.sum, 9.0);
        assert_eq!(s.avg, 2.25);
    }

    #[test]
    fn equi_width_histogram() {
        let vals: Vec<Value> = (0..100).map(|i| Value::Int(i)).collect();
        let h = EquiWidthHistogram::build(&vals, 10);
        assert_eq!(h.buckets.len(), 10);
        let total: u64 = h.buckets.iter().map(|b| b.count).sum();
        assert_eq!(total, 100);
        assert_eq!(h.estimate_range(&Value::Int(0), &Value::Int(50)), 60);
    }

    #[test]
    fn equi_depth_histogram_balanced() {
        let vals: Vec<Value> = (0..100).map(|i| Value::Int(i)).collect();
        let h = EquiDepthHistogram::build(&vals, 4);
        assert_eq!(h.buckets.len(), 4);
        for b in &h.buckets {
            assert!(b.count == 25);
        }
    }

    #[test]
    fn most_frequent_top_k() {
        let vals = vec![
            Value::Int(1), Value::Int(1), Value::Int(1),
            Value::Int(2), Value::Int(2),
            Value::Int(3),
        ];
        let mfv = most_frequent(&vals, 2);
        assert_eq!(mfv[0], (Value::Int(1), 3));
        assert_eq!(mfv[1], (Value::Int(2), 2));
    }

    #[test]
    fn range_selectivity() {
        let vals: Vec<Value> = (0..100).map(|i| Value::Int(i)).collect();
        let s = ColumnStats::compute(&vals);
        let sel = s.range_selectivity(&Value::Int(25), &Value::Int(75));
        assert!(sel > 0.4 && sel < 0.6);
    }
}
