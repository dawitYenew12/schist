//! Column profiling / `ANALYZE`.
//!
//! Before the cost model can estimate anything, someone has to gather the
//! per-column statistics it consults: row and null counts, min/max, an
//! approximate distinct count, and a few quantile boundaries for a histogram.
//! This module walks a typed [`Array`] once and produces a [`ColumnProfile`]
//! combining exact aggregates with the approximate sketches from
//! [`crate::hll`] and [`crate::quantile`]. Profiling a whole [`RecordBatch`]
//! produces one profile per column.

use crate::array::{Array, RecordBatch};
use crate::hll::HyperLogLog;
use crate::quantile::QuantileSketch;
use crate::schema::ColKind;
use crate::value::Value;

/// The gathered statistics for one column.
#[derive(Debug, Clone)]
pub struct ColumnProfile {
    pub kind: ColKind,
    pub row_count: u64,
    pub null_count: u64,
    pub distinct_estimate: u64,
    pub min: Option<Value>,
    pub max: Option<Value>,
    /// Equi-depth histogram boundaries (only for integer columns).
    pub histogram: Vec<i64>,
}

impl ColumnProfile {
    /// Fraction of values that are null, in `[0, 1]`.
    pub fn null_fraction(&self) -> f64 {
        if self.row_count == 0 {
            0.0
        } else {
            self.null_count as f64 / self.row_count as f64
        }
    }

    /// The average number of rows sharing a distinct value.
    pub fn avg_rows_per_value(&self) -> f64 {
        let non_null = self.row_count.saturating_sub(self.null_count);
        if self.distinct_estimate == 0 {
            0.0
        } else {
            non_null as f64 / self.distinct_estimate as f64
        }
    }

    /// Selectivity of an equality predicate on this column (`1/ndv`).
    pub fn equality_selectivity(&self) -> f64 {
        if self.distinct_estimate == 0 {
            1.0
        } else {
            1.0 / self.distinct_estimate as f64
        }
    }
}

/// Profile a single array.
pub fn profile_array(array: &Array) -> ColumnProfile {
    let kind = array.kind();
    let mut hll = HyperLogLog::new(12);
    let mut sketch = QuantileSketch::new(256);
    let mut min: Option<Value> = None;
    let mut max: Option<Value> = None;
    let mut null_count = 0u64;
    let row_count = array.len() as u64;

    for i in 0..array.len() {
        let v = array.value(i);
        if v.is_null() {
            null_count += 1;
            continue;
        }
        // Distinct estimate keyed off a stable hash of the value.
        hll.add_i64(value_hash_key(&v));
        // Min/max via the total order.
        min = Some(match min {
            Some(m) if m.total_cmp(&v).is_le() => m,
            _ => v,
        });
        max = Some(match max {
            Some(m) if m.total_cmp(&v).is_ge() => m,
            _ => v,
        });
        if let Some(iv) = v.as_int() {
            sketch.add(iv);
        }
    }

    let histogram = if kind == ColKind::Int && sketch.count() > 0 {
        build_histogram(&mut sketch, 8)
    } else {
        Vec::new()
    };

    ColumnProfile {
        kind,
        row_count,
        null_count,
        distinct_estimate: hll.estimate_rounded().min(row_count.saturating_sub(null_count)),
        min,
        max,
        histogram,
    }
}

/// Profile every column of a batch, returning `(name, profile)` pairs.
pub fn profile_batch(batch: &RecordBatch) -> Vec<(String, ColumnProfile)> {
    batch
        .names()
        .iter()
        .zip(batch.columns().iter())
        .map(|(name, col)| (name.clone(), profile_array(col)))
        .collect()
}

fn build_histogram(sketch: &mut QuantileSketch, buckets: usize) -> Vec<i64> {
    let mut bounds = Vec::with_capacity(buckets + 1);
    for b in 0..=buckets {
        let q = b as f64 / buckets as f64;
        if let Some(v) = sketch.quantile(q) {
            bounds.push(v);
        }
    }
    bounds.dedup();
    bounds
}

fn value_hash_key(v: &Value) -> i64 {
    match v {
        Value::Null => 0,
        Value::Bool(b) => *b as i64 + 1,
        Value::Int(i) => *i,
        Value::Real(r) => r.to_bits() as i64,
        Value::Text(id) => *id as i64 + 0x1_0000_0000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_integer_column() {
        let mut values = Vec::new();
        for i in 0..1000i64 {
            values.push(Value::Int(i % 100));
        }
        let arr = Array::from_values(ColKind::Int, &values);
        let p = profile_array(&arr);
        assert_eq!(p.row_count, 1000);
        assert_eq!(p.null_count, 0);
        assert_eq!(p.min, Some(Value::Int(0)));
        assert_eq!(p.max, Some(Value::Int(99)));
        // ~100 distinct.
        assert!((p.distinct_estimate as i64 - 100).abs() < 20, "ndv {}", p.distinct_estimate);
        assert!(!p.histogram.is_empty());
    }

    #[test]
    fn counts_nulls() {
        let values = vec![Value::Int(1), Value::Null, Value::Int(3), Value::Null];
        let arr = Array::from_values(ColKind::Int, &values);
        let p = profile_array(&arr);
        assert_eq!(p.null_count, 2);
        assert_eq!(p.null_fraction(), 0.5);
    }

    #[test]
    fn distinct_selectivity() {
        let values: Vec<Value> = (0..400).map(|i| Value::Int(i % 4)).collect();
        let arr = Array::from_values(ColKind::Int, &values);
        let p = profile_array(&arr);
        // 4 distinct → selectivity ~0.25, avg 100 rows/value.
        assert!((p.equality_selectivity() - 0.25).abs() < 0.1);
        assert!(p.avg_rows_per_value() > 50.0);
    }

    #[test]
    fn histogram_is_monotone() {
        let values: Vec<Value> = (0..1000).map(Value::Int).collect();
        let arr = Array::from_values(ColKind::Int, &values);
        let p = profile_array(&arr);
        for w in p.histogram.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn profiles_batch() {
        let a = Array::from_values(ColKind::Int, &[Value::Int(1), Value::Int(2)]);
        let b = Array::from_values(ColKind::Bool, &[Value::Bool(true), Value::Bool(false)]);
        let batch = RecordBatch::new(vec![("n".into(), a), ("b".into(), b)]);
        let profiles = profile_batch(&batch);
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].0, "n");
        assert_eq!(profiles[1].1.kind, ColKind::Bool);
    }
}
