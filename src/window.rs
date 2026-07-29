//! Window functions.
//!
//! Window functions compute a value per row over a "frame" of related rows:
//! running sums, row numbers, ranks, and offsets. The input is assumed sorted
//! by the partition + order keys (the caller sorts first); each function walks
//! the sorted rows and emits one output value per row.

use crate::value::Value;
use std::cmp::Ordering;

/// The window function to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunc {
    /// 1-based row number within the partition.
    RowNumber,
    /// Rank with gaps: ties share a rank, the next non-tie skips.
    Rank,
    /// Dense rank without gaps.
    DenseRank,
    /// `lag(col, offset)`: the value `offset` rows before.
    Lag(u32),
    /// `lead(col, offset)`: the value `offset` rows after.
    Lead(u32),
    /// Running sum of the column.
    RunningSum,
    /// Running count of non-null values.
    RunningCount,
    /// Running min.
    RunningMin,
    /// Running max.
    RunningMax,
    /// Running average (as a real).
    RunningAvg,
    /// First value in the partition.
    FirstValue,
    /// Last value in the partition.
    LastValue,
    /// Nth value in the partition (1-based).
    NthValue(u32),
    /// Percent rank within the partition (0.0 .. 1.0).
    PercentRank,
    /// Cume dist: cumulative distribution.
    CumeDist,
    /// NTile: divide the partition into `n` roughly equal groups.
    NTile(u32),
}

/// Compute a window function over a sorted partition, reading from `col`.
pub fn compute(func: WindowFunc, rows: &[Vec<Value>], col: usize) -> Vec<Value> {
    let n = rows.len();
    let mut out = Vec::with_capacity(n);
    match func {
        WindowFunc::RowNumber => {
            for i in 0..n {
                out.push(Value::Int((i + 1) as i64));
            }
        }
        WindowFunc::Rank => {
            let mut rank = 0i64;
            for i in 0..n {
                if i == 0 || rows[i].get(col) != rows[i - 1].get(col) {
                    rank = (i + 1) as i64;
                }
                out.push(Value::Int(rank));
            }
        }
        WindowFunc::DenseRank => {
            let mut rank = 0i64;
            for i in 0..n {
                if i == 0 || rows[i].get(col) != rows[i - 1].get(col) {
                    rank += 1;
                }
                out.push(Value::Int(rank));
            }
        }
        WindowFunc::Lag(off) => {
            let off = off as usize;
            for i in 0..n {
                if i >= off {
                    out.push(rows[i - off].get(col).copied().unwrap_or(Value::Null));
                } else {
                    out.push(Value::Null);
                }
            }
        }
        WindowFunc::Lead(off) => {
            let off = off as usize;
            for i in 0..n {
                if i + off < n {
                    out.push(rows[i + off].get(col).copied().unwrap_or(Value::Null));
                } else {
                    out.push(Value::Null);
                }
            }
        }
        WindowFunc::RunningSum => {
            let mut acc: i128 = 0;
            for r in rows {
                if let Some(i) = r.get(col).and_then(|v| v.as_int()) {
                    acc += i as i128;
                }
                out.push(Value::Int(acc as i64));
            }
        }
        WindowFunc::RunningCount => {
            let mut acc = 0i64;
            for r in rows {
                if r.get(col).map_or(false, |v| !v.is_null()) {
                    acc += 1;
                }
                out.push(Value::Int(acc));
            }
        }
        WindowFunc::RunningMin => {
            let mut cur = Value::Null;
            for r in rows {
                let v = r.get(col).copied().unwrap_or(Value::Null);
                if !v.is_null() {
                    cur = match cur {
                        Value::Null => v,
                        m if v.total_cmp(&m) == Ordering::Less => v,
                        m => m,
                    };
                }
                out.push(cur);
            }
        }
        WindowFunc::RunningMax => {
            let mut cur = Value::Null;
            for r in rows {
                let v = r.get(col).copied().unwrap_or(Value::Null);
                if !v.is_null() {
                    cur = match cur {
                        Value::Null => v,
                        m if v.total_cmp(&m) == Ordering::Greater => v,
                        m => m,
                    };
                }
                out.push(cur);
            }
        }
        WindowFunc::RunningAvg => {
            let mut sum: f64 = 0.0;
            let mut count = 0u64;
            for r in rows {
                if let Some(x) = r.get(col).and_then(|v| v.as_real()) {
                    sum += x;
                    count += 1;
                }
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                out.push(Value::Real(avg));
            }
        }
        WindowFunc::FirstValue => {
            let first = rows.first().and_then(|r| r.get(col)).copied().unwrap_or(Value::Null);
            for _ in 0..n {
                out.push(first);
            }
        }
        WindowFunc::LastValue => {
            let last = rows.last().and_then(|r| r.get(col)).copied().unwrap_or(Value::Null);
            for _ in 0..n {
                out.push(last);
            }
        }
        WindowFunc::NthValue(k) => {
            let v = rows
                .get(k as usize - 1)
                .and_then(|r| r.get(col))
                .copied()
                .unwrap_or(Value::Null);
            for _ in 0..n {
                out.push(v);
            }
        }
        WindowFunc::PercentRank => {
            let denom = (n.saturating_sub(1)) as f64;
            let mut rank = 0i64;
            for i in 0..n {
                if i == 0 || rows[i].get(col) != rows[i - 1].get(col) {
                    rank = i as i64;
                }
                let pr = if denom > 0.0 { rank as f64 / denom } else { 0.0 };
                out.push(Value::Real(pr));
            }
        }
        WindowFunc::CumeDist => {
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && rows[j].get(col) == rows[i].get(col) {
                    j += 1;
                }
                let dist = j as f64 / n as f64;
                for _ in i..j {
                    out.push(Value::Real(dist));
                }
                i = j;
            }
        }
        WindowFunc::NTile(tiles) => {
            let tiles = tiles.max(1) as usize;
            let base = n / tiles;
            let extra = n % tiles;
            let mut idx = 0;
            for t in 0..tiles {
                let size = base + if t < extra { 1 } else { 0 };
                for _ in 0..size {
                    out.push(Value::Int((t + 1) as i64));
                    idx += 1;
                }
            }
            let _ = idx;
        }
    }
    out
}

/// Partition a sorted input into `(start, end)` ranges by the partition key
/// column.
pub fn partitions(rows: &[Vec<Value>], part_col: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if rows.is_empty() {
        return out;
    }
    let mut start = 0;
    for i in 1..rows.len() {
        if rows[i].get(part_col) != rows[start].get(part_col) {
            out.push((start, i));
            start = i;
        }
    }
    out.push((start, rows.len()));
    out
}

/// Compute a window function over each partition of a sorted input, returning
/// the concatenated per-row outputs.
pub fn compute_partitioned(
    func: WindowFunc,
    rows: &[Vec<Value>],
    part_col: usize,
    val_col: usize,
) -> Vec<Value> {
    let parts = partitions(rows, part_col);
    let mut out = Vec::with_capacity(rows.len());
    for (s, e) in parts {
        let slice = &rows[s..e];
        out.extend(compute(func, slice, val_col));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<Vec<Value>> {
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(1), Value::Int(20)],
            vec![Value::Int(2), Value::Int(5)],
            vec![Value::Int(2), Value::Int(5)],
        ]
    }

    #[test]
    fn row_number_per_partition() {
        let r = rows();
        let out = compute_partitioned(WindowFunc::RowNumber, &r, 0, 1);
        let nums: Vec<i64> = out.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(nums, vec![1, 2, 3, 1, 2]);
    }

    #[test]
    fn rank_with_gaps() {
        let r = rows();
        let out = compute_partitioned(WindowFunc::Rank, &r, 0, 1);
        let nums: Vec<i64> = out.iter().map(|v| v.as_int().unwrap()).collect();
        // partition 1: 10,10,20 -> ranks 1,1,3 ; partition 2: 5,5 -> 1,1
        assert_eq!(nums, vec![1, 1, 3, 1, 1]);
    }

    #[test]
    fn dense_rank_no_gaps() {
        let r = rows();
        let out = compute_partitioned(WindowFunc::DenseRank, &r, 0, 1);
        let nums: Vec<i64> = out.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(nums, vec![1, 1, 2, 1, 1]);
    }

    #[test]
    fn running_sum() {
        let r = rows();
        let out = compute_partitioned(WindowFunc::RunningSum, &r, 0, 1);
        let nums: Vec<i64> = out.iter().map(|v| v.as_int().unwrap()).collect();
        assert_eq!(nums, vec![10, 20, 40, 5, 10]);
    }

    #[test]
    fn lag_lead() {
        let r = vec![
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
        ];
        let lag = compute(WindowFunc::Lag(1), &r, 0);
        assert_eq!(lag, vec![Value::Null, Value::Int(1), Value::Int(2)]);
        let lead = compute(WindowFunc::Lead(1), &r, 0);
        assert_eq!(lead, vec![Value::Int(2), Value::Int(3), Value::Null]);
    }

    #[test]
    fn ntile_even() {
        let r: Vec<Vec<Value>> = (0..10).map(|i| vec![Value::Int(i)]).collect();
        let out = compute(WindowFunc::NTile(3), &r, 0);
        let groups: Vec<i64> = out.iter().map(|v| v.as_int().unwrap()).collect();
        // 10 rows into 3 tiles: 4,3,3
        assert_eq!(groups.iter().filter(|&&g| g == 1).count(), 4);
        assert_eq!(groups.iter().filter(|&&g| g == 2).count(), 3);
        assert_eq!(groups.iter().filter(|&&g| g == 3).count(), 3);
    }

    #[test]
    fn first_last_value() {
        let r = rows();
        let first = compute_partitioned(WindowFunc::FirstValue, &r, 0, 1);
        let last = compute_partitioned(WindowFunc::LastValue, &r, 0, 1);
        assert_eq!(first[0], Value::Int(10));
        assert_eq!(last[2], Value::Int(20));
        assert_eq!(first[3], Value::Int(5));
        assert_eq!(last[4], Value::Int(5));
    }
}
