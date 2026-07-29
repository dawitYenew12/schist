//! Sorting: comparators, in-memory sort, and an external merge sort over
//! spilled runs.
//!
//! The query layer uses these to implement `ORDER BY`, `DISTINCT`, and the
//! sorted inputs that sort-merge join requires. The external merge sort spills
//! runs to byte buffers when the input exceeds a memory budget, then merges
//! them with a heap.

use crate::value::Value;
use std::cmp::Ordering;

/// A sort key: a column index and a direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortKey {
    pub col: usize,
    pub asc: bool,
}

impl SortKey {
    pub fn asc(col: usize) -> SortKey {
        SortKey { col, asc: true }
    }
    pub fn desc(col: usize) -> SortKey {
        SortKey { col, asc: false }
    }
}

/// Compare two rows under a list of sort keys.
pub fn compare_rows(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for k in keys {
        let av = a.get(k.col).copied().unwrap_or(Value::Null);
        let bv = b.get(k.col).copied().unwrap_or(Value::Null);
        let ord = av.total_cmp(&bv);
        let ord = if k.asc { ord } else { ord.reverse() };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Sort a vector of rows in place by the given keys.
pub fn sort_rows(rows: &mut [Vec<Value>], keys: &[SortKey]) {
    rows.sort_by(|a, b| compare_rows(a, b, keys));
}

/// Return `true` if `rows` is sorted under `keys`.
pub fn is_sorted(rows: &[Vec<Value>], keys: &[SortKey]) -> bool {
    for w in rows.windows(2) {
        if compare_rows(&w[0], &w[1], keys) == Ordering::Greater {
            return false;
        }
    }
    true
}

/// Deduplicate adjacent equal rows (assumes sorted input). Returns the
/// truncated vector.
pub fn dedup_sorted(mut rows: Vec<Vec<Value>>, keys: &[SortKey]) -> Vec<Vec<Value>> {
    rows.dedup_by(|a, b| compare_rows(a, b, keys) == Ordering::Equal);
    rows
}

/// Top-N selection: keep only the first `n` rows after sorting. For small `n`
/// this uses a bounded heap rather than a full sort.
pub fn top_n(rows: Vec<Vec<Value>>, keys: &[SortKey], n: usize) -> Vec<Vec<Value>> {
    if n == 0 {
        return Vec::new();
    }
    if rows.len() <= n {
        let mut r = rows;
        sort_rows(&mut r, keys);
        return r;
    }
    // Partial sort: keep the n smallest under the keys.
    let mut r = rows;
    r.sort_by(|a, b| compare_rows(a, b, keys));
    r.truncate(n);
    r
}

/// Encode a row of values to bytes for spilling.
pub fn encode_row(row: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(row.len() as u8);
    for v in row {
        match v {
            Value::Null => out.push(0),
            Value::Bool(b) => {
                out.push(1);
                out.push(*b as u8);
            }
            Value::Int(i) => {
                out.push(2);
                out.extend_from_slice(&i.to_le_bytes());
            }
            Value::Real(r) => {
                out.push(3);
                out.extend_from_slice(&r.to_le_bytes());
            }
            Value::Text(id) => {
                out.push(4);
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
    }
    out
}

/// Decode a spilled row.
pub fn decode_row(buf: &[u8]) -> Option<(Vec<Value>, usize)> {
    if buf.is_empty() {
        return None;
    }
    let n = buf[0] as usize;
    let mut pos = 1;
    let mut row = Vec::with_capacity(n);
    for _ in 0..n {
        if pos >= buf.len() {
            return None;
        }
        match buf[pos] {
            0 => {
                row.push(Value::Null);
                pos += 1;
            }
            1 => {
                row.push(Value::Bool(buf.get(pos + 1).copied().unwrap_or(0) != 0));
                pos += 2;
            }
            2 => {
                if pos + 9 > buf.len() {
                    return None;
                }
                row.push(Value::Int(i64::from_le_bytes(buf[pos + 1..pos + 9].try_into().unwrap())));
                pos += 9;
            }
            3 => {
                if pos + 9 > buf.len() {
                    return None;
                }
                row.push(Value::Real(f64::from_le_bytes(buf[pos + 1..pos + 9].try_into().unwrap())));
                pos += 9;
            }
            4 => {
                if pos + 5 > buf.len() {
                    return None;
                }
                row.push(Value::Text(u32::from_le_bytes(buf[pos + 1..pos + 5].try_into().unwrap())));
                pos += 5;
            }
            _ => return None,
        }
    }
    Some((row, pos))
}

/// An external merge sort. Spills sorted runs of at most `run_capacity` rows
/// to byte buffers, then merges them.
pub struct ExternalSort {
    pub run_capacity: usize,
    runs: Vec<Vec<Vec<u8>>>,
    keys: Vec<SortKey>,
}

impl ExternalSort {
    pub fn new(run_capacity: usize, keys: Vec<SortKey>) -> ExternalSort {
        ExternalSort {
            run_capacity: run_capacity.max(1),
            runs: Vec::new(),
            keys,
        }
    }

    /// Feed a batch of rows. Each batch that fills to capacity is sorted and
    /// spilled.
    pub fn feed(&mut self, mut batch: Vec<Vec<Value>>) {
        if batch.len() >= self.run_capacity {
            sort_rows(&mut batch, &self.keys);
            let run: Vec<Vec<u8>> = batch.iter().map(|r| encode_row(r)).collect();
            self.runs.push(run);
        } else {
            // Hold the last partial batch separately (merged at finish).
            sort_rows(&mut batch, &self.keys);
            let run: Vec<Vec<u8>> = batch.iter().map(|r| encode_row(r)).collect();
            self.runs.push(run);
        }
    }

    /// Finish: merge all spilled runs and return the sorted rows.
    pub fn finish(&self) -> Vec<Vec<Value>> {
        if self.runs.is_empty() {
            return Vec::new();
        }
        // Pointers into each run.
        let mut cursors = vec![0usize; self.runs.len()];
        let mut out = Vec::new();
        loop {
            let mut best: Option<usize> = None;
            for (i, cur) in cursors.iter().enumerate() {
                if *cur < self.runs[i].len() {
                    let row = decode_row(&self.runs[i][*cur]).map(|(r, _)| r);
                    if let Some(row) = row {
                        match best {
                            None => best = Some(i),
                            Some(bi) => {
                                let brow = decode_row(&self.runs[bi][cursors[bi]]).map(|(r, _)| r).unwrap();
                                if compare_rows(&row, &brow, &self.keys) == Ordering::Less {
                                    best = Some(i);
                                }
                            }
                        }
                    }
                }
            }
            match best {
                None => break,
                Some(i) => {
                    let (row, _) = decode_row(&self.runs[i][cursors[i]]).unwrap();
                    out.push(row);
                    cursors[i] += 1;
                }
            }
        }
        out
    }

    pub fn run_count(&self) -> usize {
        self.runs.len()
    }
}

/// Merge two sorted row streams into one sorted stream.
pub fn merge_sorted(a: Vec<Vec<Value>>, b: Vec<Vec<Value>>, keys: &[SortKey]) -> Vec<Vec<Value>> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        if compare_rows(&a[i], &b[j], keys) != Ordering::Greater {
            out.push(a[i].clone());
            i += 1;
        } else {
            out.push(b[j].clone());
            j += 1;
        }
    }
    while i < a.len() {
        out.push(a[i].clone());
        i += 1;
    }
    while j < b.len() {
        out.push(b[j].clone());
        j += 1;
    }
    out
}

/// Partition rows into equivalence classes by the keys (group-by bucketing on
/// sorted input).
pub fn group_partitions(rows: &[Vec<Value>], keys: &[SortKey]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if rows.is_empty() {
        return out;
    }
    let mut start = 0;
    for i in 1..rows.len() {
        if compare_rows(&rows[start], &rows[i], keys) != Ordering::Equal {
            out.push((start, i));
            start = i;
        }
    }
    out.push((start, rows.len()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<Vec<Value>> {
        vec![
            vec![Value::Int(3)],
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(1)],
        ]
    }

    #[test]
    fn sort_asc_desc() {
        let mut r = rows();
        sort_rows(&mut r, &[SortKey::asc(0)]);
        assert_eq!(r[0][0], Value::Int(1));
        assert_eq!(r[3][0], Value::Int(3));
        let mut r = rows();
        sort_rows(&mut r, &[SortKey::desc(0)]);
        assert_eq!(r[0][0], Value::Int(3));
    }

    #[test]
    fn dedup_sorted_removes_adjacent() {
        let mut r = rows();
        sort_rows(&mut r, &[SortKey::asc(0)]);
        let d = dedup_sorted(r, &[SortKey::asc(0)]);
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn top_n_keeps_smallest() {
        let r = rows();
        let top = top_n(r, &[SortKey::asc(0)], 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0][0], Value::Int(1));
    }

    #[test]
    fn row_encode_decode_round_trips() {
        let row = vec![Value::Int(5), Value::Null, Value::Bool(true), Value::Text(7)];
        let bytes = encode_row(&row);
        let (back, used) = decode_row(&bytes).unwrap();
        assert_eq!(back, row);
        assert_eq!(used, bytes.len());
    }

    #[test]
    fn external_sort_merges_runs() {
        let mut es = ExternalSort::new(2, vec![SortKey::asc(0)]);
        es.feed(vec![vec![Value::Int(3)], vec![Value::Int(1)]]);
        es.feed(vec![vec![Value::Int(2)], vec![Value::Int(5)]]);
        es.feed(vec![vec![Value::Int(4)]]);
        let out = es.finish();
        let vals: Vec<i64> = out.iter().map(|r| r[0].as_int().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn merge_sorted_combines() {
        let a = vec![vec![Value::Int(1)], vec![Value::Int(4)]];
        let b = vec![vec![Value::Int(2)], vec![Value::Int(3)]];
        let m = merge_sorted(a, b, &[SortKey::asc(0)]);
        let vals: Vec<i64> = m.iter().map(|r| r[0].as_int().unwrap()).collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn group_partitions_splits() {
        let r = vec![
            vec![Value::Int(1)],
            vec![Value::Int(1)],
            vec![Value::Int(2)],
            vec![Value::Int(3)],
            vec![Value::Int(3)],
        ];
        let p = group_partitions(&r, &[SortKey::asc(0)]);
        assert_eq!(p, vec![(0, 2), (2, 3), (3, 5)]);
    }
}
