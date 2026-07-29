//! Join algorithms: hash join, nested-loop join, sort-merge join, and the
//! semi/anti variants.
//!
//! Each algorithm takes two streams of rows (the *build* and *probe* sides)
//! plus an equality predicate on one column from each, and produces the joined
//! output. The caller picks the algorithm based on whether the inputs are
//! sorted, whether one side fits in memory, and whether an index is available.

use crate::sort::{merge_sorted, sort_rows, SortKey};
use crate::value::Value;
use std::cmp::Ordering;
use std::collections::HashMap;

/// The kind of join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    /// Inner: keep matching pairs.
    Inner,
    /// Left outer: keep every left row, padding the right with nulls.
    LeftOuter,
    /// Right outer: keep every right row, padding the left with nulls.
    RightOuter,
    /// Full outer.
    FullOuter,
    /// Semi: keep left rows that have at least one match (no right columns).
    Semi,
    /// Anti: keep left rows with no match.
    Anti,
}

/// A joined row: left columns followed by right columns (for non-semi/anti
/// joins).
pub type Joined = Vec<Value>;

/// Hash join on `left[left_col] == right[right_col]`. The left side is the
/// build side (hashed); the right side is probed.
pub fn hash_join(
    left: &[Vec<Value>],
    right: &[Vec<Value>],
    left_col: usize,
    right_col: usize,
    kind: JoinKind,
) -> Vec<Joined> {
    let mut build: HashMap<JoinKey, Vec<usize>> = HashMap::new();
    for (i, row) in left.iter().enumerate() {
        let key = JoinKey::from(row.get(left_col).copied().unwrap_or(Value::Null));
        build.entry(key).or_default().push(i);
    }
    let mut out = Vec::new();
    let mut matched_left = vec![false; left.len()];
    for rrow in right {
        let key = JoinKey::from(rrow.get(right_col).copied().unwrap_or(Value::Null));
        if let Some(indices) = build.get(&key) {
            for &i in indices {
                matched_left[i] = true;
                emit_join(&mut out, &left[i], rrow, kind);
            }
        } else if matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter) {
            let null_left = vec![Value::Null; left.first().map_or(0, |r| r.len())];
            emit_join(&mut out, &null_left, rrow, kind);
        }
    }
    if matches!(kind, JoinKind::LeftOuter | JoinKind::FullOuter) {
        let null_right = vec![Value::Null; right.first().map_or(0, |r| r.len())];
        for (i, lrow) in left.iter().enumerate() {
            if !matched_left[i] {
                emit_join(&mut out, lrow, &null_right, kind);
            }
        }
    }
    out
}

fn emit_join(out: &mut Vec<Joined>, left: &[Value], right: &[Value], kind: JoinKind) {
    match kind {
        JoinKind::Semi | JoinKind::Anti => out.push(left.to_vec()),
        _ => {
            let mut row = Vec::with_capacity(left.len() + right.len());
            row.extend_from_slice(left);
            row.extend_from_slice(right);
            out.push(row);
        }
    }
}

/// Nested-loop join: works for any predicate given as a comparator, but is
/// O(n*m). Used for small inputs or non-equi joins.
pub fn nested_loop_join(
    left: &[Vec<Value>],
    right: &[Vec<Value>],
    left_col: usize,
    right_col: usize,
    op: crate::value::CmpOp,
    kind: JoinKind,
) -> Vec<Joined> {
    let mut out = Vec::new();
    let mut matched_left = vec![false; left.len()];
    for (i, lrow) in left.iter().enumerate() {
        let lv = lrow.get(left_col).copied().unwrap_or(Value::Null);
        let mut any = false;
        for rrow in right {
            let rv = rrow.get(right_col).copied().unwrap_or(Value::Null);
            if op.apply(&lv, &rv) {
                any = true;
                matched_left[i] = true;
                // Semi/anti are resolved after the loop based on `any`.
                if !matches!(kind, JoinKind::Semi | JoinKind::Anti) {
                    emit_join(&mut out, lrow, rrow, kind);
                }
            }
        }
        if !any && matches!(kind, JoinKind::LeftOuter | JoinKind::FullOuter) {
            let null_right = vec![Value::Null; right.first().map_or(0, |r| r.len())];
            emit_join(&mut out, lrow, &null_right, kind);
        }
    }
    // Resolve semi/anti after the full scan.
    if matches!(kind, JoinKind::Semi) {
        for (i, lrow) in left.iter().enumerate() {
            if matched_left[i] {
                emit_join(&mut out, lrow, &[], kind);
            }
        }
    } else if matches!(kind, JoinKind::Anti) {
        for (i, lrow) in left.iter().enumerate() {
            if !matched_left[i] {
                emit_join(&mut out, lrow, &[], kind);
            }
        }
    }
    out
}

/// Sort-merge join on an equality predicate. Both inputs must be sorted on the
/// join columns (this function sorts copies if needed).
pub fn sort_merge_join(
    left: Vec<Vec<Value>>,
    right: Vec<Vec<Value>>,
    left_col: usize,
    right_col: usize,
    kind: JoinKind,
) -> Vec<Joined> {
    let mut left = left;
    let mut right = right;
    sort_rows(&mut left, &[SortKey::asc(left_col)]);
    sort_rows(&mut right, &[SortKey::asc(right_col)]);
    let mut out = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let lv = left[i].get(left_col).copied().unwrap_or(Value::Null);
        let rv = right[j].get(right_col).copied().unwrap_or(Value::Null);
        match lv.total_cmp(&rv) {
            Ordering::Less => {
                if matches!(kind, JoinKind::LeftOuter | JoinKind::FullOuter) {
                    let null_right = vec![Value::Null; right.first().map_or(0, |r| r.len())];
                    emit_join(&mut out, &left[i], &null_right, kind);
                }
                i += 1;
            }
            Ordering::Greater => {
                if matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter) {
                    let null_left = vec![Value::Null; left.first().map_or(0, |r| r.len())];
                    emit_join(&mut out, &null_left, &right[j], kind);
                }
                j += 1;
            }
            Ordering::Equal => {
                // Collect the run of equal keys on both sides.
                let lstart = i;
                while i < left.len() && left[i].get(left_col).copied().unwrap_or(Value::Null) == lv {
                    i += 1;
                }
                let rstart = j;
                while j < right.len() && right[j].get(right_col).copied().unwrap_or(Value::Null) == rv {
                    j += 1;
                }
                for a in lstart..i {
                    for b in rstart..j {
                        emit_join(&mut out, &left[a], &right[b], kind);
                    }
                }
            }
        }
    }
    while i < left.len() {
        if matches!(kind, JoinKind::LeftOuter | JoinKind::FullOuter) {
            let null_right = vec![Value::Null; right.first().map_or(0, |r| r.len())];
            emit_join(&mut out, &left[i], &null_right, kind);
        }
        i += 1;
    }
    while j < right.len() {
        if matches!(kind, JoinKind::RightOuter | JoinKind::FullOuter) {
            let null_left = vec![Value::Null; left.first().map_or(0, |r| r.len())];
            emit_join(&mut out, &null_left, &right[j], kind);
        }
        j += 1;
    }
    out
}

/// Cross (Cartesian) join: every left row with every right row.
pub fn cross_join(left: &[Vec<Value>], right: &[Vec<Value>]) -> Vec<Joined> {
    let mut out = Vec::with_capacity(left.len() * right.len());
    for l in left {
        for r in right {
            let mut row = Vec::with_capacity(l.len() + r.len());
            row.extend_from_slice(l);
            row.extend_from_slice(r);
            out.push(row);
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct JoinKey(Vec<u8>);

impl From<Value> for JoinKey {
    fn from(v: Value) -> JoinKey {
        let mut buf = Vec::new();
        match v {
            Value::Null => buf.push(0),
            Value::Bool(b) => {
                buf.push(1);
                buf.push(b as u8);
            }
            Value::Int(i) => {
                buf.push(2);
                buf.extend_from_slice(&i.to_le_bytes());
            }
            Value::Real(r) => {
                buf.push(3);
                buf.extend_from_slice(&r.to_le_bytes());
            }
            Value::Text(id) => {
                buf.push(4);
                buf.extend_from_slice(&id.to_le_bytes());
            }
        }
        JoinKey(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lr() -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
        let left = vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
        ];
        let right = vec![
            vec![Value::Int(1), Value::Int(100)],
            vec![Value::Int(1), Value::Int(101)],
            vec![Value::Int(3), Value::Int(300)],
        ];
        (left, right)
    }

    #[test]
    fn hash_join_inner() {
        let (l, r) = lr();
        let out = hash_join(&l, &r, 0, 0, JoinKind::Inner);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn hash_join_left_outer() {
        let (l, r) = lr();
        let out = hash_join(&l, &r, 0, 0, JoinKind::LeftOuter);
        assert_eq!(out.len(), 4); // 2 from id=1, 1 from id=3, 1 unmatched id=2
    }

    #[test]
    fn nested_loop_semi() {
        let (l, r) = lr();
        let out = nested_loop_join(&l, &r, 0, 0, crate::value::CmpOp::Eq, JoinKind::Semi);
        assert_eq!(out.len(), 2); // ids 1 and 3 match
    }

    #[test]
    fn nested_loop_anti() {
        let (l, r) = lr();
        let out = nested_loop_join(&l, &r, 0, 0, crate::value::CmpOp::Eq, JoinKind::Anti);
        assert_eq!(out.len(), 1); // id 2 has no match
        assert_eq!(out[0][0], Value::Int(2));
    }

    #[test]
    fn sort_merge_join_inner() {
        let (l, r) = lr();
        let out = sort_merge_join(l, r, 0, 0, JoinKind::Inner);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn cross_join_cardinality() {
        let l = vec![vec![Value::Int(1)], vec![Value::Int(2)]];
        let r = vec![vec![Value::Int(9)], vec![Value::Int(8)], vec![Value::Int(7)]];
        assert_eq!(cross_join(&l, &r).len(), 6);
    }
}
