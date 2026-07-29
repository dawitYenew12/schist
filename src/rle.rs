//! Run-length encoding helpers.
//!
//! An RLE column page stores a sequence of runs. Each run is a `(value, count)`
//! pair: the value repeats `count` times consecutively. The on-disk layout of
//! a run depends on the column kind:
//!
//! - `Int`/`Real`: 8-byte value, 4-byte count.
//! - `Text`: 4-byte dictionary id, 4-byte count.
//! - `Bool`: 1-byte value, 4-byte count.
//!
//! This module only deals with run arithmetic and (de)serialization; it knows
//! nothing about pages or slots.

use crate::error::{DecodeError, Result};
use crate::value::Value;

/// An in-memory run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Run {
    pub value: Value,
    pub count: u32,
}

impl Run {
    pub fn new(value: Value, count: u32) -> Self {
        Run { value, count }
    }
}

/// The fixed byte width of one run for a given value kind name.
pub fn run_width(kind: &str) -> usize {
    match kind {
        "bool" => 1 + 4,
        "int" | "real" => 8 + 4,
        "text" => 4 + 4,
        _ => 8 + 4,
    }
}

/// Encode a slice of runs into bytes.
pub fn encode_runs(runs: &[Run]) -> Vec<u8> {
    let mut out = Vec::with_capacity(runs.len() * 12);
    for r in runs {
        match r.value {
            Value::Bool(b) => {
                out.push(b as u8);
                out.extend_from_slice(&r.count.to_le_bytes());
            }
            Value::Int(i) => {
                out.extend_from_slice(&i.to_le_bytes());
                out.extend_from_slice(&r.count.to_le_bytes());
            }
            Value::Real(x) => {
                out.extend_from_slice(&x.to_le_bytes());
                out.extend_from_slice(&r.count.to_le_bytes());
            }
            Value::Text(id) => {
                out.extend_from_slice(&id.to_le_bytes());
                out.extend_from_slice(&r.count.to_le_bytes());
            }
            Value::Null => {
                out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&r.count.to_le_bytes());
            }
        }
    }
    out
}

/// Decode runs from bytes for a given column kind.
pub fn decode_runs(buf: &[u8], kind: &str) -> Result<Vec<Run>> {
    let w = run_width(kind);
    if buf.len() % w != 0 {
        return Err(DecodeError::BadEncoding.into());
    }
    let mut runs = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let count = u32::from_le_bytes(buf[pos + w - 4..pos + w].try_into().unwrap());
        let value = match kind {
            "bool" => Value::Bool(buf[pos] != 0),
            "int" => Value::Int(i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap())),
            "real" => Value::Real(f64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap())),
            "text" => Value::Text(u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap())),
            _ => Value::Null,
        };
        runs.push(Run::new(value, count));
        pos += w;
    }
    Ok(runs)
}

/// The total number of rows covered by a run list.
pub fn run_total(runs: &[Run]) -> u64 {
    runs.iter().map(|r| r.count as u64).sum()
}

/// Merge adjacent runs with equal values into a single run.
pub fn coalesce(runs: &[Run]) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::new();
    for r in runs {
        if let Some(last) = out.last_mut() {
            if last.value == r.value {
                last.count = last.count.saturating_add(r.count);
                continue;
            }
        }
        out.push(*r);
    }
    out
}

/// Materialize a run list into a flat vector of values. Used by scans that do
/// not care about the run structure.
pub fn materialize(runs: &[Run]) -> Vec<Value> {
    let total = run_total(runs) as usize;
    let mut out = Vec::with_capacity(total);
    for r in runs {
        for _ in 0..r.count {
            out.push(r.value);
        }
    }
    out
}

/// Drop the first `n` rows from a run list, splitting runs as needed.
pub fn drop_first(runs: &[Run], n: u64) -> Vec<Run> {
    let mut out = Vec::new();
    let mut remaining = n;
    for r in runs {
        if remaining == 0 {
            out.push(*r);
            continue;
        }
        let take = remaining.min(r.count as u64) as u32;
        remaining -= take as u64;
        if r.count > take {
            out.push(Run::new(r.value, r.count - take));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_round_trip_int() {
        let runs = vec![
            Run::new(Value::Int(5), 3),
            Run::new(Value::Int(7), 2),
        ];
        let bytes = encode_runs(&runs);
        let back = decode_runs(&bytes, "int").unwrap();
        assert_eq!(back, runs);
        assert_eq!(run_total(&back), 5);
    }

    #[test]
    fn coalesce_merges_adjacent() {
        let runs = vec![
            Run::new(Value::Int(1), 2),
            Run::new(Value::Int(1), 3),
            Run::new(Value::Int(2), 1),
        ];
        let c = coalesce(&runs);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].count, 5);
    }

    #[test]
    fn drop_first_splits() {
        let runs = vec![Run::new(Value::Int(1), 3), Run::new(Value::Int(2), 2)];
        let d = drop_first(&runs, 2);
        assert_eq!(d, vec![Run::new(Value::Int(1), 1), Run::new(Value::Int(2), 2)]);
        assert_eq!(materialize(&d), vec![Value::Int(1), Value::Int(2), Value::Int(2)]);
    }
}
