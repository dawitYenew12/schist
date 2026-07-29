//! Zone maps: per-page min/max summaries used by scan pruning.
//!
//! For every data page and every column, the zone map records the minimum and
//! maximum value present (ignoring nulls). A scan with a predicate on a column
//! can skip any page whose zone does not overlap the predicate range. Zone maps
//! are also the structure that references a page id from outside the data path
//! — the verifier checks that every zone-map entry points at a live data page.

use crate::value::Value;
use std::collections::BTreeMap;

/// One column's zone summary for one page.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zone {
    pub page: u32,
    pub min: Value,
    pub max: Value,
    pub live_rows: u32,
}

/// A zone map for a single column: `page_id -> Zone`.
#[derive(Debug, Clone, Default)]
pub struct ZoneMap {
    zones: BTreeMap<u32, Zone>,
}

impl ZoneMap {
    pub fn new() -> Self {
        ZoneMap::default()
    }

    pub fn set(&mut self, page: u32, zone: Zone) {
        self.zones.insert(page, zone);
    }

    pub fn get(&self, page: u32) -> Option<&Zone> {
        self.zones.get(&page)
    }

    pub fn remove(&mut self, page: u32) -> Option<Zone> {
        self.zones.remove(&page)
    }

    pub fn pages(&self) -> Vec<u32> {
        self.zones.keys().copied().collect()
    }

    pub fn len(&self) -> usize {
        self.zones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zones.is_empty()
    }

    /// Return the page ids whose zone might contain rows matching the given
    /// value under the given operator. This is a coarse filter: a page passing
    /// the filter may still contain no matching rows.
    pub fn candidate_pages(&self, op: crate::value::CmpOp, v: &Value) -> Vec<u32> {
        let mut out = Vec::new();
        for (&page, zone) in &self.zones {
            if zone.live_rows == 0 {
                continue;
            }
            if zone_overlaps(zone, op, v) {
                out.push(page);
            }
        }
        out
    }

    /// Recompute a page's zone from a flat list of the column's values.
    pub fn recompute(&mut self, page: u32, values: &[Value]) {
        let live = values.iter().filter(|v| !v.is_null()).count() as u32;
        let (min, max) = match values.iter().filter(|v| !v.is_null()).copied().minmax() {
            Some((mn, mx)) => (mn, mx),
            None => (Value::Null, Value::Null),
        };
        self.set(page, Zone { page, min, max, live_rows: live });
    }

    pub fn iter(&self) -> impl Iterator<Item = &Zone> + '_ {
        self.zones.values()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.zones.len() as u32).to_le_bytes());
        for zone in self.zones.values() {
            out.extend_from_slice(&zone.page.to_le_bytes());
            out.extend_from_slice(&encode_value(&zone.min));
            out.extend_from_slice(&encode_value(&zone.max));
            out.extend_from_slice(&zone.live_rows.to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> ZoneMap {
        let mut zm = ZoneMap::new();
        if buf.len() < 4 {
            return zm;
        }
        let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        for _ in 0..n {
            if pos + 4 > buf.len() {
                break;
            }
            let page = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let (min, a) = decode_value(&buf[pos..]);
            pos += a;
            let (max, b) = decode_value(&buf[pos..]);
            pos += b;
            if pos + 4 > buf.len() {
                break;
            }
            let live = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            zm.set(page, Zone { page, min, max, live_rows: live });
        }
        zm
    }
}

fn zone_overlaps(zone: &Zone, op: crate::value::CmpOp, v: &Value) -> bool {
    use crate::value::CmpOp;
    if zone.min.is_null() && zone.max.is_null() {
        return true;
    }
    match op {
        CmpOp::Eq => v.total_cmp(&zone.min) != std::cmp::Ordering::Less
            && v.total_cmp(&zone.max) != std::cmp::Ordering::Greater,
        CmpOp::Ne => true,
        CmpOp::Lt => zone.min.total_cmp(v) == std::cmp::Ordering::Less,
        CmpOp::Le => zone.min.total_cmp(v) != std::cmp::Ordering::Greater,
        CmpOp::Gt => zone.max.total_cmp(v) == std::cmp::Ordering::Greater,
        CmpOp::Ge => zone.max.total_cmp(v) != std::cmp::Ordering::Less,
    }
}

trait MinMaxExt: Iterator<Item = Value> + Sized {
    fn minmax(mut self) -> Option<(Value, Value)> {
        let first = self.next()?;
        let mut min = first;
        let mut max = first;
        for v in self {
            if v.total_cmp(&min) == std::cmp::Ordering::Less {
                min = v;
            }
            if v.total_cmp(&max) == std::cmp::Ordering::Greater {
                max = v;
            }
        }
        Some((min, max))
    }
}
impl<I: Iterator<Item = Value>> MinMaxExt for I {}

fn encode_value(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
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
    out
}

fn decode_value(buf: &[u8]) -> (Value, usize) {
    if buf.is_empty() {
        return (Value::Null, 0);
    }
    match buf[0] {
        0 => (Value::Null, 1),
        1 => (Value::Bool(buf.get(1).copied().unwrap_or(0) != 0), 2),
        2 => {
            if buf.len() < 9 {
                return (Value::Null, 1);
            }
            let i = i64::from_le_bytes(buf[1..9].try_into().unwrap());
            (Value::Int(i), 9)
        }
        3 => {
            if buf.len() < 9 {
                return (Value::Null, 1);
            }
            let r = f64::from_le_bytes(buf[1..9].try_into().unwrap());
            (Value::Real(r), 9)
        }
        4 => {
            if buf.len() < 5 {
                return (Value::Null, 1);
            }
            let id = u32::from_le_bytes(buf[1..5].try_into().unwrap());
            (Value::Text(id), 5)
        }
        _ => (Value::Null, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_pages_prunes() {
        let mut zm = ZoneMap::new();
        zm.set(0, Zone { page: 0, min: Value::Int(1), max: Value::Int(10), live_rows: 5 });
        zm.set(1, Zone { page: 1, min: Value::Int(100), max: Value::Int(200), live_rows: 5 });
        let pages = zm.candidate_pages(crate::value::CmpOp::Eq, &Value::Int(5));
        assert_eq!(pages, vec![0]);
        let pages = zm.candidate_pages(crate::value::CmpOp::Gt, &Value::Int(50));
        assert_eq!(pages, vec![1]);
    }

    #[test]
    fn recompute_from_values() {
        let mut zm = ZoneMap::new();
        zm.recompute(3, &[Value::Int(7), Value::Null, Value::Int(2), Value::Int(9)]);
        let z = zm.get(3).unwrap();
        assert_eq!(z.min, Value::Int(2));
        assert_eq!(z.max, Value::Int(9));
        assert_eq!(z.live_rows, 3);
    }
}
