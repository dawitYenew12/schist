//! A write-ahead log.
//!
//! Before a mutation is applied to the in-memory database, it is appended to a
//! write-ahead log record. On recovery, the log is replayed to reconstruct the
//! database state up to the last durable record. The log is segmented: when a
//! segment reaches its capacity, a new segment is opened.
//!
//! A record is a length-prefixed, checksummed byte blob tagged with a record
//! type. This module provides the encoder/decoder and an in-memory log buffer;
//! the mutation layer wires its operations into records.

use crate::value::Value;

/// The record type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecType {
    /// Begin a transaction.
    Begin,
    /// Commit a transaction.
    Commit,
    /// Abort a transaction.
    Abort,
    /// An insert.
    Insert,
    /// An update.
    Update,
    /// A delete.
    Delete,
    /// A checkpoint marker.
    Checkpoint,
    /// A schema change.
    Schema,
}

impl RecType {
    pub fn as_u8(self) -> u8 {
        match self {
            RecType::Begin => 0,
            RecType::Commit => 1,
            RecType::Abort => 2,
            RecType::Insert => 3,
            RecType::Update => 4,
            RecType::Delete => 5,
            RecType::Checkpoint => 6,
            RecType::Schema => 7,
        }
    }
    pub fn from_u8(b: u8) -> Option<RecType> {
        match b {
            0 => Some(RecType::Begin),
            1 => Some(RecType::Commit),
            2 => Some(RecType::Abort),
            3 => Some(RecType::Insert),
            4 => Some(RecType::Update),
            5 => Some(RecType::Delete),
            6 => Some(RecType::Checkpoint),
            7 => Some(RecType::Schema),
            _ => None,
        }
    }
}

/// A single log record.
#[derive(Debug, Clone)]
pub struct Record {
    pub lsn: u64,
    pub txn: u64,
    pub rtype: RecType,
    pub payload: Vec<u8>,
}

/// A log segment: an in-memory byte buffer with a starting LSN.
#[derive(Debug, Clone)]
pub struct Segment {
    pub start_lsn: u64,
    pub capacity: usize,
    pub buf: Vec<u8>,
    pub record_count: u32,
}

impl Segment {
    pub fn new(start_lsn: u64, capacity: usize) -> Segment {
        Segment {
            start_lsn,
            capacity,
            buf: Vec::with_capacity(capacity),
            record_count: 0,
        }
    }

    pub fn used(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buf.len() >= self.capacity
    }

    pub fn can_fit(&self, len: usize) -> bool {
        self.buf.len() + len <= self.capacity
    }
}

/// The write-ahead log: a list of segments.
#[derive(Debug, Clone)]
pub struct Wal {
    pub segments: Vec<Segment>,
    pub segment_capacity: usize,
    pub next_lsn: u64,
}

impl Wal {
    pub fn new(segment_capacity: usize) -> Wal {
        let cap = segment_capacity.max(256);
        Wal {
            segments: vec![Segment::new(0, cap)],
            segment_capacity: cap,
            next_lsn: 1,
        }
    }

    pub fn current(&self) -> &Segment {
        self.segments.last().expect("wal always has a segment")
    }

    /// Append a record. Returns its LSN. Opens a new segment if the current one
    /// cannot fit.
    pub fn append(&mut self, txn: u64, rtype: RecType, payload: &[u8]) -> u64 {
        let lsn = self.next_lsn;
        let rec_bytes = encode_record(lsn, txn, rtype, payload);
        if !self.current().can_fit(rec_bytes.len()) && !self.current().is_empty() {
            let start = self.next_lsn;
            self.segments.push(Segment::new(start, self.segment_capacity));
        }
        let seg = self.segments.last_mut().unwrap();
        seg.buf.extend_from_slice(&rec_bytes);
        seg.record_count += 1;
        self.next_lsn += 1;
        lsn
    }

    /// Iterate every record across all segments.
    pub fn records(&self) -> Vec<Record> {
        let mut out = Vec::new();
        for seg in &self.segments {
            let mut pos = 0;
            while pos < seg.buf.len() {
                match decode_record(&seg.buf[pos..]) {
                    Some((rec, used)) => {
                        out.push(rec);
                        pos += used;
                    }
                    None => break,
                }
            }
        }
        out
    }

    /// Truncate the log up to (and including) the given LSN — used after a
    /// checkpoint makes older records obsolete.
    pub fn truncate_before(&mut self, lsn: u64) {
        self.segments
            .retain(|seg| seg.start_lsn + seg.record_count as u64 > lsn);
        if self.segments.is_empty() {
            self.segments.push(Segment::new(self.next_lsn, self.segment_capacity));
        }
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn record_count(&self) -> u32 {
        self.segments.iter().map(|s| s.record_count).sum()
    }
}

/// Encode a record: `[u8 type][u64 lsn][u64 txn][u32 len][u32 checksum][payload]`.
pub fn encode_record(lsn: u64, txn: u64, rtype: RecType, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(25 + payload.len());
    out.push(rtype.as_u8());
    out.extend_from_slice(&lsn.to_le_bytes());
    out.extend_from_slice(&txn.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&checksum(payload).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Decode a record from the front of a buffer. Returns the record and bytes
/// consumed.
pub fn decode_record(buf: &[u8]) -> Option<(Record, usize)> {
    if buf.len() < 25 {
        return None;
    }
    let rtype = RecType::from_u8(buf[0])?;
    let lsn = u64::from_le_bytes(buf[1..9].try_into().unwrap());
    let txn = u64::from_le_bytes(buf[9..17].try_into().unwrap());
    let len = u32::from_le_bytes(buf[17..21].try_into().unwrap()) as usize;
    let cs = u32::from_le_bytes(buf[21..25].try_into().unwrap());
    if buf.len() < 25 + len {
        return None;
    }
    let payload = &buf[25..25 + len];
    if checksum(payload) != cs {
        return None;
    }
    Some((
        Record {
            lsn,
            txn,
            rtype,
            payload: payload.to_vec(),
        },
        25 + len,
    ))
}

fn checksum(data: &[u8]) -> u32 {
    let mut s: u32 = 0x811c9dc5;
    for &b in data {
        s = s.wrapping_mul(0x01000193).wrapping_add(b as u32);
    }
    s
}

/// Encode an insert payload: `[u64 row_id][u8 arity][cells]`.
pub fn encode_insert_payload(row_id: u64, values: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&row_id.to_le_bytes());
    out.push(values.len() as u8);
    for v in values {
        out.extend_from_slice(&encode_value(v));
    }
    out
}

/// Decode an insert payload.
pub fn decode_insert_payload(buf: &[u8]) -> Option<(u64, Vec<Value>)> {
    if buf.len() < 9 {
        return None;
    }
    let row_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let arity = buf[8] as usize;
    let mut pos = 9;
    let mut values = Vec::with_capacity(arity);
    for _ in 0..arity {
        let (v, used) = decode_value(&buf[pos..])?;
        values.push(v);
        pos += used;
    }
    Some((row_id, values))
}

/// Encode a delete payload: `[u64 row_id]`.
pub fn encode_delete_payload(row_id: u64) -> Vec<u8> {
    row_id.to_le_bytes().to_vec()
}

pub fn decode_delete_payload(buf: &[u8]) -> Option<u64> {
    if buf.len() < 8 {
        return None;
    }
    Some(u64::from_le_bytes(buf[0..8].try_into().unwrap()))
}

/// Encode an update payload: `[u64 row_id][u8 col][value]`.
pub fn encode_update_payload(row_id: u64, col: usize, value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&row_id.to_le_bytes());
    out.push(col as u8);
    out.extend_from_slice(&encode_value(value));
    out
}

pub fn decode_update_payload(buf: &[u8]) -> Option<(u64, usize, Value)> {
    if buf.len() < 9 {
        return None;
    }
    let row_id = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let col = buf[8] as usize;
    let (v, _) = decode_value(&buf[9..])?;
    Some((row_id, col, v))
}

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

fn decode_value(buf: &[u8]) -> Option<(Value, usize)> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        0 => Some((Value::Null, 1)),
        1 => Some((Value::Bool(buf.get(1).copied().unwrap_or(0) != 0), 2)),
        2 => {
            if buf.len() < 9 {
                return None;
            }
            Some((Value::Int(i64::from_le_bytes(buf[1..9].try_into().unwrap())), 9))
        }
        3 => {
            if buf.len() < 9 {
                return None;
            }
            Some((Value::Real(f64::from_le_bytes(buf[1..9].try_into().unwrap())), 9))
        }
        4 => {
            if buf.len() < 5 {
                return None;
            }
            Some((Value::Text(u32::from_le_bytes(buf[1..5].try_into().unwrap())), 5))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_read_records() {
        let mut wal = Wal::new(256);
        let lsn1 = wal.append(1, RecType::Begin, &[]);
        let lsn2 = wal.append(1, RecType::Insert, &encode_insert_payload(10, &[Value::Int(1), Value::Int(2)]));
        wal.append(1, RecType::Commit, &[]);
        let recs = wal.records();
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[0].rtype, RecType::Begin);
        assert_eq!(recs[1].rtype, RecType::Insert);
        let (rid, vals) = decode_insert_payload(&recs[1].payload).unwrap();
        assert_eq!(rid, 10);
        assert_eq!(vals, vec![Value::Int(1), Value::Int(2)]);
        assert!(lsn2 > lsn1);
    }

    #[test]
    fn segments_rotate_on_full() {
        let mut wal = Wal::new(64);
        for i in 0..20u64 {
            wal.append(i, RecType::Insert, &encode_insert_payload(i, &[Value::Int(i as i64)]));
        }
        assert!(wal.segment_count() > 1);
        assert_eq!(wal.record_count(), 20);
        let recs = wal.records();
        assert_eq!(recs.len(), 20);
    }

    #[test]
    fn truncate_removes_old_segments() {
        let mut wal = Wal::new(64);
        for i in 0..10u64 {
            wal.append(i, RecType::Insert, &[]);
        }
        wal.truncate_before(5);
        let recs = wal.records();
        assert!(recs.iter().all(|r| r.lsn > 5) || recs.is_empty());
    }

    #[test]
    fn update_payload_round_trips() {
        let p = encode_update_payload(7, 2, &Value::Int(99));
        let (rid, col, v) = decode_update_payload(&p).unwrap();
        assert_eq!(rid, 7);
        assert_eq!(col, 2);
        assert_eq!(v, Value::Int(99));
    }

    #[test]
    fn checksum_detects_corruption() {
        let mut bytes = encode_record(1, 1, RecType::Insert, &[1, 2, 3]);
        bytes[25] ^= 0xff;
        assert!(decode_record(&bytes).is_none());
    }
}
