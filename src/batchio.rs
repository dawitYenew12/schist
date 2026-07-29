//! A self-describing serialization for [`RecordBatch`].
//!
//! Result sets and spilled intermediate batches need a compact on-the-wire /
//! on-disk form that round-trips exactly. This encodes a batch as a small header
//! (magic, column count, row count) followed by, per column, its name, kind, a
//! validity bitmap, and the packed values. Values are stored with varint /
//! zig-zag coding so small integers stay short, and the whole payload can
//! optionally be run through the [`crate::compress`] block codecs.

use crate::array::{Array, ArrayBuilder, RecordBatch};
use crate::compress::{compress_best, unframe};
use crate::schema::ColKind;
use crate::value::Value;
use crate::varint::{read_ivarint, read_uvarint, write_ivarint, write_uvarint};

const BATCH_MAGIC: &[u8; 4] = b"SBTC";

/// Error decoding a serialized batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    BadMagic,
    Truncated,
    BadKind(u8),
    BadUtf8,
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::BadMagic => write!(f, "bad batch magic"),
            BatchError::Truncated => write!(f, "truncated batch"),
            BatchError::BadKind(k) => write!(f, "unknown column kind {k}"),
            BatchError::BadUtf8 => write!(f, "invalid utf-8 in column name"),
        }
    }
}

impl std::error::Error for BatchError {}

/// Encode a batch to bytes.
pub fn encode(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(BATCH_MAGIC);
    write_uvarint(&mut out, batch.width() as u64);
    write_uvarint(&mut out, batch.rows() as u64);
    for (name, col) in batch.names().iter().zip(batch.columns().iter()) {
        // Column name.
        write_uvarint(&mut out, name.len() as u64);
        out.extend_from_slice(name.as_bytes());
        // Kind.
        out.push(col.kind().as_u8());
        // Validity bitmap: one bit per row.
        let n = batch.rows();
        let mut valid = vec![0u8; n.div_ceil(8)];
        for i in 0..n {
            if !col.value(i).is_null() {
                valid[i >> 3] |= 1 << (i & 7);
            }
        }
        out.extend_from_slice(&valid);
        // Values (only non-null ones are written).
        for i in 0..n {
            let v = col.value(i);
            if v.is_null() {
                continue;
            }
            encode_value(&mut out, &v);
        }
    }
    out
}

/// Encode a batch and compress the payload.
pub fn encode_compressed(batch: &RecordBatch) -> Vec<u8> {
    compress_best(&encode(batch))
}

/// Decode a compressed batch produced by [`encode_compressed`].
pub fn decode_compressed(bytes: &[u8]) -> Result<RecordBatch, BatchError> {
    let raw = unframe(bytes).map_err(|_| BatchError::Truncated)?;
    decode(&raw)
}

fn encode_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => {}
        Value::Bool(b) => out.push(*b as u8),
        Value::Int(i) => write_ivarint(out, *i),
        Value::Real(r) => out.extend_from_slice(&r.to_le_bytes()),
        Value::Text(id) => write_uvarint(out, *id as u64),
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Cursor<'a> {
        Cursor { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], BatchError> {
        if self.pos + n > self.buf.len() {
            return Err(BatchError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn uvarint(&mut self) -> Result<u64, BatchError> {
        let (v, n) = read_uvarint(&self.buf[self.pos..]).map_err(|_| BatchError::Truncated)?;
        self.pos += n;
        Ok(v)
    }
    fn ivarint(&mut self) -> Result<i64, BatchError> {
        let (v, n) = read_ivarint(&self.buf[self.pos..]).map_err(|_| BatchError::Truncated)?;
        self.pos += n;
        Ok(v)
    }
    fn u8(&mut self) -> Result<u8, BatchError> {
        Ok(self.take(1)?[0])
    }
    fn f64(&mut self) -> Result<f64, BatchError> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes(b.try_into().unwrap()))
    }
}

/// Decode bytes produced by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<RecordBatch, BatchError> {
    let mut c = Cursor::new(bytes);
    if c.take(4)? != BATCH_MAGIC {
        return Err(BatchError::BadMagic);
    }
    let width = c.uvarint()? as usize;
    let rows = c.uvarint()? as usize;
    let mut pairs = Vec::with_capacity(width);
    for _ in 0..width {
        let name_len = c.uvarint()? as usize;
        let name_bytes = c.take(name_len)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| BatchError::BadUtf8)?
            .to_string();
        let kind = ColKind::from_u8(c.u8()?).ok_or(BatchError::BadKind(0))?;
        let valid_bytes = c.take(rows.div_ceil(8))?.to_vec();
        let mut builder = ArrayBuilder::new(kind);
        for i in 0..rows {
            let is_valid = valid_bytes[i >> 3] & (1 << (i & 7)) != 0;
            if !is_valid {
                builder.push_null();
                continue;
            }
            let v = match kind {
                ColKind::Bool => Value::Bool(c.u8()? != 0),
                ColKind::Int => Value::Int(c.ivarint()?),
                ColKind::Real => Value::Real(c.f64()?),
                ColKind::Text => Value::Text(c.uvarint()? as u32),
            };
            builder.push(v);
        }
        pairs.push((name, builder.finish()));
    }
    // An empty batch still needs its declared row count.
    if pairs.is_empty() && rows > 0 {
        return Ok(RecordBatch::new(vec![(
            "_".to_string(),
            Array::nulls(ColKind::Int, rows),
        )]));
    }
    Ok(RecordBatch::new(pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RecordBatch {
        let a = Array::from_values(
            ColKind::Int,
            &[Value::Int(1), Value::Null, Value::Int(-42), Value::Int(1000)],
        );
        let b = Array::from_values(
            ColKind::Real,
            &[Value::Real(1.5), Value::Real(2.5), Value::Null, Value::Real(9.0)],
        );
        let c = Array::from_values(
            ColKind::Bool,
            &[Value::Bool(true), Value::Bool(false), Value::Bool(true), Value::Null],
        );
        RecordBatch::new(vec![("n".into(), a), ("r".into(), b), ("flag".into(), c)])
    }

    #[test]
    fn roundtrip() {
        let batch = sample();
        let bytes = encode(&batch);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.rows(), 4);
        assert_eq!(back.column("n").unwrap().value(2), Value::Int(-42));
        assert_eq!(back.column("r").unwrap().value(2), Value::Null);
        assert_eq!(back.column("flag").unwrap().value(3), Value::Null);
    }

    #[test]
    fn compressed_roundtrip() {
        let batch = sample();
        let bytes = encode_compressed(&batch);
        let back = decode_compressed(&bytes).unwrap();
        assert_eq!(back.column("n").unwrap().value(0), Value::Int(1));
    }

    #[test]
    fn text_column() {
        let arr = Array::from_values(ColKind::Text, &[Value::Text(5), Value::Text(9)]);
        let batch = RecordBatch::new(vec![("t".into(), arr)]);
        let back = decode(&encode(&batch)).unwrap();
        assert_eq!(back.column("t").unwrap().value(1), Value::Text(9));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = encode(&sample());
        bytes[0] = b'X';
        assert_eq!(decode(&bytes), Err(BatchError::BadMagic));
    }

    #[test]
    fn rejects_truncated() {
        let bytes = encode(&sample());
        assert!(decode(&bytes[..7]).is_err());
    }

    #[test]
    fn empty_batch() {
        let arr = Array::from_values(ColKind::Int, &[]);
        let batch = RecordBatch::new(vec![("n".into(), arr)]);
        let back = decode(&encode(&batch)).unwrap();
        assert_eq!(back.rows(), 0);
    }
}
