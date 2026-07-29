//! Variable-length integer codecs and a bit reader/writer.
//!
//! The container format and the column encodings lean on compact integer
//! representations: LEB128 varints for lengths and small counts, zig-zag
//! mapping so signed values stay short near zero, and a bit-level reader/writer
//! for the packed encodings. This module gathers those primitives with strict
//! bounds checking so a truncated or over-long encoding is reported rather than
//! silently misread.

/// Error decoding a varint or reading bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarintError {
    /// The buffer ended mid-value.
    Truncated,
    /// The varint used more than 10 continuation bytes (overlong for 64 bits).
    Overlong,
}

impl std::fmt::Display for VarintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VarintError::Truncated => write!(f, "truncated varint"),
            VarintError::Overlong => write!(f, "overlong varint"),
        }
    }
}

impl std::error::Error for VarintError {}

/// Append an unsigned LEB128 varint.
pub fn write_uvarint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            out.push(byte | 0x80);
        } else {
            out.push(byte);
            break;
        }
    }
}

/// Read an unsigned LEB128 varint, returning `(value, bytes_consumed)`.
pub fn read_uvarint(buf: &[u8]) -> Result<(u64, usize), VarintError> {
    let mut result = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in buf.iter().enumerate() {
        if i >= 10 {
            return Err(VarintError::Overlong);
        }
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
    }
    Err(VarintError::Truncated)
}

/// Zig-zag encode a signed integer into an unsigned one.
pub fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Reverse [`zigzag_encode`].
pub fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

/// Append a signed varint (zig-zag + LEB128).
pub fn write_ivarint(out: &mut Vec<u8>, value: i64) {
    write_uvarint(out, zigzag_encode(value));
}

/// Read a signed varint.
pub fn read_ivarint(buf: &[u8]) -> Result<(i64, usize), VarintError> {
    let (u, n) = read_uvarint(buf)?;
    Ok((zigzag_decode(u), n))
}

/// The serialized size of an unsigned varint.
pub fn uvarint_len(value: u64) -> usize {
    let mut v = value;
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

/// A most-significant-bit-first bit writer.
#[derive(Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitWriter {
    /// A fresh writer.
    pub fn new() -> BitWriter {
        BitWriter::default()
    }

    /// Write the low `count` bits of `value` (count in 0..=32).
    pub fn write_bits(&mut self, value: u32, count: u8) {
        for i in (0..count).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.cur = (self.cur << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.cur);
                self.cur = 0;
                self.nbits = 0;
            }
        }
    }

    /// Number of bits written so far.
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.nbits as usize
    }

    /// Flush any partial byte (zero-padded) and return the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.cur <<= 8 - self.nbits;
            self.bytes.push(self.cur);
        }
        self.bytes
    }
}

/// A most-significant-bit-first bit reader.
#[derive(Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    /// A reader over `bytes`.
    pub fn new(bytes: &'a [u8]) -> BitReader<'a> {
        BitReader {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    /// Read `count` bits (0..=32) into the low bits of the result.
    pub fn read_bits(&mut self, count: u8) -> Result<u32, VarintError> {
        let mut result = 0u32;
        for _ in 0..count {
            if self.byte_pos >= self.bytes.len() {
                return Err(VarintError::Truncated);
            }
            let byte = self.bytes[self.byte_pos];
            let bit = (byte >> (7 - self.bit_pos)) & 1;
            result = (result << 1) | bit as u32;
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }
        Ok(result)
    }

    /// Number of bits still available.
    pub fn remaining_bits(&self) -> usize {
        (self.bytes.len() - self.byte_pos) * 8 - self.bit_pos as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uvarint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 16384, u64::MAX] {
            let mut buf = Vec::new();
            write_uvarint(&mut buf, v);
            assert_eq!(buf.len(), uvarint_len(v));
            let (got, n) = read_uvarint(&buf).unwrap();
            assert_eq!(got, v);
            assert_eq!(n, buf.len());
        }
    }

    #[test]
    fn zigzag_roundtrip() {
        for v in [0i64, -1, 1, -1000, 1000, i64::MIN, i64::MAX] {
            assert_eq!(zigzag_decode(zigzag_encode(v)), v);
        }
        assert_eq!(zigzag_encode(-1), 1);
        assert_eq!(zigzag_encode(1), 2);
    }

    #[test]
    fn ivarint_roundtrip() {
        for v in [-500i64, -1, 0, 1, 500, i64::MIN, i64::MAX] {
            let mut buf = Vec::new();
            write_ivarint(&mut buf, v);
            let (got, _) = read_ivarint(&buf).unwrap();
            assert_eq!(got, v);
        }
    }

    #[test]
    fn detects_truncation_and_overlong() {
        assert_eq!(read_uvarint(&[0x80]), Err(VarintError::Truncated));
        let overlong = [0x80u8; 11];
        assert_eq!(read_uvarint(&overlong), Err(VarintError::Overlong));
    }

    #[test]
    fn bit_writer_reader_roundtrip() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b1111_0000, 8);
        w.write_bits(1, 1);
        assert_eq!(w.bit_len(), 12);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(3).unwrap(), 0b101);
        assert_eq!(r.read_bits(8).unwrap(), 0b1111_0000);
        assert_eq!(r.read_bits(1).unwrap(), 1);
    }

    #[test]
    fn bit_reader_reports_truncation() {
        let mut r = BitReader::new(&[0xFF]);
        assert!(r.read_bits(8).is_ok());
        assert_eq!(r.read_bits(1), Err(VarintError::Truncated));
    }
}
