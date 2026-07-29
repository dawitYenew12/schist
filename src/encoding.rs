//! Column encodings beyond `Plain` / `Dictionary` / `Rle`.
//!
//! A real columnar engine offers a family of encodings, each suited to a
//! different data shape: bit-packing for narrow integer ranges, frame-of-
//! reference for clustered integers, delta-of-delta for monotonic sequences,
//! and null-suppression for sparse nulls. This module implements the encode and
//! decode halves of each, plus the helpers (min/max, bit-width selection,
//! zig-zag encoding) they share.
//!
//! Every encoding here is self-contained: it takes a slice of [`Value`]s (or
//! raw bytes) and produces a byte vector, and conversely. The format layer
//! chooses which encoding a column region uses and stores the encoding tag in
//! the region header; the decode half is dispatched on that tag.
//!
//! ## Layout conventions
//!
//! Each encoding's byte output begins with a 1-byte *sub-tag* so that a single
//! `Encoding::Plain`/`Dictionary`/`Rle` column region can host several sub-
//! encodings without changing the region header. The sub-tag values are:
//!
//! - `0x00` raw plain (no sub-encoding)
//! - `0x01` bit-packed
//! - `0x02` frame-of-reference
//! - `0x03` delta
//! - `0x04` delta-of-delta
//! - `0x05` null-suppressed
//! - `0x06` constant-run
//! - `0x07` zig-zag varint
//! - `0x08` byte-array dictionary inline

use crate::error::{DecodeError, Result};
use crate::value::Value;

/// The sub-encoding tag byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubTag {
    Plain,
    BitPacked,
    FrameOfRef,
    Delta,
    DeltaOfDelta,
    NullSuppressed,
    ConstantRun,
    ZigZagVarint,
    ByteArrayDict,
}

impl SubTag {
    pub fn as_u8(self) -> u8 {
        match self {
            SubTag::Plain => 0x00,
            SubTag::BitPacked => 0x01,
            SubTag::FrameOfRef => 0x02,
            SubTag::Delta => 0x03,
            SubTag::DeltaOfDelta => 0x04,
            SubTag::NullSuppressed => 0x05,
            SubTag::ConstantRun => 0x06,
            SubTag::ZigZagVarint => 0x07,
            SubTag::ByteArrayDict => 0x08,
        }
    }

    pub fn from_u8(b: u8) -> Option<SubTag> {
        match b {
            0x00 => Some(SubTag::Plain),
            0x01 => Some(SubTag::BitPacked),
            0x02 => Some(SubTag::FrameOfRef),
            0x03 => Some(SubTag::Delta),
            0x04 => Some(SubTag::DeltaOfDelta),
            0x05 => Some(SubTag::NullSuppressed),
            0x06 => Some(SubTag::ConstantRun),
            0x07 => Some(SubTag::ZigZagVarint),
            0x08 => Some(SubTag::ByteArrayDict),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SubTag::Plain => "plain",
            SubTag::BitPacked => "bitpacked",
            SubTag::FrameOfRef => "for",
            SubTag::Delta => "delta",
            SubTag::DeltaOfDelta => "dod",
            SubTag::NullSuppressed => "nullsupp",
            SubTag::ConstantRun => "const",
            SubTag::ZigZagVarint => "zigzag",
            SubTag::ByteArrayDict => "badict",
        }
    }
}

/// The number of bits needed to represent values in `[0, max]`.
pub fn bits_needed(max: u64) -> u32 {
    if max == 0 {
        return 0;
    }
    64 - max.leading_zeros()
}

/// The minimum number of bits to represent `n` distinct unsigned values.
pub fn width_for(values: &[u64]) -> u32 {
    let max = values.iter().copied().max().unwrap_or(0);
    bits_needed(max)
}

/// Zig-zag encode a signed integer into an unsigned one so that small-magnitude
/// negatives (common in deltas) stay small.
pub fn zigzag_encode(i: i64) -> u64 {
    ((i << 1) ^ (i >> 63)) as u64
}

/// Inverse of [`zigzag_encode`].
pub fn zigzag_decode(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

/// Encode an unsigned integer as a little-endian base-128 varint.
pub fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Read a varint from a buffer, returning the value and the number of bytes
/// consumed.
pub fn read_varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &b) in buf.iter().enumerate() {
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Bit-packing
// ---------------------------------------------------------------------------

/// Pack `n` values of `width` bits each into a byte vector, little-endian bit
/// order. `width` must be in `0..=64`. Returns the packed bytes; the count and
/// width are stored by the caller in the sub-header.
pub fn bitpack(values: &[u64], width: u32) -> Vec<u8> {
    if width == 0 {
        return Vec::new();
    }
    let total_bits = values.len() as u64 * width as u64;
    let nbytes = ((total_bits + 7) / 8) as usize;
    let mut out = vec![0u8; nbytes];
    let mut bit_pos = 0usize;
    for &v in values {
        for b in 0..width {
            if (v >> b) & 1 == 1 {
                let byte = bit_pos / 8;
                let bit = bit_pos % 8;
                out[byte] |= 1u8 << bit;
            }
            bit_pos += 1;
        }
    }
    out
}

/// Unpack `count` values of `width` bits each from packed bytes.
pub fn bitunpack(buf: &[u8], count: usize, width: u32) -> Vec<u64> {
    if width == 0 {
        return vec![0u64; count];
    }
    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut v: u64 = 0;
        for b in 0..width {
            let byte = bit_pos / 8;
            let bit = bit_pos % 8;
            if byte < buf.len() && (buf[byte] >> bit) & 1 == 1 {
                v |= 1u64 << b;
            }
            bit_pos += 1;
        }
        out.push(v);
    }
    out
}

/// Encode a slice of ints as a bit-packed region: sub-tag, width, count, then
/// the packed bytes.
pub fn encode_bitpacked(values: &[i64]) -> Vec<u8> {
    let max = values.iter().map(|&i| i.max(0) as u64).max().unwrap_or(0);
    let width = bits_needed(max);
    let mut out = Vec::new();
    out.push(SubTag::BitPacked.as_u8());
    out.push(width as u8);
    write_varint(&mut out, values.len() as u64);
    let u64s: Vec<u64> = values.iter().map(|&i| i.max(0) as u64).collect();
    out.extend_from_slice(&bitpack(&u64s, width));
    out
}

/// Decode a bit-packed region (without the sub-tag).
pub fn decode_bitpacked(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.len() < 2 {
        return Err(DecodeError::BadEncoding.into());
    }
    let width = buf[1] as u32;
    let (count, used) = read_varint(&buf[2..]).ok_or(DecodeError::BadEncoding)?;
    let count = count as usize;
    let packed = &buf[2 + used..];
    let u64s = bitunpack(packed, count, width);
    Ok(u64s.into_iter().map(|u| u as i64).collect())
}

// ---------------------------------------------------------------------------
// Frame of reference
// ---------------------------------------------------------------------------

/// Encode a slice of ints relative to their minimum (frame of reference), then
/// bit-pack the offsets. Layout: sub-tag, min (i64 LE), width, count, packed.
pub fn encode_for(values: &[i64]) -> Vec<u8> {
    let min = values.iter().copied().min().unwrap_or(0);
    let offsets: Vec<u64> = values.iter().map(|&i| (i - min) as u64).collect();
    let max = offsets.iter().copied().max().unwrap_or(0);
    let width = bits_needed(max);
    let mut out = Vec::new();
    out.push(SubTag::FrameOfRef.as_u8());
    out.extend_from_slice(&min.to_le_bytes());
    out.push(width as u8);
    write_varint(&mut out, values.len() as u64);
    out.extend_from_slice(&bitpack(&offsets, width));
    out
}

pub fn decode_for(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.len() < 9 {
        return Err(DecodeError::BadEncoding.into());
    }
    let min = i64::from_le_bytes(buf[1..9].try_into().unwrap());
    let width = buf[9] as u32;
    let (count, used) = read_varint(&buf[10..]).ok_or(DecodeError::BadEncoding)?;
    let count = count as usize;
    let packed = &buf[10 + used..];
    let offsets = bitunpack(packed, count, width);
    Ok(offsets.into_iter().map(|u| min + u as i64).collect())
}

// ---------------------------------------------------------------------------
// Delta
// ---------------------------------------------------------------------------

/// Encode a slice of ints as a sequence of deltas from the previous value.
/// Layout: sub-tag, first value (i64 LE), count, then varint-encoded zig-zag
/// deltas.
pub fn encode_delta(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(SubTag::Delta.as_u8());
    let first = values.first().copied().unwrap_or(0);
    out.extend_from_slice(&first.to_le_bytes());
    write_varint(&mut out, values.len() as u64);
    let mut prev = first;
    for &v in values.iter().skip(1) {
        let d = v.wrapping_sub(prev);
        write_varint(&mut out, zigzag_encode(d));
        prev = v;
    }
    out
}

pub fn decode_delta(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.len() < 9 {
        return Err(DecodeError::BadEncoding.into());
    }
    let first = i64::from_le_bytes(buf[1..9].try_into().unwrap());
    let (count, used) = read_varint(&buf[9..]).ok_or(DecodeError::BadEncoding)?;
    let count = count as usize;
    let mut out = Vec::with_capacity(count);
    out.push(first);
    let mut prev = first;
    let mut pos = 9 + used;
    for _ in 1..count {
        let (zz, n) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
        let d = zigzag_decode(zz);
        let v = prev.wrapping_add(d);
        out.push(v);
        prev = v;
        pos += n;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Delta-of-delta
// ---------------------------------------------------------------------------

/// Encode a slice of ints as delta-of-deltas (effective for monotonic or
/// near-monotonic sequences like timestamps). Layout: sub-tag, first (i64 LE),
/// second (i64 LE), count, then varint zig-zag delta-of-deltas.
pub fn encode_dod(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(SubTag::DeltaOfDelta.as_u8());
    let first = values.first().copied().unwrap_or(0);
    let second = values.get(1).copied().unwrap_or(first);
    out.extend_from_slice(&first.to_le_bytes());
    out.extend_from_slice(&second.to_le_bytes());
    write_varint(&mut out, values.len() as u64);
    let mut prev = second;
    let mut prev_delta = second.wrapping_sub(first);
    for &v in values.iter().skip(2) {
        let delta = v.wrapping_sub(prev);
        let dod = delta.wrapping_sub(prev_delta);
        write_varint(&mut out, zigzag_encode(dod));
        prev_delta = delta;
        prev = v;
    }
    out
}

pub fn decode_dod(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.len() < 17 {
        return Err(DecodeError::BadEncoding.into());
    }
    let first = i64::from_le_bytes(buf[1..9].try_into().unwrap());
    let second = i64::from_le_bytes(buf[9..17].try_into().unwrap());
    let (count, used) = read_varint(&buf[17..]).ok_or(DecodeError::BadEncoding)?;
    let count = count as usize;
    let mut out = Vec::with_capacity(count);
    out.push(first);
    if count > 1 {
        out.push(second);
    }
    let mut prev = second;
    let mut prev_delta = second.wrapping_sub(first);
    let mut pos = 17 + used;
    for _ in 2..count {
        let (zz, n) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
        let dod = zigzag_decode(zz);
        let delta = prev_delta.wrapping_add(dod);
        let v = prev.wrapping_add(delta);
        out.push(v);
        prev_delta = delta;
        prev = v;
        pos += n;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Null suppression
// ---------------------------------------------------------------------------

/// Encode ints with a null bitmap plus a packed list of the non-null values.
/// Layout: sub-tag, count, null bitmap (ceil(count/8) bytes), then the non-null
/// i64 values back to back.
pub fn encode_nullsupp(values: &[Option<i64>]) -> Vec<u8> {
    let count = values.len();
    let bitmap_bytes = (count + 7) / 8;
    let mut out = Vec::new();
    out.push(SubTag::NullSuppressed.as_u8());
    write_varint(&mut out, count as u64);
    let mut bitmap = vec![0u8; bitmap_bytes];
    let mut present = Vec::new();
    for (i, v) in values.iter().enumerate() {
        if let Some(x) = v {
            bitmap[i / 8] |= 1u8 << (i % 8);
            present.push(x);
        }
    }
    out.extend_from_slice(&bitmap);
    for x in present {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

pub fn decode_nullsupp(buf: &[u8]) -> Result<Vec<Option<i64>>> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    let (count, used) = read_varint(&buf[1..]).ok_or(DecodeError::BadEncoding)?;
    let count = count as usize;
    let bitmap_bytes = (count + 7) / 8;
    if 1 + used + bitmap_bytes > buf.len() {
        return Err(DecodeError::BadEncoding.into());
    }
    let bitmap = &buf[1 + used..1 + used + bitmap_bytes];
    let mut pos = 1 + used + bitmap_bytes;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        if (bitmap[i / 8] >> (i % 8)) & 1 == 1 {
            if pos + 8 > buf.len() {
                return Err(DecodeError::Truncated.into());
            }
            let x = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            out.push(Some(x));
            pos += 8;
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Constant run
// ---------------------------------------------------------------------------

/// Encode a slice that is entirely one repeated value. Layout: sub-tag, count,
/// value (i64 LE).
pub fn encode_const(value: i64, count: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(SubTag::ConstantRun.as_u8());
    write_varint(&mut out, count as u64);
    out.extend_from_slice(&value.to_le_bytes());
    out
}

pub fn decode_const(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.len() < 1 {
        return Err(DecodeError::BadEncoding.into());
    }
    let (count, used) = read_varint(&buf[1..]).ok_or(DecodeError::BadEncoding)?;
    if 1 + used + 8 > buf.len() {
        return Err(DecodeError::Truncated.into());
    }
    let value = i64::from_le_bytes(buf[1 + used..1 + used + 8].try_into().unwrap());
    Ok(vec![value; count as usize])
}

// ---------------------------------------------------------------------------
// Zig-zag varint
// ---------------------------------------------------------------------------

/// Encode a slice of ints as zig-zag varints. Layout: sub-tag, count, then the
/// varints.
pub fn encode_zigzag(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(SubTag::ZigZagVarint.as_u8());
    write_varint(&mut out, values.len() as u64);
    for &v in values {
        write_varint(&mut out, zigzag_encode(v));
    }
    out
}

pub fn decode_zigzag(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    let (count, used) = read_varint(&buf[1..]).ok_or(DecodeError::BadEncoding)?;
    let count = count as usize;
    let mut out = Vec::with_capacity(count);
    let mut pos = 1 + used;
    for _ in 0..count {
        let (zz, n) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
        out.push(zigzag_decode(zz));
        pos += n;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Byte-array dictionary inline
// ---------------------------------------------------------------------------

/// Encode a slice of byte-arrays with an inline dictionary: each distinct byte
/// array is interned to an id, and the ids are stored as varints. Layout:
/// sub-tag, count, dict_size, then dict entries `[varint len][bytes]`, then the
/// per-row id varints.
pub fn encode_badict(values: &[Vec<u8>]) -> Vec<u8> {
    use std::collections::BTreeMap;
    let mut dict: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
    let mut entries: Vec<Vec<u8>> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(values.len());
    for v in values {
        let id = if let Some(&id) = dict.get(v) {
            id
        } else {
            let id = entries.len() as u32;
            dict.insert(v.clone(), id);
            entries.push(v.clone());
            id
        };
        ids.push(id);
    }
    let mut out = Vec::new();
    out.push(SubTag::ByteArrayDict.as_u8());
    write_varint(&mut out, values.len() as u64);
    write_varint(&mut out, entries.len() as u64);
    for e in &entries {
        write_varint(&mut out, e.len() as u64);
        out.extend_from_slice(e);
    }
    for id in ids {
        write_varint(&mut out, id as u64);
    }
    out
}

pub fn decode_badict(buf: &[u8]) -> Result<Vec<Vec<u8>>> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    let mut pos = 1;
    let (count, used) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
    pos += used;
    let (dict_size, used) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
    pos += used;
    let mut entries: Vec<Vec<u8>> = Vec::with_capacity(dict_size as usize);
    for _ in 0..dict_size {
        let (len, used) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
        pos += used;
        let len = len as usize;
        if pos + len > buf.len() {
            return Err(DecodeError::Truncated.into());
        }
        entries.push(buf[pos..pos + len].to_vec());
        pos += len;
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (id, used) = read_varint(&buf[pos..]).ok_or(DecodeError::BadEncoding)?;
        pos += used;
        let id = id as usize;
        if id >= entries.len() {
            return Err(DecodeError::BadEncoding.into());
        }
        out.push(entries[id].clone());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Pick the best sub-encoding for a slice of non-null ints, returning the
/// encoded bytes. The heuristic prefers constant-run, then frame-of-reference,
/// then delta/delta-of-delta for monotonic data, then bit-packed, then plain
/// varint as a fallback.
pub fn encode_ints_best(values: &[i64]) -> Vec<u8> {
    if values.is_empty() {
        return encode_const(0, 0);
    }
    let first = values[0];
    if values.iter().all(|&v| v == first) {
        return encode_const(first, values.len());
    }
    let deltas: Vec<i64> = values
        .windows(2)
        .map(|w| w[1].wrapping_sub(w[0]))
        .collect();
    let dod: Vec<i64> = deltas
        .windows(2)
        .map(|w| w[1].wrapping_sub(w[0]))
        .collect();
    let dod_constant = dod.windows(2).all(|w| w[0] == w[1]);
    if dod_constant && values.len() > 3 {
        return encode_dod(values);
    }
    let delta_constant = deltas.windows(2).all(|w| w[0] == w[1]);
    if delta_constant && values.len() > 2 {
        return encode_delta(values);
    }
    let min = values.iter().copied().min().unwrap_or(0);
    let max = values.iter().copied().max().unwrap_or(0);
    let range = (max as i128) - (min as i128);
    if range <= 0xFFFF && values.len() > 1 {
        return encode_for(values);
    }
    let max_abs = values
        .iter()
        .map(|&i| zigzag_encode(i))
        .max()
        .unwrap_or(0);
    if bits_needed(max_abs) <= 16 {
        return encode_bitpacked(values);
    }
    encode_zigzag(values)
}

/// Decode ints from any sub-encoded buffer by dispatching on the sub-tag.
pub fn decode_ints(buf: &[u8]) -> Result<Vec<i64>> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    let tag = SubTag::from_u8(buf[0]).ok_or(DecodeError::BadEncoding)?;
    match tag {
        SubTag::Plain => Ok(Vec::new()),
        SubTag::BitPacked => decode_bitpacked(buf),
        SubTag::FrameOfRef => decode_for(buf),
        SubTag::Delta => decode_delta(buf),
        SubTag::DeltaOfDelta => decode_dod(buf),
        SubTag::NullSuppressed => decode_nullsupp(buf).map(|v| v.into_iter().map(|x| x.unwrap_or(0)).collect()),
        SubTag::ConstantRun => decode_const(buf),
        SubTag::ZigZagVarint => decode_zigzag(buf),
        SubTag::ByteArrayDict => Err(DecodeError::BadEncoding.into()),
    }
}

/// Encode a slice of `Value`s (ints only) with the best sub-encoding.
pub fn encode_values_best(values: &[Value]) -> Vec<u8> {
    let ints: Vec<i64> = values
        .iter()
        .map(|v| v.as_int().unwrap_or(0))
        .collect();
    encode_ints_best(&ints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_round_trips() {
        for i in [-1000i64, -1, 0, 1, 1000, i64::MAX / 2, i64::MIN / 2] {
            assert_eq!(zigzag_decode(zigzag_encode(i)), i);
        }
    }

    #[test]
    fn varint_round_trips() {
        for v in [0u64, 1, 127, 128, 255, 16384, 1 << 35, u64::MAX] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let (back, n) = read_varint(&buf).unwrap();
            assert_eq!(back, v);
            assert_eq!(n, buf.len());
        }
    }

    #[test]
    fn bitpack_round_trips() {
        let vals = vec![0u64, 1, 3, 7, 5, 2, 6, 4];
        let packed = bitpack(&vals, 3);
        let back = bitunpack(&packed, vals.len(), 3);
        assert_eq!(back, vals);
    }

    #[test]
    fn encode_best_const() {
        let bytes = encode_ints_best(&[5, 5, 5, 5]);
        let back = decode_ints(&bytes).unwrap();
        assert_eq!(back, vec![5, 5, 5, 5]);
    }

    #[test]
    fn encode_best_for() {
        let bytes = encode_ints_best(&[100, 101, 102, 103]);
        let back = decode_ints(&bytes).unwrap();
        assert_eq!(back, vec![100, 101, 102, 103]);
    }

    #[test]
    fn encode_best_delta() {
        let bytes = encode_ints_best(&[10, 20, 30, 40, 50]);
        let back = decode_ints(&bytes).unwrap();
        assert_eq!(back, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn encode_best_dod() {
        let bytes = encode_ints_best(&[1, 4, 9, 16, 25]);
        let back = decode_ints(&bytes).unwrap();
        assert_eq!(back, vec![1, 4, 9, 16, 25]);
    }

    #[test]
    fn nullsupp_round_trips() {
        let vals = vec![Some(1), None, Some(3), None, Some(5)];
        let bytes = encode_nullsupp(&vals);
        let back = decode_nullsupp(&bytes).unwrap();
        assert_eq!(back, vals);
    }

    #[test]
    fn badict_round_trips() {
        let vals = vec![b"alpha".to_vec(), b"beta".to_vec(), b"alpha".to_vec()];
        let bytes = encode_badict(&vals);
        let back = decode_badict(&bytes).unwrap();
        assert_eq!(back, vals);
    }

    #[test]
    fn sub_tag_round_trips() {
        for s in [
            SubTag::Plain,
            SubTag::BitPacked,
            SubTag::FrameOfRef,
            SubTag::Delta,
            SubTag::DeltaOfDelta,
            SubTag::NullSuppressed,
            SubTag::ConstantRun,
            SubTag::ZigZagVarint,
            SubTag::ByteArrayDict,
        ] {
            assert_eq!(SubTag::from_u8(s.as_u8()), Some(s));
        }
    }
}
