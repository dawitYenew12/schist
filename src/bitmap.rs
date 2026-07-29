//! A bitmap index.
//!
//! A bitmap index stores, for each distinct value in a column, a bitmap whose
//! `i`-th bit is set iff row `i` carries that value. Bitmap indexes are ideal
//! for low-cardinality columns and for combining multiple equality predicates
//! with bitwise AND/OR. This implementation stores bitmaps as run-length-
//! encoded word arrays (a simple WAH-like scheme) so sparse bitmaps stay small.

/// A 64-bit word. The high bit is a flag; the low 63 bits are either a literal
/// run of bits or a fill-run count.
pub const FLAG_BIT: u64 = 1u64 << 63;
pub const LITERAL_MAX: u64 = u64::MAX >> 1;

/// A compressed bitmap.
#[derive(Debug, Clone, Default)]
pub struct Bitmap {
    words: Vec<u64>,
    /// The logical bit length (number of rows this bitmap covers).
    len: u64,
}

impl Bitmap {
    pub fn new() -> Bitmap {
        Bitmap::default()
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Build a bitmap from a list of set-bit positions, up to `len` rows.
    pub fn from_set_bits(set: &[u64], len: u64) -> Bitmap {
        let mut b = Bitmap::new();
        b.len = len;
        let words_needed = (len + 63) / 64;
        let mut lit = vec![0u64; words_needed as usize];
        for &pos in set {
            if pos < len {
                lit[(pos / 64) as usize] |= 1u64 << (pos % 64);
            }
        }
        b.words = compress(&lit);
        b
    }

    /// Materialize the bitmap into a flat set of set-bit positions.
    pub fn to_set_bits(&self) -> Vec<u64> {
        let lit = decompress(&self.words, self.len);
        let mut out = Vec::new();
        for (wi, &w) in lit.iter().enumerate() {
            for b in 0..64 {
                if (w >> b) & 1 == 1 {
                    let pos = (wi as u64) * 64 + b;
                    if pos < self.len {
                        out.push(pos);
                    }
                }
            }
        }
        out
    }

    /// Bitwise AND of two bitmaps (intersection of row sets).
    pub fn and(&self, other: &Bitmap) -> Bitmap {
        let len = self.len.min(other.len);
        let a = decompress(&self.words, len);
        let b = decompress(&other.words, len);
        let mut lit = vec![0u64; a.len().max(b.len())];
        for i in 0..lit.len() {
            let x = a.get(i).copied().unwrap_or(0);
            let y = b.get(i).copied().unwrap_or(0);
            lit[i] = x & y;
        }
        let mut out = Bitmap::new();
        out.len = len;
        out.words = compress(&lit);
        out
    }

    /// Bitwise OR of two bitmaps (union of row sets).
    pub fn or(&self, other: &Bitmap) -> Bitmap {
        let len = self.len.max(other.len);
        let a = decompress(&self.words, len);
        let b = decompress(&other.words, len);
        let mut lit = vec![0u64; a.len().max(b.len())];
        for i in 0..lit.len() {
            let x = a.get(i).copied().unwrap_or(0);
            let y = b.get(i).copied().unwrap_or(0);
            lit[i] = x | y;
        }
        let mut out = Bitmap::new();
        out.len = len;
        out.words = compress(&lit);
        out
    }

    /// Bitwise NOT (complement), up to `len` rows.
    pub fn not(&self) -> Bitmap {
        let lit = decompress(&self.words, self.len);
        let mut out_lit = vec![0u64; lit.len()];
        for (i, &w) in lit.iter().enumerate() {
            out_lit[i] = !w & LITERAL_MAX;
        }
        let mut out = Bitmap::new();
        out.len = self.len;
        out.words = compress(&out_lit);
        out
    }

    /// Population count (number of set bits).
    pub fn popcount(&self) -> u64 {
        let lit = decompress(&self.words, self.len);
        let mut total = 0u64;
        for (i, &w) in lit.iter().enumerate() {
            let mut c = w.count_ones() as u64;
            // Mask off bits beyond len in the last word.
            if (i as u64 + 1) * 64 > self.len {
                let excess = (i as u64 + 1) * 64 - self.len;
                c = c.saturating_sub(excess);
            }
            total += c;
        }
        total
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&(self.words.len() as u32).to_le_bytes());
        for &w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Option<Bitmap> {
        if buf.len() < 12 {
            return None;
        }
        let len = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let n = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let mut words = Vec::with_capacity(n);
        let mut pos = 12;
        for _ in 0..n {
            if pos + 8 > buf.len() {
                break;
            }
            words.push(u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()));
            pos += 8;
        }
        Some(Bitmap { words, len })
    }
}

/// Compress a literal word array into a WAH-like run-length form. A word whose
/// high bit is 0 is a literal. A word whose high bit is 1 encodes a fill run:
/// bit 62 is the fill value, bits 0..61 are the run length (number of words).
fn compress(lit: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lit.len() {
        let w = lit[i] & LITERAL_MAX;
        let all_zero = w == 0;
        let all_one = w == LITERAL_MAX;
        if all_zero || all_one {
            let mut run = 1usize;
            while i + run < lit.len()
                && (lit[i + run] & LITERAL_MAX) == w
                && run < (u64::MAX >> 2) as usize
            {
                run += 1;
            }
            let fill_val = if all_one { 1u64 << 62 } else { 0u64 };
            out.push(FLAG_BIT | fill_val | run as u64);
            i += run;
        } else {
            out.push(w);
            i += 1;
        }
    }
    out
}

/// Decompress back to a literal word array of `ceil(len/64)` words.
fn decompress(words: &[u64], len: u64) -> Vec<u64> {
    let words_needed = ((len + 63) / 64) as usize;
    let mut out = vec![0u64; words_needed];
    let mut out_idx = 0usize;
    for &w in words {
        if w & FLAG_BIT != 0 {
            let fill_val = (w >> 62) & 1;
            let run = (w & ((1u64 << 62) - 1)) as usize;
            for _ in 0..run {
                if out_idx >= out.len() {
                    break;
                }
                out[out_idx] = if fill_val == 1 { LITERAL_MAX } else { 0 };
                out_idx += 1;
            }
        } else {
            if out_idx < out.len() {
                out[out_idx] = w;
                out_idx += 1;
            }
        }
    }
    out
}

/// A bitmap index: `value -> Bitmap`.
#[derive(Debug, Clone)]
pub struct BitmapIndex {
    pub maps: std::collections::BTreeMap<i64, Bitmap>,
    pub row_count: u64,
}

impl BitmapIndex {
    pub fn new(row_count: u64) -> BitmapIndex {
        BitmapIndex {
            maps: std::collections::BTreeMap::new(),
            row_count,
        }
    }

    /// Build a bitmap index from a column of `(row_index, value)` pairs.
    pub fn build(rows: &[(u64, i64)], row_count: u64) -> BitmapIndex {
        let mut groups: std::collections::BTreeMap<i64, Vec<u64>> = std::collections::BTreeMap::new();
        for &(pos, v) in rows {
            groups.entry(v).or_default().push(pos);
        }
        let mut idx = BitmapIndex::new(row_count);
        for (v, set) in groups {
            idx.maps.insert(v, Bitmap::from_set_bits(&set, row_count));
        }
        idx
    }

    /// Look up the bitmap for a value.
    pub fn get(&self, value: i64) -> Option<&Bitmap> {
        self.maps.get(&value)
    }

    /// The row positions matching `value`.
    pub fn lookup(&self, value: i64) -> Vec<u64> {
        self.get(value).map_or(Vec::new(), |b| b.to_set_bits())
    }

    /// Rows matching `value1 AND value2` (intersection).
    pub fn lookup_and(&self, v1: i64, v2: i64) -> Vec<u64> {
        match (self.get(v1), self.get(v2)) {
            (Some(a), Some(b)) => a.and(b).to_set_bits(),
            _ => Vec::new(),
        }
    }

    /// Rows matching `value1 OR value2` (union).
    pub fn lookup_or(&self, v1: i64, v2: i64) -> Vec<u64> {
        let empty = Bitmap::new();
        let a = self.get(v1).unwrap_or(&empty);
        let b = self.get(v2).unwrap_or(&empty);
        a.or(b).to_set_bits()
    }

    pub fn cardinality(&self) -> usize {
        self.maps.len()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.row_count.to_le_bytes());
        out.extend_from_slice(&(self.maps.len() as u32).to_le_bytes());
        for (&v, b) in &self.maps {
            out.extend_from_slice(&v.to_le_bytes());
            let enc = b.encode();
            out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
            out.extend_from_slice(&enc);
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Option<BitmapIndex> {
        if buf.len() < 12 {
            return None;
        }
        let row_count = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let n = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let mut maps = std::collections::BTreeMap::new();
        let mut pos = 12;
        for _ in 0..n {
            if pos + 12 > buf.len() {
                break;
            }
            let v = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            let l = u32::from_le_bytes(buf[pos + 8..pos + 12].try_into().unwrap()) as usize;
            pos += 12;
            if pos + l > buf.len() {
                break;
            }
            let b = Bitmap::decode(&buf[pos..pos + l])?;
            maps.insert(v, b);
            pos += l;
        }
        Some(BitmapIndex { maps, row_count })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmap_set_and_get() {
        let b = Bitmap::from_set_bits(&[0, 3, 7, 100], 200);
        assert_eq!(b.to_set_bits(), vec![0, 3, 7, 100]);
        assert_eq!(b.popcount(), 4);
    }

    #[test]
    fn bitmap_and_or() {
        let a = Bitmap::from_set_bits(&[1, 2, 3, 4], 64);
        let b = Bitmap::from_set_bits(&[2, 3, 5], 64);
        assert_eq!(a.and(&b).to_set_bits(), vec![2, 3]);
        assert_eq!(a.or(&b).to_set_bits(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn bitmap_compression_fill_runs() {
        let mut lit = vec![0u64; 200];
        for i in 0..200 {
            lit[i] = LITERAL_MAX;
        }
        let comp = compress(&lit);
        assert!(comp.len() < lit.len(), "fill runs should compress");
        let back = decompress(&comp, 200 * 64);
        assert_eq!(back, lit);
    }

    #[test]
    fn bitmap_index_build_lookup() {
        let rows = vec![(0, 10), (1, 20), (2, 10), (3, 30), (4, 10)];
        let idx = BitmapIndex::build(&rows, 5);
        assert_eq!(idx.lookup(10), vec![0, 2, 4]);
        assert_eq!(idx.lookup_and(10, 30), Vec::<u64>::new());
        assert_eq!(idx.lookup_or(10, 20), vec![0, 1, 2, 4]);
    }

    #[test]
    fn bitmap_index_encode_decode() {
        let rows = vec![(0, 1), (1, 1), (2, 2)];
        let idx = BitmapIndex::build(&rows, 3);
        let bytes = idx.encode();
        let idx2 = BitmapIndex::decode(&bytes).unwrap();
        assert_eq!(idx2.lookup(1), vec![0, 1]);
        assert_eq!(idx2.lookup(2), vec![2]);
    }
}
