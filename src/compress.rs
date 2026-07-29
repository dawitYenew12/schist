//! Block compression codecs for page bodies and checkpoint segments.
//!
//! These are byte-oriented, general-purpose codecs, distinct from the
//! column-aware encodings in [`crate::encoding`] (which understand value
//! widths and null suppression). A page body or a checkpoint segment is an
//! opaque byte string; before it is written it can be run through one of these
//! codecs and framed with a small header recording the codec id and the
//! original length, so the reader can restore the exact bytes.
//!
//! Three codecs are provided:
//!
//! - [`Codec::Store`] — no compression, the identity transform.
//! - [`Codec::Rle`] — byte run-length encoding, good for sparse/zeroed pages.
//! - [`Codec::Lz`] — a small LZ77-style compressor with a fixed window, good
//!   for pages with repeated structure.
//!
//! The [`frame`] / [`unframe`] pair chooses nothing automatically; callers pick
//! a codec (or call [`compress_best`], which tries all three and keeps the
//! smallest). Every framed block round-trips exactly.

/// The compression codec used for a framed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// Identity: the block is stored verbatim.
    Store,
    /// Byte run-length encoding.
    Rle,
    /// LZ77-style sliding-window compression.
    Lz,
}

impl Codec {
    /// The on-disk tag byte.
    pub fn tag(self) -> u8 {
        match self {
            Codec::Store => 0,
            Codec::Rle => 1,
            Codec::Lz => 2,
        }
    }

    /// Parse a tag byte.
    pub fn from_tag(b: u8) -> Option<Codec> {
        match b {
            0 => Some(Codec::Store),
            1 => Some(Codec::Rle),
            2 => Some(Codec::Lz),
            _ => None,
        }
    }
}

/// Error returned when a framed block cannot be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompressError {
    /// The frame header was truncated.
    ShortHeader,
    /// The codec tag byte was not recognized.
    BadCodec(u8),
    /// The compressed body was truncated or malformed.
    Truncated,
    /// The decompressed length did not match the framed original length.
    LengthMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for DecompressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressError::ShortHeader => write!(f, "short compression frame header"),
            DecompressError::BadCodec(b) => write!(f, "unknown codec tag {b}"),
            DecompressError::Truncated => write!(f, "truncated compressed body"),
            DecompressError::LengthMismatch { expected, got } => {
                write!(f, "decompressed length {got} != framed {expected}")
            }
        }
    }
}

impl std::error::Error for DecompressError {}

const FRAME_MAGIC: u8 = 0xC5;

/// Frame a compressed block: `[magic][codec][u32 orig_len][body]`.
pub fn frame(codec: Codec, input: &[u8]) -> Vec<u8> {
    let body = match codec {
        Codec::Store => input.to_vec(),
        Codec::Rle => rle_compress(input),
        Codec::Lz => lz_compress(input),
    };
    let mut out = Vec::with_capacity(body.len() + 6);
    out.push(FRAME_MAGIC);
    out.push(codec.tag());
    out.extend_from_slice(&(input.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Compress with every codec and keep the smallest frame.
pub fn compress_best(input: &[u8]) -> Vec<u8> {
    let candidates = [
        frame(Codec::Store, input),
        frame(Codec::Rle, input),
        frame(Codec::Lz, input),
    ];
    candidates
        .into_iter()
        .min_by_key(|c| c.len())
        .unwrap_or_default()
}

/// Decode a framed block back to the original bytes.
pub fn unframe(input: &[u8]) -> Result<Vec<u8>, DecompressError> {
    if input.len() < 6 {
        return Err(DecompressError::ShortHeader);
    }
    if input[0] != FRAME_MAGIC {
        return Err(DecompressError::ShortHeader);
    }
    let codec = Codec::from_tag(input[1]).ok_or(DecompressError::BadCodec(input[1]))?;
    let mut lenbuf = [0u8; 4];
    lenbuf.copy_from_slice(&input[2..6]);
    let orig_len = u32::from_le_bytes(lenbuf) as usize;
    let body = &input[6..];
    let out = match codec {
        Codec::Store => body.to_vec(),
        Codec::Rle => rle_decompress(body)?,
        Codec::Lz => lz_decompress(body, orig_len)?,
    };
    if out.len() != orig_len {
        return Err(DecompressError::LengthMismatch {
            expected: orig_len,
            got: out.len(),
        });
    }
    Ok(out)
}

// ----- Byte RLE ------------------------------------------------------------

/// Byte run-length encoding.
///
/// Encoding is a sequence of tokens. A control byte `c`:
/// - `c < 128`: a literal run — the next `c + 1` bytes are copied verbatim.
/// - `c >= 128`: a repeat run — the following byte is repeated `c - 128 + 3`
///   times (repeats are only emitted for runs of length >= 3).
pub fn rle_compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    let n = input.len();
    let mut literal_start = 0usize;

    let flush_literals = |out: &mut Vec<u8>, input: &[u8], from: usize, to: usize| {
        let mut s = from;
        while s < to {
            let chunk = (to - s).min(128);
            out.push((chunk - 1) as u8);
            out.extend_from_slice(&input[s..s + chunk]);
            s += chunk;
        }
    };

    while i < n {
        // Measure the run length at i.
        let b = input[i];
        let mut run = 1;
        while i + run < n && input[i + run] == b && run < (127 + 3) {
            run += 1;
        }
        if run >= 3 {
            // Flush pending literals, then emit a repeat.
            flush_literals(&mut out, input, literal_start, i);
            out.push(128 + (run - 3) as u8);
            out.push(b);
            i += run;
            literal_start = i;
        } else {
            i += 1;
        }
    }
    flush_literals(&mut out, input, literal_start, n);
    out
}

/// Inverse of [`rle_compress`].
pub fn rle_decompress(input: &[u8]) -> Result<Vec<u8>, DecompressError> {
    let mut out = Vec::new();
    let mut i = 0;
    let n = input.len();
    while i < n {
        let c = input[i];
        i += 1;
        if c < 128 {
            let count = c as usize + 1;
            if i + count > n {
                return Err(DecompressError::Truncated);
            }
            out.extend_from_slice(&input[i..i + count]);
            i += count;
        } else {
            let count = (c - 128) as usize + 3;
            if i >= n {
                return Err(DecompressError::Truncated);
            }
            let b = input[i];
            i += 1;
            out.extend(std::iter::repeat(b).take(count));
        }
    }
    Ok(out)
}

// ----- LZ77 ----------------------------------------------------------------

const LZ_WINDOW: usize = 4096;
const LZ_MIN_MATCH: usize = 4;
const LZ_MAX_MATCH: usize = 255 + LZ_MIN_MATCH;

/// A small LZ77-style compressor.
///
/// The token stream alternates a control byte with 8 following units. Each bit
/// of the control byte (LSB first) says whether the corresponding unit is a
/// literal (0) or a back-reference (1). A literal is one byte. A back-reference
/// is a 2-byte little-endian distance followed by a 1-byte `(len - MIN_MATCH)`.
pub fn lz_compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let n = input.len();
    let mut i = 0;
    while i < n {
        let mut control_idx = out.len();
        out.push(0u8);
        let mut control = 0u8;
        for bit in 0..8 {
            if i >= n {
                break;
            }
            let (dist, len) = longest_match(input, i);
            if len >= LZ_MIN_MATCH {
                control |= 1 << bit;
                out.extend_from_slice(&(dist as u16).to_le_bytes());
                out.push((len - LZ_MIN_MATCH) as u8);
                i += len;
            } else {
                out.push(input[i]);
                i += 1;
            }
        }
        out[control_idx] = control;
        let _ = &mut control_idx;
    }
    out
}

/// Find the longest back-reference for the suffix starting at `pos`.
fn longest_match(input: &[u8], pos: usize) -> (usize, usize) {
    let n = input.len();
    let window_start = pos.saturating_sub(LZ_WINDOW);
    let max_len = (n - pos).min(LZ_MAX_MATCH);
    if max_len < LZ_MIN_MATCH {
        return (0, 0);
    }
    let mut best_len = 0;
    let mut best_dist = 0;
    let mut cand = window_start;
    while cand < pos {
        let mut l = 0;
        while l < max_len && input[cand + l] == input[pos + l] {
            l += 1;
        }
        if l > best_len {
            best_len = l;
            best_dist = pos - cand;
            if l == max_len {
                break;
            }
        }
        cand += 1;
    }
    (best_dist, best_len)
}

/// Inverse of [`lz_compress`].
pub fn lz_decompress(input: &[u8], hint: usize) -> Result<Vec<u8>, DecompressError> {
    let mut out = Vec::with_capacity(hint);
    let mut i = 0;
    let n = input.len();
    while i < n {
        let control = input[i];
        i += 1;
        for bit in 0..8 {
            if i >= n {
                break;
            }
            let is_ref = (control >> bit) & 1 == 1;
            if is_ref {
                if i + 3 > n {
                    return Err(DecompressError::Truncated);
                }
                let dist = u16::from_le_bytes([input[i], input[i + 1]]) as usize;
                let len = input[i + 2] as usize + LZ_MIN_MATCH;
                i += 3;
                if dist == 0 || dist > out.len() {
                    return Err(DecompressError::Truncated);
                }
                let start = out.len() - dist;
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            } else {
                out.push(input[i]);
                i += 1;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(codec: Codec, data: &[u8]) {
        let framed = frame(codec, data);
        let back = unframe(&framed).expect("unframe");
        assert_eq!(back, data, "codec {codec:?} failed round trip");
    }

    #[test]
    fn store_roundtrip() {
        roundtrip(Codec::Store, b"hello world");
        roundtrip(Codec::Store, &[]);
    }

    #[test]
    fn rle_roundtrip() {
        roundtrip(Codec::Rle, &[0u8; 100]);
        roundtrip(Codec::Rle, b"aaaaabbbbbcccccabcabcabc");
        roundtrip(Codec::Rle, b"no repeats here!");
        let mut mixed = vec![7u8; 50];
        mixed.extend_from_slice(b"literals");
        mixed.extend(std::iter::repeat(9u8).take(40));
        roundtrip(Codec::Rle, &mixed);
    }

    #[test]
    fn rle_shrinks_runs() {
        let data = vec![0u8; 1000];
        let framed = frame(Codec::Rle, &data);
        assert!(framed.len() < 100, "rle should shrink a zeroed block");
    }

    #[test]
    fn lz_roundtrip() {
        roundtrip(Codec::Lz, b"abcabcabcabcabcabcabc");
        roundtrip(Codec::Lz, b"the quick brown fox the quick brown fox");
        roundtrip(Codec::Lz, &[]);
        let mut big = Vec::new();
        for _ in 0..200 {
            big.extend_from_slice(b"schist columnar ");
        }
        roundtrip(Codec::Lz, &big);
    }

    #[test]
    fn lz_shrinks_repetitive() {
        let mut data = Vec::new();
        for _ in 0..500 {
            data.extend_from_slice(b"0123456789");
        }
        let framed = frame(Codec::Lz, &data);
        assert!(framed.len() < data.len() / 2);
    }

    #[test]
    fn compress_best_picks_smallest() {
        let data = vec![42u8; 4096];
        let best = compress_best(&data);
        assert!(best.len() < 64);
        assert_eq!(unframe(&best).unwrap(), data);
    }

    #[test]
    fn unframe_rejects_garbage() {
        assert!(unframe(&[]).is_err());
        assert!(unframe(&[0, 0, 0, 0, 0, 0]).is_err());
        assert!(matches!(
            unframe(&[FRAME_MAGIC, 9, 0, 0, 0, 0]),
            Err(DecompressError::BadCodec(9))
        ));
    }
}
