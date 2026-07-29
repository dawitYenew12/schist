//! Rabin-Karp rolling hashes and content-defined chunking.
//!
//! A rolling hash lets a fixed-width window slide over a byte stream while its
//! hash is updated in O(1) per step, which powers substring search (Rabin-Karp)
//! and content-defined chunking (splitting a stream at positions where the hash
//! matches a mask, so insertions shift only nearby boundaries). This module
//! provides both, plus a whole-buffer polynomial fingerprint used to key cached
//! page contents.

/// A polynomial rolling hash over a fixed window width.
#[derive(Debug, Clone)]
pub struct RollingHash {
    base: u64,
    modulus: u64,
    /// `base^(width-1) mod modulus`, used to remove the leaving byte.
    top_power: u64,
    width: usize,
    hash: u64,
    filled: usize,
}

const DEFAULT_BASE: u64 = 257;
const DEFAULT_MOD: u64 = 1_000_000_007;

impl RollingHash {
    /// A rolling hash for a window of `width` bytes.
    pub fn new(width: usize) -> RollingHash {
        RollingHash::with_params(width, DEFAULT_BASE, DEFAULT_MOD)
    }

    /// A rolling hash with explicit base and modulus.
    pub fn with_params(width: usize, base: u64, modulus: u64) -> RollingHash {
        let width = width.max(1);
        let mut top_power = 1u64;
        for _ in 0..width - 1 {
            top_power = (top_power * base) % modulus;
        }
        RollingHash {
            base,
            modulus,
            top_power,
            width,
            hash: 0,
            filled: 0,
        }
    }

    /// The window width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The current hash value.
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// `true` once at least `width` bytes have been fed.
    pub fn is_full(&self) -> bool {
        self.filled >= self.width
    }

    /// Push one byte into the window (before it is full); use [`roll`] once full.
    pub fn push(&mut self, byte: u8) {
        self.hash = (self.hash * self.base + byte as u64) % self.modulus;
        self.filled += 1;
    }

    /// Slide the window: add `incoming`, remove `outgoing` (the byte that was
    /// `width` positions back).
    pub fn roll(&mut self, outgoing: u8, incoming: u8) {
        let out = (outgoing as u64 * self.top_power) % self.modulus;
        self.hash = (self.hash + self.modulus - out) % self.modulus;
        self.hash = (self.hash * self.base + incoming as u64) % self.modulus;
    }

    /// Hash a full window from scratch.
    pub fn hash_window(&self, window: &[u8]) -> u64 {
        let mut h = 0u64;
        for &b in window.iter().take(self.width) {
            h = (h * self.base + b as u64) % self.modulus;
        }
        h
    }
}

/// Find all start offsets where `needle` occurs in `haystack` (Rabin-Karp).
pub fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    if needle.is_empty() || needle.len() > haystack.len() {
        return out;
    }
    let width = needle.len();
    let rh = RollingHash::new(width);
    let needle_hash = rh.hash_window(needle);

    let mut window = RollingHash::new(width);
    for &b in &haystack[..width] {
        window.push(b);
    }
    if window.hash() == needle_hash && &haystack[..width] == needle {
        out.push(0);
    }
    for i in width..haystack.len() {
        window.roll(haystack[i - width], haystack[i]);
        let start = i - width + 1;
        if window.hash() == needle_hash && &haystack[start..=i] == needle {
            out.push(start);
        }
    }
    out
}

/// `true` if `needle` occurs anywhere in `haystack`.
pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    !find_all(haystack, needle).is_empty()
}

/// A polynomial fingerprint of a whole buffer.
pub fn fingerprint(data: &[u8]) -> u64 {
    let mut h = 1469598103934665603u64; // FNV offset basis
    for &b in data {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    h
}

/// Content-defined chunk boundaries: split `data` where the rolling hash over a
/// `window`-byte window has its low `mask_bits` bits zero, subject to
/// `[min_size, max_size]` bounds. Returns the end offset of each chunk.
pub fn content_defined_chunks(
    data: &[u8],
    window: usize,
    mask_bits: u32,
    min_size: usize,
    max_size: usize,
) -> Vec<usize> {
    let mut boundaries = Vec::new();
    if data.is_empty() {
        return boundaries;
    }
    let mask = (1u64 << mask_bits) - 1;
    let mut rh = RollingHash::new(window);
    let mut chunk_start = 0usize;
    for (i, &b) in data.iter().enumerate() {
        if rh.is_full() {
            rh.roll(data[i - window], b);
        } else {
            rh.push(b);
        }
        let size = i - chunk_start + 1;
        let hit = rh.is_full() && (rh.hash() & mask) == 0 && size >= min_size;
        if hit || size >= max_size {
            boundaries.push(i + 1);
            chunk_start = i + 1;
        }
    }
    if chunk_start < data.len() {
        boundaries.push(data.len());
    }
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_matches_fresh_hash() {
        let data = b"abcdefghij";
        let width = 4;
        let mut rh = RollingHash::new(width);
        for &b in &data[..width] {
            rh.push(b);
        }
        assert_eq!(rh.hash(), rh.hash_window(&data[..width]));
        for i in width..data.len() {
            rh.roll(data[i - width], data[i]);
            let start = i - width + 1;
            assert_eq!(rh.hash(), rh.hash_window(&data[start..=i]));
        }
    }

    #[test]
    fn rabin_karp_search() {
        let hay = b"the quick brown fox the quick";
        assert_eq!(find_all(hay, b"quick"), vec![4, 24]);
        assert_eq!(find_all(hay, b"the"), vec![0, 20]);
        assert!(find_all(hay, b"zebra").is_empty());
        assert!(contains(hay, b"brown"));
    }

    #[test]
    fn search_edge_cases() {
        assert!(find_all(b"", b"x").is_empty());
        assert!(find_all(b"ab", b"abc").is_empty());
        assert_eq!(find_all(b"aaaa", b"aa"), vec![0, 1, 2]);
    }

    #[test]
    fn fingerprint_distinguishes() {
        assert_eq!(fingerprint(b"hello"), fingerprint(b"hello"));
        assert_ne!(fingerprint(b"hello"), fingerprint(b"hellp"));
        assert_ne!(fingerprint(b""), fingerprint(b"a"));
    }

    #[test]
    fn chunking_covers_data() {
        let data: Vec<u8> = (0..2000).map(|i| (i * 7 % 251) as u8).collect();
        let bounds = content_defined_chunks(&data, 16, 6, 32, 512);
        // Boundaries strictly increasing and ending at data.len().
        assert!(bounds.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*bounds.last().unwrap(), data.len());
        // Every chunk within max_size.
        let mut prev = 0;
        for &b in &bounds {
            assert!(b - prev <= 512);
            prev = b;
        }
    }

    #[test]
    fn chunking_is_stable_to_prefix_insert() {
        // Inserting bytes near the front should not change most later boundaries.
        let base: Vec<u8> = (0..3000).map(|i| (i * 13 % 251) as u8).collect();
        let b1 = content_defined_chunks(&base, 16, 5, 16, 256);
        let mut modified = vec![9u8, 9, 9];
        modified.extend_from_slice(&base);
        let b2 = content_defined_chunks(&modified, 16, 5, 16, 256);
        // The two boundary sets should share many cut points (shifted by 3).
        let shifted: std::collections::HashSet<usize> = b1.iter().map(|&x| x + 3).collect();
        let common = b2.iter().filter(|b| shifted.contains(b)).count();
        assert!(common > b1.len() / 2, "chunking not stable enough");
    }
}
