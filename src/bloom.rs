//! Bloom and counting-bloom filters for page/segment pruning.
//!
//! A per-page bloom filter lets a point-lookup skip a page whose filter says a
//! value is definitely absent, without touching the page body. The filters are
//! sized from an expected element count and a target false-positive rate, and
//! use double hashing (two 64-bit hashes combined) to synthesize `k` probe
//! positions, which is the standard Kirsch–Mitzenmacher construction.
//!
//! The [`CountingBloom`] variant stores 4-bit counters instead of single bits,
//! so entries can be removed (a deletion decrements the counters); it is used
//! where a page's live set shrinks over its lifetime.

/// A classic bit-array bloom filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bloom {
    bits: Vec<u64>,
    nbits: usize,
    k: u32,
}

impl Bloom {
    /// Build a filter sized for `expected` elements at false-positive rate `fpr`
    /// (clamped to a sane range).
    pub fn with_capacity(expected: usize, fpr: f64) -> Bloom {
        let expected = expected.max(1);
        let fpr = fpr.clamp(1e-6, 0.5);
        // m = -n ln p / (ln 2)^2 ; k = m/n ln 2
        let ln2 = std::f64::consts::LN_2;
        let m = (-(expected as f64) * fpr.ln() / (ln2 * ln2)).ceil() as usize;
        let m = m.max(64);
        let k = ((m as f64 / expected as f64) * ln2).round().max(1.0) as u32;
        let words = m.div_ceil(64);
        Bloom {
            bits: vec![0u64; words],
            nbits: words * 64,
            k: k.min(30),
        }
    }

    /// Build with explicit sizing.
    pub fn new(nbits: usize, k: u32) -> Bloom {
        let words = nbits.div_ceil(64).max(1);
        Bloom {
            bits: vec![0u64; words],
            nbits: words * 64,
            k: k.max(1).min(30),
        }
    }

    /// Number of hash probes.
    pub fn hashes(&self) -> u32 {
        self.k
    }

    /// Number of addressable bits.
    pub fn bit_capacity(&self) -> usize {
        self.nbits
    }

    fn positions(&self, key: u64) -> impl Iterator<Item = usize> + '_ {
        let h1 = splitmix64(key);
        let h2 = splitmix64(h1 ^ 0x9E3779B97F4A7C15) | 1;
        (0..self.k as u64).map(move |i| {
            let combined = h1.wrapping_add(i.wrapping_mul(h2));
            (combined % self.nbits as u64) as usize
        })
    }

    /// Insert a 64-bit key.
    pub fn insert(&mut self, key: u64) {
        for p in self.positions(key).collect::<Vec<_>>() {
            self.bits[p >> 6] |= 1u64 << (p & 63);
        }
    }

    /// Insert an integer element.
    pub fn insert_i64(&mut self, v: i64) {
        self.insert(v as u64);
    }

    /// Insert a byte string.
    pub fn insert_bytes(&mut self, b: &[u8]) {
        self.insert(fnv1a(b));
    }

    /// `true` if the key may be present (`false` means definitely absent).
    pub fn contains(&self, key: u64) -> bool {
        self.positions(key).all(|p| self.bits[p >> 6] & (1u64 << (p & 63)) != 0)
    }

    /// Membership query for an integer element.
    pub fn contains_i64(&self, v: i64) -> bool {
        self.contains(v as u64)
    }

    /// Membership query for a byte string.
    pub fn contains_bytes(&self, b: &[u8]) -> bool {
        self.contains(fnv1a(b))
    }

    /// Estimated number of set bits, as a saturation measure.
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Union another filter of identical geometry into this one.
    pub fn union(&mut self, other: &Bloom) -> bool {
        if self.nbits != other.nbits || self.k != other.k {
            return false;
        }
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
        true
    }

    /// Serialize to bytes: `[u32 nbits][u32 k][words...]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.bits.len() * 8);
        out.extend_from_slice(&(self.nbits as u32).to_le_bytes());
        out.extend_from_slice(&self.k.to_le_bytes());
        for w in &self.bits {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Deserialize a filter produced by [`encode`](Self::encode).
    pub fn decode(buf: &[u8]) -> Option<Bloom> {
        if buf.len() < 8 {
            return None;
        }
        let nbits = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
        let k = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        let words = nbits.div_ceil(64);
        if buf.len() < 8 + words * 8 {
            return None;
        }
        let mut bits = Vec::with_capacity(words);
        for i in 0..words {
            let off = 8 + i * 8;
            bits.push(u64::from_le_bytes(buf[off..off + 8].try_into().ok()?));
        }
        Some(Bloom {
            bits,
            nbits: words * 64,
            k,
        })
    }
}

/// A counting bloom filter with 4-bit saturating counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountingBloom {
    /// Two counters per byte (low nibble = even slot, high nibble = odd slot).
    counters: Vec<u8>,
    nslots: usize,
    k: u32,
}

impl CountingBloom {
    /// Build sized for `expected` elements at rate `fpr`.
    pub fn with_capacity(expected: usize, fpr: f64) -> CountingBloom {
        let template = Bloom::with_capacity(expected, fpr);
        let nslots = template.nbits;
        CountingBloom {
            counters: vec![0u8; nslots.div_ceil(2)],
            nslots,
            k: template.k,
        }
    }

    fn get(&self, slot: usize) -> u8 {
        let byte = self.counters[slot >> 1];
        if slot & 1 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    fn adjust(&mut self, slot: usize, up: bool) {
        let idx = slot >> 1;
        let byte = self.counters[idx];
        let (mut lo, mut hi) = (byte & 0x0F, byte >> 4);
        let target = if slot & 1 == 0 { &mut lo } else { &mut hi };
        if up {
            if *target < 15 {
                *target += 1;
            }
        } else if *target > 0 {
            *target -= 1;
        }
        self.counters[idx] = (hi << 4) | lo;
    }

    fn positions(&self, key: u64) -> Vec<usize> {
        let h1 = splitmix64(key);
        let h2 = splitmix64(h1 ^ 0x9E3779B97F4A7C15) | 1;
        (0..self.k as u64)
            .map(|i| {
                let combined = h1.wrapping_add(i.wrapping_mul(h2));
                (combined % self.nslots as u64) as usize
            })
            .collect()
    }

    /// Insert a key.
    pub fn insert(&mut self, key: u64) {
        for p in self.positions(key) {
            self.adjust(p, true);
        }
    }

    /// Remove a key (decrements counters; a no-op for an absent key up to
    /// counter saturation).
    pub fn remove(&mut self, key: u64) {
        for p in self.positions(key) {
            self.adjust(p, false);
        }
    }

    /// `true` if the key may be present.
    pub fn contains(&self, key: u64) -> bool {
        self.positions(key).into_iter().all(|p| self.get(p) > 0)
    }
}

/// The SplitMix64 finalizer — a fast, good-quality integer hash.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// FNV-1a hash of a byte string.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xCBF29CE484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let mut b = Bloom::with_capacity(1000, 0.01);
        for i in 0..1000i64 {
            b.insert_i64(i * 7);
        }
        for i in 0..1000i64 {
            assert!(b.contains_i64(i * 7), "missing {i}");
        }
    }

    #[test]
    fn fpr_is_reasonable() {
        let mut b = Bloom::with_capacity(2000, 0.01);
        for i in 0..2000i64 {
            b.insert_i64(i);
        }
        let mut fp = 0;
        for i in 10_000i64..12_000 {
            if b.contains_i64(i) {
                fp += 1;
            }
        }
        // Allow generous slack over the 1% target.
        assert!(fp < 100, "false positive rate too high: {fp}/2000");
    }

    #[test]
    fn bytes_membership() {
        let mut b = Bloom::with_capacity(16, 0.01);
        b.insert_bytes(b"alpha");
        b.insert_bytes(b"beta");
        assert!(b.contains_bytes(b"alpha"));
        assert!(b.contains_bytes(b"beta"));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let mut b = Bloom::with_capacity(64, 0.02);
        for i in 0..64i64 {
            b.insert_i64(i);
        }
        let bytes = b.encode();
        let b2 = Bloom::decode(&bytes).unwrap();
        assert_eq!(b, b2);
    }

    #[test]
    fn union_merges() {
        let mut a = Bloom::new(1024, 4);
        let mut b = Bloom::new(1024, 4);
        a.insert(1);
        b.insert(2);
        assert!(a.union(&b));
        assert!(a.contains(1));
        assert!(a.contains(2));
    }

    #[test]
    fn counting_remove() {
        let mut c = CountingBloom::with_capacity(100, 0.01);
        c.insert(42);
        assert!(c.contains(42));
        c.remove(42);
        assert!(!c.contains(42));
    }
}
