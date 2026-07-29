//! Checksums and hashes used across the storage layer.
//!
//! The container format checksums each directory body with a CRC-32 so that a
//! corrupt page is rejected at decode. This module gathers the integrity
//! primitives in one place: a table-driven CRC-32 (IEEE polynomial), the
//! Adler-32 rolling checksum, and a 64-bit FxHash-style mixer used by the
//! in-memory hash structures.

/// CRC-32 (IEEE 802.3 polynomial, reflected), computed with a precomputed
/// table built once on first use.
pub struct Crc32 {
    table: [u32; 256],
}

impl Crc32 {
    /// Build the lookup table.
    pub fn new() -> Crc32 {
        let mut table = [0u32; 256];
        let poly = 0xEDB88320u32;
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 == 1 {
                    crc = (crc >> 1) ^ poly;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        Crc32 { table }
    }

    /// Checksum a byte slice.
    pub fn checksum(&self, data: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFFu32;
        for &b in data {
            let idx = ((crc ^ b as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ self.table[idx];
        }
        crc ^ 0xFFFFFFFF
    }

    /// Incrementally update a running CRC with more data.
    pub fn update(&self, mut crc: u32, data: &[u8]) -> u32 {
        crc ^= 0xFFFFFFFF;
        for &b in data {
            let idx = ((crc ^ b as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ self.table[idx];
        }
        crc ^ 0xFFFFFFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Crc32::new()
    }
}

thread_local! {
    static CRC: Crc32 = Crc32::new();
}

/// Convenience: CRC-32 of a slice using a thread-local table.
pub fn crc32(data: &[u8]) -> u32 {
    CRC.with(|c| c.checksum(data))
}

/// Adler-32 rolling checksum.
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in data {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// A 64-bit FxHash-style mixing hasher (the rustc-internal style), useful for
/// hashing integer keys into buckets.
#[derive(Clone, Default)]
pub struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    /// A fresh hasher.
    pub fn new() -> FxHasher {
        FxHasher { hash: 0 }
    }

    /// Fold one 64-bit word into the state.
    pub fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }

    /// Fold a byte string into the state (8 bytes at a time).
    pub fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let word = u64::from_le_bytes(c.try_into().unwrap());
            self.write_u64(word);
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rem.len()].copy_from_slice(rem);
            self.write_u64(u64::from_le_bytes(buf));
        }
    }

    /// The current hash value.
    pub fn finish(&self) -> u64 {
        self.hash
    }
}

/// One-shot FxHash of a byte slice.
pub fn fxhash(bytes: &[u8]) -> u64 {
    let mut h = FxHasher::new();
    h.write(bytes);
    h.finish()
}

/// One-shot FxHash of a 64-bit integer.
pub fn fxhash_u64(i: u64) -> u64 {
    let mut h = FxHasher::new();
    h.write_u64(i);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vectors() {
        // CRC-32/IEEE of "123456789" is 0xCBF43926.
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
        assert_eq!(crc32(b""), 0x00000000);
    }

    #[test]
    fn crc32_incremental_matches_oneshot() {
        let c = Crc32::new();
        let data = b"the quick brown fox jumps over the lazy dog";
        let one = c.checksum(data);
        let mut crc = 0u32;
        crc = c.update(crc, &data[..10]);
        crc = c.update(crc, &data[10..]);
        assert_eq!(one, crc);
    }

    #[test]
    fn adler32_known_vector() {
        // Adler-32 of "Wikipedia" is 0x11E60398.
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn fxhash_is_deterministic_and_distinct() {
        assert_eq!(fxhash(b"hello"), fxhash(b"hello"));
        assert_ne!(fxhash(b"hello"), fxhash(b"world"));
        assert_ne!(fxhash_u64(1), fxhash_u64(2));
    }
}
