//! LSD radix sort for integer and dictionary-id keys.
//!
//! Sorting a column of integers or dictionary ids is a hot path for `ORDER BY`,
//! sort-merge join, and run generation in external sort. For fixed-width integer
//! keys, a least-significant-digit radix sort beats comparison sorts: it makes a
//! fixed number of counting-sort passes (one per byte), is stable, and touches
//! each element a constant number of times. This module sorts `u64`/`i64` keys
//! and key/payload pairs, handling signed keys by flipping the sign bit so the
//! unsigned byte order matches the signed order.

const RADIX: usize = 256;

/// Stable LSD radix sort of unsigned 64-bit keys.
///
/// Runs all eight byte passes so that the ping-pong buffering ends with the
/// result back in the caller's slice (an even number of swaps), avoiding a
/// final copy.
pub fn sort_u64(keys: &mut [u64]) {
    let n = keys.len();
    if n < 2 {
        return;
    }
    let mut buffer = vec![0u64; n];
    let mut src = keys;
    let mut dst = &mut buffer[..];
    for byte in 0..8 {
        let shift = byte * 8;
        let mut counts = [0usize; RADIX + 1];
        for &k in src.iter() {
            let bucket = ((k >> shift) & 0xFF) as usize;
            counts[bucket + 1] += 1;
        }
        for i in 0..RADIX {
            counts[i + 1] += counts[i];
        }
        for &k in src.iter() {
            let bucket = ((k >> shift) & 0xFF) as usize;
            dst[counts[bucket]] = k;
            counts[bucket] += 1;
        }
        std::mem::swap(&mut src, &mut dst);
    }
    // Eight swaps is even, so `src` is the caller's slice holding the result.
}

/// Map a signed key to an unsigned one preserving order (flip the sign bit).
pub fn to_sortable(k: i64) -> u64 {
    (k as u64) ^ (1u64 << 63)
}

/// Inverse of [`to_sortable`].
pub fn from_sortable(u: u64) -> i64 {
    (u ^ (1u64 << 63)) as i64
}

/// Stable radix sort of signed 64-bit keys.
pub fn sort_i64(keys: &mut [i64]) {
    let mut mapped: Vec<u64> = keys.iter().map(|&k| to_sortable(k)).collect();
    sort_u64(&mut mapped);
    for (dst, &u) in keys.iter_mut().zip(mapped.iter()) {
        *dst = from_sortable(u);
    }
}

/// Stable radix sort of `(key, payload)` pairs by unsigned key.
pub fn sort_pairs(pairs: &mut Vec<(u64, u32)>) {
    let n = pairs.len();
    if n < 2 {
        return;
    }
    let mut buffer = vec![(0u64, 0u32); n];
    for byte in 0..8 {
        let shift = byte * 8;
        let mut counts = [0usize; RADIX + 1];
        for &(k, _) in pairs.iter() {
            let bucket = ((k >> shift) & 0xFF) as usize;
            counts[bucket + 1] += 1;
        }
        if counts.iter().skip(1).any(|&c| c == n) {
            continue;
        }
        for i in 0..RADIX {
            counts[i + 1] += counts[i];
        }
        for &pair in pairs.iter() {
            let bucket = ((pair.0 >> shift) & 0xFF) as usize;
            buffer[counts[bucket]] = pair;
            counts[bucket] += 1;
        }
        std::mem::swap(pairs, &mut buffer);
    }
}

/// `true` if the slice is sorted ascending.
pub fn is_sorted_u64(keys: &[u64]) -> bool {
    keys.windows(2).all(|w| w[0] <= w[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts_unsigned() {
        let mut keys = vec![5u64, 3, 8, 1, 9, 2, 7, 0, u64::MAX, 100];
        sort_u64(&mut keys);
        assert!(is_sorted_u64(&keys));
        assert_eq!(keys[0], 0);
        assert_eq!(keys[keys.len() - 1], u64::MAX);
    }

    #[test]
    fn sorts_signed() {
        let mut keys = vec![3i64, -5, 0, i64::MIN, i64::MAX, -1, 2];
        sort_i64(&mut keys);
        let mut expected = keys.clone();
        expected.sort();
        assert_eq!(keys, expected);
    }

    #[test]
    fn sortable_mapping_preserves_order() {
        assert!(to_sortable(-1) < to_sortable(0));
        assert!(to_sortable(0) < to_sortable(1));
        assert!(to_sortable(i64::MIN) < to_sortable(i64::MAX));
        assert_eq!(from_sortable(to_sortable(-42)), -42);
    }

    #[test]
    fn sorts_pairs_stably() {
        let mut pairs = vec![(3u64, 0u32), (1, 1), (3, 2), (2, 3), (1, 4)];
        sort_pairs(&mut pairs);
        let keys: Vec<u64> = pairs.iter().map(|(k, _)| *k).collect();
        assert!(is_sorted_u64(&keys));
        // Stability: equal keys keep their original payload order.
        let ones: Vec<u32> = pairs.iter().filter(|(k, _)| *k == 1).map(|(_, p)| *p).collect();
        assert_eq!(ones, vec![1, 4]);
    }

    #[test]
    fn handles_trivial() {
        let mut empty: Vec<u64> = vec![];
        sort_u64(&mut empty);
        let mut one = vec![42u64];
        sort_u64(&mut one);
        assert_eq!(one, vec![42]);
    }

    #[test]
    fn large_random() {
        let mut keys = Vec::new();
        let mut state = 12345u64;
        for _ in 0..5000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            keys.push(state >> 33);
        }
        let mut expected = keys.clone();
        expected.sort();
        sort_u64(&mut keys);
        assert_eq!(keys, expected);
    }
}
