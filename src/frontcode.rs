//! Front-coded string dictionary blocks.
//!
//! Sorted string dictionaries (the payload of a dictionary-encoded text column)
//! compress well with front coding: within a block, each string is stored as
//! the length of the prefix it shares with the previous string plus the
//! remaining suffix. Every `stride`-th string is stored in full (a "restart")
//! so that binary search and random access do not have to decode from the very
//! start of the block. This is the layout used by column stores and by search
//! engines' term dictionaries.

use crate::varint::{read_uvarint, write_uvarint};

/// A front-coded, sorted block of strings.
#[derive(Debug, Clone)]
pub struct FrontCodedBlock {
    data: Vec<u8>,
    /// Byte offsets of each restart entry within `data`.
    restarts: Vec<u32>,
    stride: usize,
    count: usize,
}

impl FrontCodedBlock {
    /// Build a block from a *sorted* list of strings, with a restart every
    /// `stride` entries.
    pub fn build(sorted: &[&str], stride: usize) -> FrontCodedBlock {
        let stride = stride.max(1);
        let mut data = Vec::new();
        let mut restarts = Vec::new();
        let mut prev: &str = "";
        for (i, s) in sorted.iter().enumerate() {
            let is_restart = i % stride == 0;
            let shared = if is_restart {
                restarts.push(data.len() as u32);
                0
            } else {
                common_prefix_len(prev, s)
            };
            let suffix = &s.as_bytes()[shared..];
            write_uvarint(&mut data, shared as u64);
            write_uvarint(&mut data, suffix.len() as u64);
            data.extend_from_slice(suffix);
            prev = s;
        }
        FrontCodedBlock {
            data,
            restarts,
            stride,
            count: sorted.len(),
        }
    }

    /// Number of strings in the block.
    pub fn len(&self) -> usize {
        self.count
    }

    /// `true` if the block is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Encoded byte size.
    pub fn byte_size(&self) -> usize {
        self.data.len() + self.restarts.len() * 4
    }

    /// Decode every string in order.
    pub fn decode_all(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.count);
        let mut prev = Vec::new();
        let mut pos = 0;
        for i in 0..self.count {
            let is_restart = i % self.stride == 0;
            let (shared, n1) = read_uvarint(&self.data[pos..]).unwrap_or((0, 0));
            pos += n1;
            let (suffix_len, n2) = read_uvarint(&self.data[pos..]).unwrap_or((0, 0));
            pos += n2;
            let suffix = &self.data[pos..pos + suffix_len as usize];
            pos += suffix_len as usize;
            let mut s = if is_restart {
                Vec::new()
            } else {
                prev[..shared as usize].to_vec()
            };
            s.extend_from_slice(suffix);
            out.push(String::from_utf8_lossy(&s).into_owned());
            prev = s;
        }
        out
    }

    /// Random-access the `i`-th string by decoding forward from its restart.
    pub fn get(&self, i: usize) -> Option<String> {
        if i >= self.count {
            return None;
        }
        let restart_block = i / self.stride;
        let start_idx = restart_block * self.stride;
        let mut pos = self.restarts[restart_block] as usize;
        let mut prev: Vec<u8> = Vec::new();
        for j in start_idx..=i {
            let is_restart = j % self.stride == 0;
            let (shared, n1) = read_uvarint(&self.data[pos..]).ok()?;
            pos += n1;
            let (suffix_len, n2) = read_uvarint(&self.data[pos..]).ok()?;
            pos += n2;
            let suffix = &self.data[pos..pos + suffix_len as usize];
            pos += suffix_len as usize;
            let mut s = if is_restart {
                Vec::new()
            } else {
                prev[..shared as usize].to_vec()
            };
            s.extend_from_slice(suffix);
            prev = s;
        }
        Some(String::from_utf8_lossy(&prev).into_owned())
    }

    /// Binary-search for `target`, returning its index if present. Uses the
    /// restart array to narrow to a block, then scans forward.
    pub fn find(&self, target: &str) -> Option<usize> {
        // Binary search over restart points by their full string.
        let mut lo = 0usize;
        let mut hi = self.restarts.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let s = self.restart_string(mid);
            if s.as_str() <= target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // The candidate block is lo-1.
        let block = lo.saturating_sub(1);
        let start = block * self.stride;
        let end = ((block + 1) * self.stride).min(self.count);
        for i in start..end {
            if let Some(s) = self.get(i) {
                if s == target {
                    return Some(i);
                }
                if s.as_str() > target {
                    break;
                }
            }
        }
        None
    }

    fn restart_string(&self, restart_block: usize) -> String {
        let idx = restart_block * self.stride;
        self.get(idx).unwrap_or_default()
    }
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    let (ab, bb) = (a.as_bytes(), b.as_bytes());
    let mut i = 0;
    while i < ab.len() && i < bb.len() && ab[i] == bb[i] {
        i += 1;
    }
    // Do not split a multi-byte UTF-8 sequence.
    while i > 0 && (bb[i - 1] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<&'static str> {
        vec![
            "apple", "application", "apply", "banana", "band", "bandana", "cat", "catalog",
        ]
    }

    #[test]
    fn decode_all_roundtrips() {
        let strings = sample();
        let block = FrontCodedBlock::build(&strings, 4);
        assert_eq!(block.decode_all(), strings);
        assert_eq!(block.len(), 8);
    }

    #[test]
    fn random_access() {
        let strings = sample();
        let block = FrontCodedBlock::build(&strings, 4);
        for (i, s) in strings.iter().enumerate() {
            assert_eq!(block.get(i).as_deref(), Some(*s));
        }
        assert_eq!(block.get(100), None);
    }

    #[test]
    fn find_present_and_absent() {
        let strings = sample();
        let block = FrontCodedBlock::build(&strings, 3);
        assert_eq!(block.find("bandana"), Some(5));
        assert_eq!(block.find("apple"), Some(0));
        assert_eq!(block.find("catalog"), Some(7));
        assert_eq!(block.find("dog"), None);
        assert_eq!(block.find("aardvark"), None);
    }

    #[test]
    fn compresses_shared_prefixes() {
        let strings = vec![
            "internationalization",
            "internationalize",
            "internationalized",
            "internationally",
        ];
        let block = FrontCodedBlock::build(&strings, 8);
        let raw: usize = strings.iter().map(|s| s.len()).sum();
        assert!(block.byte_size() < raw, "front coding should shrink");
        assert_eq!(block.decode_all(), strings);
    }

    #[test]
    fn utf8_safe_prefixes() {
        let strings = vec!["café", "cafés", "cafétéria"];
        let block = FrontCodedBlock::build(&strings, 4);
        assert_eq!(block.decode_all(), strings);
        assert_eq!(block.get(2).as_deref(), Some("cafétéria"));
    }
}
