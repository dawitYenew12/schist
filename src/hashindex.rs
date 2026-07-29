//! A paged hash index.
//!
//! A hash index maps a hash of the key to a bucket of `(key, row_id)` pairs.
//! It is the right structure for exact-match lookups on columns whose values
//! are not naturally ordered (text, in particular, where the key is the
//! dictionary id hashed to a bucket). The index uses linear probing within a
//! fixed bucket count and chains overflow buckets when a bucket fills.
//!
//! ## Layout
//!
//! The index owns a vector of buckets. Each bucket is a `Vec<Entry>`; when a
//! bucket grows past [`BUCKET_CAPACITY`], an overflow bucket is allocated and
//! chained. The whole structure serializes as: bucket count, then per bucket
//! the entry count and the entries, then the overflow chain.

use std::collections::BTreeMap;

/// Entries per bucket before an overflow bucket is chained.
pub const BUCKET_CAPACITY: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub key: i64,
    pub row_id: u64,
}

#[derive(Debug, Clone)]
pub struct Bucket {
    pub entries: Vec<Entry>,
    pub overflow: Option<usize>,
}

impl Bucket {
    pub fn new() -> Bucket {
        Bucket {
            entries: Vec::new(),
            overflow: None,
        }
    }
}

impl Default for Bucket {
    fn default() -> Self {
        Bucket::new()
    }
}

/// A hash index keyed on `i64`.
#[derive(Debug, Clone)]
pub struct HashIndex {
    pub buckets: Vec<Bucket>,
    pub overflow: BTreeMap<usize, Bucket>,
    pub next_overflow: usize,
    pub len: usize,
}

impl HashIndex {
    pub fn new(bucket_count: usize) -> HashIndex {
        let bucket_count = bucket_count.max(4);
        let mut buckets = Vec::with_capacity(bucket_count);
        for _ in 0..bucket_count {
            buckets.push(Bucket::new());
        }
        HashIndex {
            buckets,
            overflow: BTreeMap::new(),
            next_overflow: 0,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// FNV-1a-derived mix for an i64 key.
    fn bucket_of(&self, key: i64) -> usize {
        let mut h: u64 = 0xcbf29ce484222325;
        let bytes = key.to_le_bytes();
        for &b in &bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        (h as usize) % self.buckets.len()
    }

    /// Insert a `(key, row_id)` pair.
    pub fn insert(&mut self, key: i64, row_id: u64) {
        let b = self.bucket_of(key);
        self.insert_into(b, key, row_id);
        self.len += 1;
    }

    fn insert_into(&mut self, bucket_idx: usize, key: i64, row_id: u64) {
        // Try the primary bucket.
        let needs_overflow = {
            let bucket = &mut self.buckets[bucket_idx];
            if bucket.entries.len() < BUCKET_CAPACITY {
                bucket.entries.push(Entry { key, row_id });
                return;
            }
            true
        };
        if needs_overflow {
            let chain = self.buckets[bucket_idx].overflow;
            let new_head = self.insert_overflow(chain, key, row_id);
            // If a brand-new overflow bucket was created at the head of the
            // chain, link it from the primary bucket.
            if chain.is_none() && new_head.is_some() {
                self.buckets[bucket_idx].overflow = new_head;
            }
        }
    }

    /// Insert into the overflow chain starting at `chain_head`. Returns
    /// `Some(id)` if a new overflow bucket was created (and should be linked
    /// from the head), `None` if an existing bucket absorbed the entry.
    fn insert_overflow(&mut self, chain_head: Option<usize>, key: i64, row_id: u64) -> Option<usize> {
        match chain_head {
            None => {
                // Allocate a new overflow bucket.
                let id = self.next_overflow;
                self.next_overflow += 1;
                let mut ob = Bucket::new();
                ob.entries.push(Entry { key, row_id });
                self.overflow.insert(id, ob);
                Some(id)
            }
            Some(head) => {
                let full = {
                    let ob = self.overflow.get_mut(&head).unwrap();
                    if ob.entries.len() < BUCKET_CAPACITY {
                        ob.entries.push(Entry { key, row_id });
                        return None;
                    }
                    true
                };
                if full {
                    let next = self.overflow.get(&head).unwrap().overflow;
                    let new_id = self.insert_overflow(next, key, row_id);
                    if next.is_none() && new_id.is_some() {
                        self.overflow.get_mut(&head).unwrap().overflow = new_id;
                    }
                    None
                } else {
                    None
                }
            }
        }
    }

    /// Look up all row ids for a key.
    pub fn lookup(&self, key: i64) -> Vec<u64> {
        let b = self.bucket_of(key);
        let mut out = Vec::new();
        for e in &self.buckets[b].entries {
            if e.key == key {
                out.push(e.row_id);
            }
        }
        let mut cur = self.buckets[b].overflow;
        while let Some(id) = cur {
            let ob = self.overflow.get(&id).unwrap();
            for e in &ob.entries {
                if e.key == key {
                    out.push(e.row_id);
                }
            }
            cur = ob.overflow;
        }
        out
    }

    /// Remove a `(key, row_id)` pair. Returns `true` if found.
    pub fn remove(&mut self, key: i64, row_id: u64) -> bool {
        let b = self.bucket_of(key);
        let mut removed = false;
        {
            let bucket = &mut self.buckets[b];
            let before = bucket.entries.len();
            bucket.entries.retain(|e| !(e.key == key && e.row_id == row_id));
            removed |= bucket.entries.len() < before;
        }
        let mut cur = self.buckets[b].overflow;
        while let Some(id) = cur {
            let ob = self.overflow.get_mut(&id).unwrap();
            let before = ob.entries.len();
            ob.entries.retain(|e| !(e.key == key && e.row_id == row_id));
            removed |= ob.entries.len() < before;
            cur = ob.overflow;
        }
        if removed {
            self.len -= 1;
        }
        removed
    }

    /// All entries, in no particular order.
    pub fn iter(&self) -> Vec<Entry> {
        let mut out = Vec::with_capacity(self.len);
        for b in &self.buckets {
            out.extend_from_slice(&b.entries);
            let mut cur = b.overflow;
            while let Some(id) = cur {
                let ob = self.overflow.get(&id).unwrap();
                out.extend_from_slice(&ob.entries);
                cur = ob.overflow;
            }
        }
        out
    }

    /// Rehash into a new bucket count (grow/shrink).
    pub fn rehash(&mut self, new_count: usize) {
        let entries = self.iter();
        *self = HashIndex::new(new_count);
        for e in entries {
            self.insert(e.key, e.row_id);
        }
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.buckets.len() as u32).to_le_bytes());
        for b in &self.buckets {
            out.extend_from_slice(&(b.entries.len() as u32).to_le_bytes());
            for e in &b.entries {
                out.extend_from_slice(&e.key.to_le_bytes());
                out.extend_from_slice(&e.row_id.to_le_bytes());
            }
            out.extend_from_slice(&(b.overflow.unwrap_or(usize::MAX) as u32).to_le_bytes());
        }
        out.extend_from_slice(&(self.overflow.len() as u32).to_le_bytes());
        for (&id, ob) in &self.overflow {
            out.extend_from_slice(&(id as u32).to_le_bytes());
            out.extend_from_slice(&(ob.entries.len() as u32).to_le_bytes());
            for e in &ob.entries {
                out.extend_from_slice(&e.key.to_le_bytes());
                out.extend_from_slice(&e.row_id.to_le_bytes());
            }
            out.extend_from_slice(&(ob.overflow.unwrap_or(usize::MAX) as u32).to_le_bytes());
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Option<HashIndex> {
        if buf.len() < 4 {
            return None;
        }
        let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let mut pos = 4;
        let mut buckets = Vec::with_capacity(n);
        for _ in 0..n {
            if pos + 4 > buf.len() {
                break;
            }
            let ec = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let mut entries = Vec::with_capacity(ec);
            for _ in 0..ec {
                if pos + 16 > buf.len() {
                    break;
                }
                let key = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                let row_id = u64::from_le_bytes(buf[pos + 8..pos + 16].try_into().unwrap());
                pos += 16;
                entries.push(Entry { key, row_id });
            }
            if pos + 4 > buf.len() {
                break;
            }
            let ov = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            buckets.push(Bucket {
                entries,
                overflow: (ov != u32::MAX).then_some(ov as usize),
            });
        }
        if pos + 4 > buf.len() {
            return Some(HashIndex {
                buckets,
                overflow: BTreeMap::new(),
                next_overflow: 0,
                len: 0,
            });
        }
        let oc = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut overflow = BTreeMap::new();
        let mut max_id = 0usize;
        for _ in 0..oc {
            if pos + 8 > buf.len() {
                break;
            }
            let id = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
            let ec = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let mut entries = Vec::with_capacity(ec);
            for _ in 0..ec {
                if pos + 16 > buf.len() {
                    break;
                }
                let key = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                let row_id = u64::from_le_bytes(buf[pos + 8..pos + 16].try_into().unwrap());
                pos += 16;
                entries.push(Entry { key, row_id });
            }
            if pos + 4 > buf.len() {
                break;
            }
            let ov = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            pos += 4;
            overflow.insert(
                id,
                Bucket {
                    entries,
                    overflow: (ov != u32::MAX).then_some(ov as usize),
                },
            );
            max_id = max_id.max(id);
        }
        let len = buckets.iter().map(|b| b.entries.len()).sum::<usize>()
            + overflow.values().map(|b| b.entries.len()).sum::<usize>();
        Some(HashIndex {
            buckets,
            overflow,
            next_overflow: max_id + 1,
            len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_lookup_remove() {
        let mut h = HashIndex::new(8);
        h.insert(5, 1);
        h.insert(5, 2);
        h.insert(100, 3);
        assert_eq!(h.lookup(5), vec![1, 2]);
        assert_eq!(h.lookup(100), vec![3]);
        assert!(h.remove(5, 1));
        assert_eq!(h.lookup(5), vec![2]);
    }

    #[test]
    fn overflow_chains() {
        let mut h = HashIndex::new(4);
        for i in 0..120u64 {
            h.insert(i as i64, i);
        }
        for i in 0..120u64 {
            assert_eq!(h.lookup(i as i64), vec![i]);
        }
        assert!(!h.overflow.is_empty());
    }

    #[test]
    fn rehash_preserves_entries() {
        let mut h = HashIndex::new(4);
        for i in 0..40u64 {
            h.insert(i as i64, i);
        }
        h.rehash(16);
        for i in 0..40u64 {
            assert_eq!(h.lookup(i as i64), vec![i]);
        }
    }

    #[test]
    fn encode_decode_round_trips() {
        let mut h = HashIndex::new(4);
        for i in 0..30u64 {
            h.insert(i as i64, i * 3);
        }
        let bytes = h.encode();
        let h2 = HashIndex::decode(&bytes).unwrap();
        assert_eq!(h2.len, h.len);
        for i in 0..30u64 {
            assert_eq!(h2.lookup(i as i64), vec![i * 3]);
        }
    }
}
