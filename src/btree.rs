//! A paged B+tree secondary index.
//!
//! A real columnar engine keeps secondary indexes in a paged structure so they
//! survive checkpointing and stay balanced as rows are inserted. This module
//! implements a classic B+tree keyed on `i64` (the index key is the column
//! value's integer projection; text values key on their dictionary id). Keys
//! map to lists of row ids.
//!
//! The tree is *unclustered*: a key may map to many row ids, and the row ids
//! are stored in the leaves. Internal nodes store separator keys and child
//! page ids. Pages are fixed-size byte buffers with a simple header so they can
//! live in the pager alongside data and dictionary pages.
//!
//! ## Page layout
//!
//! Every node page begins with:
//!
//! ```text
//!   [u8 node_type]   // 0 = leaf, 1 = internal
//!   [u32 key_count]
//!   [u32 next]       // leaf next-leaf pointer; u32::MAX if none
//! ```
//!
//! A leaf page then stores `key_count` entries of:
//!
//! ```text
//!   [i64 key][u32 row_count][row_count * u64 row_id]
//! ```
//!
//! An internal page stores `key_count` separator keys and `key_count + 1`
//! child pointers:
//!
//! ```text
//!   [i64 key] * key_count
//!   [u32 child] * (key_count + 1)
//! ```

use std::cmp::Ordering;

/// The fan-out target: a node holds up to this many keys before splitting.
pub const FANOUT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Leaf,
    Internal,
}

impl NodeType {
    pub fn as_u8(self) -> u8 {
        match self {
            NodeType::Leaf => 0,
            NodeType::Internal => 1,
        }
    }
    pub fn from_u8(b: u8) -> NodeType {
        if b == 1 {
            NodeType::Internal
        } else {
            NodeType::Leaf
        }
    }
}

/// A key-to-rows entry in a leaf.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafEntry {
    pub key: i64,
    pub rows: Vec<u64>,
}

/// An in-memory node (leaf or internal).
#[derive(Debug, Clone)]
pub enum Node {
    Leaf {
        id: u32,
        entries: Vec<LeafEntry>,
        next: Option<u32>,
    },
    Internal {
        id: u32,
        keys: Vec<i64>,
        children: Vec<u32>,
    },
}

impl Node {
    pub fn id(&self) -> u32 {
        match self {
            Node::Leaf { id, .. } => *id,
            Node::Internal { id, .. } => *id,
        }
    }

    pub fn node_type(&self) -> NodeType {
        match self {
            Node::Leaf { .. } => NodeType::Leaf,
            Node::Internal { .. } => NodeType::Internal,
        }
    }

    pub fn is_full(&self) -> bool {
        match self {
            Node::Leaf { entries, .. } => entries.len() >= FANOUT,
            Node::Internal { keys, .. } => keys.len() >= FANOUT,
        }
    }

    /// Encode a node into a page buffer.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Node::Leaf { id: _, entries, next } => {
                out.push(NodeType::Leaf.as_u8());
                out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
                out.extend_from_slice(&next.unwrap_or(u32::MAX).to_le_bytes());
                for e in entries {
                    out.extend_from_slice(&e.key.to_le_bytes());
                    out.extend_from_slice(&(e.rows.len() as u32).to_le_bytes());
                    for &r in &e.rows {
                        out.extend_from_slice(&r.to_le_bytes());
                    }
                }
            }
            Node::Internal { id: _, keys, children } => {
                out.push(NodeType::Internal.as_u8());
                out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
                out.extend_from_slice(&u32::MAX.to_le_bytes());
                for &k in keys {
                    out.extend_from_slice(&k.to_le_bytes());
                }
                for &c in children {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        out
    }

    /// Decode a node from a page buffer, given its id.
    pub fn decode(id: u32, buf: &[u8]) -> Option<Node> {
        if buf.is_empty() {
            return None;
        }
        let nt = NodeType::from_u8(buf[0]);
        if buf.len() < 9 {
            return None;
        }
        let key_count = u32::from_le_bytes(buf[1..5].try_into().unwrap()) as usize;
        let next_raw = u32::from_le_bytes(buf[5..9].try_into().unwrap());
        let next = (next_raw != u32::MAX).then_some(next_raw);
        let mut pos = 9;
        match nt {
            NodeType::Leaf => {
                let mut entries = Vec::with_capacity(key_count);
                for _ in 0..key_count {
                    if pos + 12 > buf.len() {
                        break;
                    }
                    let key = i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
                    let rc = u32::from_le_bytes(buf[pos + 8..pos + 12].try_into().unwrap()) as usize;
                    pos += 12;
                    let mut rows = Vec::with_capacity(rc);
                    for _ in 0..rc {
                        if pos + 8 > buf.len() {
                            break;
                        }
                        rows.push(u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()));
                        pos += 8;
                    }
                    entries.push(LeafEntry { key, rows });
                }
                Some(Node::Leaf { id, entries, next })
            }
            NodeType::Internal => {
                let mut keys = Vec::with_capacity(key_count);
                for _ in 0..key_count {
                    if pos + 8 > buf.len() {
                        break;
                    }
                    keys.push(i64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap()));
                    pos += 8;
                }
                let child_count = key_count + 1;
                let mut children = Vec::with_capacity(child_count);
                for _ in 0..child_count {
                    if pos + 4 > buf.len() {
                        break;
                    }
                    children.push(u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()));
                    pos += 4;
                }
                Some(Node::Internal { id, keys, children })
            }
        }
    }
}

/// The B+tree. Nodes live in an in-memory page table keyed by id; the tree
/// owns the root id.
#[derive(Debug, Clone)]
pub struct BTree {
    pages: std::collections::BTreeMap<u32, Node>,
    root: u32,
    next_id: u32,
    height: usize,
}

impl BTree {
    pub fn new() -> BTree {
        let root = 0u32;
        let mut pages = std::collections::BTreeMap::new();
        pages.insert(
            root,
            Node::Leaf {
                id: root,
                entries: Vec::new(),
                next: None,
            },
        );
        BTree {
            pages,
            root,
            next_id: 1,
            height: 1,
        }
    }

    pub fn root(&self) -> u32 {
        self.root
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    fn alloc(&mut self, node: Node) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.pages.insert(id, node);
        id
    }

    fn get(&self, id: u32) -> &Node {
        self.pages.get(&id).expect("btree page missing")
    }

    fn get_mut(&mut self, id: u32) -> &mut Node {
        self.pages.get_mut(&id).expect("btree page missing")
    }

    /// Insert `(key, row_id)`. Splits propagate upward as needed.
    pub fn insert(&mut self, key: i64, row_id: u64) {
        let split = self.insert_rec(self.root, key, row_id);
        if let Some((sep_key, new_right_id)) = split {
            let new_root_id = self.alloc(Node::Internal {
                id: 0,
                keys: vec![sep_key],
                children: vec![self.root, new_right_id],
            });
            self.root = new_root_id;
            self.height += 1;
        }
    }

    /// Returns `Some((separator_key, new_page_id))` if the subtree rooted at
    /// `id` split.
    fn insert_rec(&mut self, id: u32, key: i64, row_id: u64) -> Option<(i64, u32)> {
        let node_type = self.get(id).node_type();
        match node_type {
            NodeType::Leaf => {
                let split = {
                    let node = self.get_mut(id);
                    if let Node::Leaf { entries, .. } = node {
                        let pos = entries
                            .binary_search_by(|e| e.key.cmp(&key))
                            .unwrap_or_else(|p| p);
                        if pos < entries.len() && entries[pos].key == key {
                            if !entries[pos].rows.contains(&row_id) {
                                entries[pos].rows.push(row_id);
                            }
                        } else {
                            entries.insert(pos, LeafEntry { key, rows: vec![row_id] });
                        }
                        entries.len() > FANOUT
                    } else {
                        false
                    }
                };
                if split {
                    let (left, right, sep) = self.split_leaf(id);
                    self.pages.insert(id, left);
                    let new_id = self.alloc(right);
                    Some((sep, new_id))
                } else {
                    None
                }
            }
            NodeType::Internal => {
                let child_id = {
                    let node = self.get(id);
                    if let Node::Internal { keys, children, .. } = node {
                        let pos = keys.partition_point(|&k| k <= key);
                        children[pos.min(children.len() - 1)]
                    } else {
                        unreachable!()
                    }
                };
                let child_split = self.insert_rec(child_id, key, row_id);
                if let Some((sep_key, new_child_id)) = child_split {
                    let needs_split = {
                        let node = self.get_mut(id);
                        if let Node::Internal { keys, children, .. } = node {
                            let pos = keys.partition_point(|&k| k <= sep_key);
                            keys.insert(pos, sep_key);
                            children.insert(pos + 1, new_child_id);
                            keys.len() > FANOUT
                        } else {
                            false
                        }
                    };
                    if needs_split {
                        let (left, right, sep) = self.split_internal(id);
                        self.pages.insert(id, left);
                        let new_id = self.alloc(right);
                        Some((sep, new_id))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    fn split_leaf(&self, id: u32) -> (Node, Node, i64) {
        let node = self.get(id);
        if let Node::Leaf { id: _, entries, next } = node {
            let mid = entries.len() / 2;
            let left_entries = entries[..mid].to_vec();
            let right_entries = entries[mid..].to_vec();
            let sep = right_entries[0].key;
            let old_next = *next;
            let left = Node::Leaf {
                id,
                entries: left_entries,
                next: Some(self.next_id),
            };
            let right = Node::Leaf {
                id: self.next_id,
                entries: right_entries,
                next: old_next,
            };
            (left, right, sep)
        } else {
            unreachable!()
        }
    }

    fn split_internal(&self, id: u32) -> (Node, Node, i64) {
        let node = self.get(id);
        if let Node::Internal { id: _, keys, children } = node {
            let mid = keys.len() / 2;
            let sep = keys[mid];
            let left_keys = keys[..mid].to_vec();
            let right_keys = keys[mid + 1..].to_vec();
            let left_children = children[..=mid].to_vec();
            let right_children = children[mid + 1..].to_vec();
            let left = Node::Internal {
                id,
                keys: left_keys,
                children: left_children,
            };
            let right = Node::Internal {
                id: self.next_id,
                keys: right_keys,
                children: right_children,
            };
            (left, right, sep)
        } else {
            unreachable!()
        }
    }

    /// Look up all row ids for a key.
    pub fn lookup(&self, key: i64) -> Vec<u64> {
        let mut id = self.root;
        loop {
            match self.get(id) {
                Node::Leaf { entries, .. } => {
                    if let Ok(pos) = entries.binary_search_by(|e| e.key.cmp(&key)) {
                        return entries[pos].rows.clone();
                    }
                    return Vec::new();
                }
                Node::Internal { keys, children, .. } => {
                    let pos = keys.partition_point(|&k| k <= key);
                    id = children[pos.min(children.len() - 1)];
                }
            }
        }
    }

    /// Range scan: all row ids whose key is in `[lo, hi]`.
    pub fn range(&self, lo: i64, hi: i64) -> Vec<(i64, u64)> {
        let mut out = Vec::new();
        let mut id = self.root;
        // Descend to the leaf that would hold `lo`.
        loop {
            match self.get(id) {
                Node::Leaf { .. } => break,
                Node::Internal { keys, children, .. } => {
                    let pos = keys.partition_point(|&k| k <= lo);
                    id = children[pos.min(children.len() - 1)];
                }
            }
        }
        // Walk leaves via the next pointer.
        while let Node::Leaf { entries, next, .. } = self.get(id) {
            for e in entries {
                if e.key >= lo && e.key <= hi {
                    for &r in &e.rows {
                        out.push((e.key, r));
                    }
                }
                if e.key > hi {
                    return out;
                }
            }
            match next {
                Some(n) => id = *n,
                None => break,
            }
        }
        out
    }

    /// Remove a single `(key, row_id)` pair. Returns `true` if it was present.
    pub fn remove(&mut self, key: i64, row_id: u64) -> bool {
        self.remove_rec(self.root, key, row_id)
    }

    fn remove_rec(&mut self, id: u32, key: i64, row_id: u64) -> bool {
        let node_type = self.get(id).node_type();
        match node_type {
            NodeType::Leaf => {
                let node = self.get_mut(id);
                if let Node::Leaf { entries, .. } = node {
                    if let Ok(pos) = entries.binary_search_by(|e| e.key.cmp(&key)) {
                        let before = entries[pos].rows.len();
                        entries[pos].rows.retain(|&r| r != row_id);
                        let after = entries[pos].rows.len();
                        if entries[pos].rows.is_empty() {
                            entries.remove(pos);
                        }
                        return after < before;
                    }
                }
                false
            }
            NodeType::Internal => {
                let child_id = {
                    let node = self.get(id);
                    if let Node::Internal { keys, children, .. } = node {
                        let pos = keys.partition_point(|&k| k <= key);
                        children[pos.min(children.len() - 1)]
                    } else {
                        unreachable!()
                    }
                };
                self.remove_rec(child_id, key, row_id)
            }
        }
    }

    /// Total number of key entries across all leaves.
    pub fn len(&self) -> usize {
        let mut total = 0;
        for node in self.pages.values() {
            if let Node::Leaf { entries, .. } = node {
                total += entries.len();
            }
        }
        total
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All (key, row_id) pairs in ascending key order.
    pub fn iter(&self) -> Vec<(i64, u64)> {
        let mut out = Vec::new();
        let mut id = self.root;
        loop {
            match self.get(id) {
                Node::Leaf { .. } => break,
                Node::Internal { children, .. } => id = children[0],
            }
        }
        while let Node::Leaf { entries, next, .. } = self.get(id) {
            for e in entries {
                for &r in &e.rows {
                    out.push((e.key, r));
                }
            }
            match next {
                Some(n) => id = *n,
                None => break,
            }
        }
        out
    }

    /// Serialize the whole tree: root id, next_id, height, page count, then
    /// each page as `[u32 id][u32 len][bytes]`.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.root.to_le_bytes());
        out.extend_from_slice(&self.next_id.to_le_bytes());
        out.extend_from_slice(&(self.height as u32).to_le_bytes());
        out.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        for (&id, node) in &self.pages {
            let bytes = node.encode();
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    pub fn decode(buf: &[u8]) -> Option<BTree> {
        if buf.len() < 16 {
            return None;
        }
        let root = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let next_id = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        let mut pages = std::collections::BTreeMap::new();
        let mut pos = 16;
        for _ in 0..count {
            if pos + 8 > buf.len() {
                break;
            }
            let id = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let len = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            if pos + len > buf.len() {
                break;
            }
            let node = Node::decode(id, &buf[pos..pos + len])?;
            pages.insert(id, node);
            pos += len;
        }
        Some(BTree {
            pages,
            root,
            next_id,
            height,
        })
    }
}

impl Default for BTree {
    fn default() -> Self {
        BTree::new()
    }
}

/// Compare two keys with the same convention as [`Ordering`].
pub fn cmp_keys(a: i64, b: i64) -> Ordering {
    a.cmp(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let mut t = BTree::new();
        t.insert(5, 1);
        t.insert(5, 2);
        t.insert(10, 3);
        t.insert(1, 4);
        assert_eq!(t.lookup(5), vec![1, 2]);
        assert_eq!(t.lookup(10), vec![3]);
        assert_eq!(t.lookup(1), vec![4]);
        assert_eq!(t.lookup(7), Vec::<u64>::new());
    }

    #[test]
    fn many_inserts_split() {
        let mut t = BTree::new();
        for i in 0..200u64 {
            t.insert(i as i64, i);
        }
        for i in 0..200u64 {
            assert_eq!(t.lookup(i as i64), vec![i], "key {} missing", i);
        }
        assert!(t.height() >= 2);
        assert!(t.page_count() > 1);
    }

    #[test]
    fn range_scan() {
        let mut t = BTree::new();
        for i in 0..100u64 {
            t.insert(i as i64, i);
        }
        let r = t.range(10, 15);
        let keys: Vec<i64> = r.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn remove_drops_row() {
        let mut t = BTree::new();
        t.insert(5, 1);
        t.insert(5, 2);
        assert!(t.remove(5, 1));
        assert_eq!(t.lookup(5), vec![2]);
        assert!(t.remove(5, 2));
        assert!(t.lookup(5).is_empty());
    }

    #[test]
    fn encode_decode_round_trips() {
        let mut t = BTree::new();
        for i in 0..50u64 {
            t.insert(i as i64, i * 2);
        }
        let bytes = t.encode();
        let t2 = BTree::decode(&bytes).unwrap();
        assert_eq!(t2.len(), t.len());
        for i in 0..50u64 {
            assert_eq!(t2.lookup(i as i64), vec![i * 2]);
        }
    }

    #[test]
    fn iter_is_sorted() {
        let mut t = BTree::new();
        for i in [50, 10, 30, 20, 40] {
            t.insert(i, i as u64);
        }
        let pairs = t.iter();
        let keys: Vec<i64> = pairs.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![10, 20, 30, 40, 50]);
    }
}
