//! A byte-wise radix trie for string keys.
//!
//! Dictionary encoding and text indexes benefit from a prefix-sharing map from
//! byte strings to integer ids: it deduplicates common prefixes, supports
//! ordered iteration, and answers prefix-range queries (`WHERE name LIKE
//! 'foo%'`) directly. This is a compressed (path-collapsing) radix trie — an
//! edge can carry a multi-byte label — with values stored at the nodes that end
//! a key.

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
struct RadixNode {
    /// The compressed edge label leading into this node (empty at the root).
    prefix: Vec<u8>,
    value: Option<i64>,
    children: BTreeMap<u8, Box<RadixNode>>,
}

impl RadixNode {
    fn new(prefix: Vec<u8>) -> RadixNode {
        RadixNode {
            prefix,
            value: None,
            children: BTreeMap::new(),
        }
    }
}

/// A map from byte strings to `i64` values.
#[derive(Debug, Clone)]
pub struct RadixTree {
    root: RadixNode,
    len: usize,
}

fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

impl RadixTree {
    /// A new empty tree.
    pub fn new() -> RadixTree {
        RadixTree {
            root: RadixNode::new(Vec::new()),
            len: 0,
        }
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a key; returns the previous value if the key existed.
    pub fn insert(&mut self, key: &[u8], value: i64) -> Option<i64> {
        let (old, added) = Self::insert_into(&mut self.root, key, value);
        if added {
            self.len += 1;
        }
        old
    }

    /// Convenience for string keys.
    pub fn insert_str(&mut self, key: &str, value: i64) -> Option<i64> {
        self.insert(key.as_bytes(), value)
    }

    fn insert_into(node: &mut RadixNode, key: &[u8], value: i64) -> (Option<i64>, bool) {
        if key.is_empty() {
            let old = node.value.replace(value);
            return (old, old.is_none());
        }
        let first = key[0];
        if let Some(child) = node.children.get_mut(&first) {
            let cp = common_prefix(&child.prefix, key);
            if cp == child.prefix.len() {
                // Full edge consumed; descend.
                return Self::insert_into(child, &key[cp..], value);
            }
            // Split the edge at cp.
            let mut split = RadixNode::new(child.prefix[..cp].to_vec());
            let mut old_child = std::mem::replace(child, Box::new(RadixNode::new(Vec::new())));
            old_child.prefix = old_child.prefix[cp..].to_vec();
            let old_first = old_child.prefix[0];
            split.children.insert(old_first, old_child);
            if cp == key.len() {
                split.value = Some(value);
                *child = Box::new(split);
                return (None, true);
            }
            let rest = &key[cp..];
            let mut leaf = RadixNode::new(rest.to_vec());
            leaf.value = Some(value);
            split.children.insert(rest[0], Box::new(leaf));
            *child = Box::new(split);
            (None, true)
        } else {
            let mut leaf = RadixNode::new(key.to_vec());
            leaf.value = Some(value);
            node.children.insert(first, Box::new(leaf));
            (None, true)
        }
    }

    /// Look up a key.
    pub fn get(&self, key: &[u8]) -> Option<i64> {
        let mut node = &self.root;
        let mut rest = key;
        loop {
            if rest.is_empty() {
                return node.value;
            }
            let child = node.children.get(&rest[0])?;
            let cp = common_prefix(&child.prefix, rest);
            if cp != child.prefix.len() {
                return None;
            }
            rest = &rest[cp..];
            node = child;
        }
    }

    /// Convenience for string keys.
    pub fn get_str(&self, key: &str) -> Option<i64> {
        self.get(key.as_bytes())
    }

    /// `true` if the exact key is present.
    pub fn contains(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    /// Collect all `(key, value)` pairs in ascending key order.
    pub fn entries(&self) -> Vec<(Vec<u8>, i64)> {
        let mut out = Vec::with_capacity(self.len);
        let mut buf = Vec::new();
        Self::collect(&self.root, &mut buf, &mut out);
        out
    }

    fn collect(node: &RadixNode, buf: &mut Vec<u8>, out: &mut Vec<(Vec<u8>, i64)>) {
        let start = buf.len();
        buf.extend_from_slice(&node.prefix);
        if let Some(v) = node.value {
            out.push((buf.clone(), v));
        }
        for child in node.children.values() {
            Self::collect(child, buf, out);
        }
        buf.truncate(start);
    }

    /// Collect all `(key, value)` pairs whose key starts with `prefix`.
    pub fn with_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, i64)> {
        // Walk down to the node covering `prefix`.
        let mut node = &self.root;
        let mut consumed: Vec<u8> = Vec::new();
        let mut rest = prefix;
        loop {
            if rest.is_empty() {
                break;
            }
            let child = match node.children.get(&rest[0]) {
                Some(c) => c,
                None => return Vec::new(),
            };
            let cp = common_prefix(&child.prefix, rest);
            if cp == rest.len() {
                // The prefix ends inside this edge; the subtree at child matches.
                consumed.extend_from_slice(&child.prefix);
                let mut out = Vec::new();
                let mut buf = consumed[..consumed.len() - child.prefix.len()].to_vec();
                Self::collect(child, &mut buf, &mut out);
                return out;
            }
            if cp != child.prefix.len() {
                return Vec::new();
            }
            consumed.extend_from_slice(&child.prefix);
            rest = &rest[cp..];
            node = child;
        }
        let mut out = Vec::new();
        let mut buf = consumed.clone();
        // Collect the subtree rooted at `node` but without re-adding its prefix.
        if let Some(v) = node.value {
            out.push((buf.clone(), v));
        }
        for child in node.children.values() {
            Self::collect(child, &mut buf, &mut out);
        }
        out
    }

    /// The number of internal nodes (a structural measure of prefix sharing).
    pub fn node_count(&self) -> usize {
        fn count(n: &RadixNode) -> usize {
            1 + n.children.values().map(|c| count(c)).sum::<usize>()
        }
        count(&self.root)
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        RadixTree::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut t = RadixTree::new();
        assert_eq!(t.insert_str("apple", 1), None);
        assert_eq!(t.insert_str("app", 2), None);
        assert_eq!(t.insert_str("application", 3), None);
        assert_eq!(t.get_str("apple"), Some(1));
        assert_eq!(t.get_str("app"), Some(2));
        assert_eq!(t.get_str("application"), Some(3));
        assert_eq!(t.get_str("ap"), None);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn update_returns_old() {
        let mut t = RadixTree::new();
        t.insert_str("key", 10);
        assert_eq!(t.insert_str("key", 20), Some(10));
        assert_eq!(t.get_str("key"), Some(20));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn ordered_entries() {
        let mut t = RadixTree::new();
        for (k, v) in [("banana", 1), ("apple", 2), ("cherry", 3), ("apricot", 4)] {
            t.insert_str(k, v);
        }
        let keys: Vec<String> = t
            .entries()
            .into_iter()
            .map(|(k, _)| String::from_utf8(k).unwrap())
            .collect();
        assert_eq!(keys, vec!["apple", "apricot", "banana", "cherry"]);
    }

    #[test]
    fn prefix_query() {
        let mut t = RadixTree::new();
        for (k, v) in [("foobar", 1), ("foobaz", 2), ("foo", 3), ("frob", 4)] {
            t.insert_str(k, v);
        }
        let mut got: Vec<String> = t
            .with_prefix(b"foo")
            .into_iter()
            .map(|(k, _)| String::from_utf8(k).unwrap())
            .collect();
        got.sort();
        assert_eq!(got, vec!["foo", "foobar", "foobaz"]);
        assert!(t.with_prefix(b"xyz").is_empty());
    }

    #[test]
    fn edge_split_preserves_values() {
        let mut t = RadixTree::new();
        t.insert_str("test", 1);
        t.insert_str("team", 2);
        // 'te' edge should split.
        assert_eq!(t.get_str("test"), Some(1));
        assert_eq!(t.get_str("team"), Some(2));
        assert!(t.node_count() >= 4);
    }
}
