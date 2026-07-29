//! A generic LRU cache with a capacity bound.
//!
//! Beyond the page buffer pool, several places want a small bounded cache keyed
//! by something other than a page id — resolved dictionary lookups, compiled
//! predicate programs, plan fragments. This is a straightforward
//! hash-map-plus-intrusive-order LRU: O(1) get/put, evicting the
//! least-recently-used entry when the capacity is exceeded. The recency order
//! is kept as a doubly linked list threaded through a slab of optional nodes so
//! no per-entry heap allocation is needed and freed slots hold no value.

use std::collections::HashMap;
use std::hash::Hash;

const NIL: usize = usize::MAX;

struct Node<K, V> {
    key: K,
    value: V,
    prev: usize,
    next: usize,
}

/// A bounded least-recently-used cache.
pub struct LruCache<K: Eq + Hash + Clone, V> {
    map: HashMap<K, usize>,
    nodes: Vec<Option<Node<K, V>>>,
    free: Vec<usize>,
    head: usize, // most recently used
    tail: usize, // least recently used
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    /// A cache holding at most `capacity` entries (min 1).
    pub fn new(capacity: usize) -> LruCache<K, V> {
        LruCache {
            map: HashMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
        }
    }

    /// Number of resident entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The capacity bound.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// `(hits, misses)` since creation.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    fn node(&self, idx: usize) -> &Node<K, V> {
        self.nodes[idx].as_ref().expect("live node")
    }

    fn node_mut(&mut self, idx: usize) -> &mut Node<K, V> {
        self.nodes[idx].as_mut().expect("live node")
    }

    fn detach(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.node(idx);
            (n.prev, n.next)
        };
        if prev != NIL {
            self.node_mut(prev).next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.node_mut(next).prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        {
            let n = self.node_mut(idx);
            n.prev = NIL;
            n.next = old_head;
        }
        if old_head != NIL {
            self.node_mut(old_head).prev = idx;
        }
        self.head = idx;
        if self.tail == NIL {
            self.tail = idx;
        }
    }

    /// Look up a key, marking it most-recently-used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = match self.map.get(key) {
            Some(&i) => i,
            None => {
                self.misses += 1;
                return None;
            }
        };
        self.hits += 1;
        self.detach(idx);
        self.push_front(idx);
        Some(&self.node(idx).value)
    }

    /// `true` if a key is present (does not affect recency).
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Peek without changing recency.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|&i| &self.node(i).value)
    }

    /// Insert or update a key. Returns the evicted `(key, value)` if the
    /// insertion pushed the cache over capacity.
    pub fn put(&mut self, key: K, value: V) -> Option<(K, V)> {
        if let Some(&idx) = self.map.get(&key) {
            self.node_mut(idx).value = value;
            self.detach(idx);
            self.push_front(idx);
            return None;
        }
        let mut evicted = None;
        if self.map.len() >= self.capacity {
            evicted = self.evict_lru();
        }
        let node = Node {
            key: key.clone(),
            value,
            prev: NIL,
            next: NIL,
        };
        let idx = if let Some(i) = self.free.pop() {
            self.nodes[i] = Some(node);
            i
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        };
        self.map.insert(key, idx);
        self.push_front(idx);
        evicted
    }

    fn evict_lru(&mut self) -> Option<(K, V)> {
        if self.tail == NIL {
            return None;
        }
        let idx = self.tail;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.map.remove(&node.key);
        self.free.push(idx);
        Some((node.key, node.value))
    }

    /// Remove a key, returning its value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.map.remove(key)?;
        self.detach(idx);
        let node = self.nodes[idx].take()?;
        self.free.push(idx);
        Some(node.value)
    }

    /// Keys from most- to least-recently used.
    pub fn keys_mru(&self) -> Vec<K> {
        let mut out = Vec::with_capacity(self.map.len());
        let mut cur = self.head;
        while cur != NIL {
            let n = self.node(cur);
            out.push(n.key.clone());
            cur = n.next;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_put_basic() {
        let mut c: LruCache<i32, &str> = LruCache::new(2);
        assert_eq!(c.put(1, "a"), None);
        assert_eq!(c.put(2, "b"), None);
        assert_eq!(c.get(&1), Some(&"a"));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn evicts_least_recently_used() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        // Touch 1 so 2 becomes LRU.
        let _ = c.get(&1);
        let evicted = c.put(3, 30);
        assert_eq!(evicted, Some((2, 20)));
        assert!(!c.contains(&2));
        assert!(c.contains(&1));
        assert!(c.contains(&3));
    }

    #[test]
    fn update_moves_to_front() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        c.put(1, 1);
        c.put(2, 2);
        c.put(1, 100); // update, 1 now MRU
        c.put(3, 3); // evicts 2
        assert_eq!(c.peek(&1), Some(&100));
        assert!(!c.contains(&2));
    }

    #[test]
    fn remove_frees_slot() {
        let mut c: LruCache<i32, &str> = LruCache::new(4);
        c.put(1, "x");
        c.put(2, "y");
        assert_eq!(c.remove(&1), Some("x"));
        assert!(!c.contains(&1));
        assert_eq!(c.len(), 1);
        // Reinsert reuses the freed slot.
        c.put(3, "z");
        assert_eq!(c.peek(&3), Some(&"z"));
    }

    #[test]
    fn mru_order_and_stats() {
        let mut c: LruCache<i32, i32> = LruCache::new(3);
        c.put(1, 1);
        c.put(2, 2);
        c.put(3, 3);
        let _ = c.get(&1);
        assert_eq!(c.keys_mru(), vec![1, 3, 2]);
        let _ = c.get(&99);
        let (hits, misses) = c.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn holds_owned_values() {
        let mut c: LruCache<String, String> = LruCache::new(1);
        c.put("k".into(), "v".into());
        let evicted = c.put("k2".into(), "v2".into());
        assert_eq!(evicted, Some(("k".to_string(), "v".to_string())));
    }
}
