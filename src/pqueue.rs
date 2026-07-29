//! A binary min-heap priority queue with keyed decrease-key.
//!
//! The external merge sort and the top-N operator both want a priority queue,
//! and the deadlock/scheduler code wants one that supports lowering an entry's
//! priority in place (decrease-key), which a plain `BinaryHeap` cannot do. This
//! is a binary heap over `(priority, key)` pairs with a side table mapping each
//! key to its current heap position, so `decrease_key` and `contains` are
//! supported without a linear scan.

use std::collections::HashMap;

/// A min-priority queue keyed by `u32`, ordered by `i64` priority.
#[derive(Debug, Default)]
pub struct PriorityQueue {
    heap: Vec<(i64, u32)>,
    /// key -> index in `heap`.
    pos: HashMap<u32, usize>,
}

impl PriorityQueue {
    /// An empty queue.
    pub fn new() -> PriorityQueue {
        PriorityQueue {
            heap: Vec::new(),
            pos: HashMap::new(),
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// `true` if the key is present.
    pub fn contains(&self, key: u32) -> bool {
        self.pos.contains_key(&key)
    }

    /// The current priority of a key, if present.
    pub fn priority_of(&self, key: u32) -> Option<i64> {
        self.pos.get(&key).map(|&i| self.heap[i].0)
    }

    /// Insert a key with a priority, or update it (taking the lower priority if
    /// already present).
    pub fn push(&mut self, key: u32, priority: i64) {
        if let Some(&i) = self.pos.get(&key) {
            if priority < self.heap[i].0 {
                self.heap[i].0 = priority;
                self.sift_up(i);
            } else if priority > self.heap[i].0 {
                self.heap[i].0 = priority;
                self.sift_down(i);
            }
            return;
        }
        let i = self.heap.len();
        self.heap.push((priority, key));
        self.pos.insert(key, i);
        self.sift_up(i);
    }

    /// Lower a key's priority; ignored if the new priority is not lower or the
    /// key is absent.
    pub fn decrease_key(&mut self, key: u32, priority: i64) -> bool {
        if let Some(&i) = self.pos.get(&key) {
            if priority < self.heap[i].0 {
                self.heap[i].0 = priority;
                self.sift_up(i);
                return true;
            }
        }
        false
    }

    /// Peek the minimum `(priority, key)`.
    pub fn peek(&self) -> Option<(i64, u32)> {
        self.heap.first().copied()
    }

    /// Remove and return the minimum `(priority, key)`.
    pub fn pop(&mut self) -> Option<(i64, u32)> {
        if self.heap.is_empty() {
            return None;
        }
        let last = self.heap.len() - 1;
        self.heap.swap(0, last);
        let (prio, key) = self.heap.pop().unwrap();
        self.pos.remove(&key);
        if !self.heap.is_empty() {
            let moved = self.heap[0].1;
            self.pos.insert(moved, 0);
            self.sift_down(0);
        }
        Some((prio, key))
    }

    /// Remove a specific key regardless of position.
    pub fn remove(&mut self, key: u32) -> Option<i64> {
        let i = *self.pos.get(&key)?;
        let last = self.heap.len() - 1;
        let prio = self.heap[i].0;
        self.heap.swap(i, last);
        self.heap.pop();
        self.pos.remove(&key);
        if i < self.heap.len() {
            let moved = self.heap[i].1;
            self.pos.insert(moved, i);
            self.sift_up(i);
            self.sift_down(i);
        }
        Some(prio)
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 {
                self.swap(i, parent);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut smallest = i;
            if l < n && self.heap[l].0 < self.heap[smallest].0 {
                smallest = l;
            }
            if r < n && self.heap[r].0 < self.heap[smallest].0 {
                smallest = r;
            }
            if smallest != i {
                self.swap(i, smallest);
                i = smallest;
            } else {
                break;
            }
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.pos.insert(self.heap[a].1, a);
        self.pos.insert(self.heap[b].1, b);
    }

    /// Drain the queue in ascending priority order.
    pub fn into_sorted(mut self) -> Vec<(i64, u32)> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(x) = self.pop() {
            out.push(x);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_order() {
        let mut pq = PriorityQueue::new();
        pq.push(1, 30);
        pq.push(2, 10);
        pq.push(3, 20);
        assert_eq!(pq.pop(), Some((10, 2)));
        assert_eq!(pq.pop(), Some((20, 3)));
        assert_eq!(pq.pop(), Some((30, 1)));
        assert!(pq.is_empty());
    }

    #[test]
    fn decrease_key_reorders() {
        let mut pq = PriorityQueue::new();
        pq.push(1, 100);
        pq.push(2, 50);
        assert!(pq.decrease_key(1, 10));
        assert_eq!(pq.peek(), Some((10, 1)));
        // A non-decrease is ignored.
        assert!(!pq.decrease_key(1, 20));
    }

    #[test]
    fn push_existing_updates() {
        let mut pq = PriorityQueue::new();
        pq.push(5, 40);
        pq.push(5, 15);
        assert_eq!(pq.len(), 1);
        assert_eq!(pq.priority_of(5), Some(15));
    }

    #[test]
    fn remove_arbitrary() {
        let mut pq = PriorityQueue::new();
        for (k, p) in [(1, 5), (2, 3), (3, 8), (4, 1)] {
            pq.push(k, p);
        }
        assert_eq!(pq.remove(3), Some(8));
        assert!(!pq.contains(3));
        assert_eq!(pq.len(), 3);
    }

    #[test]
    fn into_sorted_is_ascending() {
        let mut pq = PriorityQueue::new();
        for (k, p) in [(1, 9), (2, 2), (3, 7), (4, 4), (5, 1)] {
            pq.push(k, p);
        }
        let sorted = pq.into_sorted();
        let prios: Vec<i64> = sorted.iter().map(|(p, _)| *p).collect();
        assert_eq!(prios, vec![1, 2, 4, 7, 9]);
    }
}
