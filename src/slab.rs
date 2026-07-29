//! A slab allocator handing out reusable integer slots.
//!
//! Several pools (page frames, cursor handles, transaction slots) allocate and
//! free fixed-size records identified by an integer id, reusing freed ids to
//! keep the id space dense. A slab is the minimal structure for that: it stores
//! values in a `Vec`, keeps a free list of vacated indices, and hands each live
//! value a stable `usize` key. Unlike the generational [`crate::arena::Arena`]
//! it does not detect stale keys — it is used where the caller guarantees a key
//! is not used after free, in exchange for a smaller per-slot footprint.

/// A slab of `T` with reusable integer keys.
#[derive(Debug, Clone)]
pub struct Slab<T> {
    entries: Vec<Entry<T>>,
    free_head: usize,
    len: usize,
}

#[derive(Debug, Clone)]
enum Entry<T> {
    Occupied(T),
    /// A vacant slot pointing at the next free index (or `usize::MAX`).
    Vacant(usize),
}

const NIL: usize = usize::MAX;

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Slab {
            entries: Vec::new(),
            free_head: NIL,
            len: 0,
        }
    }
}

impl<T> Slab<T> {
    /// A new empty slab.
    pub fn new() -> Slab<T> {
        Slab::default()
    }

    /// A slab pre-sized for `cap` slots.
    pub fn with_capacity(cap: usize) -> Slab<T> {
        Slab {
            entries: Vec::with_capacity(cap),
            free_head: NIL,
            len: 0,
        }
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total slot count (live + vacant).
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Insert a value, returning its key.
    pub fn insert(&mut self, value: T) -> usize {
        self.len += 1;
        if self.free_head != NIL {
            let key = self.free_head;
            if let Entry::Vacant(next) = self.entries[key] {
                self.free_head = next;
            }
            self.entries[key] = Entry::Occupied(value);
            key
        } else {
            let key = self.entries.len();
            self.entries.push(Entry::Occupied(value));
            key
        }
    }

    /// Borrow the value at `key`.
    pub fn get(&self, key: usize) -> Option<&T> {
        match self.entries.get(key) {
            Some(Entry::Occupied(v)) => Some(v),
            _ => None,
        }
    }

    /// Mutably borrow the value at `key`.
    pub fn get_mut(&mut self, key: usize) -> Option<&mut T> {
        match self.entries.get_mut(key) {
            Some(Entry::Occupied(v)) => Some(v),
            _ => None,
        }
    }

    /// `true` if `key` refers to a live slot.
    pub fn contains(&self, key: usize) -> bool {
        matches!(self.entries.get(key), Some(Entry::Occupied(_)))
    }

    /// Remove and return the value at `key`.
    pub fn remove(&mut self, key: usize) -> Option<T> {
        match self.entries.get_mut(key) {
            Some(entry @ Entry::Occupied(_)) => {
                let old = std::mem::replace(entry, Entry::Vacant(self.free_head));
                self.free_head = key;
                self.len -= 1;
                if let Entry::Occupied(v) = old {
                    Some(v)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Iterate live `(key, &value)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.entries.iter().enumerate().filter_map(|(i, e)| match e {
            Entry::Occupied(v) => Some((i, v)),
            Entry::Vacant(_) => None,
        })
    }

    /// Drop all entries, keeping capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.free_head = NIL;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut s: Slab<&str> = Slab::new();
        let k = s.insert("hi");
        assert_eq!(s.get(k), Some(&"hi"));
        assert_eq!(s.len(), 1);
        assert_eq!(s.remove(k), Some("hi"));
        assert!(!s.contains(k));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn reuses_freed_slots() {
        let mut s: Slab<i32> = Slab::new();
        let a = s.insert(1);
        let b = s.insert(2);
        s.remove(a);
        let c = s.insert(3);
        assert_eq!(c, a); // reused
        assert_eq!(s.capacity(), 2);
        assert!(s.contains(b));
        assert!(s.contains(c));
    }

    #[test]
    fn mutation() {
        let mut s: Slab<i32> = Slab::new();
        let k = s.insert(10);
        *s.get_mut(k).unwrap() += 5;
        assert_eq!(s.get(k), Some(&15));
    }

    #[test]
    fn iteration() {
        let mut s: Slab<i32> = Slab::new();
        let a = s.insert(1);
        s.insert(2);
        s.insert(3);
        s.remove(a);
        let mut vals: Vec<i32> = s.iter().map(|(_, v)| *v).collect();
        vals.sort();
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn free_list_chains() {
        let mut s: Slab<i32> = Slab::new();
        let a = s.insert(1);
        let b = s.insert(2);
        let c = s.insert(3);
        s.remove(a);
        s.remove(c);
        // Two free slots; two inserts reuse them without growing.
        s.insert(4);
        s.insert(5);
        assert_eq!(s.capacity(), 3);
        assert!(s.contains(b));
    }
}
