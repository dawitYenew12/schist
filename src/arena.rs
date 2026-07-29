//! A generational arena allocator.
//!
//! Several subsystems want stable handles into a pool of objects that can be
//! freed and the slot reused, without dangling references surviving the free.
//! A generational arena solves this: each slot carries a generation counter
//! that is bumped on free, and a [`Handle`] stores the slot index plus the
//! generation it was minted at. A stale handle (one whose generation no longer
//! matches the slot) resolves to `None` instead of aliasing a newer occupant.

use std::marker::PhantomData;

/// A stable, generation-checked handle into an [`Arena`].
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    /// The raw slot index (for debugging / serialization).
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The generation this handle was minted at.
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

#[derive(Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// A generational arena of `T`.
#[derive(Debug)]
pub struct Arena<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }
}

impl<T> Arena<T> {
    /// A new empty arena.
    pub fn new() -> Arena<T> {
        Arena::default()
    }

    /// Number of live objects.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if there are no live objects.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total slot capacity (live + freed).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Insert a value and return its handle.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            Handle {
                index,
                generation: slot.generation,
                _marker: PhantomData,
            }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            Handle {
                index,
                generation: 0,
                _marker: PhantomData,
            }
        }
    }

    /// `true` if the handle still refers to a live object.
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Borrow the object behind a handle, or `None` if the handle is stale.
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_ref()
    }

    /// Mutably borrow the object behind a handle.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.value.as_mut()
    }

    /// Remove and return the object behind a handle, bumping the slot's
    /// generation so the handle can never resolve again.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(handle.index);
        self.len -= 1;
        Some(value)
    }

    /// Iterate over live objects and their handles.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            slot.value.as_ref().map(|v| {
                (
                    Handle {
                        index: i as u32,
                        generation: slot.generation,
                        _marker: PhantomData,
                    },
                    v,
                )
            })
        })
    }

    /// Drop every object, keeping the allocated capacity.
    pub fn clear(&mut self) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.value.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(i as u32);
            }
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut a: Arena<String> = Arena::new();
        let h = a.insert("hello".to_string());
        assert_eq!(a.get(h).map(|s| s.as_str()), Some("hello"));
        assert_eq!(a.len(), 1);
        assert_eq!(a.remove(h).unwrap(), "hello");
        assert!(a.get(h).is_none());
        assert_eq!(a.len(), 0);
    }

    #[test]
    fn stale_handle_after_reuse() {
        let mut a: Arena<i32> = Arena::new();
        let h1 = a.insert(1);
        a.remove(h1);
        let h2 = a.insert(2);
        // Slot is reused but the generation differs.
        assert_eq!(h1.index(), h2.index());
        assert_ne!(h1.generation(), h2.generation());
        assert!(a.get(h1).is_none());
        assert_eq!(a.get(h2), Some(&2));
    }

    #[test]
    fn mutation_through_handle() {
        let mut a: Arena<i32> = Arena::new();
        let h = a.insert(10);
        *a.get_mut(h).unwrap() += 5;
        assert_eq!(a.get(h), Some(&15));
    }

    #[test]
    fn iteration_skips_freed() {
        let mut a: Arena<i32> = Arena::new();
        let h1 = a.insert(1);
        let _h2 = a.insert(2);
        let h3 = a.insert(3);
        a.remove(h1);
        let mut vals: Vec<i32> = a.iter().map(|(_, v)| *v).collect();
        vals.sort();
        assert_eq!(vals, vec![2, 3]);
        assert!(a.contains(h3));
    }

    #[test]
    fn clear_invalidates_all() {
        let mut a: Arena<i32> = Arena::new();
        let h = a.insert(1);
        a.insert(2);
        a.clear();
        assert_eq!(a.len(), 0);
        assert!(a.get(h).is_none());
        // Capacity is retained for reuse.
        assert!(a.capacity() >= 2);
    }
}
