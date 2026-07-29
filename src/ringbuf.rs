//! A fixed-capacity ring buffer (circular queue).
//!
//! Morsel-driven execution passes batches between operators through bounded
//! queues, and the WAL group-commit path batches records in a rolling window.
//! A ring buffer gives O(1) push/pop with no reallocation and a hard capacity
//! bound, optionally overwriting the oldest element when full (useful for
//! fixed-size history buffers).

/// A bounded circular buffer of `T`.
pub struct RingBuffer<T> {
    buf: Vec<Option<T>>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// A buffer holding up to `capacity` elements.
    pub fn new(capacity: usize) -> RingBuffer<T> {
        let capacity = capacity.max(1);
        let mut buf = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buf.push(None);
        }
        RingBuffer {
            buf,
            head: 0,
            len: 0,
            capacity,
        }
    }

    /// Number of elements currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` if at capacity.
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// The capacity bound.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn tail(&self) -> usize {
        (self.head + self.len) % self.capacity
    }

    /// Push to the back. Returns `Err(value)` if the buffer is full.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }
        let t = self.tail();
        self.buf[t] = Some(value);
        self.len += 1;
        Ok(())
    }

    /// Push to the back, overwriting (and returning) the oldest element if full.
    pub fn push_overwrite(&mut self, value: T) -> Option<T> {
        if self.is_full() {
            let evicted = self.pop();
            let t = self.tail();
            self.buf[t] = Some(value);
            self.len += 1;
            evicted
        } else {
            let t = self.tail();
            self.buf[t] = Some(value);
            self.len += 1;
            None
        }
    }

    /// Pop from the front.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value = self.buf[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        value
    }

    /// Borrow the front without removing it.
    pub fn peek_front(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.buf[self.head].as_ref()
        }
    }

    /// Borrow the back without removing it.
    pub fn peek_back(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let last = (self.head + self.len - 1) % self.capacity;
            self.buf[last].as_ref()
        }
    }

    /// Element at logical offset `i` from the front.
    pub fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len {
            return None;
        }
        let idx = (self.head + i) % self.capacity;
        self.buf[idx].as_ref()
    }

    /// Remove all elements.
    pub fn clear(&mut self) {
        while self.pop().is_some() {}
        self.head = 0;
    }

    /// Collect elements front-to-back.
    pub fn to_vec(&self) -> Vec<&T> {
        (0..self.len).filter_map(|i| self.get(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_fifo() {
        let mut rb = RingBuffer::new(3);
        rb.push(1).unwrap();
        rb.push(2).unwrap();
        rb.push(3).unwrap();
        assert!(rb.is_full());
        assert_eq!(rb.push(4), Err(4));
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.len(), 1);
    }

    #[test]
    fn wraps_around() {
        let mut rb = RingBuffer::new(3);
        rb.push(1).unwrap();
        rb.push(2).unwrap();
        assert_eq!(rb.pop(), Some(1));
        rb.push(3).unwrap();
        rb.push(4).unwrap(); // wraps into slot 0
        assert_eq!(rb.to_vec(), vec![&2, &3, &4]);
    }

    #[test]
    fn overwrite_evicts_oldest() {
        let mut rb = RingBuffer::new(2);
        rb.push_overwrite(1);
        rb.push_overwrite(2);
        assert_eq!(rb.push_overwrite(3), Some(1));
        assert_eq!(rb.to_vec(), vec![&2, &3]);
    }

    #[test]
    fn peek_and_index() {
        let mut rb = RingBuffer::new(4);
        for i in 10..14 {
            rb.push(i).unwrap();
        }
        assert_eq!(rb.peek_front(), Some(&10));
        assert_eq!(rb.peek_back(), Some(&13));
        assert_eq!(rb.get(2), Some(&12));
        assert_eq!(rb.get(9), None);
    }

    #[test]
    fn clear_resets() {
        let mut rb = RingBuffer::new(3);
        rb.push(1).unwrap();
        rb.push(2).unwrap();
        rb.clear();
        assert!(rb.is_empty());
        rb.push(9).unwrap();
        assert_eq!(rb.peek_front(), Some(&9));
    }
}
