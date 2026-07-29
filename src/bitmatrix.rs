//! A dense bit matrix.
//!
//! Reachability closures over a small plan graph and cross-product match masks
//! in a nested-loop join are naturally expressed as a matrix of bits. This is a
//! row-major dense bit matrix with per-row word packing, supporting the usual
//! bit operations plus a transitive-closure pass (Warshall's algorithm) used to
//! compute reachability among a bounded set of nodes.

/// A `rows x cols` matrix of bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitMatrix {
    rows: usize,
    cols: usize,
    words_per_row: usize,
    data: Vec<u64>,
}

impl BitMatrix {
    /// A `rows x cols` matrix of zeros.
    pub fn new(rows: usize, cols: usize) -> BitMatrix {
        let words_per_row = cols.div_ceil(64).max(1);
        BitMatrix {
            rows,
            cols,
            words_per_row,
            data: vec![0u64; rows * words_per_row],
        }
    }

    /// A square identity matrix of order `n`.
    pub fn identity(n: usize) -> BitMatrix {
        let mut m = BitMatrix::new(n, n);
        for i in 0..n {
            m.set(i, i, true);
        }
        m
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    fn index(&self, r: usize, c: usize) -> (usize, u64) {
        let word = r * self.words_per_row + (c >> 6);
        (word, 1u64 << (c & 63))
    }

    /// Get bit `(r, c)`.
    pub fn get(&self, r: usize, c: usize) -> bool {
        if r >= self.rows || c >= self.cols {
            return false;
        }
        let (word, mask) = self.index(r, c);
        self.data[word] & mask != 0
    }

    /// Set bit `(r, c)`.
    pub fn set(&mut self, r: usize, c: usize, value: bool) {
        if r >= self.rows || c >= self.cols {
            return;
        }
        let (word, mask) = self.index(r, c);
        if value {
            self.data[word] |= mask;
        } else {
            self.data[word] &= !mask;
        }
    }

    /// Number of set bits in row `r`.
    pub fn row_popcount(&self, r: usize) -> usize {
        if r >= self.rows {
            return 0;
        }
        let start = r * self.words_per_row;
        self.data[start..start + self.words_per_row]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Total set bits.
    pub fn popcount(&self) -> usize {
        self.data.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// OR another row's bits into row `r` (both same width). Returns whether any
    /// bit changed.
    fn or_row_into(&mut self, dst: usize, src: usize) -> bool {
        let mut changed = false;
        let dbase = dst * self.words_per_row;
        let sbase = src * self.words_per_row;
        for i in 0..self.words_per_row {
            let before = self.data[dbase + i];
            let after = before | self.data[sbase + i];
            if after != before {
                changed = true;
            }
            self.data[dbase + i] = after;
        }
        changed
    }

    /// In-place transitive closure via Warshall's algorithm (square matrices
    /// only). Interprets the matrix as an adjacency matrix and adds all
    /// reachable edges.
    pub fn transitive_closure(&mut self) {
        assert_eq!(self.rows, self.cols, "closure needs a square matrix");
        for k in 0..self.rows {
            for i in 0..self.rows {
                if self.get(i, k) {
                    self.or_row_into(i, k);
                }
            }
        }
    }

    /// `true` if `to` is reachable from `from` after a closure pass.
    pub fn reachable(&self, from: usize, to: usize) -> bool {
        self.get(from, to)
    }

    /// Row `r` as the list of set column indices.
    pub fn row_indices(&self, r: usize) -> Vec<usize> {
        (0..self.cols).filter(|&c| self.get(r, c)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get() {
        let mut m = BitMatrix::new(3, 100);
        m.set(1, 70, true);
        assert!(m.get(1, 70));
        assert!(!m.get(1, 71));
        assert_eq!(m.row_popcount(1), 1);
        m.set(1, 70, false);
        assert!(!m.get(1, 70));
    }

    #[test]
    fn identity_matrix() {
        let m = BitMatrix::identity(4);
        assert_eq!(m.popcount(), 4);
        assert!(m.get(2, 2));
        assert!(!m.get(2, 3));
    }

    #[test]
    fn transitive_closure_reachability() {
        // 0 -> 1 -> 2 -> 3, plus self-loops from identity semantics.
        let mut m = BitMatrix::new(4, 4);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 3, true);
        m.transitive_closure();
        assert!(m.reachable(0, 3));
        assert!(m.reachable(1, 3));
        assert!(!m.reachable(3, 0));
    }

    #[test]
    fn closure_with_cycle() {
        let mut m = BitMatrix::new(3, 3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 0, true);
        m.transitive_closure();
        // Everything reaches everything.
        for i in 0..3 {
            for j in 0..3 {
                assert!(m.reachable(i, j));
            }
        }
    }

    #[test]
    fn row_indices_listed() {
        let mut m = BitMatrix::new(2, 10);
        m.set(0, 1, true);
        m.set(0, 5, true);
        m.set(0, 9, true);
        assert_eq!(m.row_indices(0), vec![1, 5, 9]);
        assert!(m.row_indices(1).is_empty());
    }

    #[test]
    fn out_of_bounds_safe() {
        let mut m = BitMatrix::new(2, 2);
        m.set(5, 5, true); // no-op
        assert!(!m.get(5, 5));
        assert_eq!(m.row_popcount(9), 0);
    }
}
