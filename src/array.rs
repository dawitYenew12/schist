//! Typed in-memory columnar arrays.
//!
//! Where [`crate::value::Value`] is a boxed, one-cell-at-a-time representation,
//! an [`Array`] is the batch-friendly form: a contiguous, type-homogeneous
//! buffer plus an optional validity bitmap. The vectorized executor
//! ([`crate::vexec`]) works on `ColumnVector`s, but many of the analytical
//! helpers — filtering, casting, dictionary materialization, statistics —
//! are most naturally expressed over these typed arrays, and the import/export
//! paths ([`crate::csvio`], [`crate::jsonio`]) build and consume them directly.
//!
//! The arrays here are intentionally simple and owned: each carries its own
//! `Vec` of values and its own validity bitmap. They are cheap to slice
//! (slicing shares nothing — it copies the window) and cheap to build through
//! the per-type builders.

use crate::schema::ColKind;
use crate::value::Value;
use std::fmt;

/// A validity bitmap: one bit per slot, `true` meaning "valid / non-null".
///
/// The bitmap is stored little-endian within each byte (bit 0 is the lowest
/// slot in the byte). An empty bitmap is treated as "all valid".
#[derive(Clone, PartialEq, Eq)]
pub struct Validity {
    bits: Vec<u8>,
    len: usize,
    /// Number of set (valid) bits, cached so `null_count` is O(1).
    set: usize,
}

impl Validity {
    /// A validity map of `len` slots, all valid.
    pub fn all_valid(len: usize) -> Validity {
        Validity {
            bits: Vec::new(),
            len,
            set: len,
        }
    }

    /// A validity map of `len` slots, all null.
    pub fn all_null(len: usize) -> Validity {
        let nbytes = len.div_ceil(8);
        Validity {
            bits: vec![0u8; nbytes],
            len,
            set: 0,
        }
    }

    /// Build from an explicit iterator of booleans.
    pub fn from_bools<I: IntoIterator<Item = bool>>(iter: I) -> Validity {
        let mut v = Validity {
            bits: Vec::new(),
            len: 0,
            set: 0,
        };
        for b in iter {
            v.push(b);
        }
        v
    }

    /// Number of slots covered.
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if there are no slots.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of null (unset) slots.
    pub fn null_count(&self) -> usize {
        self.len - self.set
    }

    /// `true` if every slot is valid.
    pub fn all_set(&self) -> bool {
        self.set == self.len
    }

    /// Query slot `i` (out of range reads as null).
    pub fn get(&self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        if self.bits.is_empty() {
            return true;
        }
        let byte = self.bits[i >> 3];
        (byte >> (i & 7)) & 1 == 1
    }

    /// Append one slot.
    pub fn push(&mut self, valid: bool) {
        let i = self.len;
        // Materialize the backing bytes lazily: as soon as we need to store a
        // `false`, or once we have any explicit bytes, keep them dense.
        if self.bits.is_empty() && valid {
            self.len += 1;
            self.set += 1;
            return;
        }
        if self.bits.is_empty() {
            // First null in an otherwise-all-valid run: densify.
            let nbytes = (self.len + 1).div_ceil(8);
            self.bits = vec![0u8; nbytes];
            for j in 0..self.len {
                self.bits[j >> 3] |= 1 << (j & 7);
            }
        }
        let need = (i + 1).div_ceil(8);
        if self.bits.len() < need {
            self.bits.resize(need, 0);
        }
        if valid {
            self.bits[i >> 3] |= 1 << (i & 7);
            self.set += 1;
        }
        self.len += 1;
    }

    /// Set slot `i`'s validity (grows if needed).
    pub fn set(&mut self, i: usize, valid: bool) {
        while self.len <= i {
            self.push(true);
        }
        let cur = self.get(i);
        if cur == valid {
            return;
        }
        if self.bits.is_empty() {
            // Was all-valid; densify.
            self.bits = vec![0u8; self.len.div_ceil(8)];
            for j in 0..self.len {
                self.bits[j >> 3] |= 1 << (j & 7);
            }
        }
        if valid {
            self.bits[i >> 3] |= 1 << (i & 7);
            self.set += 1;
        } else {
            self.bits[i >> 3] &= !(1 << (i & 7));
            self.set -= 1;
        }
    }

    /// Produce a validity map for the slice `[start, start+len)`.
    pub fn slice(&self, start: usize, len: usize) -> Validity {
        let mut out = Validity::from_bools((start..start + len).map(|i| self.get(i)));
        // Preserve len for a fully-out-of-range slice.
        while out.len < len {
            out.push(false);
        }
        out
    }

    /// Iterate validities.
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.get(i))
    }

    /// Bitwise AND with another validity map (element-wise), truncating to the
    /// shorter length.
    pub fn and(&self, other: &Validity) -> Validity {
        let n = self.len.min(other.len);
        Validity::from_bools((0..n).map(|i| self.get(i) && other.get(i)))
    }
}

impl fmt::Debug for Validity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Validity[{}/{} valid]", self.set, self.len)
    }
}

/// A typed, contiguous column of values.
#[derive(Clone, Debug, PartialEq)]
pub enum Array {
    Bool(BoolArray),
    Int(IntArray),
    Real(RealArray),
    Text(TextArray),
}

impl Array {
    /// The element kind of this array.
    pub fn kind(&self) -> ColKind {
        match self {
            Array::Bool(_) => ColKind::Bool,
            Array::Int(_) => ColKind::Int,
            Array::Real(_) => ColKind::Real,
            Array::Text(_) => ColKind::Text,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Array::Bool(a) => a.len(),
            Array::Int(a) => a.len(),
            Array::Real(a) => a.len(),
            Array::Text(a) => a.len(),
        }
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of null elements.
    pub fn null_count(&self) -> usize {
        match self {
            Array::Bool(a) => a.validity.null_count(),
            Array::Int(a) => a.validity.null_count(),
            Array::Real(a) => a.validity.null_count(),
            Array::Text(a) => a.validity.null_count(),
        }
    }

    /// Read element `i` as a [`Value`].
    pub fn value(&self, i: usize) -> Value {
        match self {
            Array::Bool(a) => a.value(i),
            Array::Int(a) => a.value(i),
            Array::Real(a) => a.value(i),
            Array::Text(a) => a.value(i),
        }
    }

    /// Collect all elements as `Value`s.
    pub fn to_values(&self) -> Vec<Value> {
        (0..self.len()).map(|i| self.value(i)).collect()
    }

    /// Slice `[start, start+len)` into a new owned array.
    pub fn slice(&self, start: usize, len: usize) -> Array {
        match self {
            Array::Bool(a) => Array::Bool(a.slice(start, len)),
            Array::Int(a) => Array::Int(a.slice(start, len)),
            Array::Real(a) => Array::Real(a.slice(start, len)),
            Array::Text(a) => Array::Text(a.slice(start, len)),
        }
    }

    /// Keep only elements where `mask[i]` is true.
    pub fn filter(&self, mask: &[bool]) -> Array {
        match self {
            Array::Bool(a) => Array::Bool(a.filter(mask)),
            Array::Int(a) => Array::Int(a.filter(mask)),
            Array::Real(a) => Array::Real(a.filter(mask)),
            Array::Text(a) => Array::Text(a.filter(mask)),
        }
    }

    /// Gather elements at the given indices (indices out of range become null).
    pub fn take(&self, idx: &[usize]) -> Array {
        match self {
            Array::Bool(a) => Array::Bool(a.take(idx)),
            Array::Int(a) => Array::Int(a.take(idx)),
            Array::Real(a) => Array::Real(a.take(idx)),
            Array::Text(a) => Array::Text(a.take(idx)),
        }
    }

    /// Build an array of `n` nulls of the given kind.
    pub fn nulls(kind: ColKind, n: usize) -> Array {
        match kind {
            ColKind::Bool => Array::Bool(BoolArray {
                data: vec![false; n],
                validity: Validity::all_null(n),
            }),
            ColKind::Int => Array::Int(IntArray {
                data: vec![0; n],
                validity: Validity::all_null(n),
            }),
            ColKind::Real => Array::Real(RealArray {
                data: vec![0.0; n],
                validity: Validity::all_null(n),
            }),
            ColKind::Text => Array::Text(TextArray {
                data: vec![0; n],
                validity: Validity::all_null(n),
            }),
        }
    }

    /// Build from a slice of `Value`s, coercing to `kind`.
    pub fn from_values(kind: ColKind, values: &[Value]) -> Array {
        let mut b = ArrayBuilder::new(kind);
        for v in values {
            b.push(*v);
        }
        b.finish()
    }
}

/// Boolean column.
#[derive(Clone, Debug, PartialEq)]
pub struct BoolArray {
    data: Vec<bool>,
    validity: Validity,
}

/// 64-bit signed integer column.
#[derive(Clone, Debug, PartialEq)]
pub struct IntArray {
    data: Vec<i64>,
    validity: Validity,
}

/// 64-bit float column.
#[derive(Clone, Debug, PartialEq)]
pub struct RealArray {
    data: Vec<f64>,
    validity: Validity,
}

/// Dictionary-id text column (ids reference a dictionary page).
#[derive(Clone, Debug, PartialEq)]
pub struct TextArray {
    data: Vec<u32>,
    validity: Validity,
}

macro_rules! prim_array_impl {
    ($ty:ty, $arr:ident, $variant:ident, $val:expr, $back:expr) => {
        impl $arr {
            /// Number of elements.
            pub fn len(&self) -> usize {
                self.data.len()
            }
            /// `true` if empty.
            pub fn is_empty(&self) -> bool {
                self.data.is_empty()
            }
            /// The raw backing slice (includes garbage for null slots).
            pub fn values(&self) -> &[$ty] {
                &self.data
            }
            /// The validity map.
            pub fn validity(&self) -> &Validity {
                &self.validity
            }
            /// Read raw element `i` without checking validity.
            pub fn raw(&self, i: usize) -> $ty {
                self.data[i]
            }
            /// Read element `i` as a [`Value`] (null-aware).
            pub fn value(&self, i: usize) -> Value {
                if i >= self.data.len() || !self.validity.get(i) {
                    return Value::Null;
                }
                let x = self.data[i];
                $val(x)
            }
            /// Slice `[start, start+len)`.
            pub fn slice(&self, start: usize, len: usize) -> $arr {
                let end = (start + len).min(self.data.len());
                let s = start.min(self.data.len());
                $arr {
                    data: self.data[s..end].to_vec(),
                    validity: self.validity.slice(s, end - s),
                }
            }
            /// Filter by boolean mask.
            pub fn filter(&self, mask: &[bool]) -> $arr {
                let mut data = Vec::new();
                let mut validity = Validity::all_valid(0);
                for i in 0..self.data.len() {
                    if mask.get(i).copied().unwrap_or(false) {
                        data.push(self.data[i]);
                        validity.push(self.validity.get(i));
                    }
                }
                $arr { data, validity }
            }
            /// Gather at indices.
            pub fn take(&self, idx: &[usize]) -> $arr {
                let mut data = Vec::with_capacity(idx.len());
                let mut validity = Validity::all_valid(0);
                for &j in idx {
                    if j < self.data.len() && self.validity.get(j) {
                        data.push(self.data[j]);
                        validity.push(true);
                    } else {
                        data.push(Default::default());
                        validity.push(false);
                    }
                }
                $arr { data, validity }
            }
        }
    };
}

prim_array_impl!(bool, BoolArray, Bool, Value::Bool, |v: Value| v.as_int().map(|i| i != 0));
prim_array_impl!(i64, IntArray, Int, Value::Int, |v: Value| v.as_int());
prim_array_impl!(f64, RealArray, Real, Value::Real, |v: Value| v.as_real());
prim_array_impl!(u32, TextArray, Text, Value::Text, |v: Value| v.as_text_id());

impl IntArray {
    /// Sum of the non-null elements.
    pub fn sum(&self) -> i64 {
        let mut s = 0i64;
        for i in 0..self.data.len() {
            if self.validity.get(i) {
                s = s.wrapping_add(self.data[i]);
            }
        }
        s
    }

    /// Minimum non-null element.
    pub fn min(&self) -> Option<i64> {
        let mut m: Option<i64> = None;
        for i in 0..self.data.len() {
            if self.validity.get(i) {
                m = Some(m.map_or(self.data[i], |x| x.min(self.data[i])));
            }
        }
        m
    }

    /// Maximum non-null element.
    pub fn max(&self) -> Option<i64> {
        let mut m: Option<i64> = None;
        for i in 0..self.data.len() {
            if self.validity.get(i) {
                m = Some(m.map_or(self.data[i], |x| x.max(self.data[i])));
            }
        }
        m
    }
}

impl RealArray {
    /// Sum of the non-null elements.
    pub fn sum(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.data.len() {
            if self.validity.get(i) {
                s += self.data[i];
            }
        }
        s
    }
}

/// Builds a typed [`Array`] one [`Value`] at a time, coercing to a fixed kind.
pub struct ArrayBuilder {
    kind: ColKind,
    bools: Vec<bool>,
    ints: Vec<i64>,
    reals: Vec<f64>,
    texts: Vec<u32>,
    validity: Validity,
}

impl ArrayBuilder {
    /// A new builder producing an array of `kind`.
    pub fn new(kind: ColKind) -> ArrayBuilder {
        ArrayBuilder {
            kind,
            bools: Vec::new(),
            ints: Vec::new(),
            reals: Vec::new(),
            texts: Vec::new(),
            validity: Validity::all_valid(0),
        }
    }

    /// The element kind.
    pub fn kind(&self) -> ColKind {
        self.kind
    }

    /// Number of elements pushed so far.
    pub fn len(&self) -> usize {
        self.validity.len()
    }

    /// `true` if nothing has been pushed.
    pub fn is_empty(&self) -> bool {
        self.validity.is_empty()
    }

    /// Append one value, coercing to the builder's kind. A value that cannot be
    /// coerced (or an explicit null) is appended as null.
    pub fn push(&mut self, v: Value) {
        match self.kind {
            ColKind::Bool => {
                if let Some(i) = v.as_int() {
                    self.bools.push(i != 0);
                    self.validity.push(!v.is_null());
                } else {
                    self.bools.push(false);
                    self.validity.push(false);
                }
            }
            ColKind::Int => match v.as_int() {
                Some(i) if !v.is_null() => {
                    self.ints.push(i);
                    self.validity.push(true);
                }
                _ => {
                    self.ints.push(0);
                    self.validity.push(false);
                }
            },
            ColKind::Real => match v.as_real() {
                Some(r) if !v.is_null() => {
                    self.reals.push(r);
                    self.validity.push(true);
                }
                _ => {
                    self.reals.push(0.0);
                    self.validity.push(false);
                }
            },
            ColKind::Text => match v.as_text_id() {
                Some(id) => {
                    self.texts.push(id);
                    self.validity.push(true);
                }
                None => {
                    self.texts.push(0);
                    self.validity.push(false);
                }
            },
        }
    }

    /// Append an explicit null.
    pub fn push_null(&mut self) {
        self.push(Value::Null);
    }

    /// Append a raw integer (only valid for an int builder).
    pub fn push_int(&mut self, i: i64) {
        debug_assert_eq!(self.kind, ColKind::Int);
        self.ints.push(i);
        self.validity.push(true);
    }

    /// Consume the builder and produce the array.
    pub fn finish(self) -> Array {
        match self.kind {
            ColKind::Bool => Array::Bool(BoolArray {
                data: self.bools,
                validity: self.validity,
            }),
            ColKind::Int => Array::Int(IntArray {
                data: self.ints,
                validity: self.validity,
            }),
            ColKind::Real => Array::Real(RealArray {
                data: self.reals,
                validity: self.validity,
            }),
            ColKind::Text => Array::Text(TextArray {
                data: self.texts,
                validity: self.validity,
            }),
        }
    }
}

/// A horizontal collection of equal-length arrays with column names — the
/// in-memory batch that scans yield and that the import/export code exchanges.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordBatch {
    names: Vec<String>,
    columns: Vec<Array>,
    rows: usize,
}

impl RecordBatch {
    /// Build a batch from `(name, array)` pairs. All arrays must have equal
    /// length; the shorter ones are padded with trailing nulls.
    pub fn new(pairs: Vec<(String, Array)>) -> RecordBatch {
        let rows = pairs.iter().map(|(_, a)| a.len()).max().unwrap_or(0);
        let mut names = Vec::with_capacity(pairs.len());
        let mut columns = Vec::with_capacity(pairs.len());
        for (n, a) in pairs {
            let a = if a.len() < rows {
                let kind = a.kind();
                let mut b = ArrayBuilder::new(kind);
                for i in 0..a.len() {
                    b.push(a.value(i));
                }
                for _ in a.len()..rows {
                    b.push_null();
                }
                b.finish()
            } else {
                a
            };
            names.push(n);
            columns.push(a);
        }
        RecordBatch {
            names,
            columns,
            rows,
        }
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// `true` if there are no rows.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Column names.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Column arrays.
    pub fn columns(&self) -> &[Array] {
        &self.columns
    }

    /// Column by name.
    pub fn column(&self, name: &str) -> Option<&Array> {
        self.names.iter().position(|n| n == name).map(|i| &self.columns[i])
    }

    /// Column by position.
    pub fn column_at(&self, i: usize) -> Option<&Array> {
        self.columns.get(i)
    }

    /// Materialize row `r` as a vector of values.
    pub fn row(&self, r: usize) -> Vec<Value> {
        self.columns.iter().map(|c| c.value(r)).collect()
    }

    /// Slice a window of rows into a new batch.
    pub fn slice(&self, start: usize, len: usize) -> RecordBatch {
        let cols = self
            .columns
            .iter()
            .map(|c| c.slice(start, len))
            .collect::<Vec<_>>();
        let rows = cols.first().map(|c| c.len()).unwrap_or(0);
        RecordBatch {
            names: self.names.clone(),
            columns: cols,
            rows,
        }
    }

    /// Filter every column by the same boolean mask.
    pub fn filter(&self, mask: &[bool]) -> RecordBatch {
        let cols = self.columns.iter().map(|c| c.filter(mask)).collect::<Vec<_>>();
        let rows = cols.first().map(|c| c.len()).unwrap_or(0);
        RecordBatch {
            names: self.names.clone(),
            columns: cols,
            rows,
        }
    }

    /// Concatenate two batches with identical schema.
    pub fn concat(&self, other: &RecordBatch) -> RecordBatch {
        let mut pairs = Vec::new();
        for (i, name) in self.names.iter().enumerate() {
            let kind = self.columns[i].kind();
            let mut b = ArrayBuilder::new(kind);
            for r in 0..self.rows {
                b.push(self.columns[i].value(r));
            }
            if let Some(oc) = other.column(name) {
                for r in 0..other.rows {
                    b.push(oc.value(r));
                }
            }
            pairs.push((name.clone(), b.finish()));
        }
        RecordBatch::new(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validity_push_and_query() {
        let mut v = Validity::all_valid(0);
        v.push(true);
        v.push(false);
        v.push(true);
        assert_eq!(v.len(), 3);
        assert!(v.get(0));
        assert!(!v.get(1));
        assert!(v.get(2));
        assert_eq!(v.null_count(), 1);
    }

    #[test]
    fn validity_set_densifies() {
        let mut v = Validity::all_valid(4);
        assert!(v.all_set());
        v.set(2, false);
        assert!(!v.get(2));
        assert_eq!(v.null_count(), 1);
        v.set(2, true);
        assert!(v.all_set());
    }

    #[test]
    fn int_array_stats() {
        let a = Array::from_values(ColKind::Int, &[Value::Int(3), Value::Null, Value::Int(5)]);
        if let Array::Int(ia) = &a {
            assert_eq!(ia.sum(), 8);
            assert_eq!(ia.min(), Some(3));
            assert_eq!(ia.max(), Some(5));
        } else {
            panic!("expected int array");
        }
        assert_eq!(a.null_count(), 1);
        assert_eq!(a.value(1), Value::Null);
    }

    #[test]
    fn array_filter_and_take() {
        let a = Array::from_values(
            ColKind::Int,
            &[Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)],
        );
        let f = a.filter(&[true, false, true, false]);
        assert_eq!(f.len(), 2);
        assert_eq!(f.value(0), Value::Int(1));
        assert_eq!(f.value(1), Value::Int(3));
        let t = a.take(&[3, 0, 9]);
        assert_eq!(t.value(0), Value::Int(4));
        assert_eq!(t.value(1), Value::Int(1));
        assert_eq!(t.value(2), Value::Null);
    }

    #[test]
    fn record_batch_ops() {
        let a = Array::from_values(ColKind::Int, &[Value::Int(1), Value::Int(2)]);
        let b = Array::from_values(ColKind::Bool, &[Value::Bool(true), Value::Bool(false)]);
        let batch = RecordBatch::new(vec![("n".into(), a), ("b".into(), b)]);
        assert_eq!(batch.rows(), 2);
        assert_eq!(batch.width(), 2);
        assert_eq!(batch.row(0), vec![Value::Int(1), Value::Bool(true)]);
        let f = batch.filter(&[false, true]);
        assert_eq!(f.rows(), 1);
        assert_eq!(f.row(0), vec![Value::Int(2), Value::Bool(false)]);
    }

    #[test]
    fn record_batch_concat() {
        let a = Array::from_values(ColKind::Int, &[Value::Int(1)]);
        let b = Array::from_values(ColKind::Int, &[Value::Int(2)]);
        let ba = RecordBatch::new(vec![("n".into(), a)]);
        let bb = RecordBatch::new(vec![("n".into(), b)]);
        let c = ba.concat(&bb);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.row(1), vec![Value::Int(2)]);
    }
}
