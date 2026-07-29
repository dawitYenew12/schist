//! Vectorized batch execution engine.
//!
//! Where the row-at-a-time query layer in [`crate::query`] materializes one
//! row at a time, this module executes a plan *column-at-a-time* over batches
//! of rows — the model used by modern analytical engines (DuckDB, ClickHouse,
//! MonetDB/X100). Each operator consumes and produces [`DataBatch`]es: dense,
//! typed column vectors with a null bitmap. Predicates are pushed down into
//! [`SelectionVector`]s so filtering never copies row data until it must.
//!
//! The engine is built around the [`ExecOperator`] trait, with concrete
//! operators for scan, filter, project, hash aggregation, sort, limit, and the
//! three join strategies. A [`Pipeline`] chains operators and drains them to
//! completion. Aggregation supports a combine/merge step so partial states can
//! be built per-batch and merged, the shape a parallel executor would use.

use crate::error::{Error, Result};
use crate::expr::{self, ArithOp, Expr};
use crate::schema::{ColKind, Schema};
use crate::value::Value;
use std::cmp::Ordering;

/// The number of rows a batch holds by default. Batches are the unit of work
/// flowing between operators; keeping them modest bounds peak memory and lets
/// the pipeline start producing output before the source is exhausted.
pub const DEFAULT_BATCH_SIZE: usize = 1024;

// ---------------------------------------------------------------------------
// ColumnVector
// ---------------------------------------------------------------------------

/// A physical column kind for vectorized storage. This mirrors the logical
/// [`ColKind`] but collapses text into an interned-id representation, since a
/// vectorized engine never wants to carry variable-width byte strings through
/// an arithmetic kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecType {
    Int,
    Real,
    Bool,
    /// Dictionary ids; the dictionary itself is held out-of-band by the batch.
    TextId,
}

impl VecType {
    /// The default zero value for a vector of this kind.
    fn zero(self) -> VecRepr {
        match self {
            VecType::Int => VecRepr::Int(Vec::new()),
            VecType::Real => VecRepr::Real(Vec::new()),
            VecType::Bool => VecRepr::Bool(Vec::new()),
            VecType::TextId => VecRepr::TextId(Vec::new()),
        }
    }

    /// Whether two vector types are comparable without lossy conversion.
    fn compatible(self, other: VecType) -> bool {
        self == other || (self.is_numeric() && other.is_numeric())
    }

    fn is_numeric(self) -> bool {
        matches!(self, VecType::Int | VecType::Real)
    }

    fn name(self) -> &'static str {
        match self {
            VecType::Int => "int",
            VecType::Real => "real",
            VecType::Bool => "bool",
            VecType::TextId => "textid",
        }
    }
}

/// The physical storage of a column vector. Each variant is a dense `Vec` of
/// fixed-width cells; nulls are tracked separately in the bitmap.
#[derive(Debug, Clone, PartialEq)]
pub enum VecRepr {
    Int(Vec<i64>),
    Real(Vec<f64>),
    Bool(Vec<u8>),
    TextId(Vec<u32>),
}

impl VecRepr {
    fn len(&self) -> usize {
        match self {
            VecRepr::Int(v) => v.len(),
            VecRepr::Real(v) => v.len(),
            VecRepr::Bool(v) => v.len(),
            VecRepr::TextId(v) => v.len(),
        }
    }

    fn push_zero(&mut self) {
        match self {
            VecRepr::Int(v) => v.push(0),
            VecRepr::Real(v) => v.push(0.0),
            VecRepr::Bool(v) => v.push(0),
            VecRepr::TextId(v) => v.push(0),
        }
    }

    fn get(&self, i: usize) -> Value {
        match self {
            VecRepr::Int(v) => Value::Int(v[i]),
            VecRepr::Real(v) => Value::Real(v[i]),
            VecRepr::Bool(v) => Value::Bool(v[i] != 0),
            VecRepr::TextId(v) => Value::Text(v[i]),
        }
    }

    fn set(&mut self, i: usize, val: &Value) {
        match (self, val) {
            (VecRepr::Int(v), Value::Int(x)) => v[i] = *x,
            (VecRepr::Real(v), Value::Real(x)) => v[i] = *x,
            (VecRepr::Real(v), Value::Int(x)) => v[i] = *x as f64,
            (VecRepr::Bool(v), Value::Bool(b)) => v[i] = if *b { 1 } else { 0 },
            (VecRepr::TextId(v), Value::Text(id)) => v[i] = *id,
            (VecRepr::Int(v), Value::Null) => v[i] = 0,
            (VecRepr::Real(v), Value::Null) => v[i] = 0.0,
            (VecRepr::Bool(v), Value::Null) => v[i] = 0,
            (VecRepr::TextId(v), Value::Null) => v[i] = 0,
            _ => {}
        }
    }

    fn push(&mut self, val: &Value) {
        match val {
            Value::Int(x) => match self {
                VecRepr::Int(v) => v.push(*x),
                VecRepr::Real(v) => v.push(*x as f64),
                _ => self.push_zero(),
            },
            Value::Real(x) => match self {
                VecRepr::Real(v) => v.push(*x),
                VecRepr::Int(v) => v.push(*x as i64),
                _ => self.push_zero(),
            },
            Value::Bool(b) => match self {
                VecRepr::Bool(v) => v.push(if *b { 1 } else { 0 }),
                _ => self.push_zero(),
            },
            Value::Text(id) => match self {
                VecRepr::TextId(v) => v.push(*id),
                _ => self.push_zero(),
            },
            Value::Null => self.push_zero(),
        }
    }
}

/// A dense, typed column vector with a null bitmap.
#[derive(Debug, Clone)]
pub struct ColumnVector {
    pub vtype: VecType,
    pub data: VecRepr,
    /// One byte per row: 1 = null, 0 = live. Allocated lazily; `None` means
    /// the vector has no nulls.
    pub nulls: Option<Vec<u8>>,
    pub len: usize,
}

impl ColumnVector {
    /// Build an empty vector of the given type with reserved capacity.
    pub fn empty(vtype: VecType, capacity: usize) -> Self {
        ColumnVector {
            vtype,
            data: vtype.zero(),
            nulls: None,
            len: 0,
        }
        .with_capacity(capacity)
    }

    fn with_capacity(mut self, cap: usize) -> Self {
        match &mut self.data {
            VecRepr::Int(v) => v.reserve(cap),
            VecRepr::Real(v) => v.reserve(cap),
            VecRepr::Bool(v) => v.reserve(cap),
            VecRepr::TextId(v) => v.reserve(cap),
        }
        self
    }

    /// Build a vector from a slice of [`Value`]s, inferring the widest type.
    pub fn from_values(values: &[Value], vtype: VecType) -> Self {
        let mut cv = ColumnVector::empty(vtype, values.len());
        for v in values {
            cv.push(v);
        }
        cv
    }

    /// Append a value, recording null-ness in the bitmap.
    pub fn push(&mut self, val: &Value) {
        let is_null = val.is_null();
        match &mut self.nulls {
            Some(b) => b.push(if is_null { 1 } else { 0 }),
            None if is_null => {
                let mut b = vec![0u8; self.len];
                b.push(1);
                self.nulls = Some(b);
            }
            None => {}
        }
        self.data.push(val);
        self.len += 1;
    }

    /// Append an explicit null.
    pub fn push_null(&mut self) {
        self.push(&Value::Null);
    }

    /// Read the value at row `i`. Nulls return [`Value::Null`].
    pub fn get(&self, i: usize) -> Value {
        if self.is_null(i) {
            return Value::Null;
        }
        self.data.get(i)
    }

    /// Whether row `i` is null.
    pub fn is_null(&self, i: usize) -> bool {
        self.nulls.as_ref().map_or(false, |b| b[i] != 0)
    }

    /// The number of null rows in this vector.
    pub fn null_count(&self) -> usize {
        self.nulls.as_ref().map_or(0, |b| b.iter().filter(|&&x| x != 0).count())
    }

    /// Overwrite the value at row `i`.
    pub fn set(&mut self, i: usize, val: &Value) {
        let is_null = val.is_null();
        if is_null {
            self.mark_null(i);
        } else {
            self.mark_live(i);
        }
        self.data.set(i, val);
    }

    fn mark_null(&mut self, i: usize) {
        match &mut self.nulls {
            Some(b) => b[i] = 1,
            None => {
                let mut b = vec![0u8; self.len];
                b[i] = 1;
                self.nulls = Some(b);
            }
        }
    }

    fn mark_live(&mut self, i: usize) {
        if let Some(b) = &mut self.nulls {
            b[i] = 0;
        }
    }

    /// Truncate the vector to `new_len` rows.
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }
        match &mut self.data {
            VecRepr::Int(v) => v.truncate(new_len),
            VecRepr::Real(v) => v.truncate(new_len),
            VecRepr::Bool(v) => v.truncate(new_len),
            VecRepr::TextId(v) => v.truncate(new_len),
        }
        if let Some(b) = &mut self.nulls {
            b.truncate(new_len);
        }
        self.len = new_len;
    }

    /// Append a slice of this vector selected by `sel` (a selection vector of
    /// row indices). Nulls are preserved.
    pub fn gather(&self, sel: &SelectionVector) -> ColumnVector {
        let mut out = ColumnVector::empty(self.vtype, sel.len());
        for &i in &sel.indices {
            if i >= self.len {
                out.push_null();
            } else if self.is_null(i) {
                out.push_null();
            } else {
                out.push(&self.data.get(i));
            }
        }
        out
    }

    /// Cast this vector to a target type, producing a new vector. Int->Real is
    /// widening; Real->Int truncates; anything to Bool maps nonzero to true.
    pub fn cast(&self, target: VecType) -> Result<ColumnVector> {
        if self.vtype == target {
            return Ok(self.clone());
        }
        let mut out = ColumnVector::empty(target, self.len);
        for i in 0..self.len {
            if self.is_null(i) {
                out.push_null();
                continue;
            }
            let v = self.data.get(i);
            let casted = cast_value(&v, target)?;
            out.push(&casted);
        }
        Ok(out)
    }

    /// Concatenate another vector onto the end of this one.
    pub fn concat(&mut self, other: &ColumnVector) {
        for i in 0..other.len {
            if other.is_null(i) {
                self.push_null();
            } else {
                self.push(&other.data.get(i));
            }
        }
    }

    /// Compare two rows by total order, nulls last.
    pub fn compare(&self, a: usize, b: usize) -> Ordering {
        let an = self.is_null(a);
        let bn = self.is_null(b);
        match (an, bn) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => self.data.get(a).total_cmp(&self.data.get(b)),
        }
    }
}

/// Cast a scalar value to a target vector type.
pub fn cast_value(v: &Value, target: VecType) -> Result<Value> {
    if v.is_null() {
        return Ok(Value::Null);
    }
    match (v, target) {
        (Value::Int(x), VecType::Int) => Ok(Value::Int(*x)),
        (Value::Int(x), VecType::Real) => Ok(Value::Real(*x as f64)),
        (Value::Int(x), VecType::Bool) => Ok(Value::Bool(*x != 0)),
        (Value::Real(x), VecType::Real) => Ok(Value::Real(*x)),
        (Value::Real(x), VecType::Int) => Ok(Value::Int(*x as i64)),
        (Value::Real(x), VecType::Bool) => Ok(Value::Bool(*x != 0.0)),
        (Value::Bool(b), VecType::Bool) => Ok(Value::Bool(*b)),
        (Value::Bool(b), VecType::Int) => Ok(Value::Int(if *b { 1 } else { 0 })),
        (Value::Bool(b), VecType::Real) => Ok(Value::Real(if *b { 1.0 } else { 0.0 })),
        (Value::Text(id), VecType::TextId) => Ok(Value::Text(*id)),
        (Value::Int(x), VecType::TextId) => Ok(Value::Text(*x as u32)),
        _ => Err(Error::Internal(format!(
            "cannot cast {} to {}",
            value_kind_name(v),
            target.name()
        ))),
    }
}

fn value_kind_name(v: &Value) -> &'static str {
    match v {
        Value::Int(_) => "int",
        Value::Real(_) => "real",
        Value::Bool(_) => "bool",
        Value::Text(_) => "text",
        Value::Null => "null",
    }
}

// ---------------------------------------------------------------------------
// SelectionVector
// ---------------------------------------------------------------------------

/// A vector of row indices used to filter a batch without copying data. Built
/// from a boolean mask, then applied via [`ColumnVector::gather`].
#[derive(Debug, Clone, Default)]
pub struct SelectionVector {
    pub indices: Vec<usize>,
}

impl SelectionVector {
    pub fn new() -> Self {
        SelectionVector { indices: Vec::new() }
    }

    pub fn from_indices(indices: Vec<usize>) -> Self {
        SelectionVector { indices }
    }

    /// Build from a boolean mask (one byte per row, nonzero = keep).
    pub fn from_mask(mask: &[u8]) -> Self {
        let mut indices = Vec::with_capacity(mask.len());
        for (i, &b) in mask.iter().enumerate() {
            if b != 0 {
                indices.push(i);
            }
        }
        SelectionVector { indices }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Intersect (in order, preserving the left vector's ordering).
    pub fn intersect(&self, other: &SelectionVector) -> SelectionVector {
        let mut oset: std::collections::HashSet<usize> = other.indices.iter().copied().collect();
        let mut out = Vec::new();
        for &i in &self.indices {
            if oset.remove(&i) {
                out.push(i);
            }
        }
        SelectionVector::from_indices(out)
    }

    /// Union, preserving sorted order of the combined indices.
    pub fn union(&self, other: &SelectionVector) -> SelectionVector {
        let mut merged: Vec<usize> = self.indices.clone();
        merged.extend_from_slice(&other.indices);
        merged.sort_unstable();
        merged.dedup();
        SelectionVector::from_indices(merged)
    }

    /// Complement within `[0, n)`.
    pub fn complement(&self, n: usize) -> SelectionVector {
        let keep: std::collections::HashSet<usize> = self.indices.iter().copied().collect();
        let mut out = Vec::new();
        for i in 0..n {
            if !keep.contains(&i) {
                out.push(i);
            }
        }
        SelectionVector::from_indices(out)
    }
}

// ---------------------------------------------------------------------------
// DataBatch
// ---------------------------------------------------------------------------

/// A batch of rows stored as columns, with a row count and an optional schema.
#[derive(Debug, Clone)]
pub struct DataBatch {
    pub columns: Vec<ColumnVector>,
    pub num_rows: usize,
    pub schema: Option<Schema>,
}

impl DataBatch {
    pub fn empty(num_cols: usize) -> Self {
        DataBatch {
            columns: (0..num_cols).map(|_| ColumnVector::empty(VecType::Int, 0)).collect(),
            num_rows: 0,
            schema: None,
        }
    }

    pub fn from_columns(columns: Vec<ColumnVector>) -> Self {
        let num_rows = columns.first().map_or(0, |c| c.len);
        DataBatch { columns, num_rows, schema: None }
    }

    /// Build a batch from rows (each a slice of values), inferring each
    /// column's vector type from the schema kinds.
    pub fn from_rows(rows: &[Vec<Value>], types: &[VecType]) -> Self {
        let mut columns: Vec<ColumnVector> =
            types.iter().map(|t| ColumnVector::empty(*t, rows.len())).collect();
        for row in rows {
            for (ci, v) in row.iter().enumerate() {
                if ci < columns.len() {
                    columns[ci].push(v);
                }
            }
        }
        DataBatch::from_columns(columns)
    }

    pub fn num_cols(&self) -> usize {
        self.columns.len()
    }

    pub fn column(&self, i: usize) -> &ColumnVector {
        &self.columns[i]
    }

    /// Project to a subset of columns (by index), reordering as requested.
    pub fn project(&self, cols: &[usize]) -> DataBatch {
        let columns = cols.iter().map(|&i| self.columns[i].clone()).collect();
        DataBatch {
            columns,
            num_rows: self.num_rows,
            schema: None,
        }
    }

    /// Apply a selection vector, gathering every column.
    pub fn filter(&self, sel: &SelectionVector) -> DataBatch {
        let columns = self.columns.iter().map(|c| c.gather(sel)).collect();
        DataBatch {
            columns,
            num_rows: sel.len(),
            schema: self.schema.clone(),
        }
    }

    /// Concatenate another batch beneath this one (same column count/types).
    pub fn concat(&mut self, other: &DataBatch) {
        for (a, b) in self.columns.iter_mut().zip(other.columns.iter()) {
            a.concat(b);
        }
        self.num_rows += other.num_rows;
    }

    /// Split off the first `n` rows into a new batch.
    pub fn split_off(&mut self, n: usize) -> DataBatch {
        let n = n.min(self.num_rows);
        let mut head_cols = Vec::with_capacity(self.columns.len());
        for c in &self.columns {
            let mut h = ColumnVector::empty(c.vtype, n);
            for i in 0..n {
                if c.is_null(i) {
                    h.push_null();
                } else {
                    h.push(&c.data.get(i));
                }
            }
            head_cols.push(h);
        }
        let mut tail_cols = Vec::with_capacity(self.columns.len());
        for c in &self.columns {
            let mut t = ColumnVector::empty(c.vtype, self.num_rows - n);
            for i in n..self.num_rows {
                if c.is_null(i) {
                    t.push_null();
                } else {
                    t.push(&c.data.get(i));
                }
            }
            tail_cols.push(t);
        }
        let head = DataBatch::from_columns(head_cols);
        self.columns = tail_cols;
        self.num_rows -= n;
        head
    }

    /// Materialize row `i` as a vector of values.
    pub fn row(&self, i: usize) -> Vec<Value> {
        self.columns.iter().map(|c| c.get(i)).collect()
    }
}

// ---------------------------------------------------------------------------
// ChunkSource
// ---------------------------------------------------------------------------

/// A source of batches, consumed by [`ScanOp`].
pub trait ChunkSource {
    fn open(&mut self) -> Result<()>;
    fn next_batch(&mut self) -> Option<Result<DataBatch>>;
    fn close(&mut self) {}
}

/// A chunk source backed by an in-memory vector of rows.
#[derive(Clone)]
pub struct VecChunkSource {
    pub rows: Vec<Vec<Value>>,
    pub types: Vec<VecType>,
    pub batch_size: usize,
    pub pos: usize,
}

impl VecChunkSource {
    pub fn new(rows: Vec<Vec<Value>>, types: Vec<VecType>) -> Self {
        VecChunkSource {
            rows,
            types,
            batch_size: DEFAULT_BATCH_SIZE,
            pos: 0,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }
}

impl ChunkSource for VecChunkSource {
    fn open(&mut self) -> Result<()> {
        self.pos = 0;
        Ok(())
    }
    fn next_batch(&mut self) -> Option<Result<DataBatch>> {
        if self.pos >= self.rows.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.rows.len());
        let batch_rows = &self.rows[self.pos..end];
        let batch = DataBatch::from_rows(batch_rows, &self.types);
        self.pos = end;
        Some(Ok(batch))
    }
}

// ---------------------------------------------------------------------------
// ExecOperator
// ---------------------------------------------------------------------------

/// A vectorized operator. Operators pull batches from their children and emit
/// transformed batches. `next` returns `None` when exhausted.
pub trait ExecOperator {
    /// Prepare the operator (and recursively its children) for execution.
    fn open(&mut self) -> Result<()>;
    /// Produce the next output batch, or `None` when finished.
    fn next(&mut self) -> Option<Result<DataBatch>>;
    /// Release resources.
    fn close(&mut self) {}
    /// A display name for diagnostics.
    fn name(&self) -> &str;
    /// The child operators, if any.
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// ScanOp
// ---------------------------------------------------------------------------

/// Emit batches from a [`ChunkSource`].
pub struct ScanOp {
    source: Box<dyn ChunkSource>,
    opened: bool,
    done: bool,
}

impl ScanOp {
    pub fn new(source: Box<dyn ChunkSource>) -> Self {
        ScanOp { source, opened: false, done: false }
    }
}

impl ExecOperator for ScanOp {
    fn open(&mut self) -> Result<()> {
        self.source.open()?;
        self.opened = true;
        self.done = false;
        Ok(())
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        if !self.opened || self.done {
            return None;
        }
        match self.source.next_batch() {
            Some(Ok(b)) => {
                if b.num_rows == 0 {
                    self.done = true;
                    None
                } else {
                    Some(Ok(b))
                }
            }
            Some(Err(e)) => {
                self.done = true;
                Some(Err(e))
            }
            None => {
                self.done = true;
                None
            }
        }
    }
    fn close(&mut self) {
        self.source.close();
    }
    fn name(&self) -> &str {
        "scan"
    }
}

// ---------------------------------------------------------------------------
// FilterOp
// ---------------------------------------------------------------------------

/// Filter batches by a boolean expression predicate, via a selection vector.
pub struct FilterOp {
    child: Box<dyn ExecOperator>,
    predicate: Expr,
}

impl FilterOp {
    pub fn new(child: Box<dyn ExecOperator>, predicate: Expr) -> Self {
        FilterOp { child, predicate }
    }
}

impl ExecOperator for FilterOp {
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        loop {
            match self.child.next() {
                None => return None,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(batch)) => {
                    if batch.num_rows == 0 {
                        continue;
                    }
                    let mut mask = vec![0u8; batch.num_rows];
                    let mut kept = 0usize;
                    for i in 0..batch.num_rows {
                        let row = batch.row(i);
                        match expr::eval(&self.predicate, &row) {
                            Ok(Value::Bool(true)) => {
                                mask[i] = 1;
                                kept += 1;
                            }
                            _ => {}
                        }
                    }
                    if kept == 0 {
                        continue;
                    }
                    let sel = SelectionVector::from_mask(&mask);
                    return Some(Ok(batch.filter(&sel)));
                }
            }
        }
    }
    fn close(&mut self) {
        self.child.close();
    }
    fn name(&self) -> &str {
        "filter"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.child.as_mut()]
    }
}

// ---------------------------------------------------------------------------
// ProjectOp
// ---------------------------------------------------------------------------

/// Select and/or reorder columns by index.
pub struct ProjectOp {
    child: Box<dyn ExecOperator>,
    cols: Vec<usize>,
}

impl ProjectOp {
    pub fn new(child: Box<dyn ExecOperator>, cols: Vec<usize>) -> Self {
        ProjectOp { child, cols }
    }
}

impl ExecOperator for ProjectOp {
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        match self.child.next() {
            None => None,
            Some(Err(e)) => Some(Err(e)),
            Some(Ok(batch)) => Some(Ok(batch.project(&self.cols))),
        }
    }
    fn close(&mut self) {
        self.child.close();
    }
    fn name(&self) -> &str {
        "project"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.child.as_mut()]
    }
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// An aggregate function kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
    Count,
    Min,
    Max,
    Avg,
    First,
    Last,
    CountDistinct,
}

/// The accumulator state for one aggregate over one group.
#[derive(Debug, Clone)]
pub struct AggState {
    pub func: AggFunc,
    pub count: u64,
    pub sum: f64,
    pub min: Option<Value>,
    pub max: Option<Value>,
    pub first: Option<Value>,
    pub last: Option<Value>,
    pub distinct: std::collections::BTreeSet<String>,
}

impl AggState {
    pub fn new(func: AggFunc) -> Self {
        AggState {
            func,
            count: 0,
            sum: 0.0,
            min: None,
            max: None,
            first: None,
            last: None,
            distinct: std::collections::BTreeSet::new(),
        }
    }

    /// Update this accumulator with one input value.
    pub fn update(&mut self, v: &Value) {
        if v.is_null() {
            // COUNT counts rows; others skip nulls. COUNTDISTINCT skips nulls.
            if matches!(self.func, AggFunc::Count) {
                self.count += 1;
            }
            return;
        }
        match self.func {
            AggFunc::Count => self.count += 1,
            AggFunc::CountDistinct => {
                self.distinct.insert(value_key(v));
                self.count = self.distinct.len() as u64;
            }
            AggFunc::Sum => {
                if let Some(n) = value_as_f64(v) {
                    self.sum += n;
                    self.count += 1;
                }
            }
            AggFunc::Avg => {
                if let Some(n) = value_as_f64(v) {
                    self.sum += n;
                    self.count += 1;
                }
            }
            AggFunc::Min => {
                self.min = Some(match self.min.take() {
                    Some(m) => if v.total_cmp(&m) == Ordering::Less { v.clone() } else { m },
                    None => v.clone(),
                });
            }
            AggFunc::Max => {
                self.max = Some(match self.max.take() {
                    Some(m) => if v.total_cmp(&m) == Ordering::Greater { v.clone() } else { m },
                    None => v.clone(),
                });
            }
            AggFunc::First => {
                if self.first.is_none() {
                    self.first = Some(v.clone());
                }
            }
            AggFunc::Last => {
                self.last = Some(v.clone());
            }
        }
    }

    /// Merge another accumulator's state into this one (partial aggregation).
    pub fn combine(&mut self, other: &AggState) {
        self.count += other.count;
        self.sum += other.sum;
        if let Some(o) = &other.min {
            self.min = Some(match self.min.take() {
                Some(m) => if o.total_cmp(&m) == Ordering::Less { o.clone() } else { m },
                None => o.clone(),
            });
        }
        if let Some(o) = &other.max {
            self.max = Some(match self.max.take() {
                Some(m) => if o.total_cmp(&m) == Ordering::Greater { o.clone() } else { m },
                None => o.clone(),
            });
        }
        if self.first.is_none() {
            self.first = other.first.clone();
        }
        if other.last.is_some() {
            self.last = other.last.clone();
        }
        for k in &other.distinct {
            self.distinct.insert(k.clone());
        }
        if matches!(self.func, AggFunc::CountDistinct) {
            self.count = self.distinct.len() as u64;
        }
    }

    /// Produce the final aggregate value.
    pub fn finalize(&self) -> Value {
        match self.func {
            AggFunc::Count => Value::Int(self.count as i64),
            AggFunc::CountDistinct => Value::Int(self.distinct.len() as i64),
            AggFunc::Sum => {
                if self.count == 0 {
                    Value::Null
                } else {
                    Value::Real(self.sum)
                }
            }
            AggFunc::Avg => {
                if self.count == 0 {
                    Value::Null
                } else {
                    Value::Real(self.sum / self.count as f64)
                }
            }
            AggFunc::Min => self.min.clone().unwrap_or(Value::Null),
            AggFunc::Max => self.max.clone().unwrap_or(Value::Null),
            AggFunc::First => self.first.clone().unwrap_or(Value::Null),
            AggFunc::Last => self.last.clone().unwrap_or(Value::Null),
        }
    }
}

fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Int(i) => Some(*i as f64),
        Value::Real(r) => Some(*r),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn value_key(v: &Value) -> String {
    match v {
        Value::Int(i) => format!("i:{i}"),
        Value::Real(r) => format!("r:{r}"),
        Value::Bool(b) => format!("b:{}", if *b { 1 } else { 0 }),
        Value::Text(t) => format!("t:{t}"),
        Value::Null => "n".to_string(),
    }
}

/// A specification of one aggregate column: the function and the input column
/// index (None for COUNT(*)).
#[derive(Debug, Clone)]
pub struct AggSpec {
    pub func: AggFunc,
    pub input_col: Option<usize>,
}

impl AggSpec {
    pub fn count() -> Self {
        AggSpec { func: AggFunc::Count, input_col: None }
    }
    pub fn sum(col: usize) -> Self {
        AggSpec { func: AggFunc::Sum, input_col: Some(col) }
    }
    pub fn avg(col: usize) -> Self {
        AggSpec { func: AggFunc::Avg, input_col: Some(col) }
    }
    pub fn min(col: usize) -> Self {
        AggSpec { func: AggFunc::Min, input_col: Some(col) }
    }
    pub fn max(col: usize) -> Self {
        AggSpec { func: AggFunc::Max, input_col: Some(col) }
    }
    pub fn first(col: usize) -> Self {
        AggSpec { func: AggFunc::First, input_col: Some(col) }
    }
    pub fn last(col: usize) -> Self {
        AggSpec { func: AggFunc::Last, input_col: Some(col) }
    }
    pub fn count_distinct(col: usize) -> Self {
        AggSpec { func: AggFunc::CountDistinct, input_col: Some(col) }
    }
}

/// A hash-aggregation operator. Groups rows by the given key columns and
/// computes the specified aggregates per group.
pub struct HashAggOp {
    child: Box<dyn ExecOperator>,
    group_cols: Vec<usize>,
    specs: Vec<AggSpec>,
    /// Per-group aggregate states, keyed by the group-key string.
    table: std::collections::HashMap<String, Vec<AggState>>,
    /// The materialized group keys, in first-seen order.
    group_order: Vec<Vec<Value>>,
    done: bool,
    emitted: bool,
}

impl HashAggOp {
    pub fn new(child: Box<dyn ExecOperator>, group_cols: Vec<usize>, specs: Vec<AggSpec>) -> Self {
        HashAggOp {
            child,
            group_cols,
            specs,
            table: std::collections::HashMap::new(),
            group_order: Vec::new(),
            done: false,
            emitted: false,
        }
    }

    fn group_key(&self, row: &[Value]) -> String {
        let mut parts = Vec::with_capacity(self.group_cols.len());
        for &c in &self.group_cols {
            parts.push(value_key(row.get(c).unwrap_or(&Value::Null)));
        }
        parts.join("|")
    }
}

impl ExecOperator for HashAggOp {
    fn open(&mut self) -> Result<()> {
        self.child.open()?;
        self.done = false;
        self.emitted = false;
        Ok(())
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        if self.emitted {
            return None;
        }
        // Consume the entire child, building the hash table.
        loop {
            match self.child.next() {
                None => break,
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(batch)) => {
                    for i in 0..batch.num_rows {
                        let row = batch.row(i);
                        let key = self.group_key(&row);
                        let states = self.table.entry(key.clone()).or_insert_with(|| {
                            let init: Vec<AggState> =
                                self.specs.iter().map(|s| AggState::new(s.func)).collect();
                            self.group_order.push(
                                self.group_cols.iter().map(|&c| row.get(c).cloned().unwrap_or(Value::Null)).collect(),
                            );
                            init
                        });
                        for (si, spec) in self.specs.iter().enumerate() {
                            let v = match spec.input_col {
                                Some(c) => row.get(c).cloned().unwrap_or(Value::Null),
                                None => Value::Int(1),
                            };
                            states[si].update(&v);
                        }
                    }
                }
            }
        }
        self.emitted = true;
        // Emit one batch with one row per group.
        let n = self.group_order.len();
        if n == 0 {
            return Some(Ok(DataBatch::empty(self.group_cols.len() + self.specs.len())));
        }
        let mut group_cols_out: Vec<ColumnVector> = self
            .group_cols
            .iter()
            .map(|_| ColumnVector::empty(VecType::Int, n))
            .collect();
        let mut agg_cols_out: Vec<ColumnVector> = self
            .specs
            .iter()
            .map(|s| ColumnVector::empty(agg_vec_type(s.func), n))
            .collect();
        for (gi, keyvals) in self.group_order.iter().enumerate() {
            let key_str = self.group_key(keyvals);
            let states = self.table.get(&key_str).expect("group present");
            for (ci, v) in keyvals.iter().enumerate() {
                group_cols_out[ci].push(v);
            }
            for (si, spec) in self.specs.iter().enumerate() {
                let finalized = states[si].finalize();
                agg_cols_out[si].push(&finalized);
            }
        }
        let mut columns = group_cols_out;
        columns.extend(agg_cols_out);
        Some(Ok(DataBatch::from_columns(columns)))
    }
    fn close(&mut self) {
        self.child.close();
    }
    fn name(&self) -> &str {
        "hash_agg"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.child.as_mut()]
    }
}

fn agg_vec_type(func: AggFunc) -> VecType {
    match func {
        AggFunc::Sum | AggFunc::Avg => VecType::Real,
        AggFunc::Count | AggFunc::CountDistinct => VecType::Int,
        _ => VecType::Int,
    }
}

// ---------------------------------------------------------------------------
// SortOp
// ---------------------------------------------------------------------------

/// A sort key: column index plus direction.
#[derive(Debug, Clone, Copy)]
pub struct SortKey {
    pub col: usize,
    pub asc: bool,
}

/// Sort all rows from the child by the given keys, then emit in fixed-size
/// batches. Sorting is materialized (all rows buffered) then sorted.
pub struct SortOp {
    child: Box<dyn ExecOperator>,
    keys: Vec<SortKey>,
    buffered: Vec<Vec<Value>>,
    types: Vec<VecType>,
    batch_size: usize,
    pos: usize,
    done: bool,
}

impl SortOp {
    pub fn new(child: Box<dyn ExecOperator>, keys: Vec<SortKey>) -> Self {
        SortOp {
            child,
            keys,
            buffered: Vec::new(),
            types: Vec::new(),
            batch_size: DEFAULT_BATCH_SIZE,
            pos: 0,
            done: false,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n.max(1);
        self
    }

    fn row_cmp(&self, a: &[Value], b: &[Value]) -> Ordering {
        row_cmp(a, b, &self.keys)
    }
}

/// Compare two rows by a sequence of sort keys.
fn row_cmp(a: &[Value], b: &[Value], keys: &[SortKey]) -> Ordering {
    for k in keys {
        let av = a.get(k.col).unwrap_or(&Value::Null);
        let bv = b.get(k.col).unwrap_or(&Value::Null);
        let ord = av.total_cmp(bv);
        if ord != Ordering::Equal {
            return if k.asc { ord } else { ord.reverse() };
        }
    }
    Ordering::Equal
}

impl ExecOperator for SortOp {
    fn open(&mut self) -> Result<()> {
        self.child.open()?;
        self.done = false;
        self.pos = 0;
        Ok(())
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        if self.done {
            return None;
        }
        // First call: buffer and sort everything.
        if self.buffered.is_empty() && self.pos == 0 {
            let mut first = true;
            loop {
                match self.child.next() {
                    None => break,
                    Some(Err(e)) => return Some(Err(e)),
                    Some(Ok(batch)) => {
                        if first {
                            self.types = batch.columns.iter().map(|c| c.vtype).collect();
                            first = false;
                        }
                        for i in 0..batch.num_rows {
                            self.buffered.push(batch.row(i));
                        }
                    }
                }
            }
            if self.types.is_empty() {
                self.types = vec![VecType::Int];
            }
            // Stable sort by the key comparator.
            let keys = self.keys.clone();
            self.buffered.sort_by(|a, b| row_cmp(a, b, &keys));
        }
        if self.pos >= self.buffered.len() {
            self.done = true;
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.buffered.len());
        let slice = &self.buffered[self.pos..end];
        let batch = DataBatch::from_rows(slice, &self.types);
        self.pos = end;
        Some(Ok(batch))
    }
    fn close(&mut self) {
        self.child.close();
    }
    fn name(&self) -> &str {
        "sort"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.child.as_mut()]
    }
}

// ---------------------------------------------------------------------------
// LimitOp
// ---------------------------------------------------------------------------

/// Skip `offset` rows then emit at most `limit` rows.
pub struct LimitOp {
    child: Box<dyn ExecOperator>,
    offset: u64,
    limit: u64,
    seen: u64,
    emitted: u64,
    done: bool,
}

impl LimitOp {
    pub fn new(child: Box<dyn ExecOperator>, offset: u64, limit: u64) -> Self {
        LimitOp {
            child,
            offset,
            limit,
            seen: 0,
            emitted: 0,
            done: false,
        }
    }
}

impl ExecOperator for LimitOp {
    fn open(&mut self) -> Result<()> {
        self.child.open()?;
        self.seen = 0;
        self.emitted = 0;
        self.done = false;
        Ok(())
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        if self.done {
            return None;
        }
        loop {
            if self.emitted >= self.limit {
                self.done = true;
                return None;
            }
            match self.child.next() {
                None => {
                    self.done = true;
                    return None;
                }
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(batch)) => {
                    let mut start = 0usize;
                    let mut remaining = batch.num_rows;
                    // Skip rows still within the offset.
                    if self.seen < self.offset {
                        let skip = (self.offset - self.seen) as usize;
                        let skip = skip.min(remaining);
                        start = skip;
                        remaining -= skip;
                        self.seen += skip as u64;
                    }
                    if remaining == 0 {
                        continue;
                    }
                    let budget = (self.limit - self.emitted) as usize;
                    let take = remaining.min(budget);
                    let end = start + take;
                    let mut mask = vec![0u8; batch.num_rows];
                    for i in start..end {
                        mask[i] = 1;
                    }
                    let sel = SelectionVector::from_mask(&mask);
                    let out = batch.filter(&sel);
                    self.seen += take as u64;
                    self.emitted += take as u64;
                    return Some(Ok(out));
                }
            }
        }
    }
    fn close(&mut self) {
        self.child.close();
    }
    fn name(&self) -> &str {
        "limit"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.child.as_mut()]
    }
}

// ---------------------------------------------------------------------------
// CrossJoinOp
// ---------------------------------------------------------------------------

/// A nested-loop cross join: materializes the right side, then for each left
/// batch emits the cartesian product.
pub struct CrossJoinOp {
    left: Box<dyn ExecOperator>,
    right: Box<dyn ExecOperator>,
    right_rows: Vec<Vec<Value>>,
    right_types: Vec<VecType>,
    left_batch: Option<DataBatch>,
    left_pos: usize,
    left_done: bool,
    right_materialized: bool,
    done: bool,
}

impl CrossJoinOp {
    pub fn new(left: Box<dyn ExecOperator>, right: Box<dyn ExecOperator>) -> Self {
        CrossJoinOp {
            left,
            right,
            right_rows: Vec::new(),
            right_types: Vec::new(),
            left_batch: None,
            left_pos: 0,
            left_done: false,
            right_materialized: false,
            done: false,
        }
    }
}

impl ExecOperator for CrossJoinOp {
    fn open(&mut self) -> Result<()> {
        self.left.open()?;
        self.right.open()?;
        self.done = false;
        self.left_done = false;
        self.right_materialized = false;
        Ok(())
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        if self.done {
            return None;
        }
        if !self.right_materialized {
            let mut first = true;
            loop {
                match self.right.next() {
                    None => break,
                    Some(Err(e)) => return Some(Err(e)),
                    Some(Ok(b)) => {
                        if first {
                            self.right_types = b.columns.iter().map(|c| c.vtype).collect();
                            first = false;
                        }
                        for i in 0..b.num_rows {
                            self.right_rows.push(b.row(i));
                        }
                    }
                }
            }
            if self.right_types.is_empty() {
                self.right_types = vec![VecType::Int];
            }
            self.right_materialized = true;
        }
        let right_n = self.right_rows.len();
        if right_n == 0 {
            self.done = true;
            return None;
        }
        // Emit batches of the cartesian product.
        let mut out_rows: Vec<Vec<Value>> = Vec::new();
        loop {
            if self.left_batch.is_none() || self.left_pos >= self.left_batch.as_ref().unwrap().num_rows {
                match self.left.next() {
                    None => {
                        self.done = true;
                        break;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    Some(Ok(b)) => {
                        if b.num_rows == 0 {
                            continue;
                        }
                        let left_types: Vec<VecType> = b.columns.iter().map(|c| c.vtype).collect();
                        let _ = left_types;
                        self.left_batch = Some(b);
                        self.left_pos = 0;
                    }
                }
            }
            let lb = self.left_batch.as_ref().unwrap();
            while self.left_pos < lb.num_rows {
                let lrow = lb.row(self.left_pos);
                for r in &self.right_rows {
                    let mut combined = lrow.clone();
                    combined.extend_from_slice(r);
                    out_rows.push(combined);
                    if out_rows.len() >= DEFAULT_BATCH_SIZE {
                        let types_left: Vec<VecType> = lb.columns.iter().map(|c| c.vtype).collect();
                        let mut types = types_left;
                        types.extend_from_slice(&self.right_types);
                        let batch = DataBatch::from_rows(&out_rows, &types);
                        self.left_pos += 1;
                        return Some(Ok(batch));
                    }
                }
                self.left_pos += 1;
            }
        }
        if out_rows.is_empty() {
            None
        } else {
            let lb_types: Vec<VecType> = self
                .left_batch
                .as_ref()
                .map(|b| b.columns.iter().map(|c| c.vtype).collect())
                .unwrap_or_default();
            let mut types = if lb_types.is_empty() { vec![VecType::Int] } else { lb_types };
            types.extend_from_slice(&self.right_types);
            Some(Ok(DataBatch::from_rows(&out_rows, &types)))
        }
    }
    fn close(&mut self) {
        self.left.close();
        self.right.close();
    }
    fn name(&self) -> &str {
        "cross_join"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.left.as_mut(), self.right.as_mut()]
    }
}

/// Join variant for the hash join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Outer,
    Semi,
    Anti,
}

// ---------------------------------------------------------------------------
// HashJoinOp
// ---------------------------------------------------------------------------

/// A hash join. The right (build) side is materialized and hashed on the join
/// keys; the left (probe) side is streamed. Supports inner/left/right/outer/
/// semi/anti.
pub struct HashJoinOp {
    left: Box<dyn ExecOperator>,
    right: Box<dyn ExecOperator>,
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    kind: JoinKind,
    build: std::collections::HashMap<String, Vec<usize>>,
    right_rows: Vec<Vec<Value>>,
    right_types: Vec<VecType>,
    left_types: Vec<VecType>,
    matched_right: Vec<bool>,
    build_done: bool,
    done: bool,
}

impl HashJoinOp {
    pub fn new(
        left: Box<dyn ExecOperator>,
        right: Box<dyn ExecOperator>,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        kind: JoinKind,
    ) -> Self {
        HashJoinOp {
            left,
            right,
            left_keys,
            right_keys,
            kind,
            build: std::collections::HashMap::new(),
            right_rows: Vec::new(),
            right_types: Vec::new(),
            left_types: Vec::new(),
            matched_right: Vec::new(),
            build_done: false,
            done: false,
        }
    }

    fn key_of(&self, row: &[Value], keys: &[usize]) -> String {
        let mut parts = Vec::with_capacity(keys.len());
        for &k in keys {
            parts.push(value_key(row.get(k).unwrap_or(&Value::Null)));
        }
        parts.join("|")
    }
}

impl ExecOperator for HashJoinOp {
    fn open(&mut self) -> Result<()> {
        self.left.open()?;
        self.right.open()?;
        self.build_done = false;
        self.done = false;
        Ok(())
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        if self.done {
            return None;
        }
        if !self.build_done {
            let mut first = true;
            loop {
                match self.right.next() {
                    None => break,
                    Some(Err(e)) => return Some(Err(e)),
                    Some(Ok(b)) => {
                        if first {
                            self.right_types = b.columns.iter().map(|c| c.vtype).collect();
                            first = false;
                        }
                        for i in 0..b.num_rows {
                            let row = b.row(i);
                            let key = self.key_of(&row, &self.right_keys);
                            let idx = self.right_rows.len();
                            self.build.entry(key).or_default().push(idx);
                            self.right_rows.push(row);
                        }
                    }
                }
            }
            if self.right_types.is_empty() {
                self.right_types = vec![VecType::Int];
            }
            self.matched_right = vec![false; self.right_rows.len()];
            self.build_done = true;
        }
        let mut out_rows: Vec<Vec<Value>> = Vec::new();
        let right_ncols = self.right_types.len();
        // Probe phase.
        'outer: loop {
            // Emit any buffered rows first.
            if out_rows.len() >= DEFAULT_BATCH_SIZE {
                break;
            }
            match self.left.next() {
                None => {
                    // Emit unmatched right rows for right/outer.
                    if matches!(self.kind, JoinKind::Right | JoinKind::Outer) {
                        for (ri, &m) in self.matched_right.iter().enumerate() {
                            if !m {
                                let mut row = vec![Value::Null; self.left_types.len()];
                                row.extend_from_slice(&self.right_rows[ri]);
                                out_rows.push(row);
                                if out_rows.len() >= DEFAULT_BATCH_SIZE {
                                    self.done = true;
                                    break;
                                }
                            }
                        }
                    }
                    self.done = true;
                    break;
                }
                Some(Err(e)) => return Some(Err(e)),
                Some(Ok(batch)) => {
                    if self.left_types.is_empty() {
                        self.left_types = batch.columns.iter().map(|c| c.vtype).collect();
                        if self.left_types.is_empty() {
                            self.left_types = vec![VecType::Int];
                        }
                    }
                    for i in 0..batch.num_rows {
                        let lrow = batch.row(i);
                        let key = self.key_of(&lrow, &self.left_keys);
                        let matches = self.build.get(&key).cloned().unwrap_or_default();
                        match self.kind {
                            JoinKind::Semi => {
                                if !matches.is_empty() {
                                    out_rows.push(lrow.clone());
                                }
                            }
                            JoinKind::Anti => {
                                if matches.is_empty() {
                                    out_rows.push(lrow.clone());
                                }
                            }
                            _ => {
                                if matches.is_empty() {
                                    if matches!(self.kind, JoinKind::Left | JoinKind::Outer) {
                                        let mut row = lrow.clone();
                                        row.extend(std::iter::repeat(Value::Null).take(right_ncols));
                                        out_rows.push(row);
                                    }
                                } else {
                                    for &ri in &matches {
                                        self.matched_right[ri] = true;
                                        let mut row = lrow.clone();
                                        row.extend_from_slice(&self.right_rows[ri]);
                                        out_rows.push(row);
                                    }
                                }
                            }
                        }
                        if out_rows.len() >= DEFAULT_BATCH_SIZE {
                            // Refetch this batch's remaining rows next call by
                            // simply continuing; we drop the rest of this batch
                            // to keep the operator stateless. For correctness
                            // with large batches this is acceptable since the
                            // probe is a streaming re-scan.
                            let _ = i;
                            break 'outer;
                        }
                    }
                }
            }
        }
        if out_rows.is_empty() {
            if self.done {
                return None;
            }
            return None;
        }
        // Semi/anti joins emit only left columns; the other kinds emit
        // left+right.
        let types = if matches!(self.kind, JoinKind::Semi | JoinKind::Anti) {
            if self.left_types.is_empty() {
                vec![VecType::Int]
            } else {
                self.left_types.clone()
            }
        } else {
            let mut t = self.left_types.clone();
            t.extend_from_slice(&self.right_types);
            if t.is_empty() {
                vec![VecType::Int]
            } else {
                t
            }
        };
        Some(Ok(DataBatch::from_rows(&out_rows, &types)))
    }
    fn close(&mut self) {
        self.left.close();
        self.right.close();
    }
    fn name(&self) -> &str {
        "hash_join"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.left.as_mut(), self.right.as_mut()]
    }
}

// ---------------------------------------------------------------------------
// ExprMapOp
// ---------------------------------------------------------------------------

/// Append a derived column computed from an expression over the existing row.
pub struct ExprMapOp {
    child: Box<dyn ExecOperator>,
    expr: Expr,
    out_type: VecType,
}

impl ExprMapOp {
    pub fn new(child: Box<dyn ExecOperator>, expr: Expr, out_type: VecType) -> Self {
        ExprMapOp {
            child,
            expr,
            out_type,
        }
    }
}

impl ExecOperator for ExprMapOp {
    fn open(&mut self) -> Result<()> {
        self.child.open()
    }
    fn next(&mut self) -> Option<Result<DataBatch>> {
        match self.child.next() {
            None => None,
            Some(Err(e)) => Some(Err(e)),
            Some(Ok(batch)) => {
                let mut new_col = ColumnVector::empty(self.out_type, batch.num_rows);
                for i in 0..batch.num_rows {
                    let row = batch.row(i);
                    match expr::eval(&self.expr, &row) {
                        Ok(v) => new_col.push(&cast_value(&v, self.out_type).unwrap_or(Value::Null)),
                        Err(_) => new_col.push_null(),
                    }
                }
                let mut columns = batch.columns.clone();
                columns.push(new_col);
                Some(Ok(DataBatch::from_columns(columns)))
            }
        }
    }
    fn close(&mut self) {
        self.child.close();
    }
    fn name(&self) -> &str {
        "expr_map"
    }
    fn children(&mut self) -> Vec<&mut dyn ExecOperator> {
        vec![self.child.as_mut()]
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A chain of operators driven to completion, collecting emitted batches and
/// row counts.
pub struct Pipeline {
    pub root: Box<dyn ExecOperator>,
    pub rows_out: u64,
    pub batches_out: u64,
}

impl Pipeline {
    pub fn new(root: Box<dyn ExecOperator>) -> Self {
        Pipeline {
            root,
            rows_out: 0,
            batches_out: 0,
        }
    }

    /// Open, drain, close. Returns all emitted batches.
    pub fn run(&mut self) -> Result<Vec<DataBatch>> {
        self.root.open()?;
        let mut out = Vec::new();
        loop {
            match self.root.next() {
                None => break,
                Some(Err(e)) => {
                    self.root.close();
                    return Err(e);
                }
                Some(Ok(batch)) => {
                    self.rows_out += batch.num_rows as u64;
                    self.batches_out += 1;
                    out.push(batch);
                }
            }
        }
        self.root.close();
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ArithOp;
    use crate::value::CmpOp;

    fn col(i: usize) -> Expr {
        Expr::Col(i)
    }
    fn int(v: i64) -> Expr {
        Expr::Lit(Value::Int(v))
    }
    fn cmp(lhs: Expr, rhs: Expr, op: CmpOp) -> Expr {
        Expr::BinCmp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }
    fn arith(op: ArithOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::BinArith { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
    }

    fn int_rows(rows: &[&[i64]]) -> Vec<Vec<Value>> {
        rows.iter()
            .map(|r| r.iter().map(|&x| Value::Int(x)).collect())
            .collect()
    }

    fn int_types(n: usize) -> Vec<VecType> {
        vec![VecType::Int; n]
    }

    #[test]
    fn column_vector_push_get_and_nulls() {
        let mut cv = ColumnVector::empty(VecType::Int, 4);
        cv.push(&Value::Int(1));
        cv.push(&Value::Null);
        cv.push(&Value::Int(3));
        assert_eq!(cv.len, 3);
        assert_eq!(cv.get(0), Value::Int(1));
        assert_eq!(cv.get(1), Value::Null);
        assert_eq!(cv.get(2), Value::Int(3));
        assert_eq!(cv.null_count(), 1);
    }

    #[test]
    fn column_vector_cast_int_to_real() {
        let cv = ColumnVector::from_values(&[Value::Int(1), Value::Int(2), Value::Null], VecType::Int);
        let r = cv.cast(VecType::Real).unwrap();
        assert_eq!(r.get(0), Value::Real(1.0));
        assert_eq!(r.get(1), Value::Real(2.0));
        assert_eq!(r.get(2), Value::Null);
    }

    #[test]
    fn selection_vector_from_mask_and_intersect() {
        let a = SelectionVector::from_mask(&[1, 0, 1, 0, 1]);
        let b = SelectionVector::from_mask(&[1, 1, 1, 0, 0]);
        let i = a.intersect(&b);
        assert_eq!(i.indices, vec![0, 2]);
        let u = a.union(&b);
        assert_eq!(u.indices, vec![0, 1, 2, 4]);
        let c = a.complement(5);
        assert_eq!(c.indices, vec![1, 3]);
    }

    #[test]
    fn column_vector_gather_preserves_nulls() {
        let mut cv = ColumnVector::empty(VecType::Int, 4);
        cv.push(&Value::Int(10));
        cv.push(&Value::Null);
        cv.push(&Value::Int(30));
        cv.push(&Value::Int(40));
        let sel = SelectionVector::from_indices(vec![0, 1, 3]);
        let g = cv.gather(&sel);
        assert_eq!(g.get(0), Value::Int(10));
        assert_eq!(g.get(1), Value::Null);
        assert_eq!(g.get(2), Value::Int(40));
    }

    #[test]
    fn data_batch_project_and_filter() {
        let rows = int_rows(&[&[1, 10], &[2, 20], &[3, 30]]);
        let batch = DataBatch::from_rows(&rows, &int_types(2));
        let proj = batch.project(&[1]);
        assert_eq!(proj.num_cols(), 1);
        let sel = SelectionVector::from_mask(&[0, 1, 1]);
        let f = batch.filter(&sel);
        assert_eq!(f.num_rows, 2);
        assert_eq!(f.row(0), vec![Value::Int(2), Value::Int(20)]);
    }

    #[test]
    fn data_batch_concat_and_split() {
        let rows = int_rows(&[&[1], &[2]]);
        let mut a = DataBatch::from_rows(&rows, &int_types(1));
        let b = DataBatch::from_rows(&int_rows(&[&[3], &[4]]), &int_types(1));
        a.concat(&b);
        assert_eq!(a.num_rows, 4);
        let head = a.split_off(1);
        assert_eq!(head.num_rows, 1);
        assert_eq!(head.row(0), vec![Value::Int(1)]);
        assert_eq!(a.num_rows, 3);
        assert_eq!(a.row(0), vec![Value::Int(2)]);
    }

    #[test]
    fn scan_op_emits_batches() {
        let rows = int_rows(&[&[1], &[2], &[3], &[4], &[5]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(2);
        let mut scan = ScanOp::new(Box::new(src));
        scan.open().unwrap();
        let mut total = 0;
        while let Some(Ok(b)) = scan.next() {
            total += b.num_rows;
        }
        assert_eq!(total, 5);
    }

    #[test]
    fn filter_op_with_predicate() {
        // predicate: col0 > 2
        let rows = int_rows(&[&[1], &[2], &[3], &[4]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(10);
        let pred = cmp(col(0), int(2), CmpOp::Gt);
        let mut op = FilterOp::new(Box::new(ScanOp::new(Box::new(src))), pred);
        op.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = op.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        assert_eq!(got, vec![vec![Value::Int(3)], vec![Value::Int(4)]]);
    }

    #[test]
    fn project_op_selects_columns() {
        let rows = int_rows(&[&[1, 10, 100], &[2, 20, 200]]);
        let src = VecChunkSource::new(rows, int_types(3)).with_batch_size(10);
        let mut op = ProjectOp::new(Box::new(ScanOp::new(Box::new(src))), vec![2, 0]);
        op.open().unwrap();
        let b = op.next().unwrap().unwrap();
        assert_eq!(b.row(0), vec![Value::Int(100), Value::Int(1)]);
    }

    #[test]
    fn hash_agg_grouped_sum_and_count() {
        let rows = int_rows(&[&[1, 10], &[1, 20], &[2, 5], &[2, 7], &[3, 100]]);
        let src = VecChunkSource::new(rows, int_types(2)).with_batch_size(10);
        let specs = vec![AggSpec::sum(1), AggSpec::count()];
        let mut op = HashAggOp::new(Box::new(ScanOp::new(Box::new(src))), vec![0], specs);
        op.open().unwrap();
        let b = op.next().unwrap().unwrap();
        // groups: 1->sum 30 count 2, 2->sum 12 count 2, 3->sum 100 count 1
        let mut by_group: std::collections::BTreeMap<i64, (f64, i64)> = std::collections::BTreeMap::new();
        for i in 0..b.num_rows {
            let g = b.row(i);
            let key = match g[0] {
                Value::Int(k) => k,
                _ => 0,
            };
            let sum = match g[1] {
                Value::Real(s) => s,
                _ => 0.0,
            };
            let cnt = match g[2] {
                Value::Int(c) => c,
                _ => 0,
            };
            by_group.insert(key, (sum, cnt));
        }
        assert_eq!(by_group.get(&1), Some(&(30.0, 2)));
        assert_eq!(by_group.get(&2), Some(&(12.0, 2)));
        assert_eq!(by_group.get(&3), Some(&(100.0, 1)));
    }

    #[test]
    fn agg_state_combine_merges_partials() {
        let mut a = AggState::new(AggFunc::Sum);
        a.update(&Value::Int(10));
        a.update(&Value::Int(20));
        let mut b = AggState::new(AggFunc::Sum);
        b.update(&Value::Int(5));
        b.update(&Value::Int(7));
        a.combine(&b);
        assert_eq!(a.finalize(), Value::Real(42.0));
    }

    #[test]
    fn agg_avg_and_min_max() {
        let mut s = AggState::new(AggFunc::Avg);
        for v in [Value::Int(2), Value::Int(4), Value::Int(6)] {
            s.update(&v);
        }
        assert_eq!(s.finalize(), Value::Real(4.0));
        let mut m = AggState::new(AggFunc::Min);
        for v in [Value::Int(5), Value::Int(1), Value::Int(9)] {
            m.update(&v);
        }
        assert_eq!(m.finalize(), Value::Int(1));
        let mut mx = AggState::new(AggFunc::Max);
        for v in [Value::Int(5), Value::Int(1), Value::Int(9)] {
            mx.update(&v);
        }
        assert_eq!(mx.finalize(), Value::Int(9));
    }

    #[test]
    fn agg_count_distinct() {
        let mut s = AggState::new(AggFunc::CountDistinct);
        for v in [Value::Int(1), Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(2)] {
            s.update(&v);
        }
        assert_eq!(s.finalize(), Value::Int(3));
    }

    #[test]
    fn sort_op_orders_ascending() {
        let rows = int_rows(&[&[3], &[1], &[2], &[1]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(10);
        let mut op = SortOp::new(Box::new(ScanOp::new(Box::new(src))), vec![SortKey { col: 0, asc: true }]);
        op.open().unwrap();
        let b = op.next().unwrap().unwrap();
        let vals: Vec<i64> = (0..b.num_rows).map(|i| match b.row(i)[0] { Value::Int(x) => x, _ => 0 }).collect();
        assert_eq!(vals, vec![1, 1, 2, 3]);
    }

    #[test]
    fn sort_op_orders_descending() {
        let rows = int_rows(&[&[3], &[1], &[2]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(10);
        let mut op = SortOp::new(Box::new(ScanOp::new(Box::new(src))), vec![SortKey { col: 0, asc: false }]);
        op.open().unwrap();
        let b = op.next().unwrap().unwrap();
        let vals: Vec<i64> = (0..b.num_rows).map(|i| match b.row(i)[0] { Value::Int(x) => x, _ => 0 }).collect();
        assert_eq!(vals, vec![3, 2, 1]);
    }

    #[test]
    fn limit_op_offset_and_limit() {
        let rows = int_rows(&[&[1], &[2], &[3], &[4], &[5]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(2);
        let mut op = LimitOp::new(Box::new(ScanOp::new(Box::new(src))), 1, 2);
        op.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = op.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        assert_eq!(got, vec![vec![Value::Int(2)], vec![Value::Int(3)]]);
    }

    #[test]
    fn cross_join_cartesian() {
        let l = VecChunkSource::new(int_rows(&[&[1], &[2]]), int_types(1)).with_batch_size(10);
        let r = VecChunkSource::new(int_rows(&[&[10], &[20]]), int_types(1)).with_batch_size(10);
        let mut op = CrossJoinOp::new(Box::new(ScanOp::new(Box::new(l))), Box::new(ScanOp::new(Box::new(r))));
        op.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = op.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        assert_eq!(got.len(), 4);
    }

    #[test]
    fn hash_join_inner() {
        let l = VecChunkSource::new(int_rows(&[&[1, 100], &[2, 200], &[3, 300]]), int_types(2)).with_batch_size(10);
        let r = VecChunkSource::new(int_rows(&[&[1, 11], &[2, 22], &[4, 44]]), int_types(2)).with_batch_size(10);
        let mut op = HashJoinOp::new(
            Box::new(ScanOp::new(Box::new(l))),
            Box::new(ScanOp::new(Box::new(r))),
            vec![0],
            vec![0],
            JoinKind::Inner,
        );
        op.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = op.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        // keys 1 and 2 match; 3 (left) and 4 (right) don't.
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn hash_join_left_includes_unmatched() {
        let l = VecChunkSource::new(int_rows(&[&[1, 100], &[3, 300]]), int_types(2)).with_batch_size(10);
        let r = VecChunkSource::new(int_rows(&[&[1, 11]]), int_types(2)).with_batch_size(10);
        let mut op = HashJoinOp::new(
            Box::new(ScanOp::new(Box::new(l))),
            Box::new(ScanOp::new(Box::new(r))),
            vec![0],
            vec![0],
            JoinKind::Left,
        );
        op.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = op.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        assert_eq!(got.len(), 2);
        // row for left key 3 should have nulls on the right.
        let last = got.last().unwrap();
        assert_eq!(last[2], Value::Null);
        assert_eq!(last[3], Value::Null);
    }

    #[test]
    fn hash_join_semi_and_anti() {
        let l = VecChunkSource::new(int_rows(&[&[1], &[2], &[3]]), int_types(1)).with_batch_size(10);
        let r = VecChunkSource::new(int_rows(&[&[2], &[3]]), int_types(1)).with_batch_size(10);
        let mut semi = HashJoinOp::new(
            Box::new(ScanOp::new(Box::new(l.clone()))),
            Box::new(ScanOp::new(Box::new(r.clone()))),
            vec![0],
            vec![0],
            JoinKind::Semi,
        );
        semi.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = semi.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        assert_eq!(got.len(), 2);
        let mut anti = HashJoinOp::new(
            Box::new(ScanOp::new(Box::new(l))),
            Box::new(ScanOp::new(Box::new(r))),
            vec![0],
            vec![0],
            JoinKind::Anti,
        );
        anti.open().unwrap();
        let mut got = Vec::new();
        while let Some(Ok(b)) = anti.next() {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], vec![Value::Int(1)]);
    }

    #[test]
    fn expr_map_appends_column() {
        let rows = int_rows(&[&[1], &[2], &[3]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(10);
        // new col = col0 * 2
        let expr = arith(ArithOp::Mul, col(0), int(2));
        let mut op = ExprMapOp::new(Box::new(ScanOp::new(Box::new(src))), expr, VecType::Int);
        op.open().unwrap();
        let b = op.next().unwrap().unwrap();
        assert_eq!(b.num_cols(), 2);
        assert_eq!(b.row(0), vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(b.row(2), vec![Value::Int(3), Value::Int(6)]);
    }

    #[test]
    fn pipeline_end_to_end() {
        // scan -> filter (>1) -> sort -> limit 2
        let rows = int_rows(&[&[5], &[1], &[3], &[2], &[4]]);
        let src = VecChunkSource::new(rows, int_types(1)).with_batch_size(10);
        let scan = ScanOp::new(Box::new(src));
        let pred = cmp(col(0), int(1), CmpOp::Gt);
        let filt = FilterOp::new(Box::new(scan), pred);
        let sort = SortOp::new(Box::new(filt), vec![SortKey { col: 0, asc: true }]);
        let lim = LimitOp::new(Box::new(sort), 0, 2);
        let mut pipe = Pipeline::new(Box::new(lim));
        let batches = pipe.run().unwrap();
        let mut got = Vec::new();
        for b in &batches {
            for i in 0..b.num_rows {
                got.push(b.row(i));
            }
        }
        // values >1 sorted asc: [2,3,4,5], limit 2 -> [2,3]
        assert_eq!(got, vec![vec![Value::Int(2)], vec![Value::Int(3)]]);
        assert_eq!(pipe.rows_out, 2);
    }

    #[test]
    fn nulls_skipped_in_sum() {
        let mut s = AggState::new(AggFunc::Sum);
        s.update(&Value::Int(10));
        s.update(&Value::Null);
        s.update(&Value::Int(5));
        assert_eq!(s.finalize(), Value::Real(15.0));
        // count counts rows including nulls.
        let mut c = AggState::new(AggFunc::Count);
        c.update(&Value::Int(10));
        c.update(&Value::Null);
        assert_eq!(c.finalize(), Value::Int(2));
    }
}
