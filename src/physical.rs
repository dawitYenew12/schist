//! Volcano-style physical operators.
//!
//! A physical plan is a tree of iterators, each pulling rows from its child on
//! demand through a uniform `next` interface (the "Volcano" model). This module
//! implements the row-at-a-time operators the planner lowers to — scan, filter,
//! project, limit, sort, and hash aggregate — over rows of [`Value`]. Filter and
//! project reuse the compiled [`crate::exprvm`] programs, so predicate and
//! projection evaluation is the flat bytecode path rather than tree walking.

use crate::exprvm::{eval, Program};
use crate::value::Value;
use std::collections::HashMap;

/// A row is a fixed-width vector of values.
pub type Row = Vec<Value>;

/// The pull-based operator interface.
pub trait Operator {
    /// Produce the next row, or `None` at end of stream.
    fn next(&mut self) -> Option<Row>;

    /// Drain the operator into a vector (for tests and materialization).
    fn collect_rows(&mut self) -> Vec<Row> {
        let mut out = Vec::new();
        while let Some(row) = self.next() {
            out.push(row);
        }
        out
    }
}

/// A leaf scan over an in-memory row set.
pub struct Scan {
    rows: std::vec::IntoIter<Row>,
}

impl Scan {
    /// Scan the given rows.
    pub fn new(rows: Vec<Row>) -> Scan {
        Scan {
            rows: rows.into_iter(),
        }
    }
}

impl Operator for Scan {
    fn next(&mut self) -> Option<Row> {
        self.rows.next()
    }
}

/// A filter applying a compiled boolean predicate.
pub struct Filter<C: Operator> {
    child: C,
    predicate: Program,
}

impl<C: Operator> Filter<C> {
    /// Filter `child` by `predicate`.
    pub fn new(child: C, predicate: Program) -> Filter<C> {
        Filter { child, predicate }
    }
}

impl<C: Operator> Operator for Filter<C> {
    fn next(&mut self) -> Option<Row> {
        for row in std::iter::from_fn(|| self.child.next()) {
            if matches!(eval(&self.predicate, &row), Value::Bool(true)) {
                return Some(row);
            }
        }
        None
    }
}

/// A projection producing new columns from compiled expressions.
pub struct Project<C: Operator> {
    child: C,
    exprs: Vec<Program>,
}

impl<C: Operator> Project<C> {
    /// Project `child` through the given per-output-column programs.
    pub fn new(child: C, exprs: Vec<Program>) -> Project<C> {
        Project { child, exprs }
    }
}

impl<C: Operator> Operator for Project<C> {
    fn next(&mut self) -> Option<Row> {
        let row = self.child.next()?;
        Some(self.exprs.iter().map(|p| eval(p, &row)).collect())
    }
}

/// A limit/offset operator.
pub struct Limit<C: Operator> {
    child: C,
    remaining: usize,
    to_skip: usize,
}

impl<C: Operator> Limit<C> {
    /// Take at most `limit` rows after skipping `offset`.
    pub fn new(child: C, limit: usize, offset: usize) -> Limit<C> {
        Limit {
            child,
            remaining: limit,
            to_skip: offset,
        }
    }
}

impl<C: Operator> Operator for Limit<C> {
    fn next(&mut self) -> Option<Row> {
        while self.to_skip > 0 {
            self.child.next()?;
            self.to_skip -= 1;
        }
        if self.remaining == 0 {
            return None;
        }
        let row = self.child.next()?;
        self.remaining -= 1;
        Some(row)
    }
}

/// A blocking sort operator (materializes its input, then streams sorted rows).
pub struct Sort<C: Operator> {
    child: Option<C>,
    keys: Vec<(usize, bool)>, // (column index, ascending)
    buffer: Vec<Row>,
    pos: usize,
}

impl<C: Operator> Sort<C> {
    /// Sort `child` by `(column, ascending)` keys, in order.
    pub fn new(child: C, keys: Vec<(usize, bool)>) -> Sort<C> {
        Sort {
            child: Some(child),
            keys,
            buffer: Vec::new(),
            pos: 0,
        }
    }

    fn materialize(&mut self) {
        if let Some(mut child) = self.child.take() {
            self.buffer = child.collect_rows();
            let keys = self.keys.clone();
            self.buffer.sort_by(|a, b| {
                for &(col, asc) in &keys {
                    let va = a.get(col).copied().unwrap_or(Value::Null);
                    let vb = b.get(col).copied().unwrap_or(Value::Null);
                    let ord = va.total_cmp(&vb);
                    let ord = if asc { ord } else { ord.reverse() };
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
                std::cmp::Ordering::Equal
            });
        }
    }
}

impl<C: Operator> Operator for Sort<C> {
    fn next(&mut self) -> Option<Row> {
        if self.child.is_some() {
            self.materialize();
        }
        if self.pos < self.buffer.len() {
            let row = std::mem::take(&mut self.buffer[self.pos]);
            self.pos += 1;
            Some(row)
        } else {
            None
        }
    }
}

/// A grouping aggregate: `SUM`/`COUNT`/`MIN`/`MAX` over integer columns keyed by
/// a set of grouping columns.
pub struct HashAggregate<C: Operator> {
    child: Option<C>,
    group_cols: Vec<usize>,
    aggs: Vec<AggSpec>,
    output: std::vec::IntoIter<Row>,
}

/// One aggregate to compute.
#[derive(Debug, Clone, Copy)]
pub struct AggSpec {
    pub kind: AggKind,
    pub column: usize,
}

/// Supported aggregate kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    Count,
    Sum,
    Min,
    Max,
}

#[derive(Clone, Copy)]
struct AggState {
    count: i64,
    sum: i64,
    min: i64,
    max: i64,
    seen: bool,
}

impl AggState {
    fn new() -> AggState {
        AggState {
            count: 0,
            sum: 0,
            min: i64::MAX,
            max: i64::MIN,
            seen: false,
        }
    }

    fn update(&mut self, v: Value) {
        self.count += 1;
        if let Some(i) = v.as_int() {
            self.sum = self.sum.wrapping_add(i);
            self.min = self.min.min(i);
            self.max = self.max.max(i);
            self.seen = true;
        }
    }

    fn finalize(&self, kind: AggKind) -> Value {
        match kind {
            AggKind::Count => Value::Int(self.count),
            AggKind::Sum => Value::Int(self.sum),
            AggKind::Min => {
                if self.seen {
                    Value::Int(self.min)
                } else {
                    Value::Null
                }
            }
            AggKind::Max => {
                if self.seen {
                    Value::Int(self.max)
                } else {
                    Value::Null
                }
            }
        }
    }
}

impl<C: Operator> HashAggregate<C> {
    /// Aggregate `child` grouped by `group_cols`, computing `aggs`.
    pub fn new(child: C, group_cols: Vec<usize>, aggs: Vec<AggSpec>) -> HashAggregate<C> {
        HashAggregate {
            child: Some(child),
            group_cols,
            aggs,
            output: Vec::new().into_iter(),
        }
    }

    fn run(&mut self) {
        let mut child = match self.child.take() {
            Some(c) => c,
            None => return,
        };
        let mut groups: HashMap<Vec<i64>, Vec<AggState>> = HashMap::new();
        let mut order: Vec<Vec<i64>> = Vec::new();
        while let Some(row) = child.next() {
            let key: Vec<i64> = self
                .group_cols
                .iter()
                .map(|&c| row.get(c).and_then(|v| v.as_int()).unwrap_or(i64::MIN))
                .collect();
            let entry = groups.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                vec![AggState::new(); self.aggs.len()]
            });
            for (i, spec) in self.aggs.iter().enumerate() {
                let v = row.get(spec.column).copied().unwrap_or(Value::Null);
                entry[i].update(v);
            }
        }
        let mut out = Vec::with_capacity(order.len());
        for key in order {
            let states = &groups[&key];
            let mut row: Row = key.iter().map(|&k| Value::Int(k)).collect();
            for (i, spec) in self.aggs.iter().enumerate() {
                row.push(states[i].finalize(spec.kind));
            }
            out.push(row);
        }
        self.output = out.into_iter();
    }
}

impl<C: Operator> Operator for HashAggregate<C> {
    fn next(&mut self) -> Option<Row> {
        if self.child.is_some() {
            self.run();
        }
        self.output.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exprvm::Compiler;
    use crate::sql::{Parser, Statement};

    fn rows() -> Vec<Row> {
        vec![
            vec![Value::Int(1), Value::Int(10)],
            vec![Value::Int(2), Value::Int(20)],
            vec![Value::Int(3), Value::Int(30)],
            vec![Value::Int(1), Value::Int(40)],
        ]
    }

    fn program(expr_src: &str) -> Program {
        let stmt = Parser::new(&format!("SELECT {expr_src} FROM t"))
            .unwrap()
            .parse_statement()
            .unwrap();
        let expr = match stmt {
            Statement::Select(s) => s.items[0].expr.clone(),
            _ => panic!(),
        };
        let mut cols = HashMap::new();
        cols.insert("a".to_string(), 0);
        cols.insert("b".to_string(), 1);
        Compiler::new(&cols).compile(&expr).unwrap()
    }

    #[test]
    fn scan_filter() {
        let scan = Scan::new(rows());
        let mut op = Filter::new(scan, program("b > 15"));
        let out = op.collect_rows();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn project_expressions() {
        let scan = Scan::new(rows());
        let mut op = Project::new(scan, vec![program("a + b")]);
        let out = op.collect_rows();
        assert_eq!(out[0], vec![Value::Int(11)]);
        assert_eq!(out[3], vec![Value::Int(41)]);
    }

    #[test]
    fn limit_offset() {
        let scan = Scan::new(rows());
        let mut op = Limit::new(scan, 2, 1);
        let out = op.collect_rows();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0][0], Value::Int(2));
    }

    #[test]
    fn sort_descending() {
        let scan = Scan::new(rows());
        let mut op = Sort::new(scan, vec![(1, false)]);
        let out = op.collect_rows();
        let col1: Vec<Value> = out.iter().map(|r| r[1]).collect();
        assert_eq!(col1, vec![Value::Int(40), Value::Int(30), Value::Int(20), Value::Int(10)]);
    }

    #[test]
    fn hash_aggregate_group_sum() {
        let scan = Scan::new(rows());
        let mut op = HashAggregate::new(
            scan,
            vec![0],
            vec![
                AggSpec { kind: AggKind::Sum, column: 1 },
                AggSpec { kind: AggKind::Count, column: 1 },
            ],
        );
        let mut out = op.collect_rows();
        out.sort_by_key(|r| r[0].as_int().unwrap());
        // Group 1 -> sum 50, count 2.
        assert_eq!(out[0], vec![Value::Int(1), Value::Int(50), Value::Int(2)]);
        assert_eq!(out[1], vec![Value::Int(2), Value::Int(20), Value::Int(1)]);
    }

    #[test]
    fn pipeline_composition() {
        // Scan -> Filter -> Project -> Sort
        let scan = Scan::new(rows());
        let filt = Filter::new(scan, program("b >= 20"));
        let proj = Project::new(filt, vec![program("a"), program("b")]);
        let mut sort = Sort::new(proj, vec![(1, true)]);
        let out = sort.collect_rows();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0][1], Value::Int(20));
        assert_eq!(out[2][1], Value::Int(40));
    }
}
