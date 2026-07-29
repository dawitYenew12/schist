//! A cardinality and cost model for logical plans.
//!
//! Choosing between physical operators (hash vs. merge join, index vs. full
//! scan) needs an estimate of how many rows each plan node produces and how
//! much work it costs. This module walks a [`LogicalPlan`] bottom-up, deriving
//! output cardinalities from base-table statistics and predicate selectivity,
//! and assigns each node a cost in abstract units (CPU touches plus a page-I/O
//! surcharge). The selectivity rules are the textbook defaults: equality is
//! `1/ndv`, range is a third, conjunction multiplies, disjunction adds and
//! subtracts the overlap.

use crate::logical::LogicalPlan;
use crate::sql::{BinOp, Expr, UnaryOp};
use std::collections::HashMap;

/// Per-table statistics the estimator consults.
#[derive(Debug, Clone)]
pub struct TableStats {
    pub row_count: f64,
    /// Distinct value counts per column name (for equality selectivity).
    pub ndv: HashMap<String, f64>,
}

impl TableStats {
    /// Stats for a table of `rows` rows with no per-column detail.
    pub fn new(rows: u64) -> TableStats {
        TableStats {
            row_count: rows as f64,
            ndv: HashMap::new(),
        }
    }

    /// Builder: set a column's distinct-value count.
    pub fn with_ndv(mut self, column: &str, ndv: u64) -> TableStats {
        self.ndv.insert(column.to_string(), ndv.max(1) as f64);
        self
    }

    fn ndv_of(&self, column: &str) -> f64 {
        self.ndv
            .get(column)
            .copied()
            .unwrap_or_else(|| (self.row_count.max(1.0)).sqrt().max(1.0))
    }
}

/// The estimator, holding a catalog of table statistics.
pub struct CostModel {
    stats: HashMap<String, TableStats>,
    /// Cost multiplier for a page fault relative to a CPU row touch.
    io_weight: f64,
}

/// The estimated cardinality and cost of a plan node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub rows: f64,
    pub cost: f64,
}

impl CostModel {
    /// A model with no statistics.
    pub fn new() -> CostModel {
        CostModel {
            stats: HashMap::new(),
            io_weight: 4.0,
        }
    }

    /// Register statistics for a table.
    pub fn set_stats(&mut self, table: &str, stats: TableStats) {
        self.stats.insert(table.to_string(), stats);
    }

    /// Selectivity of a boolean predicate against a table's stats, in `[0, 1]`.
    pub fn selectivity(&self, predicate: &Expr, stats: &TableStats) -> f64 {
        match predicate {
            Expr::LitBool(true) => 1.0,
            Expr::LitBool(false) => 0.0,
            Expr::Binary(BinOp::And, a, b) => {
                self.selectivity(a, stats) * self.selectivity(b, stats)
            }
            Expr::Binary(BinOp::Or, a, b) => {
                let sa = self.selectivity(a, stats);
                let sb = self.selectivity(b, stats);
                (sa + sb - sa * sb).clamp(0.0, 1.0)
            }
            Expr::Binary(BinOp::Eq, l, r) => {
                if let Some(col) = column_name(l).or_else(|| column_name(r)) {
                    (1.0 / stats.ndv_of(&col)).clamp(0.0, 1.0)
                } else {
                    0.1
                }
            }
            Expr::Binary(BinOp::NotEq, l, r) => {
                if let Some(col) = column_name(l).or_else(|| column_name(r)) {
                    (1.0 - 1.0 / stats.ndv_of(&col)).clamp(0.0, 1.0)
                } else {
                    0.9
                }
            }
            Expr::Binary(BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq, _, _) => 0.3333,
            Expr::Between(_, _, _) => 0.25,
            Expr::InList(_, list) => {
                let each = 0.1;
                (each * list.len() as f64).clamp(0.0, 0.9)
            }
            Expr::Like(_, _) => 0.2,
            Expr::IsNull(_, false) => 0.05,
            Expr::IsNull(_, true) => 0.95,
            Expr::Unary(UnaryOp::Not, inner) => 1.0 - self.selectivity(inner, stats),
            _ => 0.5,
        }
    }

    /// Estimate the whole plan.
    pub fn estimate(&self, plan: &LogicalPlan) -> Estimate {
        match plan {
            LogicalPlan::EmptyRelation => Estimate { rows: 0.0, cost: 0.0 },
            LogicalPlan::Scan { table, .. } => {
                let rows = self
                    .stats
                    .get(table)
                    .map(|s| s.row_count)
                    .unwrap_or(1000.0);
                Estimate {
                    rows,
                    cost: rows * self.io_weight,
                }
            }
            LogicalPlan::Filter { predicate, input } => {
                let child = self.estimate(input);
                let sel = self.filter_selectivity(predicate, input);
                let rows = (child.rows * sel).max(0.0);
                Estimate {
                    rows,
                    cost: child.cost + child.rows, // one touch per input row
                }
            }
            LogicalPlan::Project { input, .. } => {
                let child = self.estimate(input);
                Estimate {
                    rows: child.rows,
                    cost: child.cost + child.rows,
                }
            }
            LogicalPlan::Aggregate {
                group_by, input, ..
            } => {
                let child = self.estimate(input);
                let groups = if group_by.is_empty() {
                    1.0
                } else {
                    // Heuristic: distinct groups ~ sqrt of input rows.
                    child.rows.sqrt().max(1.0)
                };
                Estimate {
                    rows: groups,
                    cost: child.cost + child.rows * 2.0,
                }
            }
            LogicalPlan::Join {
                left, right, on, ..
            } => {
                let l = self.estimate(left);
                let r = self.estimate(right);
                // Estimate join selectivity from the ON predicate.
                let sel = join_selectivity(on, l.rows, r.rows);
                let rows = (l.rows * r.rows * sel).max(0.0);
                Estimate {
                    rows,
                    cost: l.cost + r.cost + l.rows + r.rows,
                }
            }
            LogicalPlan::Sort { input, .. } => {
                let child = self.estimate(input);
                let n = child.rows.max(1.0);
                Estimate {
                    rows: child.rows,
                    cost: child.cost + n * n.log2().max(1.0),
                }
            }
            LogicalPlan::Limit { limit, input, .. } => {
                let child = self.estimate(input);
                let rows = match limit {
                    Some(l) => child.rows.min(*l as f64),
                    None => child.rows,
                };
                Estimate {
                    rows,
                    cost: child.cost,
                }
            }
            LogicalPlan::Distinct { input } => {
                let child = self.estimate(input);
                Estimate {
                    rows: child.rows.sqrt().max(1.0),
                    cost: child.cost + child.rows,
                }
            }
        }
    }

    fn filter_selectivity(&self, predicate: &Expr, input: &LogicalPlan) -> f64 {
        // Find the underlying scan's stats if the input is a simple scan.
        if let LogicalPlan::Scan { table, .. } = input {
            if let Some(stats) = self.stats.get(table) {
                return self.selectivity(predicate, stats);
            }
        }
        // Fall back to a stat-free default table.
        let default = TableStats::new(1000);
        self.selectivity(predicate, &default)
    }
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel::new()
    }
}

fn column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(c) => Some(c.clone()),
        Expr::Qualified(_, c) => Some(c.clone()),
        _ => None,
    }
}

fn join_selectivity(on: &Expr, left_rows: f64, right_rows: f64) -> f64 {
    // For an equi-join on keys, selectivity ~ 1 / max(ndv) ~ 1 / max(rows).
    if let Expr::Binary(BinOp::Eq, _, _) = on {
        1.0 / left_rows.max(right_rows).max(1.0)
    } else {
        // Non-equi join: assume a third of the cross product survives.
        0.3333
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::{Parser, Statement};

    fn plan(sql: &str) -> LogicalPlan {
        let stmt = Parser::new(sql).unwrap().parse_statement().unwrap();
        match stmt {
            Statement::Select(s) => LogicalPlan::from_select(&s),
            _ => panic!(),
        }
    }

    #[test]
    fn scan_uses_row_count() {
        let mut m = CostModel::new();
        m.set_stats("t", TableStats::new(5000));
        let e = m.estimate(&plan("SELECT * FROM t"));
        assert_eq!(e.rows, 5000.0);
    }

    #[test]
    fn equality_selectivity_from_ndv() {
        let mut m = CostModel::new();
        m.set_stats("t", TableStats::new(1000).with_ndv("a", 100));
        let e = m.estimate(&plan("SELECT * FROM t WHERE a = 5"));
        // 1000 * (1/100) = 10.
        assert!((e.rows - 10.0).abs() < 0.001, "rows {}", e.rows);
    }

    #[test]
    fn conjunction_multiplies() {
        let mut m = CostModel::new();
        m.set_stats("t", TableStats::new(1000).with_ndv("a", 10).with_ndv("b", 10));
        let e = m.estimate(&plan("SELECT * FROM t WHERE a = 1 AND b = 2"));
        // 1000 * 0.1 * 0.1 = 10.
        assert!((e.rows - 10.0).abs() < 0.001, "rows {}", e.rows);
    }

    #[test]
    fn range_is_a_third() {
        let mut m = CostModel::new();
        m.set_stats("t", TableStats::new(900));
        let e = m.estimate(&plan("SELECT * FROM t WHERE a > 5"));
        assert!((e.rows - 300.0).abs() < 1.0, "rows {}", e.rows);
    }

    #[test]
    fn aggregate_reduces_rows() {
        let mut m = CostModel::new();
        m.set_stats("t", TableStats::new(10_000));
        let e = m.estimate(&plan("SELECT a, COUNT(*) FROM t GROUP BY a"));
        assert!(e.rows < 10_000.0);
        assert!(e.rows >= 1.0);
    }

    #[test]
    fn limit_caps_rows() {
        let mut m = CostModel::new();
        m.set_stats("t", TableStats::new(10_000));
        let e = m.estimate(&plan("SELECT * FROM t LIMIT 25"));
        assert_eq!(e.rows, 25.0);
    }

    #[test]
    fn selectivity_bounds() {
        let m = CostModel::new();
        let stats = TableStats::new(100).with_ndv("a", 4);
        let s = m.selectivity(
            &Parser::new("SELECT * FROM t WHERE a = 1")
                .unwrap()
                .parse_statement()
                .map(|st| match st {
                    Statement::Select(sel) => sel.filter.unwrap(),
                    _ => panic!(),
                })
                .unwrap(),
            &stats,
        );
        assert!((s - 0.25).abs() < 0.001);
    }
}
