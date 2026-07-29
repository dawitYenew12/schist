//! Query planning.
//!
//! A logical plan describes *what* a query computes (scan, filter, project,
//! join, aggregate, sort). A physical plan describes *how* it is computed
//! (which scan path, which join algorithm, where to sort). The planner turns a
//! logical plan into a physical one using simple cost estimates derived from
//! column statistics.

use crate::stats::ColumnStats;
use crate::value::Value;

/// A logical operator.
#[derive(Debug, Clone)]
pub enum LogicalNode {
    /// Scan a table.
    Scan { table: String },
    /// Filter by a predicate on a column.
    Filter { input: Box<LogicalNode>, col: usize, op: String, value: Value },
    /// Project a subset of columns.
    Project { input: Box<LogicalNode>, cols: Vec<usize> },
    /// Join two inputs on equal columns.
    Join { left: Box<LogicalNode>, right: Box<LogicalNode>, left_col: usize, right_col: usize },
    /// Aggregate by a set of keys.
    Aggregate { input: Box<LogicalNode>, keys: Vec<usize>, agg_col: usize },
    /// Sort by a set of keys.
    Sort { input: Box<LogicalNode>, keys: Vec<(usize, bool)> },
    /// Limit to N rows.
    Limit { input: Box<LogicalNode>, n: usize },
    /// A materialized constant.
    Values { rows: Vec<Vec<Value>> },
}

impl LogicalNode {
    /// Pretty-print the plan indented.
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        self.pretty_into(&mut s, 0);
        s
    }
    fn pretty_into(&self, s: &mut String, indent: usize) {
        for _ in 0..indent {
            s.push(' ');
        }
        match self {
            LogicalNode::Scan { table } => s.push_str(&format!("Scan({})\n", table)),
            LogicalNode::Filter { input, col, op, value } => {
                s.push_str(&format!("Filter(col={} {} {:?})\n", col, op, value));
                input.pretty_into(s, indent + 2);
            }
            LogicalNode::Project { input, cols } => {
                s.push_str(&format!("Project({:?})\n", cols));
                input.pretty_into(s, indent + 2);
            }
            LogicalNode::Join { left, right, left_col, right_col } => {
                s.push_str(&format!("Join({}={})\n", left_col, right_col));
                left.pretty_into(s, indent + 2);
                right.pretty_into(s, indent + 2);
            }
            LogicalNode::Aggregate { input, keys, agg_col } => {
                s.push_str(&format!("Aggregate(keys={:?}, agg={})\n", keys, agg_col));
                input.pretty_into(s, indent + 2);
            }
            LogicalNode::Sort { input, keys } => {
                s.push_str(&format!("Sort({:?})\n", keys));
                input.pretty_into(s, indent + 2);
            }
            LogicalNode::Limit { input, n } => {
                s.push_str(&format!("Limit({})\n", n));
                input.pretty_into(s, indent + 2);
            }
            LogicalNode::Values { rows } => {
                s.push_str(&format!("Values({} rows)\n", rows.len()));
            }
        }
    }
}

/// A physical scan path.
#[derive(Debug, Clone)]
pub enum ScanPath {
    /// Full sequential scan.
    Sequential,
    /// Index scan via the secondary index for an equality predicate.
    IndexEq { col: usize, value: Value },
    /// Zone-map-pruned scan for a range predicate.
    ZonePruned { col: usize, lo: Value, hi: Value },
}

/// A physical operator.
#[derive(Debug, Clone)]
pub enum PhysicalNode {
    Scan { table: String, path: ScanPath, est_rows: f64 },
    Filter { input: Box<PhysicalNode>, est_rows: f64 },
    Project { input: Box<PhysicalNode>, cols: Vec<usize>, est_rows: f64 },
    HashJoin { left: Box<PhysicalNode>, right: Box<PhysicalNode>, est_rows: f64 },
    SortMergeJoin { left: Box<PhysicalNode>, right: Box<PhysicalNode>, est_rows: f64 },
    NestedLoopJoin { left: Box<PhysicalNode>, right: Box<PhysicalNode>, est_rows: f64 },
    Aggregate { input: Box<PhysicalNode>, est_rows: f64 },
    Sort { input: Box<PhysicalNode>, est_rows: f64 },
    Limit { input: Box<PhysicalNode>, n: usize, est_rows: f64 },
}

impl PhysicalNode {
    pub fn est_rows(&self) -> f64 {
        match self {
            PhysicalNode::Scan { est_rows, .. } => *est_rows,
            PhysicalNode::Filter { est_rows, .. } => *est_rows,
            PhysicalNode::Project { est_rows, .. } => *est_rows,
            PhysicalNode::HashJoin { est_rows, .. } => *est_rows,
            PhysicalNode::SortMergeJoin { est_rows, .. } => *est_rows,
            PhysicalNode::NestedLoopJoin { est_rows, .. } => *est_rows,
            PhysicalNode::Aggregate { est_rows, .. } => *est_rows,
            PhysicalNode::Sort { est_rows, .. } => *est_rows,
            PhysicalNode::Limit { est_rows, .. } => *est_rows,
        }
    }

    pub fn pretty(&self) -> String {
        let mut s = String::new();
        self.pretty_into(&mut s, 0);
        s
    }
    fn pretty_into(&self, s: &mut String, indent: usize) {
        for _ in 0..indent {
            s.push(' ');
        }
        match self {
            PhysicalNode::Scan { table, path, est_rows } => {
                s.push_str(&format!("Scan({:?} {:?} ~{:.0} rows)\n", table, path, est_rows));
            }
            PhysicalNode::Filter { est_rows, .. } => {
                s.push_str(&format!("Filter(~{:.0} rows)\n", est_rows));
                if let PhysicalNode::Filter { input, .. } = self {
                    input.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::Project { cols, est_rows, .. } => {
                s.push_str(&format!("Project({:?} ~{:.0})\n", cols, est_rows));
                if let PhysicalNode::Project { input, .. } = self {
                    input.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::HashJoin { est_rows, .. } => {
                s.push_str(&format!("HashJoin(~{:.0})\n", est_rows));
                if let PhysicalNode::HashJoin { left, right, .. } = self {
                    left.pretty_into(s, indent + 2);
                    right.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::SortMergeJoin { est_rows, .. } => {
                s.push_str(&format!("SortMergeJoin(~{:.0})\n", est_rows));
                if let PhysicalNode::SortMergeJoin { left, right, .. } = self {
                    left.pretty_into(s, indent + 2);
                    right.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::NestedLoopJoin { est_rows, .. } => {
                s.push_str(&format!("NestedLoopJoin(~{:.0})\n", est_rows));
                if let PhysicalNode::NestedLoopJoin { left, right, .. } = self {
                    left.pretty_into(s, indent + 2);
                    right.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::Aggregate { est_rows, .. } => {
                s.push_str(&format!("Aggregate(~{:.0})\n", est_rows));
                if let PhysicalNode::Aggregate { input, .. } = self {
                    input.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::Sort { est_rows, .. } => {
                s.push_str(&format!("Sort(~{:.0})\n", est_rows));
                if let PhysicalNode::Sort { input, .. } = self {
                    input.pretty_into(s, indent + 2);
                }
            }
            PhysicalNode::Limit { n, est_rows, .. } => {
                s.push_str(&format!("Limit({} ~{:.0})\n", n, est_rows));
                if let PhysicalNode::Limit { input, .. } = self {
                    input.pretty_into(s, indent + 2);
                }
            }
        }
    }
}

/// Statistics the planner consults.
#[derive(Debug, Clone, Default)]
pub struct PlanStats {
    pub table_rows: f64,
    pub column_stats: Vec<ColumnStats>,
    pub indexed_cols: Vec<bool>,
}

/// Plan a logical node into a physical node.
pub fn plan(logical: &LogicalNode, stats: &PlanStats) -> PhysicalNode {
    match logical {
        LogicalNode::Scan { table: _ } => PhysicalNode::Scan {
            table: String::new(),
            path: ScanPath::Sequential,
            est_rows: stats.table_rows,
        },
        LogicalNode::Filter { input, col, op, value } => {
            let child = plan(input, stats);
            let base = child.est_rows();
            let sel = if op == "=" {
                stats.column_stats.get(*col).map_or(0.1, |s| s.eq_selectivity())
            } else if op == "<" || op == ">" || op == "<=" || op == ">=" {
                0.3
            } else {
                0.5
            };
            let est = base * sel;
            // If the filter is an equality on an indexed column, fold it into the
            // scan path.
            if op == "=" && stats.indexed_cols.get(*col).copied().unwrap_or(false) {
                if let PhysicalNode::Scan { table, .. } = child {
                    return PhysicalNode::Scan {
                        table,
                        path: ScanPath::IndexEq { col: *col, value: *value },
                        est_rows: est,
                    };
                }
            }
            PhysicalNode::Filter { input: Box::new(child), est_rows: est }
        }
        LogicalNode::Project { input, cols } => {
            let child = plan(input, stats);
            let est = child.est_rows();
            PhysicalNode::Project { input: Box::new(child), cols: cols.clone(), est_rows: est }
        }
        LogicalNode::Join { left, right, .. } => {
            let l = plan(left, stats);
            let r = plan(right, stats);
            let est = (l.est_rows() * r.est_rows()) / stats.table_rows.max(1.0);
            // Pick the algorithm by size: hash join if one side is small, else
            // sort-merge; nested loop only for tiny inputs.
            if l.est_rows().min(r.est_rows()) < 1000.0 {
                PhysicalNode::HashJoin { left: Box::new(l), right: Box::new(r), est_rows: est }
            } else if l.est_rows() + r.est_rows() < 1_000_000.0 {
                PhysicalNode::SortMergeJoin { left: Box::new(l), right: Box::new(r), est_rows: est }
            } else {
                PhysicalNode::NestedLoopJoin { left: Box::new(l), right: Box::new(r), est_rows: est }
            }
        }
        LogicalNode::Aggregate { input, keys, .. } => {
            let child = plan(input, stats);
            let est = child.est_rows() / (keys.len().max(1) as f64 * 10.0).max(1.0);
            PhysicalNode::Aggregate { input: Box::new(child), est_rows: est.max(1.0) }
        }
        LogicalNode::Sort { input, .. } => {
            let child = plan(input, stats);
            let est = child.est_rows();
            PhysicalNode::Sort { input: Box::new(child), est_rows: est }
        }
        LogicalNode::Limit { input, n } => {
            let child = plan(input, stats);
            let est = child.est_rows().min(*n as f64);
            PhysicalNode::Limit { input: Box::new(child), n: *n, est_rows: est }
        }
        LogicalNode::Values { rows } => PhysicalNode::Scan {
            table: "values".into(),
            path: ScanPath::Sequential,
            est_rows: rows.len() as f64,
        },
    }
}

/// Apply a set of optimization rewrites to a logical plan: predicate pushdown
/// (push filters below projects), and limit pushdown.
pub fn optimize(logical: LogicalNode) -> LogicalNode {
    match logical {
        LogicalNode::Filter { input, col, op, value } => {
            let input = optimize(*input);
            // Push filter below a project (rewriting the column index is
            // skipped here for simplicity; we only push when the project does
            // not reorder).
            if let LogicalNode::Project { input: pinput, cols } = input.clone() {
                if cols.iter().enumerate().all(|(i, &c)| i == c) {
                    return LogicalNode::Project {
                        input: Box::new(LogicalNode::Filter {
                            input: pinput,
                            col,
                            op,
                            value,
                        }),
                        cols,
                    };
                }
            }
            LogicalNode::Filter { input: Box::new(input), col, op, value }
        }
        LogicalNode::Limit { input, n } => {
            let input = optimize(*input);
            if let LogicalNode::Sort { input: sinput, keys } = input.clone() {
                return LogicalNode::Sort {
                    input: Box::new(LogicalNode::Limit { input: sinput, n }),
                    keys,
                };
            }
            LogicalNode::Limit { input: Box::new(input), n }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats() -> PlanStats {
        PlanStats {
            table_rows: 10000.0,
            column_stats: vec![ColumnStats::default(), {
                let mut s = ColumnStats::default();
                s.distinct = 100;
                s
            }],
            indexed_cols: vec![false, true],
        }
    }

    #[test]
    fn plan_filter_with_index_uses_index_scan() {
        let logical = LogicalNode::Filter {
            input: Box::new(LogicalNode::Scan { table: "t".into() }),
            col: 1,
            op: "=".into(),
            value: Value::Int(5),
        };
        let phys = plan(&logical, &stats());
        match phys {
            PhysicalNode::Scan { path: ScanPath::IndexEq { col, .. }, .. } => assert_eq!(col, 1),
            _ => panic!("expected index scan"),
        }
    }

    #[test]
    fn plan_join_picks_hash_for_small() {
        let logical = LogicalNode::Join {
            left: Box::new(LogicalNode::Scan { table: "a".into() }),
            right: Box::new(LogicalNode::Scan { table: "b".into() }),
            left_col: 0,
            right_col: 0,
        };
        let phys = plan(&logical, &stats());
        assert!(matches!(phys, PhysicalNode::HashJoin { .. }));
    }

    #[test]
    fn optimize_pushes_filter_below_project() {
        let logical = LogicalNode::Filter {
            input: Box::new(LogicalNode::Project {
                input: Box::new(LogicalNode::Scan { table: "t".into() }),
                cols: vec![0, 1],
            }),
            col: 1,
            op: "=".into(),
            value: Value::Int(5),
        };
        let opt = optimize(logical);
        match opt {
            LogicalNode::Project { input, .. } => match *input {
                LogicalNode::Filter { .. } => {}
                _ => panic!("expected filter below project"),
            },
            _ => panic!("expected project on top"),
        }
    }

    #[test]
    fn pretty_print_does_not_panic() {
        let logical = LogicalNode::Limit {
            input: Box::new(LogicalNode::Sort {
                input: Box::new(LogicalNode::Scan { table: "t".into() }),
                keys: vec![(0, true)],
            }),
            n: 10,
        };
        let _ = logical.pretty();
        let phys = plan(&logical, &stats());
        let _ = phys.pretty();
    }
}
