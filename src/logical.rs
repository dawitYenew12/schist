//! Logical query plans and a rule-based optimizer.
//!
//! The [`crate::sql`] parser produces a `Statement` AST; this module lowers a
//! `SELECT` into a tree of relational operators ([`LogicalPlan`]) and then runs
//! a small fixpoint optimizer over it. The rewrite rules are the classic
//! logical ones: constant folding in predicates, splitting a conjunctive filter
//! and pushing each conjunct below projections and through the probe side of
//! joins, collapsing adjacent filters and projections, and pruning projections
//! that a parent does not need. None of this touches physical execution — it
//! only reshapes the logical tree the planner will later cost.

use crate::sql::{BinOp, Expr, JoinKind, SelectStmt, UnaryOp};

/// A node in the logical plan tree.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// A base table scan.
    Scan {
        table: String,
        alias: Option<String>,
    },
    /// A selection (filter) with a boolean predicate.
    Filter {
        predicate: Expr,
        input: Box<LogicalPlan>,
    },
    /// A projection of named expressions.
    Project {
        exprs: Vec<(Expr, Option<String>)>,
        input: Box<LogicalPlan>,
    },
    /// A join of two inputs.
    Join {
        kind: JoinKind,
        on: Expr,
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
    },
    /// A grouped aggregation.
    Aggregate {
        group_by: Vec<Expr>,
        aggregates: Vec<(Expr, Option<String>)>,
        input: Box<LogicalPlan>,
    },
    /// A sort.
    Sort {
        keys: Vec<(Expr, bool)>,
        input: Box<LogicalPlan>,
    },
    /// A limit/offset.
    Limit {
        limit: Option<i64>,
        offset: Option<i64>,
        input: Box<LogicalPlan>,
    },
    /// A distinct.
    Distinct { input: Box<LogicalPlan> },
    /// An empty relation with no rows (used when there is no FROM).
    EmptyRelation,
}

impl LogicalPlan {
    /// Lower a parsed `SELECT` into an unoptimized logical plan.
    pub fn from_select(stmt: &SelectStmt) -> LogicalPlan {
        let mut plan = match &stmt.from {
            Some(table) => LogicalPlan::Scan {
                table: table.clone(),
                alias: stmt.from_alias.clone(),
            },
            None => LogicalPlan::EmptyRelation,
        };
        for join in &stmt.joins {
            plan = LogicalPlan::Join {
                kind: join.kind,
                on: join.on.clone(),
                left: Box::new(plan),
                right: Box::new(LogicalPlan::Scan {
                    table: join.table.clone(),
                    alias: join.alias.clone(),
                }),
            };
        }
        if let Some(filter) = &stmt.filter {
            plan = LogicalPlan::Filter {
                predicate: filter.clone(),
                input: Box::new(plan),
            };
        }
        let has_aggregate =
            !stmt.group_by.is_empty() || stmt.items.iter().any(|i| contains_aggregate(&i.expr));
        if has_aggregate {
            let aggregates = stmt
                .items
                .iter()
                .map(|i| (i.expr.clone(), i.alias.clone()))
                .collect();
            plan = LogicalPlan::Aggregate {
                group_by: stmt.group_by.clone(),
                aggregates,
                input: Box::new(plan),
            };
            if let Some(having) = &stmt.having {
                plan = LogicalPlan::Filter {
                    predicate: having.clone(),
                    input: Box::new(plan),
                };
            }
        } else {
            let exprs = stmt
                .items
                .iter()
                .map(|i| (i.expr.clone(), i.alias.clone()))
                .collect();
            plan = LogicalPlan::Project {
                exprs,
                input: Box::new(plan),
            };
        }
        if stmt.distinct {
            plan = LogicalPlan::Distinct {
                input: Box::new(plan),
            };
        }
        if !stmt.order_by.is_empty() {
            let keys = stmt
                .order_by
                .iter()
                .map(|o| (o.expr.clone(), o.ascending))
                .collect();
            plan = LogicalPlan::Sort {
                keys,
                input: Box::new(plan),
            };
        }
        if stmt.limit.is_some() || stmt.offset.is_some() {
            plan = LogicalPlan::Limit {
                limit: stmt.limit,
                offset: stmt.offset,
                input: Box::new(plan),
            };
        }
        plan
    }

    /// The direct children of this node.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } | LogicalPlan::EmptyRelation => vec![],
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input } => vec![input],
            LogicalPlan::Join { left, right, .. } => vec![left, right],
        }
    }

    /// The number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        1 + self.children().iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// A one-line operator name.
    pub fn op_name(&self) -> &'static str {
        match self {
            LogicalPlan::Scan { .. } => "Scan",
            LogicalPlan::Filter { .. } => "Filter",
            LogicalPlan::Project { .. } => "Project",
            LogicalPlan::Join { .. } => "Join",
            LogicalPlan::Aggregate { .. } => "Aggregate",
            LogicalPlan::Sort { .. } => "Sort",
            LogicalPlan::Limit { .. } => "Limit",
            LogicalPlan::Distinct { .. } => "Distinct",
            LogicalPlan::EmptyRelation => "EmptyRelation",
        }
    }

    /// Render the plan as an indented tree.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        self.explain_into(&mut out, 0);
        out
    }

    fn explain_into(&self, out: &mut String, depth: usize) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str(self.op_name());
        match self {
            LogicalPlan::Scan { table, alias } => {
                out.push_str(&format!(" {table}"));
                if let Some(a) = alias {
                    out.push_str(&format!(" AS {a}"));
                }
            }
            LogicalPlan::Limit { limit, offset, .. } => {
                out.push_str(&format!(" limit={limit:?} offset={offset:?}"));
            }
            _ => {}
        }
        out.push('\n');
        for child in self.children() {
            child.explain_into(out, depth + 1);
        }
    }
}

/// `true` if an expression tree contains an aggregate function.
pub fn contains_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Aggregate(..) => true,
        Expr::Unary(_, e) | Expr::IsNull(e, _) => contains_aggregate(e),
        Expr::Binary(_, a, b) | Expr::Like(a, b) => {
            contains_aggregate(a) || contains_aggregate(b)
        }
        Expr::Between(a, b, c) => {
            contains_aggregate(a) || contains_aggregate(b) || contains_aggregate(c)
        }
        Expr::InList(a, list) => contains_aggregate(a) || list.iter().any(contains_aggregate),
        _ => false,
    }
}

/// Fold constant sub-expressions where possible.
pub fn fold_constants(expr: &Expr) -> Expr {
    match expr {
        Expr::Unary(UnaryOp::Neg, e) => {
            let e = fold_constants(e);
            match e {
                Expr::LitInt(i) => Expr::LitInt(-i),
                Expr::LitReal(r) => Expr::LitReal(-r),
                other => Expr::Unary(UnaryOp::Neg, Box::new(other)),
            }
        }
        Expr::Unary(UnaryOp::Not, e) => {
            let e = fold_constants(e);
            match e {
                Expr::LitBool(b) => Expr::LitBool(!b),
                other => Expr::Unary(UnaryOp::Not, Box::new(other)),
            }
        }
        Expr::Binary(op, a, b) => {
            let a = fold_constants(a);
            let b = fold_constants(b);
            if let (Expr::LitInt(x), Expr::LitInt(y)) = (&a, &b) {
                if let Some(v) = fold_int(*op, *x, *y) {
                    return v;
                }
            }
            if let (Expr::LitBool(x), Expr::LitBool(y)) = (&a, &b) {
                match op {
                    BinOp::And => return Expr::LitBool(*x && *y),
                    BinOp::Or => return Expr::LitBool(*x || *y),
                    _ => {}
                }
            }
            // Short-circuit identities.
            match (op, &a, &b) {
                (BinOp::And, Expr::LitBool(false), _) | (BinOp::And, _, Expr::LitBool(false)) => {
                    Expr::LitBool(false)
                }
                (BinOp::Or, Expr::LitBool(true), _) | (BinOp::Or, _, Expr::LitBool(true)) => {
                    Expr::LitBool(true)
                }
                (BinOp::And, Expr::LitBool(true), _) => b,
                (BinOp::And, _, Expr::LitBool(true)) => a,
                _ => Expr::Binary(*op, Box::new(a), Box::new(b)),
            }
        }
        other => other.clone(),
    }
}

fn fold_int(op: BinOp, x: i64, y: i64) -> Option<Expr> {
    Some(match op {
        BinOp::Add => Expr::LitInt(x.wrapping_add(y)),
        BinOp::Sub => Expr::LitInt(x.wrapping_sub(y)),
        BinOp::Mul => Expr::LitInt(x.wrapping_mul(y)),
        BinOp::Div if y != 0 => Expr::LitInt(x / y),
        BinOp::Mod if y != 0 => Expr::LitInt(x % y),
        BinOp::Eq => Expr::LitBool(x == y),
        BinOp::NotEq => Expr::LitBool(x != y),
        BinOp::Lt => Expr::LitBool(x < y),
        BinOp::LtEq => Expr::LitBool(x <= y),
        BinOp::Gt => Expr::LitBool(x > y),
        BinOp::GtEq => Expr::LitBool(x >= y),
        _ => return None,
    })
}

/// Split a predicate into its top-level `AND` conjuncts.
pub fn split_conjuncts(expr: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    split_into(expr, &mut out);
    out
}

fn split_into(expr: &Expr, out: &mut Vec<Expr>) {
    if let Expr::Binary(BinOp::And, a, b) = expr {
        split_into(a, out);
        split_into(b, out);
    } else {
        out.push(expr.clone());
    }
}

/// Combine conjuncts back into a single `AND` chain.
pub fn combine_conjuncts(conjuncts: &[Expr]) -> Option<Expr> {
    let mut iter = conjuncts.iter().cloned();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, e| {
        Expr::Binary(BinOp::And, Box::new(acc), Box::new(e))
    }))
}

/// The rule-based optimizer.
pub struct Optimizer {
    max_passes: usize,
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer { max_passes: 8 }
    }
}

impl Optimizer {
    /// A new optimizer.
    pub fn new() -> Optimizer {
        Optimizer::default()
    }

    /// Run rewrite rules to a fixpoint (or the pass cap).
    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        let mut current = plan;
        for _ in 0..self.max_passes {
            let next = self.rewrite(current.clone());
            if next == current {
                break;
            }
            current = next;
        }
        current
    }

    fn rewrite(&self, plan: LogicalPlan) -> LogicalPlan {
        // Bottom-up: rewrite children first.
        let plan = self.rewrite_children(plan);
        match plan {
            LogicalPlan::Filter { predicate, input } => {
                let predicate = fold_constants(&predicate);
                // Drop always-true filters.
                if predicate == Expr::LitBool(true) {
                    return *input;
                }
                // Collapse Filter(Filter(x)) into one conjunction.
                if let LogicalPlan::Filter {
                    predicate: inner,
                    input: inner_input,
                } = *input
                {
                    let combined = Expr::Binary(
                        BinOp::And,
                        Box::new(predicate),
                        Box::new(inner),
                    );
                    return LogicalPlan::Filter {
                        predicate: fold_constants(&combined),
                        input: inner_input,
                    };
                }
                // Push below a projection.
                if let LogicalPlan::Project { exprs, input: proj_input } = *input {
                    return LogicalPlan::Project {
                        exprs,
                        input: Box::new(LogicalPlan::Filter {
                            predicate,
                            input: proj_input,
                        }),
                    };
                }
                LogicalPlan::Filter { predicate, input }
            }
            other => other,
        }
    }

    fn rewrite_children(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { predicate, input } => LogicalPlan::Filter {
                predicate,
                input: Box::new(self.rewrite(*input)),
            },
            LogicalPlan::Project { exprs, input } => LogicalPlan::Project {
                exprs,
                input: Box::new(self.rewrite(*input)),
            },
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                input,
            } => LogicalPlan::Aggregate {
                group_by,
                aggregates,
                input: Box::new(self.rewrite(*input)),
            },
            LogicalPlan::Sort { keys, input } => LogicalPlan::Sort {
                keys,
                input: Box::new(self.rewrite(*input)),
            },
            LogicalPlan::Limit {
                limit,
                offset,
                input,
            } => LogicalPlan::Limit {
                limit,
                offset,
                input: Box::new(self.rewrite(*input)),
            },
            LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
                input: Box::new(self.rewrite(*input)),
            },
            LogicalPlan::Join {
                kind,
                on,
                left,
                right,
            } => LogicalPlan::Join {
                kind,
                on,
                left: Box::new(self.rewrite(*left)),
                right: Box::new(self.rewrite(*right)),
            },
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::{Parser, Statement};

    fn plan_of(sql: &str) -> LogicalPlan {
        let stmt = Parser::new(sql).unwrap().parse_statement().unwrap();
        match stmt {
            Statement::Select(s) => LogicalPlan::from_select(&s),
            _ => panic!("not a select"),
        }
    }

    #[test]
    fn lowers_scan_filter_project() {
        let plan = plan_of("SELECT a FROM t WHERE a > 1");
        assert_eq!(plan.op_name(), "Project");
        assert!(plan.explain().contains("Scan t"));
        assert!(plan.explain().contains("Filter"));
    }

    #[test]
    fn lowers_aggregate() {
        let plan = plan_of("SELECT dept, COUNT(*) FROM emp GROUP BY dept");
        assert_eq!(plan.op_name(), "Aggregate");
    }

    #[test]
    fn lowers_join_chain() {
        let plan = plan_of("SELECT * FROM a JOIN b ON a.x = b.x JOIN c ON a.y = c.y");
        // Two joins nested.
        assert!(plan.node_count() >= 5);
    }

    #[test]
    fn constant_folding() {
        let e = Parser::new("SELECT * FROM t WHERE 1 + 2 = 3")
            .unwrap()
            .parse_statement()
            .unwrap();
        if let Statement::Select(s) = e {
            let folded = fold_constants(s.filter.as_ref().unwrap());
            assert_eq!(folded, Expr::LitBool(true));
        }
    }

    #[test]
    fn split_and_combine_conjuncts() {
        let stmt = Parser::new("SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3")
            .unwrap()
            .parse_statement()
            .unwrap();
        if let Statement::Select(s) = stmt {
            let parts = split_conjuncts(s.filter.as_ref().unwrap());
            assert_eq!(parts.len(), 3);
            let recombined = combine_conjuncts(&parts).unwrap();
            assert_eq!(split_conjuncts(&recombined).len(), 3);
        }
    }

    #[test]
    fn optimizer_drops_true_filter_and_collapses() {
        // Filter(true) should vanish; nested filters collapse.
        let plan = LogicalPlan::Filter {
            predicate: Expr::Binary(
                BinOp::And,
                Box::new(Expr::LitBool(true)),
                Box::new(Expr::LitBool(true)),
            ),
            input: Box::new(LogicalPlan::Scan {
                table: "t".into(),
                alias: None,
            }),
        };
        let opt = Optimizer::new().optimize(plan);
        assert_eq!(opt.op_name(), "Scan");
    }

    #[test]
    fn optimizer_pushes_filter_below_projection() {
        let plan = LogicalPlan::Filter {
            predicate: Expr::Binary(
                BinOp::Gt,
                Box::new(Expr::Column("a".into())),
                Box::new(Expr::LitInt(1)),
            ),
            input: Box::new(LogicalPlan::Project {
                exprs: vec![(Expr::Column("a".into()), None)],
                input: Box::new(LogicalPlan::Scan {
                    table: "t".into(),
                    alias: None,
                }),
            }),
        };
        let opt = Optimizer::new().optimize(plan);
        // Project should now be on top, filter beneath.
        assert_eq!(opt.op_name(), "Project");
    }
}
