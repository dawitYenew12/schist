//! A stack-machine bytecode compiler and evaluator for scalar expressions.
//!
//! The row-at-a-time evaluator walks an [`Expr`] tree per row, which re-pays the
//! tree-traversal cost for every row scanned. Compiling the expression once to a
//! flat bytecode program and then running that program per row removes the
//! pointer chasing: the program is a `Vec` of [`Op`]s executed against a small
//! value stack, reading inputs from a row slice by column index. This is the
//! same idea as a query JIT, minus the machine-code emission.

use crate::sql::{BinOp, Expr, UnaryOp};
use crate::value::Value;
use std::collections::HashMap;

/// One bytecode instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Push a constant.
    PushConst(Value),
    /// Push the value of input column `n`.
    PushColumn(usize),
    /// Pop two, push the binary result.
    Binary(BinOp),
    /// Pop one, push the unary result.
    Unary(UnaryOp),
    /// Pop one; push whether it is (not) null.
    IsNull(bool),
    /// Pop three (value, lo, hi); push whether value is in `[lo, hi]`.
    Between,
    /// Pop `n` values plus the probe; push membership.
    InList(usize),
}

/// A compiled expression program.
#[derive(Debug, Clone, Default)]
pub struct Program {
    ops: Vec<Op>,
}

impl Program {
    /// The instruction stream.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Number of instructions.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// `true` if the program is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Error compiling an expression to bytecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// A column name not present in the binding map.
    UnknownColumn(String),
    /// An expression form the VM does not support (e.g. aggregates).
    Unsupported(&'static str),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::UnknownColumn(c) => write!(f, "unknown column '{c}'"),
            CompileError::Unsupported(s) => write!(f, "unsupported expression: {s}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Compiles an [`Expr`] into a [`Program`], resolving column names to indices.
pub struct Compiler<'a> {
    columns: &'a HashMap<String, usize>,
}

impl<'a> Compiler<'a> {
    /// A compiler binding column names to input indices.
    pub fn new(columns: &'a HashMap<String, usize>) -> Compiler<'a> {
        Compiler { columns }
    }

    /// Compile `expr`.
    pub fn compile(&self, expr: &Expr) -> Result<Program, CompileError> {
        let mut prog = Program::default();
        self.emit(expr, &mut prog)?;
        Ok(prog)
    }

    fn column_index(&self, name: &str) -> Result<usize, CompileError> {
        self.columns
            .get(name)
            .copied()
            .ok_or_else(|| CompileError::UnknownColumn(name.to_string()))
    }

    fn emit(&self, expr: &Expr, prog: &mut Program) -> Result<(), CompileError> {
        match expr {
            Expr::LitInt(i) => prog.ops.push(Op::PushConst(Value::Int(*i))),
            Expr::LitReal(r) => prog.ops.push(Op::PushConst(Value::Real(*r))),
            Expr::LitBool(b) => prog.ops.push(Op::PushConst(Value::Bool(*b))),
            Expr::LitStr(_) => {
                return Err(CompileError::Unsupported("string literal without dictionary"))
            }
            Expr::Null => prog.ops.push(Op::PushConst(Value::Null)),
            Expr::Column(c) => prog.ops.push(Op::PushColumn(self.column_index(c)?)),
            Expr::Qualified(_, c) => prog.ops.push(Op::PushColumn(self.column_index(c)?)),
            Expr::Unary(op, inner) => {
                self.emit(inner, prog)?;
                prog.ops.push(Op::Unary(*op));
            }
            Expr::Binary(op, a, b) => {
                self.emit(a, prog)?;
                self.emit(b, prog)?;
                prog.ops.push(Op::Binary(*op));
            }
            Expr::IsNull(inner, negated) => {
                self.emit(inner, prog)?;
                prog.ops.push(Op::IsNull(*negated));
            }
            Expr::Between(v, lo, hi) => {
                self.emit(v, prog)?;
                self.emit(lo, prog)?;
                self.emit(hi, prog)?;
                prog.ops.push(Op::Between);
            }
            Expr::InList(probe, list) => {
                self.emit(probe, prog)?;
                for item in list {
                    self.emit(item, prog)?;
                }
                prog.ops.push(Op::InList(list.len()));
            }
            Expr::Like(_, _) => return Err(CompileError::Unsupported("LIKE")),
            Expr::Aggregate(..) => return Err(CompileError::Unsupported("aggregate")),
            Expr::Star => return Err(CompileError::Unsupported("*")),
        }
        Ok(())
    }
}

/// Evaluate a compiled program against a row of input values.
pub fn eval(program: &Program, row: &[Value]) -> Value {
    let mut stack: Vec<Value> = Vec::with_capacity(8);
    for op in &program.ops {
        match op {
            Op::PushConst(v) => stack.push(*v),
            Op::PushColumn(i) => stack.push(row.get(*i).copied().unwrap_or(Value::Null)),
            Op::Unary(op) => {
                let v = stack.pop().unwrap_or(Value::Null);
                stack.push(apply_unary(*op, v));
            }
            Op::Binary(op) => {
                let b = stack.pop().unwrap_or(Value::Null);
                let a = stack.pop().unwrap_or(Value::Null);
                stack.push(apply_binary(*op, a, b));
            }
            Op::IsNull(negated) => {
                let v = stack.pop().unwrap_or(Value::Null);
                stack.push(Value::Bool(v.is_null() ^ negated));
            }
            Op::Between => {
                let hi = stack.pop().unwrap_or(Value::Null);
                let lo = stack.pop().unwrap_or(Value::Null);
                let v = stack.pop().unwrap_or(Value::Null);
                if v.is_null() || lo.is_null() || hi.is_null() {
                    stack.push(Value::Null);
                } else {
                    let in_range = v.total_cmp(&lo).is_ge() && v.total_cmp(&hi).is_le();
                    stack.push(Value::Bool(in_range));
                }
            }
            Op::InList(n) => {
                // The list items sit on top of the stack; the probe is beneath.
                let mut items = Vec::with_capacity(*n);
                for _ in 0..*n {
                    items.push(stack.pop().unwrap_or(Value::Null));
                }
                let probe = stack.pop().unwrap_or(Value::Null);
                if probe.is_null() {
                    stack.push(Value::Null);
                    continue;
                }
                let mut found = false;
                let mut saw_null = false;
                for item in items {
                    if item.is_null() {
                        saw_null = true;
                    } else if probe.total_cmp(&item).is_eq() {
                        found = true;
                        break;
                    }
                }
                // SQL semantics: a miss with a NULL present is unknown (NULL).
                if found {
                    stack.push(Value::Bool(true));
                } else if saw_null {
                    stack.push(Value::Null);
                } else {
                    stack.push(Value::Bool(false));
                }
            }
        }
    }
    stack.pop().unwrap_or(Value::Null)
}

fn apply_unary(op: UnaryOp, v: Value) -> Value {
    if v.is_null() {
        return Value::Null;
    }
    match op {
        UnaryOp::Neg => match v {
            Value::Int(i) => Value::Int(-i),
            Value::Real(r) => Value::Real(-r),
            _ => Value::Null,
        },
        UnaryOp::Not => match v {
            Value::Bool(b) => Value::Bool(!b),
            Value::Int(i) => Value::Bool(i == 0),
            _ => Value::Null,
        },
    }
}

fn apply_binary(op: BinOp, a: Value, b: Value) -> Value {
    use BinOp::*;
    // Logical operators have their own null rules; handle first.
    match op {
        And => {
            return match (as_bool(a), as_bool(b)) {
                (Some(false), _) | (_, Some(false)) => Value::Bool(false),
                (Some(true), Some(true)) => Value::Bool(true),
                _ => Value::Null,
            }
        }
        Or => {
            return match (as_bool(a), as_bool(b)) {
                (Some(true), _) | (_, Some(true)) => Value::Bool(true),
                (Some(false), Some(false)) => Value::Bool(false),
                _ => Value::Null,
            }
        }
        _ => {}
    }
    if a.is_null() || b.is_null() {
        return Value::Null;
    }
    match op {
        Add | Sub | Mul | Div | Mod => arith(op, a, b),
        Eq => Value::Bool(a.total_cmp(&b).is_eq()),
        NotEq => Value::Bool(a.total_cmp(&b).is_ne()),
        Lt => Value::Bool(a.total_cmp(&b).is_lt()),
        LtEq => Value::Bool(a.total_cmp(&b).is_le()),
        Gt => Value::Bool(a.total_cmp(&b).is_gt()),
        GtEq => Value::Bool(a.total_cmp(&b).is_ge()),
        And | Or => unreachable!(),
    }
}

fn arith(op: BinOp, a: Value, b: Value) -> Value {
    // Prefer integer arithmetic when both are integers.
    if let (Value::Int(x), Value::Int(y)) = (a, b) {
        return match op {
            BinOp::Add => Value::Int(x.wrapping_add(y)),
            BinOp::Sub => Value::Int(x.wrapping_sub(y)),
            BinOp::Mul => Value::Int(x.wrapping_mul(y)),
            BinOp::Div if y != 0 => Value::Int(x / y),
            BinOp::Mod if y != 0 => Value::Int(x % y),
            _ => Value::Null,
        };
    }
    match (a.as_real(), b.as_real()) {
        (Some(x), Some(y)) => match op {
            BinOp::Add => Value::Real(x + y),
            BinOp::Sub => Value::Real(x - y),
            BinOp::Mul => Value::Real(x * y),
            BinOp::Div if y != 0.0 => Value::Real(x / y),
            BinOp::Mod if y != 0.0 => Value::Real(x % y),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

fn as_bool(v: Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(b),
        Value::Int(i) => Some(i != 0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::Parser;

    fn cols() -> HashMap<String, usize> {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 0);
        m.insert("b".to_string(), 1);
        m
    }

    fn compile_expr(src: &str) -> Program {
        // Parse "SELECT <expr>" and pull the projected expression out.
        let stmt = Parser::new(&format!("SELECT {src} FROM t"))
            .unwrap()
            .parse_statement()
            .unwrap();
        let expr = match stmt {
            crate::sql::Statement::Select(s) => s.items[0].expr.clone(),
            _ => panic!(),
        };
        let cols = cols();
        Compiler::new(&cols).compile(&expr).unwrap()
    }

    #[test]
    fn arithmetic_program() {
        let prog = compile_expr("a + b * 2");
        let v = eval(&prog, &[Value::Int(3), Value::Int(4)]);
        assert_eq!(v, Value::Int(11));
    }

    #[test]
    fn comparison_and_logic() {
        let prog = compile_expr("a > 1 AND b < 10");
        assert_eq!(eval(&prog, &[Value::Int(2), Value::Int(5)]), Value::Bool(true));
        assert_eq!(eval(&prog, &[Value::Int(0), Value::Int(5)]), Value::Bool(false));
    }

    #[test]
    fn null_propagation() {
        let prog = compile_expr("a + b");
        assert_eq!(eval(&prog, &[Value::Null, Value::Int(1)]), Value::Null);
    }

    #[test]
    fn is_null_op() {
        let prog = compile_expr("a IS NULL");
        assert_eq!(eval(&prog, &[Value::Null, Value::Int(0)]), Value::Bool(true));
        assert_eq!(eval(&prog, &[Value::Int(1), Value::Int(0)]), Value::Bool(false));
    }

    #[test]
    fn between_op() {
        let prog = compile_expr("a BETWEEN 1 AND 10");
        assert_eq!(eval(&prog, &[Value::Int(5), Value::Null]), Value::Bool(true));
        assert_eq!(eval(&prog, &[Value::Int(20), Value::Null]), Value::Bool(false));
    }

    #[test]
    fn in_list_op() {
        let prog = compile_expr("a IN (1, 3, 5)");
        assert_eq!(eval(&prog, &[Value::Int(3), Value::Null]), Value::Bool(true));
        assert_eq!(eval(&prog, &[Value::Int(4), Value::Null]), Value::Bool(false));
    }

    #[test]
    fn unknown_column_errors() {
        let cols = cols();
        let stmt = Parser::new("SELECT c FROM t").unwrap().parse_statement().unwrap();
        let expr = match stmt {
            crate::sql::Statement::Select(s) => s.items[0].expr.clone(),
            _ => panic!(),
        };
        assert!(matches!(
            Compiler::new(&cols).compile(&expr),
            Err(CompileError::UnknownColumn(_))
        ));
    }
}
