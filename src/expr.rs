//! Expression evaluation.
//!
//! The query and script layers share a small expression language for
//! predicates and computed projections. An expression is parsed from tokens
//! into an [`Expr`] tree and evaluated against a row of [`Value`]s bound to
//! column positions. The evaluator supports arithmetic, comparisons, logical
//! operators, `CASE`-style conditionals, and a handful of scalar functions.
//!
//! ## Grammar
//!
//! ```text
//!   expr   := or
//!   or     := and ( "OR" and )*
//!   and    := not ( "AND" not )*
//!   not    := "NOT" not | cmp
//!   cmp    := add ( (=|!=|<|<=|>|>=) add )?
//!   add    := mul ( (+|-) mul )*
//!   mul    := unary ( (*|/|%) unary )*
//!   unary  := (-|+) unary | atom
//!   atom   := number | "true" | "false" | "null"
//!           | ident ( "(" arglist ")" )?          // function call or column
//!           | "(" expr ")"
//! ```

use crate::value::{CmpOp, Value};

/// An expression node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal value.
    Lit(Value),
    /// A column reference by index.
    Col(usize),
    /// A column reference by name (resolved at evaluation time against a name
    /// table supplied by the caller).
    ColName(String),
    /// Arithmetic: `op(lhs, rhs)`.
    BinArith { op: ArithOp, lhs: Box<Expr>, rhs: Box<Expr> },
    /// Comparison: `op(lhs, rhs) -> bool`.
    BinCmp { op: CmpOp, lhs: Box<Expr>, rhs: Box<Expr> },
    /// Logical AND / OR.
    BinLogic { op: LogicOp, lhs: Box<Expr>, rhs: Box<Expr> },
    /// Logical NOT.
    Not(Box<Expr>),
    /// Unary minus.
    Neg(Box<Expr>),
    /// `CASE WHEN cond THEN a ELSE b END`.
    Case { cond: Box<Expr>, then: Box<Expr>, els: Box<Expr> },
    /// `COALESCE(a, b, ...)`.
    Coalesce(Vec<Expr>),
    /// A scalar function call.
    Func { name: String, args: Vec<Expr> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl ArithOp {
    pub fn name(self) -> &'static str {
        match self {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "*",
            ArithOp::Div => "/",
            ArithOp::Mod => "%",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
}

/// A scalar function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Func {
    Abs,
    Min,
    Max,
    Length,
    Upper,
    Lower,
    IsNull,
    IsNotNull,
    IfNull,
    Cast,
    Coalesce,
}

impl Func {
    pub fn from_name(name: &str) -> Option<Func> {
        match name.to_ascii_lowercase().as_str() {
            "abs" => Some(Func::Abs),
            "min" => Some(Func::Min),
            "max" => Some(Func::Max),
            "length" => Some(Func::Length),
            "upper" => Some(Func::Upper),
            "lower" => Some(Func::Lower),
            "isnull" => Some(Func::IsNull),
            "isnotnull" => Some(Func::IsNotNull),
            "ifnull" => Some(Func::IfNull),
            "cast" => Some(Func::Cast),
            "coalesce" => Some(Func::Coalesce),
            _ => None,
        }
    }
}

/// The result of evaluating an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    /// A column name could not be resolved.
    UnknownColumn(String),
    /// A function was called with the wrong arity.
    Arity { func: String, got: usize },
    /// An unknown function.
    UnknownFunction(String),
    /// A type error.
    Type(String),
    /// Division by zero.
    DivByZero,
}

/// A name table mapping column names to positions, used to resolve `ColName`.
pub trait NameTable {
    fn resolve(&self, name: &str) -> Option<usize>;
}

/// An empty name table (for expressions that use only positional columns).
pub struct NoNames;
impl NameTable for NoNames {
    fn resolve(&self, _name: &str) -> Option<usize> {
        None
    }
}

/// Evaluate an expression against a row of values, using positional columns.
pub fn eval(expr: &Expr, row: &[Value]) -> Result<Value, EvalError> {
    eval_with(expr, row, &NoNames)
}

/// Evaluate with a name table for resolving `ColName` references.
pub fn eval_with(
    expr: &Expr,
    row: &[Value],
    names: &dyn NameTable,
) -> Result<Value, EvalError> {
    match expr {
        Expr::Lit(v) => Ok(*v),
        Expr::Col(i) => row.get(*i).copied().ok_or(EvalError::UnknownColumn(format!("#{i}"))),
        Expr::ColName(n) => {
            let i = names.resolve(n).ok_or_else(|| EvalError::UnknownColumn(n.clone()))?;
            row.get(i).copied().ok_or_else(|| EvalError::UnknownColumn(n.clone()))
        }
        Expr::BinArith { op, lhs, rhs } => {
            let l = eval_with(lhs, row, names)?;
            let r = eval_with(rhs, row, names)?;
            arith(*op, &l, &r)
        }
        Expr::BinCmp { op, lhs, rhs } => {
            let l = eval_with(lhs, row, names)?;
            let r = eval_with(rhs, row, names)?;
            Ok(Value::Bool(op.apply(&l, &r)))
        }
        Expr::BinLogic { op, lhs, rhs } => {
            let l = eval_with(lhs, row, names)?;
            if matches!(op, LogicOp::And) && !truthy(&l) {
                return Ok(Value::Bool(false));
            }
            if matches!(op, LogicOp::Or) && truthy(&l) {
                return Ok(Value::Bool(true));
            }
            let r = eval_with(rhs, row, names)?;
            Ok(Value::Bool(truthy(&r)))
        }
        Expr::Not(e) => Ok(Value::Bool(!truthy(&eval_with(e, row, names)?))),
        Expr::Neg(e) => {
            let v = eval_with(e, row, names)?;
            match v {
                Value::Int(i) => Ok(Value::Int(-i)),
                Value::Real(r) => Ok(Value::Real(-r)),
                _ => Err(EvalError::Type("neg of non-number".into())),
            }
        }
        Expr::Case { cond, then, els } => {
            if truthy(&eval_with(cond, row, names)?) {
                eval_with(then, row, names)
            } else {
                eval_with(els, row, names)
            }
        }
        Expr::Coalesce(exprs) => {
            for e in exprs {
                let v = eval_with(e, row, names)?;
                if !v.is_null() {
                    return Ok(v);
                }
            }
            Ok(Value::Null)
        }
        Expr::Func { name, args } => {
            let func = Func::from_name(name).ok_or_else(|| EvalError::UnknownFunction(name.clone()))?;
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_with(a, row, names)?);
            }
            call(func, &vals)
        }
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Real(r) => *r != 0.0,
        Value::Text(id) => *id != 0,
    }
}

fn arith(op: ArithOp, l: &Value, r: &Value) -> Result<Value, EvalError> {
    if l.is_null() || r.is_null() {
        return Ok(Value::Null);
    }
    // Promote to real if either side is real.
    if matches!(l, Value::Real(_)) || matches!(r, Value::Real(_)) {
        let a = l.as_real().ok_or(EvalError::Type("arith on non-number".into()))?;
        let b = r.as_real().ok_or(EvalError::Type("arith on non-number".into()))?;
        return Ok(Value::Real(match op {
            ArithOp::Add => a + b,
            ArithOp::Sub => a - b,
            ArithOp::Mul => a * b,
            ArithOp::Div => {
                if b == 0.0 {
                    return Err(EvalError::DivByZero);
                }
                a / b
            }
            ArithOp::Mod => {
                if b == 0.0 {
                    return Err(EvalError::DivByZero);
                }
                a % b
            }
        }));
    }
    let a = l.as_int().ok_or(EvalError::Type("arith on non-number".into()))?;
    let b = r.as_int().ok_or(EvalError::Type("arith on non-number".into()))?;
    Ok(Value::Int(match op {
        ArithOp::Add => a.wrapping_add(b),
        ArithOp::Sub => a.wrapping_sub(b),
        ArithOp::Mul => a.wrapping_mul(b),
        ArithOp::Div => {
            if b == 0 {
                return Err(EvalError::DivByZero);
            }
            a / b
        }
        ArithOp::Mod => {
            if b == 0 {
                return Err(EvalError::DivByZero);
            }
            a % b
        }
    }))
}

fn call(func: Func, args: &[Value]) -> Result<Value, EvalError> {
    match func {
        Func::Abs => {
            if args.len() != 1 {
                return Err(EvalError::Arity { func: "abs".into(), got: args.len() });
            }
            match args[0] {
                Value::Int(i) => Ok(Value::Int(i.abs())),
                Value::Real(r) => Ok(Value::Real(r.abs())),
                Value::Null => Ok(Value::Null),
                _ => Err(EvalError::Type("abs of non-number".into())),
            }
        }
        Func::Min => {
            if args.len() < 2 {
                return Err(EvalError::Arity { func: "min".into(), got: args.len() });
            }
            let mut best = args[0];
            for &v in &args[1..] {
                if v.total_cmp(&best) == std::cmp::Ordering::Less {
                    best = v;
                }
            }
            Ok(best)
        }
        Func::Max => {
            if args.len() < 2 {
                return Err(EvalError::Arity { func: "max".into(), got: args.len() });
            }
            let mut best = args[0];
            for &v in &args[1..] {
                if v.total_cmp(&best) == std::cmp::Ordering::Greater {
                    best = v;
                }
            }
            Ok(best)
        }
        Func::Length => {
            if args.len() != 1 {
                return Err(EvalError::Arity { func: "length".into(), got: args.len() });
            }
            match args[0] {
                Value::Text(id) => Ok(Value::Int(id as i64)),
                Value::Null => Ok(Value::Null),
                _ => Err(EvalError::Type("length of non-text".into())),
            }
        }
        Func::Upper | Func::Lower => {
            // Text case functions operate on dictionary ids; without the
            // dictionary here we return the id unchanged (a no-op marker).
            if args.len() != 1 {
                return Err(EvalError::Arity {
                    func: if matches!(func, Func::Upper) { "upper" } else { "lower" }.into(),
                    got: args.len(),
                });
            }
            Ok(args[0])
        }
        Func::IsNull => {
            if args.len() != 1 {
                return Err(EvalError::Arity { func: "isnull".into(), got: args.len() });
            }
            Ok(Value::Bool(args[0].is_null()))
        }
        Func::IsNotNull => {
            if args.len() != 1 {
                return Err(EvalError::Arity { func: "isnotnull".into(), got: args.len() });
            }
            Ok(Value::Bool(!args[0].is_null()))
        }
        Func::IfNull => {
            if args.len() != 2 {
                return Err(EvalError::Arity { func: "ifnull".into(), got: args.len() });
            }
            Ok(if args[0].is_null() { args[1] } else { args[0] })
        }
        Func::Cast => {
            if args.len() != 2 {
                return Err(EvalError::Arity { func: "cast".into(), got: args.len() });
            }
            // The second argument is the target kind id (int/real/bool/text).
            Ok(args[0])
        }
        Func::Coalesce => {
            if args.is_empty() {
                return Err(EvalError::Arity { func: "coalesce".into(), got: 0 });
            }
            for &v in args {
                if !v.is_null() {
                    return Ok(v);
                }
            }
            Ok(Value::Null)
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse an expression from a source string.
pub fn parse(src: &str) -> Result<Expr, String> {
    let toks = lex(src);
    let mut p = TokParser { toks, pos: 0 };
    let e = p.parse_or()?;
    if p.pos < p.toks.len() {
        return Err(format!("trailing tokens at {}", p.toks[p.pos]));
    }
    Ok(e)
}

struct TokParser {
    toks: Vec<String>,
    pos: usize,
}

impl TokParser {
    fn peek(&self) -> Option<&str> {
        self.toks.get(self.pos).map(|s| s.as_str())
    }
    fn next(&mut self) -> Option<&str> {
        let t = self.toks.get(self.pos).map(|s| s.as_str());
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_and()?;
        while let Some(t) = self.peek() {
            if t.eq_ignore_ascii_case("or") {
                self.next();
                let rhs = self.parse_and()?;
                lhs = Expr::BinLogic { op: LogicOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs) };
            } else {
                break;
            }
        }
        Ok(lhs)
    }
    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_not()?;
        while let Some(t) = self.peek() {
            if t.eq_ignore_ascii_case("and") {
                self.next();
                let rhs = self.parse_not()?;
                lhs = Expr::BinLogic { op: LogicOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs) };
            } else {
                break;
            }
        }
        Ok(lhs)
    }
    fn parse_not(&mut self) -> Result<Expr, String> {
        if let Some(t) = self.peek() {
            if t.eq_ignore_ascii_case("not") {
                self.next();
                let e = self.parse_not()?;
                return Ok(Expr::Not(Box::new(e)));
            }
        }
        self.parse_cmp()
    }
    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let lhs = self.parse_add()?;
        if let Some(t) = self.peek() {
            if let Some(op) = CmpOp::parse(t) {
                self.next();
                let rhs = self.parse_add()?;
                return Ok(Expr::BinCmp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) });
            }
        }
        Ok(lhs)
    }
    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Some("+") => ArithOp::Add,
                Some("-") => ArithOp::Sub,
                _ => break,
            };
            self.next();
            let rhs = self.parse_mul()?;
            lhs = Expr::BinArith { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }
    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some("*") => ArithOp::Mul,
                Some("/") => ArithOp::Div,
                Some("%") => ArithOp::Mod,
                _ => break,
            };
            self.next();
            let rhs = self.parse_unary()?;
            lhs = Expr::BinArith { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }
    fn parse_unary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some("-") => {
                self.next();
                let e = self.parse_unary()?;
                Ok(Expr::Neg(Box::new(e)))
            }
            Some("+") => {
                self.next();
                self.parse_unary()
            }
            _ => self.parse_atom(),
        }
    }
    fn parse_atom(&mut self) -> Result<Expr, String> {
        let t = self.next().ok_or("unexpected end of expression")?.to_string();
        if t == "(" {
            let e = self.parse_or()?;
            match self.next() {
                Some(")") => Ok(e),
                _ => Err("expected )".into()),
            }
        } else if t.eq_ignore_ascii_case("true") {
            Ok(Expr::Lit(Value::Bool(true)))
        } else if t.eq_ignore_ascii_case("false") {
            Ok(Expr::Lit(Value::Bool(false)))
        } else if t.eq_ignore_ascii_case("null") {
            Ok(Expr::Lit(Value::Null))
        } else if t.starts_with('"') {
            // String literal -> encode as text id 0 marker (caller resolves).
            Ok(Expr::Lit(Value::Text(0)))
        } else if let Ok(i) = t.parse::<i64>() {
            Ok(Expr::Lit(Value::Int(i)))
        } else if let Ok(r) = t.parse::<f64>() {
            Ok(Expr::Lit(Value::Real(r)))
        } else if self.peek() == Some("(") {
            self.next();
            let mut args = Vec::new();
            if self.peek() != Some(")") {
                args.push(self.parse_or()?);
                while self.peek() == Some(",") {
                    self.next();
                    args.push(self.parse_or()?);
                }
            }
            match self.next() {
                Some(")") => Ok(Expr::Func { name: t, args }),
                _ => Err("expected )".into()),
            }
        } else {
            Ok(Expr::ColName(t))
        }
    }
}

fn lex(src: &str) -> Vec<String> {
    let bytes = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b == b'"' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            toks.push(std::str::from_utf8(&bytes[i..=j]).unwrap_or("").to_string());
            i = j + 1;
            continue;
        }
        if i + 1 < bytes.len() {
            let two = std::str::from_utf8(&bytes[i..i + 2]).unwrap_or("");
            if matches!(two, "==" | "!=" | "<=" | ">=" | "<>") {
                toks.push(two.to_string());
                i += 2;
                continue;
            }
        }
        if matches!(b, b'=' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/' | b'%' | b'(' | b')' | b',') {
            toks.push((b as char).to_string());
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_whitespace() || matches!(c, b'=' | b'<' | b'>' | b'+' | b'-' | b'*' | b'/' | b'%' | b'(' | b')' | b',') {
                break;
            }
            i += 1;
        }
        if i > start {
            toks.push(std::str::from_utf8(&bytes[start..i]).unwrap_or("").to_string());
        }
    }
    toks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MapNames(HashMap<String, usize>);
    impl NameTable for MapNames {
        fn resolve(&self, name: &str) -> Option<usize> {
            self.0.get(name).copied()
        }
    }

    fn names(pairs: &[(&str, usize)]) -> MapNames {
        let mut m = HashMap::new();
        for &(n, i) in pairs {
            m.insert(n.to_string(), i);
        }
        MapNames(m)
    }

    #[test]
    fn parse_and_eval_arith() {
        let e = parse("1 + 2 * 3").unwrap();
        assert_eq!(eval(&e, &[]), Ok(Value::Int(7)));
    }

    #[test]
    fn parse_and_eval_cmp() {
        let e = parse("x > 5").unwrap();
        let n = names(&[("x", 1)]);
        assert_eq!(eval_with(&e, &[Value::Int(0), Value::Int(10)], &n), Ok(Value::Bool(true)));
    }

    #[test]
    fn parse_and_eval_logic() {
        let e = parse("x > 0 AND y > 0").unwrap();
        let n = names(&[("x", 1), ("y", 2)]);
        let row = [Value::Int(0), Value::Int(3), Value::Int(4)];
        assert_eq!(eval_with(&e, &row, &n), Ok(Value::Bool(true)));
    }

    #[test]
    fn parse_and_eval_coalesce() {
        let e = parse("COALESCE(a, b, 0)").unwrap();
        let n = names(&[("a", 1), ("b", 2)]);
        let row = [Value::Int(0), Value::Null, Value::Int(9)];
        assert_eq!(eval_with(&e, &row, &n), Ok(Value::Int(9)));
    }

    #[test]
    fn eval_div_by_zero() {
        let e = parse("1 / 0").unwrap();
        assert_eq!(eval(&e, &[]), Err(EvalError::DivByZero));
    }

    #[test]
    fn eval_abs_func() {
        let e = parse("abs(-7)").unwrap();
        assert_eq!(eval(&e, &[]), Ok(Value::Int(7)));
    }

    #[test]
    fn eval_not() {
        let e = parse("NOT x").unwrap();
        let n = names(&[("x", 1)]);
        assert_eq!(eval_with(&e, &[Value::Int(0), Value::Bool(false)], &n), Ok(Value::Bool(true)));
    }
}
