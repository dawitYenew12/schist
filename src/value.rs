//! Runtime value type and ordering helpers.
//!
//! A [`Value`] is what the script and query layers pass around. On disk,
//! columns are encoded according to [`crate::schema::Encoding`]; the encoded
//! bytes are turned back into `Value`s only when a scan or predicate needs to
//! inspect them.

use std::cmp::Ordering;

/// The in-memory representation of a single cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    /// A dictionary-encoded text value: the integer is the dictionary entry id.
    /// The actual bytes live in a dictionary page; comparing two `Text` ids is
    /// comparing their dictionary positions, which is stable within a page
    /// generation.
    Text(u32),
}

impl Value {
    /// The type tag of this value, matching [`crate::schema::ColKind`].
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Real(_) => "real",
            Value::Text(_) => "text",
        }
    }

    /// `true` if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Coerce to `i64` if possible (used by integer predicates).
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(true) => Some(1),
            Value::Bool(false) => Some(0),
            Value::Real(r) => Some(*r as i64),
            _ => None,
        }
    }

    /// Coerce to `f64` if possible.
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Value::Real(r) => Some(*r),
            Value::Int(i) => Some(*i as f64),
            Value::Bool(true) => Some(1.0),
            Value::Bool(false) => Some(0.0),
            _ => None,
        }
    }

    /// Coerce to the dictionary id of a text value.
    pub fn as_text_id(&self) -> Option<u32> {
        match self {
            Value::Text(id) => Some(*id),
            _ => None,
        }
    }

    /// Total ordering used by the query engine. Null sorts first.
    pub fn total_cmp(&self, other: &Value) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Real(a), Value::Real(b)) => a.total_cmp(b),
            (Value::Int(a), Value::Real(b)) => (*a as f64).total_cmp(b),
            (Value::Real(a), Value::Int(b)) => a.total_cmp(&(*b as f64)),
            (Value::Text(a), Value::Text(b)) => a.cmp(b),
            // Cross-type fallback by type tag keeps the order total.
            (a, b) => a.kind().cmp(b.kind()),
        }
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Value) -> Option<Ordering> {
        Some(self.total_cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

/// A comparison operator as written in a script predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// Parse a comparison operator from its source token.
    pub fn parse(tok: &str) -> Option<CmpOp> {
        match tok {
            "=" | "==" => Some(CmpOp::Eq),
            "!=" | "<>" => Some(CmpOp::Ne),
            "<" => Some(CmpOp::Lt),
            "<=" => Some(CmpOp::Le),
            ">" => Some(CmpOp::Gt),
            ">=" => Some(CmpOp::Ge),
            _ => None,
        }
    }

    /// Apply the operator to two values, respecting null semantics: any
    /// comparison against null is `false` except `!=` which is `true` (SQL-like
    /// three-valued logic collapsed to a boolean here).
    pub fn apply(self, lhs: &Value, rhs: &Value) -> bool {
        if lhs.is_null() || rhs.is_null() {
            return matches!(self, CmpOp::Ne);
        }
        match self {
            CmpOp::Eq => lhs == rhs,
            CmpOp::Ne => lhs != rhs,
            CmpOp::Lt => lhs.total_cmp(rhs) == Ordering::Less,
            CmpOp::Le => lhs.total_cmp(rhs) != Ordering::Greater,
            CmpOp::Gt => lhs.total_cmp(rhs) == Ordering::Greater,
            CmpOp::Ge => lhs.total_cmp(rhs) != Ordering::Less,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sorts_first() {
        assert_eq!(Value::Null.total_cmp(&Value::Int(5)), Ordering::Less);
        assert_eq!(Value::Int(5).total_cmp(&Value::Null), Ordering::Greater);
    }

    #[test]
    fn int_real_compare() {
        assert_eq!(Value::Int(3).total_cmp(&Value::Real(3.0)), Ordering::Equal);
        assert_eq!(
            Value::Real(2.5).total_cmp(&Value::Int(3)),
            Ordering::Less
        );
    }

    #[test]
    fn cmp_op_null_semantics() {
        assert!(!CmpOp::Eq.apply(&Value::Null, &Value::Int(1)));
        assert!(CmpOp::Ne.apply(&Value::Null, &Value::Int(1)));
        assert!(CmpOp::Eq.apply(&Value::Int(1), &Value::Int(1)));
        assert!(CmpOp::Lt.apply(&Value::Int(1), &Value::Int(2)));
    }
}
