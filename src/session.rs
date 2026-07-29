//! Session state: prepared statements and bound parameters.
//!
//! A connected client runs inside a session that outlives individual requests.
//! It holds prepared statements (parsed once, executed many times with
//! different parameters), a monotonic handle allocator, and simple session
//! settings. Preparing a statement parses it and records how many parameter
//! placeholders it has; executing binds a parameter list and substitutes the
//! placeholders. This module manages that lifecycle over the [`crate::sql`]
//! parser; actual execution is delegated to the caller.

use crate::sql::{Parser, SqlError, Statement};
use crate::value::Value;
use std::collections::HashMap;

/// A prepared statement: its source, parsed form, and placeholder count.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub handle: u32,
    pub sql: String,
    pub statement: Statement,
    pub param_count: usize,
}

/// A session setting value.
#[derive(Debug, Clone, PartialEq)]
pub enum Setting {
    Bool(bool),
    Int(i64),
    Text(String),
}

/// Errors from session operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionError {
    Parse(SqlError),
    UnknownHandle(u32),
    ParamCountMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Parse(e) => write!(f, "parse error: {e}"),
            SessionError::UnknownHandle(h) => write!(f, "unknown statement handle {h}"),
            SessionError::ParamCountMismatch { expected, got } => {
                write!(f, "expected {expected} parameters, got {got}")
            }
        }
    }
}

impl std::error::Error for SessionError {}

/// A client session.
#[derive(Debug, Default)]
pub struct Session {
    next_handle: u32,
    prepared: HashMap<u32, Prepared>,
    settings: HashMap<String, Setting>,
    /// Statements executed this session (for diagnostics).
    executed: u64,
}

impl Session {
    /// A fresh session with default settings.
    pub fn new() -> Session {
        let mut s = Session::default();
        s.settings.insert("autocommit".into(), Setting::Bool(true));
        s.settings.insert("max_rows".into(), Setting::Int(10_000));
        s
    }

    /// Number of live prepared statements.
    pub fn prepared_count(&self) -> usize {
        self.prepared.len()
    }

    /// Number of statements executed.
    pub fn executed_count(&self) -> u64 {
        self.executed
    }

    /// Prepare a statement, returning its handle. The parameter count is the
    /// number of `?` placeholders in the source.
    pub fn prepare(&mut self, sql: &str) -> Result<u32, SessionError> {
        // Replace `?` placeholders with distinct sentinel literals so the parser
        // accepts them, counting as we go.
        let (rewritten, count) = rewrite_placeholders(sql);
        let statement = Parser::new(&rewritten)
            .and_then(|mut p| p.parse_statement())
            .map_err(SessionError::Parse)?;
        let handle = self.next_handle;
        self.next_handle += 1;
        self.prepared.insert(
            handle,
            Prepared {
                handle,
                sql: sql.to_string(),
                statement,
                param_count: count,
            },
        );
        Ok(handle)
    }

    /// Borrow a prepared statement.
    pub fn get_prepared(&self, handle: u32) -> Option<&Prepared> {
        self.prepared.get(&handle)
    }

    /// Validate a parameter binding against a prepared statement and record an
    /// execution. Returns the bound parameters back for the caller to run.
    pub fn bind(&mut self, handle: u32, params: Vec<Value>) -> Result<Vec<Value>, SessionError> {
        let prep = self
            .prepared
            .get(&handle)
            .ok_or(SessionError::UnknownHandle(handle))?;
        if params.len() != prep.param_count {
            return Err(SessionError::ParamCountMismatch {
                expected: prep.param_count,
                got: params.len(),
            });
        }
        self.executed += 1;
        Ok(params)
    }

    /// Deallocate a prepared statement.
    pub fn deallocate(&mut self, handle: u32) -> bool {
        self.prepared.remove(&handle).is_some()
    }

    /// Set a session setting.
    pub fn set(&mut self, name: &str, value: Setting) {
        self.settings.insert(name.to_string(), value);
    }

    /// Get a session setting.
    pub fn get(&self, name: &str) -> Option<&Setting> {
        self.settings.get(name)
    }

    /// Convenience: an integer setting, with a default.
    pub fn get_int(&self, name: &str, default: i64) -> i64 {
        match self.settings.get(name) {
            Some(Setting::Int(i)) => *i,
            _ => default,
        }
    }
}

/// Replace `?` placeholders with `NULL` so the statement parses, returning the
/// rewritten SQL and the placeholder count. Placeholders inside string literals
/// are left alone.
fn rewrite_placeholders(sql: &str) -> (String, usize) {
    let mut out = String::with_capacity(sql.len());
    let mut count = 0;
    let mut in_str = false;
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_str = !in_str;
                out.push(c);
            }
            '?' if !in_str => {
                out.push_str("NULL");
                count += 1;
            }
            _ => out.push(c),
        }
    }
    (out, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_counts_params() {
        let mut s = Session::new();
        let h = s.prepare("SELECT * FROM t WHERE a = ? AND b = ?").unwrap();
        assert_eq!(s.get_prepared(h).unwrap().param_count, 2);
        assert_eq!(s.prepared_count(), 1);
    }

    #[test]
    fn bind_checks_arity() {
        let mut s = Session::new();
        let h = s.prepare("SELECT * FROM t WHERE a = ?").unwrap();
        assert!(s.bind(h, vec![Value::Int(1)]).is_ok());
        assert_eq!(
            s.bind(h, vec![]),
            Err(SessionError::ParamCountMismatch { expected: 1, got: 0 })
        );
        assert_eq!(s.executed_count(), 1);
    }

    #[test]
    fn unknown_handle() {
        let mut s = Session::new();
        assert_eq!(s.bind(99, vec![]), Err(SessionError::UnknownHandle(99)));
    }

    #[test]
    fn deallocate_removes() {
        let mut s = Session::new();
        let h = s.prepare("SELECT 1").unwrap();
        assert!(s.deallocate(h));
        assert!(!s.deallocate(h));
        assert_eq!(s.prepared_count(), 0);
    }

    #[test]
    fn placeholder_in_string_ignored() {
        let (rewritten, count) = rewrite_placeholders("SELECT '?' , a WHERE b = ?");
        assert_eq!(count, 1);
        assert!(rewritten.contains("'?'"));
    }

    #[test]
    fn settings() {
        let mut s = Session::new();
        assert_eq!(s.get("autocommit"), Some(&Setting::Bool(true)));
        s.set("max_rows", Setting::Int(500));
        assert_eq!(s.get_int("max_rows", 0), 500);
        assert_eq!(s.get_int("missing", -1), -1);
    }

    #[test]
    fn parse_error_surfaces() {
        let mut s = Session::new();
        assert!(matches!(s.prepare("NOT SQL AT ALL @@"), Err(SessionError::Parse(_))));
    }
}
