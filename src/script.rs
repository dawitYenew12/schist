//! The operation script language.
//!
//! The harness drives the engine with a tiny imperative language. A script is a
//! sequence of statements separated by `;` (or newlines). Each statement is
//! parsed and executed independently; a malformed statement is skipped so a
//! partially-valid script still exercises the engine.
//!
//! Grammar
//! -------
//!
//! ```text
//!   insert <v> <v> ...           append a row (positional, schema order)
//!   update set <col> <v> where <col> <op> <v>
//!   delete where <col> <op> <v>
//!   scan
//!   scan where <col> <op> <v>
//!   index_scan <col> <v>
//!   join
//!   compact
//!   checkpoint
//!   agg <col>
//! ```
//!
//! `<col>` is a column name or a 0-based index. `<v>` is an integer, a real, a
//! boolean (`true`/`false`), `null`, or a double-quoted string literal (interned
//! into the column's dictionary for text columns). `<op>` is one of
//! `= == != <> < <= > >=`.

use crate::compact;
use crate::error::{Error, Result, ScriptError};
use crate::mutation::{self, intern_text};
use crate::query;
use crate::value::{CmpOp, Value};
use crate::Database;

/// A parsed operand.
#[derive(Debug, Clone)]
enum Operand {
    Val(Value),
    Str(Vec<u8>),
}

/// A parsed statement.
#[derive(Debug, Clone)]
enum Stmt {
    Insert(Vec<Operand>),
    Update {
        set_col: usize,
        set_val: Operand,
        pred_col: usize,
        op: CmpOp,
        target: Operand,
    },
    Delete {
        pred_col: usize,
        op: CmpOp,
        target: Operand,
    },
    Scan,
    ScanWhere {
        col: usize,
        op: CmpOp,
        target: Operand,
    },
    IndexScan {
        col: usize,
        target: Operand,
    },
    Join,
    Compact,
    Checkpoint,
    Agg {
        col: usize,
    },
}

/// A summary of a script run.
#[derive(Debug, Clone, Default)]
pub struct RunReport {
    pub statements: usize,
    pub executed: usize,
    pub errors: usize,
    pub rows_scanned: u64,
}

/// Parse and execute a script against the database.
pub fn run_script(db: &mut Database, src: &str) -> Result<RunReport> {
    let mut report = RunReport::default();
    for raw in src.split(';') {
        let stmt_src = raw.trim();
        if stmt_src.is_empty() {
            continue;
        }
        report.statements += 1;
        let stmt = match parse_statement(db, stmt_src) {
            Ok(s) => s,
            Err(_) => {
                report.errors += 1;
                continue;
            }
        };
        match execute(db, stmt) {
            Ok(scanned) => {
                report.executed += 1;
                report.rows_scanned += scanned;
            }
            Err(_) => {
                report.errors += 1;
            }
        }
    }
    Ok(report)
}

fn execute(db: &mut Database, stmt: Stmt) -> Result<u64> {
    match stmt {
        Stmt::Insert(operands) => {
            let mut values = Vec::with_capacity(operands.len());
            for (ci, op) in operands.into_iter().enumerate() {
                values.push(resolve_operand(db, ci, op)?);
            }
            mutation::insert(db, &values)?;
            Ok(0)
        }
        Stmt::Update {
            set_col,
            set_val,
            pred_col,
            op,
            target,
        } => {
            let set_value = resolve_operand(db, set_col, set_val)?;
            let target = resolve_operand(db, pred_col, target)?;
            mutation::update_where(db, set_col, set_value, pred_col, op, target)?;
            Ok(0)
        }
        Stmt::Delete { pred_col, op, target } => {
            let target = resolve_operand(db, pred_col, target)?;
            mutation::delete_where(db, pred_col, op, target)?;
            Ok(0)
        }
        Stmt::Scan => {
            let rows = query::scan_all(db);
            Ok(rows.len() as u64)
        }
        Stmt::ScanWhere { col, op, target } => {
            let target = resolve_operand(db, col, target)?;
            let rows = query::scan_where(db, col, op, target)?;
            Ok(rows.len() as u64)
        }
        Stmt::IndexScan { col, target } => {
            let target = resolve_operand(db, col, target)?;
            let rows = query::index_scan(db, col, target)?;
            Ok(rows.len() as u64)
        }
        Stmt::Join => {
            let pairs = query::merge_join(db);
            Ok(pairs.len() as u64)
        }
        Stmt::Compact => {
            compact::compact(db)?;
            Ok(0)
        }
        Stmt::Checkpoint => {
            db.stats.checkpoints += 1;
            let _ = crate::format::encode_database(db);
            Ok(0)
        }
        Stmt::Agg { col } => {
            let agg = query::aggregate(db, col);
            Ok(agg.count)
        }
    }
}

/// Resolve an operand to a value in the context of a column. String literals
/// are interned into the column's dictionary for text columns.
fn resolve_operand(db: &mut Database, col: usize, op: Operand) -> Result<Value> {
    let col_def = db
        .schema
        .columns
        .get(col)
        .ok_or_else(|| Error::NotFound(format!("column {col}")))?;
    match op {
        Operand::Val(v) => Ok(coerce(v, col_def.kind)),
        Operand::Str(bytes) => {
            if col_def.kind == crate::schema::ColKind::Text {
                let id = intern_text(db, col, &bytes)?;
                Ok(Value::Text(id))
            } else {
                Err(Error::Script(ScriptError::BadValue(format!(
                    "string literal not valid for {} column",
                    col_def.name
                ))))
            }
        }
    }
}

fn coerce(v: Value, kind: crate::schema::ColKind) -> Value {
    use crate::schema::ColKind;
    match (kind, v) {
        (ColKind::Int, Value::Int(i)) => Value::Int(i),
        (ColKind::Int, Value::Bool(b)) => Value::Int(if b { 1 } else { 0 }),
        (ColKind::Int, Value::Real(r)) => Value::Int(r as i64),
        (ColKind::Real, Value::Real(r)) => Value::Real(r),
        (ColKind::Real, Value::Int(i)) => Value::Real(i as f64),
        (ColKind::Bool, Value::Bool(b)) => Value::Bool(b),
        (ColKind::Bool, Value::Int(i)) => Value::Bool(i != 0),
        (ColKind::Text, Value::Text(id)) => Value::Text(id),
        (_, Value::Null) => Value::Null,
        (k, other) => {
            let _ = k;
            other
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

fn parse_statement(db: &Database, src: &str) -> Result<Stmt> {
    let toks = lex(src)?;
    if toks.is_empty() {
        return Err(Error::Script(ScriptError::Parse("empty".into())));
    }
    let p = &mut Parser { toks, pos: 0, db };
    match p.peek().unwrap().as_str() {
        "insert" => parse_insert(p),
        "update" => parse_update(p),
        "delete" => parse_delete(p),
        "scan" => parse_scan(p),
        "index_scan" => parse_index_scan(p),
        "join" => {
            p.next();
            Ok(Stmt::Join)
        }
        "compact" => {
            p.next();
            Ok(Stmt::Compact)
        }
        "checkpoint" => {
            p.next();
            Ok(Stmt::Checkpoint)
        }
        "agg" => parse_agg(p),
        kw => Err(Error::Script(ScriptError::UnknownStatement(kw.to_string()))),
    }
}

struct Parser<'a> {
    toks: Vec<String>,
    pos: usize,
    db: &'a Database,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&String> {
        self.toks.get(self.pos)
    }
    fn next(&mut self) -> Option<String> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }
    fn expect(&mut self, s: &str) -> Result<()> {
        match self.next() {
            Some(t) if t == s => Ok(()),
            Some(t) => Err(Error::Script(ScriptError::Parse(format!(
                "expected {s}, got {t}"
            )))),
            None => Err(Error::Script(ScriptError::Parse(format!(
                "expected {s}, got end"
            )))),
        }
    }
    fn parse_col(&mut self) -> Result<usize> {
        let t = self.next().ok_or(Error::Script(ScriptError::EndOfInput))?;
        if let Ok(i) = t.parse::<usize>() {
            return Ok(i);
        }
        self.db
            .schema
            .find(&t)
            .ok_or_else(|| Error::NotFound(format!("column {t}")))
    }
    fn parse_op(&mut self) -> Result<CmpOp> {
        let t = self.next().ok_or(Error::Script(ScriptError::EndOfInput))?;
        CmpOp::parse(&t).ok_or_else(|| Error::Script(ScriptError::Parse(format!("bad op {t}"))))
    }
    fn parse_operand(&mut self) -> Result<Operand> {
        let t = self.next().ok_or(Error::Script(ScriptError::EndOfInput))?;
        if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
            let inner = &t[1..t.len() - 1];
            return Ok(Operand::Str(inner.as_bytes().to_vec()));
        }
        match t.as_str() {
            "null" => Ok(Operand::Val(Value::Null)),
            "true" => Ok(Operand::Val(Value::Bool(true))),
            "false" => Ok(Operand::Val(Value::Bool(false))),
            _ => {
                if let Ok(i) = t.parse::<i64>() {
                    return Ok(Operand::Val(Value::Int(i)));
                }
                if let Ok(r) = t.parse::<f64>() {
                    return Ok(Operand::Val(Value::Real(r)));
                }
                Err(Error::Script(ScriptError::BadValue(t)))
            }
        }
    }
}

fn parse_insert(p: &mut Parser<'_>) -> Result<Stmt> {
    p.expect("insert")?;
    let mut ops = Vec::new();
    while p.peek().is_some() {
        ops.push(p.parse_operand()?);
    }
    Ok(Stmt::Insert(ops))
}

fn parse_update(p: &mut Parser<'_>) -> Result<Stmt> {
    p.expect("update")?;
    p.expect("set")?;
    let set_col = p.parse_col()?;
    let set_val = p.parse_operand()?;
    p.expect("where")?;
    let pred_col = p.parse_col()?;
    let op = p.parse_op()?;
    let target = p.parse_operand()?;
    Ok(Stmt::Update {
        set_col,
        set_val,
        pred_col,
        op,
        target,
    })
}

fn parse_delete(p: &mut Parser<'_>) -> Result<Stmt> {
    p.expect("delete")?;
    p.expect("where")?;
    let pred_col = p.parse_col()?;
    let op = p.parse_op()?;
    let target = p.parse_operand()?;
    Ok(Stmt::Delete { pred_col, op, target })
}

fn parse_scan(p: &mut Parser<'_>) -> Result<Stmt> {
    p.expect("scan")?;
    if p.peek().map_or(false, |t| t == "where") {
        p.next();
        let col = p.parse_col()?;
        let op = p.parse_op()?;
        let target = p.parse_operand()?;
        Ok(Stmt::ScanWhere { col, op, target })
    } else {
        Ok(Stmt::Scan)
    }
}

fn parse_index_scan(p: &mut Parser<'_>) -> Result<Stmt> {
    p.expect("index_scan")?;
    let col = p.parse_col()?;
    let target = p.parse_operand()?;
    Ok(Stmt::IndexScan { col, target })
}

fn parse_agg(p: &mut Parser<'_>) -> Result<Stmt> {
    p.expect("agg")?;
    let col = p.parse_col()?;
    Ok(Stmt::Agg { col })
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

fn lex(src: &str) -> Result<Vec<String>> {
    let mut toks = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b == b';' {
            i += 1;
            continue;
        }
        // String literal.
        if b == b'"' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'"' {
                j += 1;
            }
            let s = std::str::from_utf8(&bytes[i..=j.min(bytes.len() - 1)])
                .unwrap_or("")
                .to_string();
            toks.push(s);
            i = j + 1;
            continue;
        }
        // Two-char operators.
        if i + 1 < bytes.len() {
            let two = std::str::from_utf8(&bytes[i..i + 2]).unwrap_or("");
            if matches!(two, "==" | "!=" | "<=" | ">=" | "<>") {
                toks.push(two.to_string());
                i += 2;
                continue;
            }
        }
        // Single-char operators.
        if matches!(b, b'=' | b'<' | b'>') {
            toks.push((b as char).to_string());
            i += 1;
            continue;
        }
        // Identifier / number: read until whitespace or operator or ; or ".
        let start = i;
        while i < bytes.len() {
            let c = bytes[i];
            if c.is_ascii_whitespace() || c == b';' || c == b'"' || matches!(c, b'=' | b'<' | b'>') {
                break;
            }
            i += 1;
        }
        if i > start {
            let s = std::str::from_utf8(&bytes[start..i]).map_err(|_| {
                Error::Script(ScriptError::Lex("non-utf8 token".into()))
            })?;
            toks.push(s.to_string());
        }
    }
    Ok(toks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ColKind, Column, Encoding, Schema};

    fn int_db() -> Database {
        let mut cols = vec![Column::row_id()];
        cols.push(Column::new("x", ColKind::Int, Encoding::Plain));
        Database::new(Schema::new(cols))
    }

    #[test]
    fn insert_and_scan_script() {
        let mut db = int_db();
        let report = run_script(&mut db, "insert 1 10; insert 2 20; scan").unwrap();
        assert_eq!(report.executed, 3);
        assert_eq!(report.rows_scanned, 2);
    }

    #[test]
    fn delete_and_compact_script() {
        let mut db = int_db();
        let report = run_script(
            &mut db,
            "insert 1 10; insert 2 20; insert 3 30; delete where x = 20; compact; scan",
        )
        .unwrap();
        assert_eq!(report.executed, 6);
    }

    #[test]
    fn malformed_statement_is_skipped() {
        let mut db = int_db();
        let report = run_script(&mut db, "insert 1 10; bogus statement; scan").unwrap();
        assert_eq!(report.executed, 2);
        assert_eq!(report.errors, 1);
    }
}
