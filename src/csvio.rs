//! CSV import and export.
//!
//! CSV is the lowest-common-denominator bulk load/dump format. This module
//! implements an RFC-4180-style reader (quoted fields, embedded delimiters and
//! newlines, doubled quotes) and a matching writer, plus a type-inference pass
//! that turns a header + rows of strings into typed columns. Text columns are
//! interned into a string dictionary so the resulting [`crate::array::Array`]s
//! carry dictionary ids the rest of the engine understands.

use crate::array::{Array, ArrayBuilder, RecordBatch};
use crate::schema::ColKind;
use crate::value::Value;
use std::collections::HashMap;

/// CSV reader/writer options.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Field delimiter (default `,`).
    pub delimiter: u8,
    /// Quote character (default `"`).
    pub quote: u8,
    /// Whether the first row is a header.
    pub has_header: bool,
    /// String that denotes a null field (default empty string).
    pub null_token: String,
}

impl Default for CsvOptions {
    fn default() -> CsvOptions {
        CsvOptions {
            delimiter: b',',
            quote: b'"',
            has_header: true,
            null_token: String::new(),
        }
    }
}

/// A CSV parse error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsvError {
    pub message: String,
    pub row: usize,
}

impl std::fmt::Display for CsvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (row {})", self.message, self.row)
    }
}

impl std::error::Error for CsvError {}

/// Parse CSV text into a header (if present) and rows of string fields.
pub fn parse_records(
    input: &str,
    opts: &CsvOptions,
) -> Result<(Option<Vec<String>>, Vec<Vec<String>>), CsvError> {
    let bytes = input.as_bytes();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut field = String::new();
    let mut record: Vec<String> = Vec::new();
    let mut in_quotes = false;
    let mut i = 0;
    let mut row_no = 0;
    let mut started = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_quotes {
            if b == opts.quote {
                if i + 1 < bytes.len() && bytes[i + 1] == opts.quote {
                    field.push(opts.quote as char);
                    i += 2;
                    continue;
                }
                in_quotes = false;
                i += 1;
            } else {
                push_utf8(&mut field, bytes, &mut i);
            }
        } else if b == opts.quote && field.is_empty() {
            in_quotes = true;
            started = true;
            i += 1;
        } else if b == opts.delimiter {
            record.push(std::mem::take(&mut field));
            started = true;
            i += 1;
        } else if b == b'\n' || b == b'\r' {
            // End of record (consume \r\n as one).
            if b == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 1;
            }
            i += 1;
            if started || !field.is_empty() || !record.is_empty() {
                record.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut record));
                row_no += 1;
            }
            started = false;
        } else {
            started = true;
            push_utf8(&mut field, bytes, &mut i);
        }
    }
    if in_quotes {
        return Err(CsvError {
            message: "unterminated quoted field".to_string(),
            row: row_no,
        });
    }
    if started || !field.is_empty() || !record.is_empty() {
        record.push(field);
        rows.push(record);
    }

    if opts.has_header && !rows.is_empty() {
        let header = rows.remove(0);
        Ok((Some(header), rows))
    } else {
        Ok((None, rows))
    }
}

fn push_utf8(field: &mut String, bytes: &[u8], i: &mut usize) {
    let b = bytes[*i];
    if b < 0x80 {
        field.push(b as char);
        *i += 1;
    } else {
        let extra = if b >= 0xF0 {
            3
        } else if b >= 0xE0 {
            2
        } else {
            1
        };
        let end = (*i + 1 + extra).min(bytes.len());
        match std::str::from_utf8(&bytes[*i..end]) {
            Ok(s) => field.push_str(s),
            Err(_) => field.push('\u{FFFD}'),
        }
        *i = end;
    }
}

/// Infer the [`ColKind`] of a column from its string samples.
pub fn infer_kind(samples: &[&str], null_token: &str) -> ColKind {
    let mut all_int = true;
    let mut all_real = true;
    let mut all_bool = true;
    let mut any = false;
    for s in samples {
        if *s == null_token {
            continue;
        }
        any = true;
        if s.parse::<i64>().is_err() {
            all_int = false;
        }
        if s.parse::<f64>().is_err() {
            all_real = false;
        }
        let lower = s.to_ascii_lowercase();
        if lower != "true" && lower != "false" && lower != "0" && lower != "1" {
            all_bool = false;
        }
    }
    if !any {
        return ColKind::Text;
    }
    if all_int {
        ColKind::Int
    } else if all_real {
        ColKind::Real
    } else if all_bool {
        ColKind::Bool
    } else {
        ColKind::Text
    }
}

/// A string dictionary that assigns stable ids in first-seen order.
#[derive(Debug, Default, Clone)]
pub struct StringDict {
    map: HashMap<String, u32>,
    values: Vec<String>,
}

impl StringDict {
    /// A fresh empty dictionary.
    pub fn new() -> StringDict {
        StringDict::default()
    }

    /// Intern a string, returning its id.
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.values.len() as u32;
        self.values.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }

    /// Resolve an id back to its string.
    pub fn resolve(&self, id: u32) -> Option<&str> {
        self.values.get(id as usize).map(|s| s.as_str())
    }

    /// Number of distinct strings.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The result of importing CSV: a typed batch plus the text dictionary used to
/// intern any text columns.
pub struct CsvImport {
    pub batch: RecordBatch,
    pub dict: StringDict,
}

/// Import CSV text into a typed [`RecordBatch`], inferring column types.
pub fn import(input: &str, opts: &CsvOptions) -> Result<CsvImport, CsvError> {
    let (header, rows) = parse_records(input, opts)?;
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let names: Vec<String> = match header {
        Some(h) => {
            let mut names = h;
            while names.len() < ncols {
                names.push(format!("col{}", names.len()));
            }
            names
        }
        None => (0..ncols).map(|i| format!("col{i}")).collect(),
    };

    let mut dict = StringDict::new();
    let mut columns: Vec<Array> = Vec::with_capacity(ncols);
    for c in 0..ncols {
        let samples: Vec<&str> = rows
            .iter()
            .map(|r| r.get(c).map(|s| s.as_str()).unwrap_or(""))
            .collect();
        let kind = infer_kind(&samples, &opts.null_token);
        let mut b = ArrayBuilder::new(kind);
        for r in &rows {
            let cell = r.get(c).map(|s| s.as_str()).unwrap_or("");
            if cell == opts.null_token {
                b.push_null();
                continue;
            }
            let v = match kind {
                ColKind::Int => cell.parse::<i64>().map(Value::Int).unwrap_or(Value::Null),
                ColKind::Real => cell.parse::<f64>().map(Value::Real).unwrap_or(Value::Null),
                ColKind::Bool => {
                    let l = cell.to_ascii_lowercase();
                    Value::Bool(l == "true" || l == "1")
                }
                ColKind::Text => Value::Text(dict.intern(cell)),
            };
            b.push(v);
        }
        columns.push(b.finish());
    }

    let pairs: Vec<(String, Array)> = names.into_iter().zip(columns).collect();
    Ok(CsvImport {
        batch: RecordBatch::new(pairs),
        dict,
    })
}

/// Export a [`RecordBatch`] to CSV, resolving text ids through `dict`.
pub fn export(batch: &RecordBatch, dict: &StringDict, opts: &CsvOptions) -> String {
    let mut out = String::new();
    if opts.has_header {
        write_row(
            &mut out,
            batch.names().iter().map(|s| s.as_str()),
            opts,
        );
    }
    for r in 0..batch.rows() {
        let cells: Vec<String> = batch
            .columns()
            .iter()
            .map(|col| cell_to_string(col, r, dict, opts))
            .collect();
        write_row(&mut out, cells.iter().map(|s| s.as_str()), opts);
    }
    out
}

fn cell_to_string(col: &Array, row: usize, dict: &StringDict, opts: &CsvOptions) -> String {
    match col.value(row) {
        Value::Null => opts.null_token.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Real(r) => r.to_string(),
        Value::Text(id) => dict.resolve(id).unwrap_or("").to_string(),
    }
}

fn write_row<'a, I: Iterator<Item = &'a str>>(out: &mut String, cells: I, opts: &CsvOptions) {
    let mut first = true;
    for cell in cells {
        if !first {
            out.push(opts.delimiter as char);
        }
        first = false;
        let needs_quote = cell.as_bytes().iter().any(|&b| {
            b == opts.delimiter || b == opts.quote || b == b'\n' || b == b'\r'
        });
        if needs_quote {
            out.push(opts.quote as char);
            for c in cell.chars() {
                if c as u32 == opts.quote as u32 {
                    out.push(opts.quote as char);
                }
                out.push(c);
            }
            out.push(opts.quote as char);
        } else {
            out.push_str(cell);
        }
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        let (h, rows) = parse_records("a,b,c\n1,2,3\n4,5,6\n", &CsvOptions::default()).unwrap();
        assert_eq!(h, Some(vec!["a".into(), "b".into(), "c".into()]));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1], vec!["4", "5", "6"]);
    }

    #[test]
    fn parse_quoted_fields() {
        let src = "name,note\n\"Smith, John\",\"line1\nline2\"\n\"quote\"\"inside\",x\n";
        let (_, rows) = parse_records(src, &CsvOptions::default()).unwrap();
        assert_eq!(rows[0][0], "Smith, John");
        assert_eq!(rows[0][1], "line1\nline2");
        assert_eq!(rows[1][0], "quote\"inside");
    }

    #[test]
    fn infers_types() {
        let imp = import("n,f,t,s\n1,1.5,true,hi\n2,2.5,false,yo\n", &CsvOptions::default())
            .unwrap();
        assert_eq!(imp.batch.column("n").unwrap().kind(), ColKind::Int);
        assert_eq!(imp.batch.column("f").unwrap().kind(), ColKind::Real);
        assert_eq!(imp.batch.column("t").unwrap().kind(), ColKind::Bool);
        assert_eq!(imp.batch.column("s").unwrap().kind(), ColKind::Text);
    }

    #[test]
    fn roundtrip_export() {
        let src = "id,label\n1,alpha\n2,beta\n";
        let imp = import(src, &CsvOptions::default()).unwrap();
        let out = export(&imp.batch, &imp.dict, &CsvOptions::default());
        let imp2 = import(&out, &CsvOptions::default()).unwrap();
        assert_eq!(imp.batch.rows(), imp2.batch.rows());
        assert_eq!(imp2.batch.column("id").unwrap().value(1), Value::Int(2));
    }

    #[test]
    fn nulls_from_empty() {
        let imp = import("a,b\n1,\n,4\n", &CsvOptions::default()).unwrap();
        assert_eq!(imp.batch.column("b").unwrap().value(0), Value::Null);
        assert_eq!(imp.batch.column("a").unwrap().value(1), Value::Null);
    }

    #[test]
    fn export_quotes_when_needed() {
        let mut dict = StringDict::new();
        let id = dict.intern("a,b");
        let arr = Array::from_values(ColKind::Text, &[Value::Text(id)]);
        let batch = RecordBatch::new(vec![("c".into(), arr)]);
        let out = export(&batch, &dict, &CsvOptions { has_header: false, ..Default::default() });
        assert!(out.contains("\"a,b\""));
    }
}
