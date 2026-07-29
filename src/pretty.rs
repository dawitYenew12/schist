//! Human-readable result-set rendering.
//!
//! The CLI and the diagnostics dumps present query results as aligned ASCII
//! tables. This module lays out a header plus rows of [`Value`]s into a
//! box-drawn grid, computing each column's width from its widest cell, aligning
//! numbers to the right and text to the left, and truncating overly wide cells
//! with an ellipsis. It also offers a compact "expanded" one-field-per-line
//! rendering for very wide rows.

use crate::value::Value;

/// Options controlling table rendering.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Maximum width of any single column before truncation.
    pub max_col_width: usize,
    /// The string used to represent null cells.
    pub null_text: String,
    /// Whether to draw the outer border.
    pub border: bool,
}

impl Default for RenderOptions {
    fn default() -> RenderOptions {
        RenderOptions {
            max_col_width: 40,
            null_text: "NULL".to_string(),
            border: true,
        }
    }
}

/// Render a value to its display string.
pub fn value_to_string(v: &Value, null_text: &str) -> String {
    match v {
        Value::Null => null_text.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Real(r) => format_real(*r),
        Value::Text(id) => format!("#{id}"),
    }
}

fn format_real(r: f64) -> String {
    if r.fract() == 0.0 && r.abs() < 1e15 {
        format!("{r:.1}")
    } else {
        format!("{r}")
    }
}

fn is_numeric(v: &Value) -> bool {
    matches!(v, Value::Int(_) | Value::Real(_))
}

fn display_width(s: &str) -> usize {
    s.chars().count()
}

fn truncate(s: &str, max: usize) -> String {
    if display_width(s) <= max {
        s.to_string()
    } else if max <= 1 {
        "…".to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

/// Render a header and rows to an aligned ASCII table.
pub fn render_table(headers: &[String], rows: &[Vec<Value>], opts: &RenderOptions) -> String {
    let ncols = headers.len();
    if ncols == 0 {
        return String::new();
    }
    // Compute cell strings and column widths.
    let mut widths: Vec<usize> = headers.iter().map(|h| display_width(h)).collect();
    let mut right_align = vec![true; ncols];
    let mut cell_rows: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for row in rows {
        let mut cells = Vec::with_capacity(ncols);
        for c in 0..ncols {
            let v = row.get(c).copied().unwrap_or(Value::Null);
            if !is_numeric(&v) && !v.is_null() {
                right_align[c] = false;
            }
            let s = truncate(&value_to_string(&v, &opts.null_text), opts.max_col_width);
            widths[c] = widths[c].max(display_width(&s));
            cells.push(s);
        }
        cell_rows.push(cells);
    }
    for (c, w) in widths.iter_mut().enumerate() {
        *w = (*w).min(opts.max_col_width.max(display_width(&headers[c])));
    }

    let mut out = String::new();
    let sep = |out: &mut String, widths: &[usize]| {
        out.push('+');
        for &w in widths {
            out.push_str(&"-".repeat(w + 2));
            out.push('+');
        }
        out.push('\n');
    };

    if opts.border {
        sep(&mut out, &widths);
    }
    // Header row (always left-aligned).
    out.push('|');
    for (c, h) in headers.iter().enumerate() {
        out.push(' ');
        out.push_str(&pad(h, widths[c], false));
        out.push_str(" |");
    }
    out.push('\n');
    sep(&mut out, &widths);

    for cells in &cell_rows {
        out.push('|');
        for (c, s) in cells.iter().enumerate() {
            out.push(' ');
            out.push_str(&pad(s, widths[c], right_align[c]));
            out.push_str(" |");
        }
        out.push('\n');
    }
    if opts.border {
        sep(&mut out, &widths);
    }
    out
}

fn pad(s: &str, width: usize, right: bool) -> String {
    let w = display_width(s);
    if w >= width {
        return s.to_string();
    }
    let fill = " ".repeat(width - w);
    if right {
        format!("{fill}{s}")
    } else {
        format!("{s}{fill}")
    }
}

/// Render each row as a block of `name: value` lines (for wide result sets).
pub fn render_expanded(headers: &[String], rows: &[Vec<Value>], null_text: &str) -> String {
    let name_width = headers.iter().map(|h| display_width(h)).max().unwrap_or(0);
    let mut out = String::new();
    for (i, row) in rows.iter().enumerate() {
        out.push_str(&format!("-[ row {} ]-\n", i + 1));
        for (c, h) in headers.iter().enumerate() {
            let v = row.get(c).copied().unwrap_or(Value::Null);
            out.push_str(&format!(
                "{:<width$} | {}\n",
                h,
                value_to_string(&v, null_text),
                width = name_width
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_aligned_table() {
        let headers = vec!["id".to_string(), "name".to_string()];
        let rows = vec![
            vec![Value::Int(1), Value::Text(0)],
            vec![Value::Int(100), Value::Text(1)],
        ];
        let table = render_table(&headers, &rows, &RenderOptions::default());
        // Every rendered line should have the same length.
        let lens: Vec<usize> = table.lines().map(|l| l.chars().count()).collect();
        assert!(lens.windows(2).all(|w| w[0] == w[1]));
        assert!(table.contains("id"));
    }

    #[test]
    fn nulls_and_numbers() {
        let headers = vec!["v".to_string()];
        let rows = vec![vec![Value::Null], vec![Value::Int(-5)], vec![Value::Real(2.0)]];
        let table = render_table(&headers, &rows, &RenderOptions::default());
        assert!(table.contains("NULL"));
        assert!(table.contains("2.0"));
    }

    #[test]
    fn truncates_wide_cells() {
        let headers = vec!["x".to_string()];
        let long = "a".repeat(100);
        // Build a text value; render via value_to_string won't be long, so test
        // truncate directly.
        assert_eq!(truncate(&long, 5).chars().count(), 5);
        assert!(truncate(&long, 5).ends_with('…'));
        let _ = headers;
    }

    #[test]
    fn expanded_layout() {
        let headers = vec!["a".to_string(), "bb".to_string()];
        let rows = vec![vec![Value::Int(1), Value::Null]];
        let out = render_expanded(&headers, &rows, "NULL");
        assert!(out.contains("row 1"));
        assert!(out.contains("bb"));
        assert!(out.contains("NULL"));
    }

    #[test]
    fn empty_headers() {
        assert_eq!(render_table(&[], &[], &RenderOptions::default()), "");
    }
}
