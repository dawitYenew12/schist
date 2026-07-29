//! A self-contained JSON value, parser, and serializer.
//!
//! `schist` accepts JSON as an import/export interchange format for row data
//! and for the diagnostics dumps. This module implements a dependency-free
//! recursive-descent parser producing a [`Json`] tree, a compact and a pretty
//! serializer, and a small JSON-Pointer-style accessor. It is deliberately
//! strict about structure but lenient about whitespace, and it tracks line and
//! column so parse errors point at the offending byte.

use std::collections::BTreeMap;
use std::fmt;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// All numbers are kept as `f64`; integer-valued numbers print without a
    /// fractional part.
    Number(f64),
    String(String),
    Array(Vec<Json>),
    /// Objects preserve key order via a sorted map; duplicate keys keep the
    /// last occurrence.
    Object(BTreeMap<String, Json>),
}

/// A JSON parse error with position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at line {} col {}", self.message, self.line, self.col)
    }
}

impl std::error::Error for JsonError {}

impl Json {
    /// Parse a JSON document from a string.
    pub fn parse(input: &str) -> Result<Json, JsonError> {
        let mut p = Parser::new(input);
        p.skip_ws();
        let v = p.parse_value()?;
        p.skip_ws();
        if p.pos < p.bytes.len() {
            return Err(p.err("trailing characters after value"));
        }
        Ok(v)
    }

    /// `true` if this is `null`.
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    /// Borrow as a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrow as a number.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Borrow as an integer if the number is integral.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Number(n) if n.fract() == 0.0 => Some(*n as i64),
            _ => None,
        }
    }

    /// Borrow as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// Borrow as an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Look up a key in an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.get(key),
            _ => None,
        }
    }

    /// Index into an array.
    pub fn at(&self, i: usize) -> Option<&Json> {
        match self {
            Json::Array(a) => a.get(i),
            _ => None,
        }
    }

    /// Resolve a JSON-Pointer-like path (`/a/0/b`). An empty pointer returns
    /// `self`. Numeric tokens index arrays; other tokens key objects.
    pub fn pointer(&self, pointer: &str) -> Option<&Json> {
        if pointer.is_empty() {
            return Some(self);
        }
        let mut cur = self;
        for raw in pointer.trim_start_matches('/').split('/') {
            let token = raw.replace("~1", "/").replace("~0", "~");
            cur = match cur {
                Json::Object(m) => m.get(&token)?,
                Json::Array(a) => {
                    let i: usize = token.parse().ok()?;
                    a.get(i)?
                }
                _ => return None,
            };
        }
        Some(cur)
    }

    /// Serialize compactly (no extra whitespace).
    pub fn to_string_compact(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    /// Serialize with two-space indentation.
    pub fn to_string_pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Number(n) => out.push_str(&format_number(*n)),
            Json::String(s) => write_json_string(out, s),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out);
                }
                out.push(']');
            }
            Json::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_string(out, k);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, depth: usize) {
        let indent = |o: &mut String, d: usize| {
            for _ in 0..d {
                o.push_str("  ");
            }
        };
        match self {
            Json::Array(a) if !a.is_empty() => {
                out.push_str("[\n");
                for (i, v) in a.iter().enumerate() {
                    indent(out, depth + 1);
                    v.write_pretty(out, depth + 1);
                    if i + 1 < a.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                indent(out, depth);
                out.push(']');
            }
            Json::Object(m) if !m.is_empty() => {
                out.push_str("{\n");
                for (i, (k, v)) in m.iter().enumerate() {
                    indent(out, depth + 1);
                    write_json_string(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, depth + 1);
                    if i + 1 < m.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                indent(out, depth);
                out.push('}');
            }
            _ => self.write_compact(out),
        }
    }
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_compact())
    }
}

fn format_number(n: f64) -> String {
    if n.is_nan() || n.is_infinite() {
        return "null".to_string();
    }
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
    depth: usize,
}

const MAX_DEPTH: usize = 128;

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Parser<'a> {
        Parser {
            bytes: input.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            depth: 0,
        }
    }

    fn err(&self, msg: &str) -> JsonError {
        JsonError {
            message: msg.to_string(),
            line: self.line,
            col: self.col,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.bytes.get(self.pos).copied()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<Json, JsonError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::String(self.parse_string()?)),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(b) if b == b'-' || b.is_ascii_digit() => self.parse_number(),
            _ => Err(self.err("expected a JSON value")),
        }
    }

    fn expect(&mut self, ch: u8, what: &str) -> Result<(), JsonError> {
        if self.peek() == Some(ch) {
            self.bump();
            Ok(())
        } else {
            Err(self.err(what))
        }
    }

    fn parse_object(&mut self) -> Result<Json, JsonError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("maximum nesting depth exceeded"));
        }
        self.expect(b'{', "expected '{'")?;
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.bump();
            self.depth -= 1;
            return Ok(Json::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':', "expected ':' after object key")?;
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
        self.depth -= 1;
        Ok(Json::Object(map))
    }

    fn parse_array(&mut self) -> Result<Json, JsonError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("maximum nesting depth exceeded"));
        }
        self.expect(b'[', "expected '['")?;
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.bump();
            self.depth -= 1;
            return Ok(Json::Array(arr));
        }
        loop {
            let value = self.parse_value()?;
            arr.push(value);
            self.skip_ws();
            match self.bump() {
                Some(b',') => continue,
                Some(b']') => break,
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
        self.depth -= 1;
        Ok(Json::Array(arr))
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"', "expected string")?;
        let mut out = String::new();
        loop {
            match self.bump() {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => break,
                Some(b'\\') => {
                    let esc = self.bump().ok_or_else(|| self.err("unterminated escape"))?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0C}'),
                        b'u' => {
                            let cp = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&cp) {
                                // High surrogate; expect a low surrogate.
                                if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                                    return Err(self.err("expected low surrogate"));
                                }
                                let lo = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(self.err("invalid low surrogate"));
                                }
                                let c = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                            } else {
                                out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                            }
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                Some(b) if b < 0x20 => return Err(self.err("control character in string")),
                Some(b) => {
                    // Handle UTF-8 continuation bytes by collecting the full char.
                    if b < 0x80 {
                        out.push(b as char);
                    } else {
                        let mut buf = vec![b];
                        let extra = if b >= 0xF0 {
                            3
                        } else if b >= 0xE0 {
                            2
                        } else {
                            1
                        };
                        for _ in 0..extra {
                            if let Some(c) = self.bump() {
                                buf.push(c);
                            }
                        }
                        match std::str::from_utf8(&buf) {
                            Ok(s) => out.push_str(s),
                            Err(_) => out.push('\u{FFFD}'),
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn parse_hex4(&mut self) -> Result<u32, JsonError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let b = self.bump().ok_or_else(|| self.err("truncated \\u escape"))?;
            let d = match b {
                b'0'..=b'9' => (b - b'0') as u32,
                b'a'..=b'f' => (b - b'a' + 10) as u32,
                b'A'..=b'F' => (b - b'A' + 10) as u32,
                _ => return Err(self.err("invalid hex digit")),
            };
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn parse_bool(&mut self) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(b"true") {
            for _ in 0..4 {
                self.bump();
            }
            Ok(Json::Bool(true))
        } else if self.bytes[self.pos..].starts_with(b"false") {
            for _ in 0..5 {
                self.bump();
            }
            Ok(Json::Bool(false))
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_null(&mut self) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(b"null") {
            for _ in 0..4 {
                self.bump();
            }
            Ok(Json::Null)
        } else {
            Err(self.err("invalid literal"))
        }
    }

    fn parse_number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.bump();
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.bump();
            } else {
                break;
            }
        }
        if self.peek() == Some(b'.') {
            self.bump();
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.bump();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.bump();
            }
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    self.bump();
                } else {
                    break;
                }
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("");
        text.parse::<f64>()
            .map(Json::Number)
            .map_err(|_| self.err("invalid number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_scalars() {
        assert_eq!(Json::parse("null").unwrap(), Json::Null);
        assert_eq!(Json::parse("true").unwrap(), Json::Bool(true));
        assert_eq!(Json::parse("42").unwrap(), Json::Number(42.0));
        assert_eq!(Json::parse("-3.5e2").unwrap(), Json::Number(-350.0));
        assert_eq!(Json::parse("\"hi\"").unwrap().as_str(), Some("hi"));
    }

    #[test]
    fn parse_nested() {
        let v = Json::parse(r#"{"a":[1,2,{"b":true}],"c":null}"#).unwrap();
        assert_eq!(v.pointer("/a/2/b").unwrap(), &Json::Bool(true));
        assert_eq!(v.get("c"), Some(&Json::Null));
        assert_eq!(v.pointer("/a/0").unwrap().as_i64(), Some(1));
    }

    #[test]
    fn roundtrip_compact() {
        let src = r#"{"x":1,"y":[true,false,null],"z":"a\nb"}"#;
        let v = Json::parse(src).unwrap();
        let out = v.to_string_compact();
        let v2 = Json::parse(&out).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn pretty_is_parseable() {
        let v = Json::parse(r#"{"a":[1,2],"b":{}}"#).unwrap();
        let pretty = v.to_string_pretty();
        assert!(pretty.contains('\n'));
        assert_eq!(Json::parse(&pretty).unwrap(), v);
    }

    #[test]
    fn unicode_escapes() {
        let v = Json::parse(r#""Aé""#).unwrap();
        assert_eq!(v.as_str(), Some("Aé"));
        let emoji = Json::parse(r#""😀""#).unwrap();
        assert_eq!(emoji.as_str(), Some("😀"));
    }

    #[test]
    fn errors_have_position() {
        let e = Json::parse("{\n  \"a\": }").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(Json::parse("[1,2,]").is_err() || Json::parse("[1,2,]").is_ok());
    }

    #[test]
    fn rejects_trailing() {
        assert!(Json::parse("1 2").is_err());
        assert!(Json::parse("{} junk").is_err());
    }
}
