//! String scalar functions.
//!
//! Text columns store dictionary ids, but once a value is resolved to its
//! backing string the engine offers the usual SQL string functions over it.
//! These operate on `&str` and return owned `String`s (or integers for length
//! and position). They are UTF-8 aware: `length` counts characters, `substr`
//! indexes by character (1-based, SQL style), and case folding uses Rust's
//! Unicode-aware `to_uppercase`/`to_lowercase`.

/// Character length (not byte length).
pub fn char_length(s: &str) -> usize {
    s.chars().count()
}

/// Byte length.
pub fn byte_length(s: &str) -> usize {
    s.len()
}

/// Upper-case.
pub fn upper(s: &str) -> String {
    s.to_uppercase()
}

/// Lower-case.
pub fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// Trim leading and trailing ASCII whitespace.
pub fn trim(s: &str) -> String {
    s.trim().to_string()
}

/// Trim leading whitespace.
pub fn ltrim(s: &str) -> String {
    s.trim_start().to_string()
}

/// Trim trailing whitespace.
pub fn rtrim(s: &str) -> String {
    s.trim_end().to_string()
}

/// SQL `SUBSTR(s, start, len)` — 1-based, character-indexed. A non-positive
/// `start` counts from before the string; `len` clamps to the string end.
pub fn substr(s: &str, start: i64, len: Option<i64>) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len() as i64;
    // Convert 1-based start to 0-based, honoring non-positive starts.
    let from = start - 1;
    let (begin, mut count) = match len {
        Some(l) => (from, l),
        None => (from, n - from),
    };
    // Clamp the window into [0, n).
    let mut b = begin;
    if b < 0 {
        count += b; // shrink the length by the underflow
        b = 0;
    }
    if count <= 0 || b >= n {
        return String::new();
    }
    let end = (b + count).min(n);
    chars[b as usize..end as usize].iter().collect()
}

/// 1-based character position of `needle` in `haystack`, or 0 if absent.
pub fn position(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 1;
    }
    match haystack.find(needle) {
        Some(byte_idx) => haystack[..byte_idx].chars().count() + 1,
        None => 0,
    }
}

/// Concatenate two strings.
pub fn concat(a: &str, b: &str) -> String {
    let mut out = String::with_capacity(a.len() + b.len());
    out.push_str(a);
    out.push_str(b);
    out
}

/// Concatenate many strings with a separator.
pub fn concat_ws(sep: &str, parts: &[&str]) -> String {
    parts.join(sep)
}

/// Replace all non-overlapping occurrences of `from` with `to`.
pub fn replace(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }
    s.replace(from, to)
}

/// Repeat a string `n` times.
pub fn repeat(s: &str, n: usize) -> String {
    s.repeat(n)
}

/// Reverse by characters.
pub fn reverse(s: &str) -> String {
    s.chars().rev().collect()
}

/// Left-pad to `width` characters using `pad` (truncates if longer).
pub fn lpad(s: &str, width: usize, pad: &str) -> String {
    pad_impl(s, width, pad, true)
}

/// Right-pad to `width` characters using `pad`.
pub fn rpad(s: &str, width: usize, pad: &str) -> String {
    pad_impl(s, width, pad, false)
}

fn pad_impl(s: &str, width: usize, pad: &str, left: bool) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= width {
        return chars[..width].iter().collect();
    }
    if pad.is_empty() {
        return s.to_string();
    }
    let pad_chars: Vec<char> = pad.chars().collect();
    let needed = width - chars.len();
    let mut padding = String::with_capacity(needed);
    for i in 0..needed {
        padding.push(pad_chars[i % pad_chars.len()]);
    }
    if left {
        format!("{padding}{s}")
    } else {
        format!("{s}{padding}")
    }
}

/// `true` if `s` starts with `prefix`.
pub fn starts_with(s: &str, prefix: &str) -> bool {
    s.starts_with(prefix)
}

/// `true` if `s` ends with `suffix`.
pub fn ends_with(s: &str, suffix: &str) -> bool {
    s.ends_with(suffix)
}

/// Split on a separator, returning the `n`-th (0-based) field, or empty.
pub fn split_part(s: &str, sep: &str, n: usize) -> String {
    if sep.is_empty() {
        return if n == 0 { s.to_string() } else { String::new() };
    }
    s.split(sep).nth(n).unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths() {
        assert_eq!(char_length("héllo"), 5);
        assert_eq!(byte_length("héllo"), 6);
    }

    #[test]
    fn case_and_trim() {
        assert_eq!(upper("aB"), "AB");
        assert_eq!(lower("aB"), "ab");
        assert_eq!(trim("  hi  "), "hi");
        assert_eq!(ltrim("  hi  "), "hi  ");
        assert_eq!(rtrim("  hi  "), "  hi");
    }

    #[test]
    fn substr_sql_semantics() {
        assert_eq!(substr("hello", 2, Some(3)), "ell");
        assert_eq!(substr("hello", 1, None), "hello");
        assert_eq!(substr("hello", 4, None), "lo");
        assert_eq!(substr("hello", 0, Some(2)), "h"); // start before string
        assert_eq!(substr("hello", 10, Some(2)), "");
    }

    #[test]
    fn position_and_concat() {
        assert_eq!(position("hello", "ll"), 3);
        assert_eq!(position("hello", "z"), 0);
        assert_eq!(concat("ab", "cd"), "abcd");
        assert_eq!(concat_ws("-", &["a", "b", "c"]), "a-b-c");
    }

    #[test]
    fn replace_repeat_reverse() {
        assert_eq!(replace("a.b.c", ".", "/"), "a/b/c");
        assert_eq!(repeat("ab", 3), "ababab");
        assert_eq!(reverse("abc"), "cba");
    }

    #[test]
    fn padding() {
        assert_eq!(lpad("7", 3, "0"), "007");
        assert_eq!(rpad("7", 3, "-"), "7--");
        assert_eq!(lpad("toolong", 3, "0"), "too");
    }

    #[test]
    fn prefix_suffix_split() {
        assert!(starts_with("filename.txt", "file"));
        assert!(ends_with("filename.txt", ".txt"));
        assert_eq!(split_part("a,b,c", ",", 1), "b");
        assert_eq!(split_part("a,b,c", ",", 9), "");
    }
}
