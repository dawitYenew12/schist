//! String edit-distance and fuzzy-match helpers.
//!
//! Fuzzy predicate matching and "did you mean" column-name resolution use edit
//! distance. This module implements Levenshtein distance with the two-row
//! rolling optimization (linear memory), Damerau-Levenshtein (adding
//! transpositions), a bounded variant that stops early once a threshold is
//! exceeded, and a normalized similarity ratio in `[0, 1]`. Distances are
//! computed over Unicode scalar values, not bytes.

/// Levenshtein edit distance (insertions, deletions, substitutions).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Levenshtein distance that returns `None` as soon as it provably exceeds
/// `max` (a cheap early-out for threshold queries).
pub fn levenshtein_bounded(a: &str, b: &str, max: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            row_min = row_min.min(cur[j + 1]);
        }
        if row_min > max {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let d = prev[b.len()];
    if d > max {
        None
    } else {
        Some(d)
    }
}

/// Damerau-Levenshtein distance (adds adjacent transpositions).
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        d[i][0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let mut best = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = best;
        }
    }
    d[n][m]
}

/// A normalized similarity ratio in `[0, 1]` (1.0 = identical).
pub fn similarity(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    1.0 - (levenshtein(a, b) as f64 / max_len as f64)
}

/// The prefix length two strings share (in characters).
pub fn common_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Find the closest candidate to `query` within `max_distance`, if any.
pub fn closest<'a>(query: &str, candidates: &'a [&'a str], max_distance: usize) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for &cand in candidates {
        if let Some(d) = levenshtein_bounded(query, cand, max_distance) {
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((cand, d));
            }
        }
    }
    best.map(|(c, _)| c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("same", "same"), 0);
        assert_eq!(levenshtein("flaw", "lawn"), 2);
    }

    #[test]
    fn bounded_early_out() {
        assert_eq!(levenshtein_bounded("kitten", "sitting", 3), Some(3));
        assert_eq!(levenshtein_bounded("kitten", "sitting", 2), None);
        assert_eq!(levenshtein_bounded("abc", "abcdefghij", 2), None);
    }

    #[test]
    fn damerau_handles_transposition() {
        assert_eq!(damerau_levenshtein("ca", "ac"), 1);
        assert_eq!(levenshtein("ca", "ac"), 2);
        assert_eq!(damerau_levenshtein("teh", "the"), 1);
    }

    #[test]
    fn similarity_ratio() {
        assert_eq!(similarity("abc", "abc"), 1.0);
        assert_eq!(similarity("", ""), 1.0);
        assert!(similarity("kitten", "sitting") > 0.5);
        assert!(similarity("abc", "xyz") < 0.1);
    }

    #[test]
    fn prefix_and_closest() {
        assert_eq!(common_prefix("prefix_a", "prefix_b"), 7);
        let cands = ["name", "age", "email", "phone"];
        assert_eq!(closest("naem", &cands, 2), Some("name"));
        assert_eq!(closest("zzzzz", &cands, 2), None);
    }

    #[test]
    fn unicode_aware() {
        assert_eq!(levenshtein("café", "cafe"), 1);
        assert_eq!(common_prefix("café", "cafétér"), 4);
    }
}
