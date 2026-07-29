//! Pattern matching for `LIKE` and a small glob/regex subset.
//!
//! Predicate pushdown needs to evaluate `LIKE` patterns and simple wildcard
//! globs against text columns. This module compiles a pattern once into a
//! token program and then matches it against many strings, which is much
//! cheaper than re-parsing per row. Three surfaces are supported:
//!
//! - SQL `LIKE`: `%` matches any run, `_` matches one character, `\` escapes.
//! - Shell glob: `*`, `?`, and `[a-z]` character classes.
//! - A tiny regex subset: literals, `.`, `*` (greedy, applied to the previous
//!   atom), `^`/`$` anchors, and `[...]` classes.

/// A compiled `LIKE` pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Like {
    tokens: Vec<LikeToken>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LikeToken {
    Literal(char),
    Any,       // `_`
    AnyRun,    // `%`
}

impl Like {
    /// Compile a `LIKE` pattern with `\` as the escape character.
    pub fn compile(pattern: &str) -> Like {
        let mut tokens = Vec::new();
        let mut chars = pattern.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\\' => {
                    if let Some(n) = chars.next() {
                        tokens.push(LikeToken::Literal(n));
                    } else {
                        tokens.push(LikeToken::Literal('\\'));
                    }
                }
                '%' => {
                    // Collapse consecutive `%`.
                    if tokens.last() != Some(&LikeToken::AnyRun) {
                        tokens.push(LikeToken::AnyRun);
                    }
                }
                '_' => tokens.push(LikeToken::Any),
                other => tokens.push(LikeToken::Literal(other)),
            }
        }
        Like { tokens }
    }

    /// `true` if `text` matches the pattern.
    pub fn matches(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        like_match(&self.tokens, 0, &chars, 0)
    }
}

fn like_match(tokens: &[LikeToken], ti: usize, text: &[char], si: usize) -> bool {
    if ti == tokens.len() {
        return si == text.len();
    }
    match &tokens[ti] {
        LikeToken::Literal(c) => {
            si < text.len() && text[si] == *c && like_match(tokens, ti + 1, text, si + 1)
        }
        LikeToken::Any => si < text.len() && like_match(tokens, ti + 1, text, si + 1),
        LikeToken::AnyRun => {
            // Try to consume zero or more characters.
            for skip in si..=text.len() {
                if like_match(tokens, ti + 1, text, skip) {
                    return true;
                }
            }
            false
        }
    }
}

/// A compiled glob pattern.
#[derive(Debug, Clone)]
pub struct Glob {
    tokens: Vec<GlobToken>,
}

#[derive(Debug, Clone)]
enum GlobToken {
    Literal(char),
    Any,   // `?`
    Star,  // `*`
    Class { negated: bool, ranges: Vec<(char, char)> },
}

impl Glob {
    /// Compile a shell-style glob.
    pub fn compile(pattern: &str) -> Glob {
        let mut tokens = Vec::new();
        let mut chars = pattern.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '*' => {
                    if !matches!(tokens.last(), Some(GlobToken::Star)) {
                        tokens.push(GlobToken::Star);
                    }
                }
                '?' => tokens.push(GlobToken::Any),
                '[' => {
                    let (tok, _) = parse_class(&mut chars);
                    tokens.push(tok);
                }
                '\\' => {
                    if let Some(n) = chars.next() {
                        tokens.push(GlobToken::Literal(n));
                    }
                }
                other => tokens.push(GlobToken::Literal(other)),
            }
        }
        Glob { tokens }
    }

    /// `true` if `text` matches.
    pub fn matches(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        glob_match(&self.tokens, 0, &chars, 0)
    }
}

fn parse_class(chars: &mut std::iter::Peekable<std::str::Chars>) -> (GlobToken, bool) {
    let mut negated = false;
    if chars.peek() == Some(&'!') || chars.peek() == Some(&'^') {
        negated = true;
        chars.next();
    }
    let mut ranges = Vec::new();
    while let Some(&c) = chars.peek() {
        if c == ']' {
            chars.next();
            break;
        }
        chars.next();
        if chars.peek() == Some(&'-') {
            chars.next();
            if let Some(&hi) = chars.peek() {
                if hi != ']' {
                    chars.next();
                    ranges.push((c, hi));
                    continue;
                }
            }
            ranges.push((c, c));
            ranges.push(('-', '-'));
        } else {
            ranges.push((c, c));
        }
    }
    (GlobToken::Class { negated, ranges }, negated)
}

fn class_matches(negated: bool, ranges: &[(char, char)], c: char) -> bool {
    let hit = ranges.iter().any(|&(lo, hi)| c >= lo && c <= hi);
    hit ^ negated
}

fn glob_match(tokens: &[GlobToken], ti: usize, text: &[char], si: usize) -> bool {
    if ti == tokens.len() {
        return si == text.len();
    }
    match &tokens[ti] {
        GlobToken::Literal(c) => {
            si < text.len() && text[si] == *c && glob_match(tokens, ti + 1, text, si + 1)
        }
        GlobToken::Any => si < text.len() && glob_match(tokens, ti + 1, text, si + 1),
        GlobToken::Class { negated, ranges } => {
            si < text.len()
                && class_matches(*negated, ranges, text[si])
                && glob_match(tokens, ti + 1, text, si + 1)
        }
        GlobToken::Star => {
            for skip in si..=text.len() {
                if glob_match(tokens, ti + 1, text, skip) {
                    return true;
                }
            }
            false
        }
    }
}

/// A compiled tiny regex.
#[derive(Debug, Clone)]
pub struct Regex {
    atoms: Vec<Atom>,
    anchored_start: bool,
    anchored_end: bool,
}

#[derive(Debug, Clone)]
struct Atom {
    kind: AtomKind,
    star: bool,
}

#[derive(Debug, Clone)]
enum AtomKind {
    Literal(char),
    Dot,
    Class { negated: bool, ranges: Vec<(char, char)> },
}

impl Regex {
    /// Compile a tiny regex subset. Returns `None` on a malformed class.
    pub fn compile(pattern: &str) -> Regex {
        let mut atoms = Vec::new();
        let mut anchored_start = false;
        let mut anchored_end = false;
        let mut chars = pattern.chars().peekable();
        if chars.peek() == Some(&'^') {
            anchored_start = true;
            chars.next();
        }
        while let Some(c) = chars.next() {
            let kind = match c {
                '$' if chars.peek().is_none() => {
                    anchored_end = true;
                    break;
                }
                '.' => AtomKind::Dot,
                '[' => {
                    let (tok, _) = parse_class(&mut chars);
                    match tok {
                        GlobToken::Class { negated, ranges } => AtomKind::Class { negated, ranges },
                        _ => AtomKind::Dot,
                    }
                }
                '\\' => {
                    let n = chars.next().unwrap_or('\\');
                    AtomKind::Literal(n)
                }
                other => AtomKind::Literal(other),
            };
            let star = chars.peek() == Some(&'*');
            if star {
                chars.next();
            }
            atoms.push(Atom { kind, star });
        }
        Regex {
            atoms,
            anchored_start,
            anchored_end,
        }
    }

    /// `true` if the regex matches anywhere in `text` (respecting anchors).
    pub fn is_match(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        if self.anchored_start {
            return self.match_at(&chars, 0);
        }
        for start in 0..=chars.len() {
            if self.match_at(&chars, start) {
                return true;
            }
        }
        false
    }

    fn match_at(&self, text: &[char], start: usize) -> bool {
        self.match_from(0, text, start)
    }

    fn match_from(&self, ai: usize, text: &[char], si: usize) -> bool {
        if ai == self.atoms.len() {
            return !self.anchored_end || si == text.len();
        }
        let atom = &self.atoms[ai];
        if atom.star {
            // Greedy: consume as many as possible, then backtrack.
            let mut count = 0;
            while si + count < text.len() && atom_matches(&atom.kind, text[si + count]) {
                count += 1;
            }
            for take in (0..=count).rev() {
                if self.match_from(ai + 1, text, si + take) {
                    return true;
                }
            }
            false
        } else if si < text.len() && atom_matches(&atom.kind, text[si]) {
            self.match_from(ai + 1, text, si + 1)
        } else {
            false
        }
    }
}

fn atom_matches(kind: &AtomKind, c: char) -> bool {
    match kind {
        AtomKind::Literal(l) => *l == c,
        AtomKind::Dot => true,
        AtomKind::Class { negated, ranges } => class_matches(*negated, ranges, c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn like_basic() {
        let p = Like::compile("a%z");
        assert!(p.matches("az"));
        assert!(p.matches("abcz"));
        assert!(!p.matches("abc"));
        let u = Like::compile("a_c");
        assert!(u.matches("abc"));
        assert!(!u.matches("ac"));
    }

    #[test]
    fn like_escape() {
        let p = Like::compile(r"100\%");
        assert!(p.matches("100%"));
        assert!(!p.matches("100x"));
    }

    #[test]
    fn like_collapses_percent() {
        let p = Like::compile("%%%abc%%%");
        assert!(p.matches("xxabcyy"));
        assert!(p.matches("abc"));
    }

    #[test]
    fn glob_star_question_class() {
        assert!(Glob::compile("*.rs").matches("main.rs"));
        assert!(Glob::compile("file?.txt").matches("file1.txt"));
        assert!(Glob::compile("[a-c]at").matches("bat"));
        assert!(!Glob::compile("[a-c]at").matches("dat"));
        assert!(Glob::compile("[!0-9]x").matches("ax"));
    }

    #[test]
    fn regex_literals_and_dot() {
        assert!(Regex::compile("a.c").is_match("abc"));
        assert!(Regex::compile("^abc$").is_match("abc"));
        assert!(!Regex::compile("^abc$").is_match("abcd"));
    }

    #[test]
    fn regex_star() {
        assert!(Regex::compile("ab*c").is_match("ac"));
        assert!(Regex::compile("ab*c").is_match("abbbc"));
        assert!(Regex::compile("^a.*z$").is_match("abcz"));
        assert!(Regex::compile("[0-9]*").is_match("123"));
    }

    #[test]
    fn regex_unanchored_search() {
        assert!(Regex::compile("cat").is_match("the cat sat"));
        assert!(!Regex::compile("dog").is_match("the cat sat"));
    }
}
