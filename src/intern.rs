//! A string interner producing small integer symbols.
//!
//! Column names, table names, and repeated text literals appear over and over
//! in plans and catalogs. Interning maps each distinct string to a compact
//! [`Symbol`] once, so downstream code compares and stores `u32`s instead of
//! re-hashing strings. The interner is append-only (symbols are never
//! invalidated) which keeps the mapping stable for the life of a query.

use std::collections::HashMap;

/// A compact handle for an interned string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(pub u32);

/// An append-only string interner.
#[derive(Debug, Default, Clone)]
pub struct Interner {
    map: HashMap<String, Symbol>,
    strings: Vec<String>,
}

impl Interner {
    /// A fresh interner.
    pub fn new() -> Interner {
        Interner::default()
    }

    /// Intern a string, returning its symbol (idempotent).
    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let sym = Symbol(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), sym);
        sym
    }

    /// Look up a symbol without interning; `None` if the string is unknown.
    pub fn get(&self, s: &str) -> Option<Symbol> {
        self.map.get(s).copied()
    }

    /// Resolve a symbol back to its string.
    pub fn resolve(&self, sym: Symbol) -> Option<&str> {
        self.strings.get(sym.0 as usize).map(|s| s.as_str())
    }

    /// `true` if the string has been interned.
    pub fn contains(&self, s: &str) -> bool {
        self.map.contains_key(s)
    }

    /// Number of distinct interned strings.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// `true` if nothing has been interned.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    /// All interned strings in symbol order.
    pub fn strings(&self) -> &[String] {
        &self.strings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_idempotent() {
        let mut it = Interner::new();
        let a = it.intern("hello");
        let b = it.intern("hello");
        assert_eq!(a, b);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn distinct_strings_distinct_symbols() {
        let mut it = Interner::new();
        let a = it.intern("a");
        let b = it.intern("b");
        assert_ne!(a, b);
        assert_eq!(it.resolve(a), Some("a"));
        assert_eq!(it.resolve(b), Some("b"));
    }

    #[test]
    fn get_and_contains() {
        let mut it = Interner::new();
        it.intern("x");
        assert!(it.contains("x"));
        assert!(it.get("x").is_some());
        assert!(it.get("y").is_none());
    }

    #[test]
    fn symbols_are_stable_order() {
        let mut it = Interner::new();
        for (i, s) in ["one", "two", "three"].iter().enumerate() {
            assert_eq!(it.intern(s), Symbol(i as u32));
        }
        assert_eq!(it.strings().len(), 3);
    }
}
