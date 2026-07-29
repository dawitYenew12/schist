//! A library of scalar functions over [`Value`].
//!
//! The expression evaluator dispatches built-in scalar functions — arithmetic
//! helpers, string manipulation, math, null handling — through this registry.
//! Keeping them here (rather than inline in the evaluator) makes the set easy to
//! enumerate for the planner and keeps their null-propagation and type-coercion
//! rules consistent: most functions return `Null` if any required argument is
//! null, following SQL semantics.

use crate::value::Value;
use std::collections::HashMap;

/// The signature and implementation of one scalar function.
pub struct ScalarFn {
    pub name: &'static str,
    pub min_args: usize,
    pub max_args: usize,
    pub func: fn(&[Value]) -> Value,
}

/// A registry of scalar functions by lowercase name.
pub struct ScalarRegistry {
    fns: HashMap<&'static str, ScalarFn>,
}

impl Default for ScalarRegistry {
    fn default() -> Self {
        ScalarRegistry::with_builtins()
    }
}

impl ScalarRegistry {
    /// An empty registry.
    pub fn new() -> ScalarRegistry {
        ScalarRegistry {
            fns: HashMap::new(),
        }
    }

    /// A registry populated with all built-in functions.
    pub fn with_builtins() -> ScalarRegistry {
        let mut r = ScalarRegistry::new();
        for f in builtins() {
            r.register(f);
        }
        r
    }

    /// Register a function.
    pub fn register(&mut self, f: ScalarFn) {
        self.fns.insert(f.name, f);
    }

    /// Look up by name (case-insensitive on ASCII).
    pub fn get(&self, name: &str) -> Option<&ScalarFn> {
        // Names are stored lowercase; callers usually pass lowercase already.
        self.fns.get(name)
    }

    /// Number of registered functions.
    pub fn len(&self) -> usize {
        self.fns.len()
    }

    /// `true` if no functions are registered.
    pub fn is_empty(&self) -> bool {
        self.fns.is_empty()
    }

    /// Call a function by name, checking arity. Returns `Null` for an unknown
    /// function or an arity mismatch.
    pub fn call(&self, name: &str, args: &[Value]) -> Value {
        let lname = name.to_ascii_lowercase();
        match self.fns.get(lname.as_str()) {
            Some(f) if args.len() >= f.min_args && args.len() <= f.max_args => (f.func)(args),
            _ => Value::Null,
        }
    }

    /// All registered function names, sorted.
    pub fn names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.fns.keys().copied().collect();
        names.sort_unstable();
        names
    }
}

fn any_null(args: &[Value]) -> bool {
    args.iter().any(|a| a.is_null())
}

fn builtins() -> Vec<ScalarFn> {
    vec![
        ScalarFn {
            name: "abs",
            min_args: 1,
            max_args: 1,
            func: |a| match a[0] {
                Value::Int(i) => Value::Int(i.abs()),
                Value::Real(r) => Value::Real(r.abs()),
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "sign",
            min_args: 1,
            max_args: 1,
            func: |a| match a[0] {
                Value::Int(i) => Value::Int(i.signum()),
                Value::Real(r) => Value::Int(if r > 0.0 {
                    1
                } else if r < 0.0 {
                    -1
                } else {
                    0
                }),
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "ceil",
            min_args: 1,
            max_args: 1,
            func: |a| match a[0].as_real() {
                Some(r) if !a[0].is_null() => Value::Int(r.ceil() as i64),
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "floor",
            min_args: 1,
            max_args: 1,
            func: |a| match a[0].as_real() {
                Some(r) if !a[0].is_null() => Value::Int(r.floor() as i64),
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "round",
            min_args: 1,
            max_args: 2,
            func: |a| {
                if a[0].is_null() {
                    return Value::Null;
                }
                let r = match a[0].as_real() {
                    Some(r) => r,
                    None => return Value::Null,
                };
                let digits = a.get(1).and_then(|v| v.as_int()).unwrap_or(0);
                let factor = 10f64.powi(digits as i32);
                Value::Real((r * factor).round() / factor)
            },
        },
        ScalarFn {
            name: "sqrt",
            min_args: 1,
            max_args: 1,
            func: |a| match a[0].as_real() {
                Some(r) if r >= 0.0 && !a[0].is_null() => Value::Real(r.sqrt()),
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "pow",
            min_args: 2,
            max_args: 2,
            func: |a| {
                if any_null(a) {
                    return Value::Null;
                }
                match (a[0].as_real(), a[1].as_real()) {
                    (Some(b), Some(e)) => Value::Real(b.powf(e)),
                    _ => Value::Null,
                }
            },
        },
        ScalarFn {
            name: "mod",
            min_args: 2,
            max_args: 2,
            func: |a| {
                if any_null(a) {
                    return Value::Null;
                }
                match (a[0].as_int(), a[1].as_int()) {
                    (Some(_), Some(0)) => Value::Null,
                    (Some(x), Some(y)) => Value::Int(x % y),
                    _ => Value::Null,
                }
            },
        },
        ScalarFn {
            name: "greatest",
            min_args: 1,
            max_args: 16,
            func: |a| {
                a.iter()
                    .filter(|v| !v.is_null())
                    .copied()
                    .reduce(|x, y| if x.total_cmp(&y).is_ge() { x } else { y })
                    .unwrap_or(Value::Null)
            },
        },
        ScalarFn {
            name: "least",
            min_args: 1,
            max_args: 16,
            func: |a| {
                a.iter()
                    .filter(|v| !v.is_null())
                    .copied()
                    .reduce(|x, y| if x.total_cmp(&y).is_le() { x } else { y })
                    .unwrap_or(Value::Null)
            },
        },
        ScalarFn {
            name: "coalesce",
            min_args: 1,
            max_args: 16,
            func: |a| a.iter().copied().find(|v| !v.is_null()).unwrap_or(Value::Null),
        },
        ScalarFn {
            name: "nullif",
            min_args: 2,
            max_args: 2,
            func: |a| {
                if !a[0].is_null() && !a[1].is_null() && a[0].total_cmp(&a[1]).is_eq() {
                    Value::Null
                } else {
                    a[0]
                }
            },
        },
        ScalarFn {
            name: "ifnull",
            min_args: 2,
            max_args: 2,
            func: |a| if a[0].is_null() { a[1] } else { a[0] },
        },
        ScalarFn {
            name: "iif",
            min_args: 3,
            max_args: 3,
            func: |a| match a[0] {
                Value::Bool(true) => a[1],
                Value::Bool(false) => a[2],
                Value::Int(i) => {
                    if i != 0 {
                        a[1]
                    } else {
                        a[2]
                    }
                }
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "not",
            min_args: 1,
            max_args: 1,
            func: |a| match a[0] {
                Value::Bool(b) => Value::Bool(!b),
                Value::Null => Value::Null,
                Value::Int(i) => Value::Bool(i == 0),
                _ => Value::Null,
            },
        },
        ScalarFn {
            name: "to_int",
            min_args: 1,
            max_args: 1,
            func: |a| a[0].as_int().map(Value::Int).unwrap_or(Value::Null),
        },
        ScalarFn {
            name: "to_real",
            min_args: 1,
            max_args: 1,
            func: |a| a[0].as_real().map(Value::Real).unwrap_or(Value::Null),
        },
        ScalarFn {
            name: "is_null",
            min_args: 1,
            max_args: 1,
            func: |a| Value::Bool(a[0].is_null()),
        },
        ScalarFn {
            name: "min2",
            min_args: 2,
            max_args: 2,
            func: |a| {
                if any_null(a) {
                    return Value::Null;
                }
                if a[0].total_cmp(&a[1]).is_le() {
                    a[0]
                } else {
                    a[1]
                }
            },
        },
        ScalarFn {
            name: "max2",
            min_args: 2,
            max_args: 2,
            func: |a| {
                if any_null(a) {
                    return Value::Null;
                }
                if a[0].total_cmp(&a[1]).is_ge() {
                    a[0]
                } else {
                    a[1]
                }
            },
        },
        ScalarFn {
            name: "clamp",
            min_args: 3,
            max_args: 3,
            func: |a| {
                if any_null(a) {
                    return Value::Null;
                }
                let mut v = a[0];
                if v.total_cmp(&a[1]).is_lt() {
                    v = a[1];
                }
                if v.total_cmp(&a[2]).is_gt() {
                    v = a[2];
                }
                v
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg() -> ScalarRegistry {
        ScalarRegistry::with_builtins()
    }

    #[test]
    fn math_functions() {
        let r = reg();
        assert_eq!(r.call("abs", &[Value::Int(-5)]), Value::Int(5));
        assert_eq!(r.call("ceil", &[Value::Real(2.1)]), Value::Int(3));
        assert_eq!(r.call("floor", &[Value::Real(2.9)]), Value::Int(2));
        assert_eq!(r.call("sqrt", &[Value::Real(9.0)]), Value::Real(3.0));
        assert_eq!(r.call("pow", &[Value::Int(2), Value::Int(10)]), Value::Real(1024.0));
    }

    #[test]
    fn round_with_digits() {
        let r = reg();
        assert_eq!(r.call("round", &[Value::Real(3.14159), Value::Int(2)]), Value::Real(3.14));
        assert_eq!(r.call("round", &[Value::Real(2.5)]), Value::Real(3.0));
    }

    #[test]
    fn null_handling() {
        let r = reg();
        assert_eq!(r.call("coalesce", &[Value::Null, Value::Null, Value::Int(7)]), Value::Int(7));
        assert_eq!(r.call("ifnull", &[Value::Null, Value::Int(1)]), Value::Int(1));
        assert_eq!(r.call("nullif", &[Value::Int(5), Value::Int(5)]), Value::Null);
        assert_eq!(r.call("is_null", &[Value::Null]), Value::Bool(true));
        assert_eq!(r.call("abs", &[Value::Null]), Value::Null);
    }

    #[test]
    fn greatest_least_clamp() {
        let r = reg();
        assert_eq!(
            r.call("greatest", &[Value::Int(3), Value::Int(9), Value::Int(1)]),
            Value::Int(9)
        );
        assert_eq!(
            r.call("least", &[Value::Int(3), Value::Null, Value::Int(1)]),
            Value::Int(1)
        );
        assert_eq!(
            r.call("clamp", &[Value::Int(15), Value::Int(0), Value::Int(10)]),
            Value::Int(10)
        );
    }

    #[test]
    fn mod_divide_by_zero_is_null() {
        let r = reg();
        assert_eq!(r.call("mod", &[Value::Int(10), Value::Int(0)]), Value::Null);
        assert_eq!(r.call("mod", &[Value::Int(10), Value::Int(3)]), Value::Int(1));
    }

    #[test]
    fn arity_and_unknown() {
        let r = reg();
        assert_eq!(r.call("abs", &[]), Value::Null); // too few args
        assert_eq!(r.call("no_such_fn", &[Value::Int(1)]), Value::Null);
        assert!(r.len() > 10);
        assert!(r.names().contains(&"coalesce"));
    }

    #[test]
    fn case_insensitive() {
        let r = reg();
        assert_eq!(r.call("ABS", &[Value::Int(-2)]), Value::Int(2));
    }
}
