//! Fixed-point decimal arithmetic.
//!
//! A [`Decimal`] is an `i128` unscaled mantissa plus a `u8` scale (number of
//! fractional digits). It gives exact base-10 arithmetic for money-like
//! columns where binary floating point would accumulate rounding error. The
//! type supports the four arithmetic operations with explicit rounding, parses
//! and formats decimal strings, and converts to and from the engine's
//! [`crate::value::Value`] (as scaled integers or reals).

use std::cmp::Ordering;
use std::fmt;

/// Rounding mode for operations that must reduce scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// Round half away from zero (commercial rounding).
    HalfUp,
    /// Round half to even (banker's rounding).
    HalfEven,
    /// Truncate toward zero.
    Down,
}

/// A fixed-point decimal number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decimal {
    mantissa: i128,
    scale: u8,
}

/// Error parsing a decimal string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDecimalError(pub String);

impl fmt::Display for ParseDecimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid decimal: {}", self.0)
    }
}

impl std::error::Error for ParseDecimalError {}

const MAX_SCALE: u8 = 38;

fn pow10(n: u8) -> i128 {
    let mut p = 1i128;
    for _ in 0..n {
        p = p.saturating_mul(10);
    }
    p
}

impl Decimal {
    /// A decimal from an unscaled mantissa and scale.
    pub fn new(mantissa: i128, scale: u8) -> Decimal {
        Decimal {
            mantissa,
            scale: scale.min(MAX_SCALE),
        }
    }

    /// Zero at scale 0.
    pub fn zero() -> Decimal {
        Decimal {
            mantissa: 0,
            scale: 0,
        }
    }

    /// An integer value at scale 0.
    pub fn from_i64(v: i64) -> Decimal {
        Decimal {
            mantissa: v as i128,
            scale: 0,
        }
    }

    /// The unscaled mantissa.
    pub fn mantissa(&self) -> i128 {
        self.mantissa
    }

    /// The scale (fractional digit count).
    pub fn scale(&self) -> u8 {
        self.scale
    }

    /// `true` if the value is zero.
    pub fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// The sign: -1, 0, or 1.
    pub fn signum(&self) -> i32 {
        match self.mantissa.cmp(&0) {
            Ordering::Less => -1,
            Ordering::Equal => 0,
            Ordering::Greater => 1,
        }
    }

    /// Re-scale to a target scale, rounding if the scale decreases.
    pub fn rescale(&self, target: u8, mode: Rounding) -> Decimal {
        let target = target.min(MAX_SCALE);
        if target == self.scale {
            return *self;
        }
        if target > self.scale {
            let factor = pow10(target - self.scale);
            Decimal {
                mantissa: self.mantissa.saturating_mul(factor),
                scale: target,
            }
        } else {
            let drop = self.scale - target;
            let factor = pow10(drop);
            let q = self.mantissa / factor;
            let r = self.mantissa % factor;
            let rounded = apply_rounding(q, r, factor, mode);
            Decimal {
                mantissa: rounded,
                scale: target,
            }
        }
    }

    fn align(a: &Decimal, b: &Decimal) -> (i128, i128, u8) {
        let scale = a.scale.max(b.scale);
        let am = a.mantissa.saturating_mul(pow10(scale - a.scale));
        let bm = b.mantissa.saturating_mul(pow10(scale - b.scale));
        (am, bm, scale)
    }

    /// Add two decimals (result scale is the larger of the two).
    pub fn add(&self, other: &Decimal) -> Decimal {
        let (a, b, scale) = Decimal::align(self, other);
        Decimal {
            mantissa: a.saturating_add(b),
            scale,
        }
    }

    /// Subtract.
    pub fn sub(&self, other: &Decimal) -> Decimal {
        let (a, b, scale) = Decimal::align(self, other);
        Decimal {
            mantissa: a.saturating_sub(b),
            scale,
        }
    }

    /// Multiply (result scale is the sum of the two scales, capped).
    pub fn mul(&self, other: &Decimal) -> Decimal {
        let scale = (self.scale as u16 + other.scale as u16).min(MAX_SCALE as u16) as u8;
        let raw = self.mantissa.saturating_mul(other.mantissa);
        // If the true scale exceeds the cap, shrink.
        let true_scale = self.scale as u16 + other.scale as u16;
        if true_scale > scale as u16 {
            let drop = (true_scale - scale as u16) as u8;
            let factor = pow10(drop);
            Decimal {
                mantissa: raw / factor,
                scale,
            }
        } else {
            Decimal {
                mantissa: raw,
                scale,
            }
        }
    }

    /// Divide to a target result scale with rounding. Returns `None` on divide
    /// by zero.
    pub fn div(&self, other: &Decimal, result_scale: u8, mode: Rounding) -> Option<Decimal> {
        if other.mantissa == 0 {
            return None;
        }
        let result_scale = result_scale.min(MAX_SCALE);
        // numerator scaled up so the quotient carries result_scale + 1 guard digit.
        let extra = result_scale + 1 + other.scale;
        let num = self.mantissa.saturating_mul(pow10(extra.saturating_sub(self.scale)));
        let q = num / other.mantissa;
        let r = num % other.mantissa;
        let guarded = apply_rounding(q, r, other.mantissa, mode);
        // guarded currently at scale result_scale + 1; drop the guard digit.
        let final_m = {
            let factor = 10i128;
            let qq = guarded / factor;
            let rr = guarded % factor;
            apply_rounding(qq, rr, factor, mode)
        };
        Some(Decimal {
            mantissa: final_m,
            scale: result_scale,
        })
    }

    /// Negate.
    pub fn neg(&self) -> Decimal {
        Decimal {
            mantissa: -self.mantissa,
            scale: self.scale,
        }
    }

    /// Absolute value.
    pub fn abs(&self) -> Decimal {
        Decimal {
            mantissa: self.mantissa.abs(),
            scale: self.scale,
        }
    }

    /// Convert to `f64` (may lose precision).
    pub fn to_f64(&self) -> f64 {
        self.mantissa as f64 / pow10(self.scale) as f64
    }

    /// Ordering comparison across differing scales.
    pub fn cmp(&self, other: &Decimal) -> Ordering {
        let (a, b, _) = Decimal::align(self, other);
        a.cmp(&b)
    }

    /// Parse a decimal string such as `-12.340`.
    pub fn parse(s: &str) -> Result<Decimal, ParseDecimalError> {
        let s = s.trim();
        if s.is_empty() {
            return Err(ParseDecimalError(s.to_string()));
        }
        let (neg, rest) = match s.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let mut int_part = String::new();
        let mut frac_part = String::new();
        let mut seen_dot = false;
        for c in rest.chars() {
            match c {
                '0'..='9' => {
                    if seen_dot {
                        frac_part.push(c);
                    } else {
                        int_part.push(c);
                    }
                }
                '.' if !seen_dot => seen_dot = true,
                '_' => {}
                _ => return Err(ParseDecimalError(s.to_string())),
            }
        }
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(ParseDecimalError(s.to_string()));
        }
        let scale = frac_part.len().min(MAX_SCALE as usize) as u8;
        let frac_part = &frac_part[..scale as usize];
        let digits = format!("{int_part}{frac_part}");
        let mut mantissa: i128 = 0;
        for c in digits.chars() {
            mantissa = mantissa
                .checked_mul(10)
                .and_then(|m| m.checked_add((c as u8 - b'0') as i128))
                .ok_or_else(|| ParseDecimalError(s.to_string()))?;
        }
        if neg {
            mantissa = -mantissa;
        }
        Ok(Decimal { mantissa, scale })
    }
}

fn apply_rounding(q: i128, r: i128, divisor: i128, mode: Rounding) -> i128 {
    if r == 0 {
        return q;
    }
    let sign = if (r < 0) ^ (divisor < 0) { -1 } else { 1 };
    let twice = (r.saturating_mul(2)).abs();
    let d = divisor.abs();
    match mode {
        Rounding::Down => q,
        Rounding::HalfUp => {
            if twice >= d {
                q + sign
            } else {
                q
            }
        }
        Rounding::HalfEven => match twice.cmp(&d) {
            Ordering::Greater => q + sign,
            Ordering::Less => q,
            Ordering::Equal => {
                if q % 2 == 0 {
                    q
                } else {
                    q + sign
                }
            }
        },
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let neg = self.mantissa < 0;
        let mag = self.mantissa.unsigned_abs();
        let s = mag.to_string();
        if self.scale == 0 {
            if neg {
                write!(f, "-{s}")
            } else {
                write!(f, "{s}")
            }
        } else {
            let scale = self.scale as usize;
            let padded = if s.len() <= scale {
                format!("{:0>width$}", s, width = scale + 1)
            } else {
                s
            };
            let dot = padded.len() - scale;
            let (int, frac) = padded.split_at(dot);
            if neg {
                write!(f, "-{int}.{frac}")
            } else {
                write!(f, "{int}.{frac}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display() {
        assert_eq!(Decimal::parse("12.34").unwrap().to_string(), "12.34");
        assert_eq!(Decimal::parse("-0.005").unwrap().to_string(), "-0.005");
        assert_eq!(Decimal::parse("100").unwrap().to_string(), "100");
        assert_eq!(Decimal::parse("0.50").unwrap().to_string(), "0.50");
        assert!(Decimal::parse("abc").is_err());
        assert!(Decimal::parse("").is_err());
    }

    #[test]
    fn add_sub_align_scales() {
        let a = Decimal::parse("1.5").unwrap();
        let b = Decimal::parse("2.25").unwrap();
        assert_eq!(a.add(&b).to_string(), "3.75");
        assert_eq!(b.sub(&a).to_string(), "0.75");
    }

    #[test]
    fn multiply_scales_add() {
        let a = Decimal::parse("1.2").unwrap();
        let b = Decimal::parse("1.3").unwrap();
        let p = a.mul(&b);
        assert_eq!(p.scale(), 2);
        assert_eq!(p.to_string(), "1.56");
    }

    #[test]
    fn divide_with_rounding() {
        let a = Decimal::from_i64(10);
        let b = Decimal::from_i64(3);
        let q = a.div(&b, 4, Rounding::HalfUp).unwrap();
        assert_eq!(q.to_string(), "3.3333");
        assert!(a.div(&Decimal::zero(), 2, Rounding::HalfUp).is_none());
    }

    #[test]
    fn rescale_rounds_half_even() {
        let a = Decimal::parse("2.5").unwrap();
        assert_eq!(a.rescale(0, Rounding::HalfEven).to_string(), "2");
        let b = Decimal::parse("3.5").unwrap();
        assert_eq!(b.rescale(0, Rounding::HalfEven).to_string(), "4");
    }

    #[test]
    fn ordering_across_scales() {
        let a = Decimal::parse("1.5").unwrap();
        let b = Decimal::parse("1.50").unwrap();
        assert_eq!(a.cmp(&b), Ordering::Equal);
        let c = Decimal::parse("1.500001").unwrap();
        assert_eq!(a.cmp(&c), Ordering::Less);
    }
}
