//! Calendar date and timestamp types.
//!
//! `schist` stores temporal columns as integers — a [`Date`] as a day count
//! from the Unix epoch (1970-01-01), a [`Timestamp`] as a microsecond count
//! from the same epoch — so they sort and range-scan as plain integers. This
//! module provides the conversion between those integers and broken-down
//! calendar fields, using the proleptic Gregorian calendar, plus ISO-8601
//! parsing and formatting and a small amount of date arithmetic.

use std::fmt;

/// A calendar date, stored as days from 1970-01-01 (may be negative).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    days: i32,
}

/// A wall-clock timestamp, stored as microseconds from the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp {
    micros: i64,
}

/// Broken-down calendar fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalendarDate {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

/// Error parsing a temporal string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseTemporalError(pub String);

impl fmt::Display for ParseTemporalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid temporal literal: {}", self.0)
    }
}

impl std::error::Error for ParseTemporalError {}

const MICROS_PER_SEC: i64 = 1_000_000;
const SECS_PER_DAY: i64 = 86_400;

/// `true` if `year` is a Gregorian leap year.
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Number of days in `month` (1..=12) of `year`.
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

// Days from civil date to epoch, after Howard Hinnant's algorithm.
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let doy = ((153 * (if m > 2 { m - 3 } else { m + 9 }) as i64 + 2) / 5 + d as i64 - 1) as i64;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era as i64 * 146097 + doe - 719468) as i32
}

fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

impl Date {
    /// Construct from a day count.
    pub fn from_days(days: i32) -> Date {
        Date { days }
    }

    /// The day count from the epoch.
    pub fn days(&self) -> i32 {
        self.days
    }

    /// Construct from calendar fields (no validation of field ranges).
    pub fn from_ymd(year: i32, month: u32, day: u32) -> Date {
        Date {
            days: days_from_civil(year, month, day),
        }
    }

    /// Break down into calendar fields.
    pub fn calendar(&self) -> CalendarDate {
        let (year, month, day) = civil_from_days(self.days);
        CalendarDate { year, month, day }
    }

    /// The weekday, 0 = Monday .. 6 = Sunday.
    pub fn weekday(&self) -> u32 {
        // 1970-01-01 was a Thursday (index 3).
        (((self.days % 7) + 7 + 3) % 7) as u32
    }

    /// Add a number of days.
    pub fn add_days(&self, n: i32) -> Date {
        Date {
            days: self.days + n,
        }
    }

    /// Add `n` months, clamping the day to the target month length.
    pub fn add_months(&self, n: i32) -> Date {
        let c = self.calendar();
        let total = (c.year * 12 + c.month as i32 - 1) + n;
        let year = total.div_euclid(12);
        let month = (total.rem_euclid(12) + 1) as u32;
        let day = c.day.min(days_in_month(year, month));
        Date::from_ymd(year, month, day)
    }

    /// Number of days from `self` to `other`.
    pub fn days_until(&self, other: &Date) -> i32 {
        other.days - self.days
    }

    /// Parse an ISO date `YYYY-MM-DD`.
    pub fn parse(s: &str) -> Result<Date, ParseTemporalError> {
        let s = s.trim();
        let parts: Vec<&str> = s.split('-').collect();
        // Handle a possible leading '-' for negative years by re-splitting.
        let (year, month, day) = if parts.len() == 3 {
            (
                parts[0].parse::<i32>(),
                parts[1].parse::<u32>(),
                parts[2].parse::<u32>(),
            )
        } else if parts.len() == 4 && parts[0].is_empty() {
            (
                parts[1].parse::<i32>().map(|y| -y),
                parts[2].parse::<u32>(),
                parts[3].parse::<u32>(),
            )
        } else {
            return Err(ParseTemporalError(s.to_string()));
        };
        match (year, month, day) {
            (Ok(y), Ok(m), Ok(d)) if (1..=12).contains(&m) && (1..=31).contains(&d) => {
                if d > days_in_month(y, m) {
                    return Err(ParseTemporalError(s.to_string()));
                }
                Ok(Date::from_ymd(y, m, d))
            }
            _ => Err(ParseTemporalError(s.to_string())),
        }
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = self.calendar();
        write!(f, "{:04}-{:02}-{:02}", c.year, c.month, c.day)
    }
}

impl Timestamp {
    /// Construct from a microsecond count.
    pub fn from_micros(micros: i64) -> Timestamp {
        Timestamp { micros }
    }

    /// The microsecond count.
    pub fn micros(&self) -> i64 {
        self.micros
    }

    /// The whole-second count (floored).
    pub fn seconds(&self) -> i64 {
        self.micros.div_euclid(MICROS_PER_SEC)
    }

    /// The date portion.
    pub fn date(&self) -> Date {
        Date::from_days(self.seconds().div_euclid(SECS_PER_DAY) as i32)
    }

    /// Wall-clock (hour, minute, second, microsecond).
    pub fn time_of_day(&self) -> (u32, u32, u32, u32) {
        let sod = self.seconds().rem_euclid(SECS_PER_DAY);
        let micro = self.micros.rem_euclid(MICROS_PER_SEC) as u32;
        let h = (sod / 3600) as u32;
        let m = ((sod % 3600) / 60) as u32;
        let s = (sod % 60) as u32;
        (h, m, s, micro)
    }

    /// Construct from date and time fields.
    pub fn from_parts(
        date: Date,
        hour: u32,
        minute: u32,
        second: u32,
        micro: u32,
    ) -> Timestamp {
        let day_secs = date.days() as i64 * SECS_PER_DAY;
        let tod = hour as i64 * 3600 + minute as i64 * 60 + second as i64;
        Timestamp {
            micros: (day_secs + tod) * MICROS_PER_SEC + micro as i64,
        }
    }

    /// Add a whole number of seconds.
    pub fn add_seconds(&self, secs: i64) -> Timestamp {
        Timestamp {
            micros: self.micros + secs * MICROS_PER_SEC,
        }
    }

    /// Microseconds between `self` and `other`.
    pub fn micros_until(&self, other: &Timestamp) -> i64 {
        other.micros - self.micros
    }

    /// Parse `YYYY-MM-DD[ T]HH:MM:SS[.ffffff]`.
    pub fn parse(s: &str) -> Result<Timestamp, ParseTemporalError> {
        let s = s.trim();
        let (date_str, time_str) = if let Some(pos) = s.find(['T', ' ']) {
            (&s[..pos], &s[pos + 1..])
        } else {
            (s, "00:00:00")
        };
        let date = Date::parse(date_str)?;
        let (hms, frac) = match time_str.split_once('.') {
            Some((h, f)) => (h, f),
            None => (time_str, ""),
        };
        let tparts: Vec<&str> = hms.split(':').collect();
        if tparts.is_empty() || tparts.len() > 3 {
            return Err(ParseTemporalError(s.to_string()));
        }
        let parse_u = |x: &str| x.parse::<u32>().map_err(|_| ParseTemporalError(s.to_string()));
        let hour = parse_u(tparts[0])?;
        let minute = if tparts.len() > 1 { parse_u(tparts[1])? } else { 0 };
        let second = if tparts.len() > 2 { parse_u(tparts[2])? } else { 0 };
        if hour > 23 || minute > 59 || second > 60 {
            return Err(ParseTemporalError(s.to_string()));
        }
        let micro = if frac.is_empty() {
            0
        } else {
            let mut f = String::from(frac);
            f.truncate(6);
            while f.len() < 6 {
                f.push('0');
            }
            f.parse::<u32>().map_err(|_| ParseTemporalError(s.to_string()))?
        };
        Ok(Timestamp::from_parts(date, hour, minute, second, micro))
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.date();
        let (h, m, s, us) = self.time_of_day();
        if us == 0 {
            write!(f, "{d} {h:02}:{m:02}:{s:02}")
        } else {
            write!(f, "{d} {h:02}:{m:02}:{s:02}.{us:06}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_zero() {
        assert_eq!(Date::from_ymd(1970, 1, 1).days(), 0);
        assert_eq!(Date::from_days(0).to_string(), "1970-01-01");
    }

    #[test]
    fn civil_roundtrips() {
        for &(y, m, d) in &[(2000, 2, 29), (1999, 12, 31), (2024, 7, 5), (1600, 1, 1)] {
            let date = Date::from_ymd(y, m, d);
            let c = date.calendar();
            assert_eq!((c.year, c.month, c.day), (y, m, d));
        }
    }

    #[test]
    fn leap_years() {
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
        assert!(is_leap_year(2024));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
    }

    #[test]
    fn weekday_known() {
        // 2024-07-05 is a Friday (index 4).
        assert_eq!(Date::from_ymd(2024, 7, 5).weekday(), 4);
        // 1970-01-01 is Thursday (index 3).
        assert_eq!(Date::from_ymd(1970, 1, 1).weekday(), 3);
    }

    #[test]
    fn add_months_clamps() {
        let d = Date::from_ymd(2024, 1, 31);
        assert_eq!(d.add_months(1).to_string(), "2024-02-29");
        assert_eq!(d.add_months(13).to_string(), "2025-02-28");
    }

    #[test]
    fn parse_date() {
        assert_eq!(Date::parse("2024-07-05").unwrap(), Date::from_ymd(2024, 7, 5));
        assert!(Date::parse("2024-13-01").is_err());
        assert!(Date::parse("2023-02-29").is_err());
    }

    #[test]
    fn timestamp_parse_and_display() {
        let ts = Timestamp::parse("2024-07-05 13:30:00").unwrap();
        assert_eq!(ts.date(), Date::from_ymd(2024, 7, 5));
        assert_eq!(ts.time_of_day(), (13, 30, 0, 0));
        assert_eq!(ts.to_string(), "2024-07-05 13:30:00");
        let ts2 = Timestamp::parse("2024-07-05T00:00:00.500000").unwrap();
        assert_eq!(ts2.time_of_day(), (0, 0, 0, 500000));
    }

    #[test]
    fn timestamp_arithmetic() {
        let ts = Timestamp::from_parts(Date::from_ymd(2024, 1, 1), 0, 0, 0, 0);
        let later = ts.add_seconds(SECS_PER_DAY);
        assert_eq!(later.date(), Date::from_ymd(2024, 1, 2));
    }
}
