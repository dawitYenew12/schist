//! Date and timestamp scalar functions.
//!
//! Once a temporal column is decoded to a [`Date`] or [`Timestamp`], queries
//! apply the usual calendar functions: extract a field, truncate to a unit, add
//! an interval, compute a difference. These build on [`crate::temporal`] and
//! return plain integers or new temporal values, so they slot into the scalar
//! evaluation path alongside the numeric functions.

use crate::temporal::{Date, Timestamp};

/// A field extractable from a date or timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePart {
    Year,
    Month,
    Day,
    Weekday,
    DayOfYear,
    Quarter,
    Hour,
    Minute,
    Second,
}

impl DatePart {
    /// Parse a part name (case-insensitive).
    pub fn parse(name: &str) -> Option<DatePart> {
        Some(match name.to_ascii_lowercase().as_str() {
            "year" | "yr" => DatePart::Year,
            "month" | "mon" => DatePart::Month,
            "day" => DatePart::Day,
            "weekday" | "dow" => DatePart::Weekday,
            "doy" | "dayofyear" => DatePart::DayOfYear,
            "quarter" | "qtr" => DatePart::Quarter,
            "hour" => DatePart::Hour,
            "minute" | "min" => DatePart::Minute,
            "second" | "sec" => DatePart::Second,
            _ => return None,
        })
    }
}

/// Extract a field from a date (time fields return 0).
pub fn extract_date(date: Date, part: DatePart) -> i64 {
    let c = date.calendar();
    match part {
        DatePart::Year => c.year as i64,
        DatePart::Month => c.month as i64,
        DatePart::Day => c.day as i64,
        DatePart::Weekday => date.weekday() as i64,
        DatePart::DayOfYear => day_of_year(date) as i64,
        DatePart::Quarter => ((c.month - 1) / 3 + 1) as i64,
        DatePart::Hour | DatePart::Minute | DatePart::Second => 0,
    }
}

/// Extract a field from a timestamp.
pub fn extract_timestamp(ts: Timestamp, part: DatePart) -> i64 {
    match part {
        DatePart::Hour | DatePart::Minute | DatePart::Second => {
            let (h, m, s, _) = ts.time_of_day();
            match part {
                DatePart::Hour => h as i64,
                DatePart::Minute => m as i64,
                DatePart::Second => s as i64,
                _ => 0,
            }
        }
        _ => extract_date(ts.date(), part),
    }
}

/// The 1-based day of the year.
pub fn day_of_year(date: Date) -> u32 {
    let c = date.calendar();
    let year_start = Date::from_ymd(c.year, 1, 1);
    (year_start.days_until(&date) + 1) as u32
}

/// Truncate a date to the first day of its unit (`year`, `quarter`, `month`).
pub fn date_trunc(date: Date, unit: &str) -> Date {
    let c = date.calendar();
    match unit.to_ascii_lowercase().as_str() {
        "year" => Date::from_ymd(c.year, 1, 1),
        "quarter" => {
            let qmonth = ((c.month - 1) / 3) * 3 + 1;
            Date::from_ymd(c.year, qmonth, 1)
        }
        "month" => Date::from_ymd(c.year, c.month, 1),
        "week" => date.add_days(-(date.weekday() as i32)),
        _ => date,
    }
}

/// Add an interval expressed in a unit to a date.
pub fn date_add(date: Date, amount: i32, unit: &str) -> Date {
    match unit.to_ascii_lowercase().as_str() {
        "day" | "days" => date.add_days(amount),
        "week" | "weeks" => date.add_days(amount * 7),
        "month" | "months" => date.add_months(amount),
        "year" | "years" => date.add_months(amount * 12),
        _ => date,
    }
}

/// Whole days from `a` to `b`.
pub fn date_diff_days(a: Date, b: Date) -> i64 {
    a.days_until(&b) as i64
}

/// Whole months between two dates (calendar months, truncated).
pub fn date_diff_months(a: Date, b: Date) -> i64 {
    let (ca, cb) = (a.calendar(), b.calendar());
    let mut months = (cb.year as i64 - ca.year as i64) * 12 + (cb.month as i64 - ca.month as i64);
    if cb.day < ca.day {
        months -= 1;
    }
    months
}

/// The last day of the month containing `date`.
pub fn last_day_of_month(date: Date) -> Date {
    let c = date.calendar();
    let next_month = if c.month == 12 {
        Date::from_ymd(c.year + 1, 1, 1)
    } else {
        Date::from_ymd(c.year, c.month + 1, 1)
    };
    next_month.add_days(-1)
}

/// `true` if the date falls on a weekend (Saturday or Sunday).
pub fn is_weekend(date: Date) -> bool {
    date.weekday() >= 5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_fields() {
        let d = Date::from_ymd(2024, 7, 5);
        assert_eq!(extract_date(d, DatePart::Year), 2024);
        assert_eq!(extract_date(d, DatePart::Month), 7);
        assert_eq!(extract_date(d, DatePart::Quarter), 3);
        assert_eq!(extract_date(d, DatePart::Day), 5);
    }

    #[test]
    fn extract_time_from_timestamp() {
        let ts = Timestamp::parse("2024-07-05 13:45:30").unwrap();
        assert_eq!(extract_timestamp(ts, DatePart::Hour), 13);
        assert_eq!(extract_timestamp(ts, DatePart::Minute), 45);
        assert_eq!(extract_timestamp(ts, DatePart::Year), 2024);
    }

    #[test]
    fn day_of_year_computed() {
        assert_eq!(day_of_year(Date::from_ymd(2024, 1, 1)), 1);
        assert_eq!(day_of_year(Date::from_ymd(2024, 12, 31)), 366); // leap year
        assert_eq!(day_of_year(Date::from_ymd(2023, 12, 31)), 365);
    }

    #[test]
    fn truncation() {
        let d = Date::from_ymd(2024, 7, 15);
        assert_eq!(date_trunc(d, "month"), Date::from_ymd(2024, 7, 1));
        assert_eq!(date_trunc(d, "quarter"), Date::from_ymd(2024, 7, 1));
        assert_eq!(date_trunc(d, "year"), Date::from_ymd(2024, 1, 1));
    }

    #[test]
    fn arithmetic() {
        let d = Date::from_ymd(2024, 1, 31);
        assert_eq!(date_add(d, 1, "month"), Date::from_ymd(2024, 2, 29));
        assert_eq!(date_add(d, 1, "day"), Date::from_ymd(2024, 2, 1));
        assert_eq!(date_diff_days(Date::from_ymd(2024, 1, 1), Date::from_ymd(2024, 1, 31)), 30);
        assert_eq!(date_diff_months(Date::from_ymd(2024, 1, 15), Date::from_ymd(2024, 4, 10)), 2);
    }

    #[test]
    fn last_day_and_weekend() {
        assert_eq!(last_day_of_month(Date::from_ymd(2024, 2, 1)), Date::from_ymd(2024, 2, 29));
        assert_eq!(last_day_of_month(Date::from_ymd(2023, 2, 15)), Date::from_ymd(2023, 2, 28));
        // 2024-07-06 is a Saturday.
        assert!(is_weekend(Date::from_ymd(2024, 7, 6)));
        assert!(!is_weekend(Date::from_ymd(2024, 7, 5)));
    }

    #[test]
    fn part_parsing() {
        assert_eq!(DatePart::parse("YEAR"), Some(DatePart::Year));
        assert_eq!(DatePart::parse("dow"), Some(DatePart::Weekday));
        assert_eq!(DatePart::parse("nonsense"), None);
    }
}
