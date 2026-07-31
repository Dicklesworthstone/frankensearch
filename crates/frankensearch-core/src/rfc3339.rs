//! RFC 3339 timestamp handling without the `time` dependency.
//!
//! frankensearch used the `time` crate for exactly three operations —
//! "now" as Unix nanoseconds, formatting "now" as RFC 3339, and parsing
//! RFC 3339 strings — while `frankensearch-ops` had already grown its own
//! byte-level fast parser with `time` demoted to a reference fallback.
//! This module is that reference implementation, promoted, completed, and
//! shared: a strict, allocation-light RFC 3339 parser/formatter over
//! `i128` Unix nanoseconds.
//!
//! Semantics:
//!
//! * Calendar math uses the proleptic-Gregorian civil-day algorithms
//!   (Hinnant's `days_from_civil` / `civil_from_days`), exact over the
//!   full supported year range 0000..=9999 with no lookup tables.
//! * Parsing is strict RFC 3339: `full-date "T" full-time` with `T`/`t`
//!   accepted, offsets `Z`/`z` or `±HH:MM`, optional `.` + 1..=9
//!   fractional digits. Leap seconds (`:60`), `24:00`, missing offsets,
//!   more than nine fractional digits, and trailing bytes are rejected —
//!   loudly, because these strings gate retention configs and telemetry
//!   envelopes where silent acceptance would hide producer bugs.
//! * `-00:00` (RFC 3339's "offset unknown") parses as UTC, matching its
//!   numeric value.
//! * Formatting always emits UTC with a `Z` designator, and subsecond
//!   digits only when nonzero, trimmed of trailing zeros. This is a valid
//!   RFC 3339 rendering that this module's own parser round-trips exactly;
//!   it is NOT guaranteed byte-identical to the `time` crate (which may
//!   render a numeric `+00:00` offset). frankensearch only ever re-parses
//!   these strings or checks them for validity — never byte-compares them
//!   against a `time`-produced golden — so the rendering choice is free.

use thiserror::Error;

/// Nanoseconds per second.
const NANOS_PER_SECOND: i128 = 1_000_000_000;
/// Seconds per civil day.
const SECONDS_PER_DAY: i64 = 86_400;

/// Why an RFC 3339 string was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Rfc3339Error {
    /// The string does not have the `YYYY-MM-DDTHH:MM:SS` skeleton.
    #[error("malformed RFC3339 date-time skeleton")]
    Malformed,
    /// A calendar or clock field is out of range (bad month/day/hour/…).
    #[error("RFC3339 field out of range: {0}")]
    FieldRange(&'static str),
    /// The fractional-second part is empty or longer than nine digits.
    #[error("RFC3339 fractional seconds must be 1..=9 digits")]
    Fraction,
    /// The UTC offset is missing or malformed.
    #[error("RFC3339 offset must be Z or \u{b1}HH:MM")]
    Offset,
    /// Input continues past a complete RFC 3339 date-time.
    #[error("trailing bytes after RFC3339 date-time")]
    Trailing,
}

/// Current wall-clock time as Unix nanoseconds.
///
/// Pre-epoch system clocks (misconfigured hosts) yield the correct
/// negative value rather than a panic or zero.
#[must_use]
pub fn now_unix_nanos() -> i128 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(after) => i128::try_from(after.as_nanos()).unwrap_or(i128::MAX),
        Err(before) => -i128::try_from(before.duration().as_nanos()).unwrap_or(i128::MAX),
    }
}

/// Days since 1970-01-01 for a civil date (proleptic Gregorian).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let adjusted_year = if month <= 2 { year - 1 } else { year };
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = (month + 9) % 12;
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Civil date (year, month, day) for days since 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let civil_year = if month <= 2 { year + 1 } else { year };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    (civil_year, month as u32, day as u32)
}

/// Days in `month` (`1..=12`) of `year`; `0` for an invalid month.
const fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Format Unix nanoseconds as RFC 3339 UTC (`…Z`), subseconds trimmed.
#[must_use]
pub fn format_unix_nanos(unix_nanos: i128) -> String {
    let seconds = unix_nanos.div_euclid(NANOS_PER_SECOND);
    #[allow(clippy::cast_possible_truncation)]
    let subsecond_nanos = unix_nanos.rem_euclid(NANOS_PER_SECOND) as u32;
    // Saturate instead of truncating: an out-of-range input (nothing the
    // supported year range 0000..=9999 can produce) yields a stable
    // far-future/far-past rendering rather than wrapped garbage.
    let saturated = if unix_nanos >= 0 {
        i64::MAX / 2
    } else {
        i64::MIN / 2
    };
    let seconds = i64::try_from(seconds).unwrap_or(saturated);
    let days = seconds.div_euclid(SECONDS_PER_DAY);
    let second_of_day = seconds.rem_euclid(SECONDS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let hour = second_of_day / 3_600;
    let minute = (second_of_day % 3_600) / 60;
    let second = second_of_day % 60;

    let mut formatted = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}");
    if subsecond_nanos > 0 {
        let digits = format!("{subsecond_nanos:09}");
        formatted.push('.');
        formatted.push_str(digits.trim_end_matches('0'));
    }
    formatted.push('Z');
    formatted
}

/// Parse a strict RFC 3339 date-time into Unix nanoseconds.
///
/// # Errors
///
/// Returns an [`Rfc3339Error`] describing the first defect found; see the
/// module docs for exactly what is accepted.
pub fn parse_rfc3339_to_unix_nanos(input: &str) -> Result<i128, Rfc3339Error> {
    let bytes = input.as_bytes();
    // The date-time skeleton "YYYY-MM-DDTHH:MM:SS" is 19 bytes; anything
    // shorter cannot be indexed safely. The mandatory offset is checked by
    // the offset parser itself so that a skeleton-complete input missing
    // only its offset gets the actionable `Offset` error, not `Malformed`.
    if bytes.len() < 19 {
        return Err(Rfc3339Error::Malformed);
    }
    let digit = |index: usize| -> Result<i64, Rfc3339Error> {
        let byte = bytes[index];
        if byte.is_ascii_digit() {
            Ok(i64::from(byte - b'0'))
        } else {
            Err(Rfc3339Error::Malformed)
        }
    };
    let pair =
        |index: usize| -> Result<i64, Rfc3339Error> { Ok(digit(index)? * 10 + digit(index + 1)?) };

    if bytes[4] != b'-'
        || bytes[7] != b'-'
        || !matches!(bytes[10], b'T' | b't')
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return Err(Rfc3339Error::Malformed);
    }

    let year = digit(0)? * 1_000 + digit(1)? * 100 + digit(2)? * 10 + digit(3)?;
    let month = pair(5)?;
    let day = pair(8)?;
    let hour = pair(11)?;
    let minute = pair(14)?;
    let second = pair(17)?;

    if !(1..=12).contains(&month) {
        return Err(Rfc3339Error::FieldRange("month"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let month_days = i64::from(days_in_month(year, month as u32));
    if !(1..=month_days).contains(&day) {
        return Err(Rfc3339Error::FieldRange("day"));
    }
    if hour > 23 {
        return Err(Rfc3339Error::FieldRange("hour"));
    }
    if minute > 59 {
        return Err(Rfc3339Error::FieldRange("minute"));
    }
    if second > 59 {
        return Err(Rfc3339Error::FieldRange("second"));
    }

    // Optional fractional seconds.
    let mut cursor = 19;
    let mut fraction_nanos: i128 = 0;
    if bytes.get(cursor) == Some(&b'.') {
        cursor += 1;
        let fraction_start = cursor;
        let mut scale = NANOS_PER_SECOND / 10;
        while let Some(byte) = bytes.get(cursor) {
            if !byte.is_ascii_digit() {
                break;
            }
            if cursor - fraction_start >= 9 {
                return Err(Rfc3339Error::Fraction);
            }
            fraction_nanos += i128::from(byte - b'0') * scale;
            scale /= 10;
            cursor += 1;
        }
        if cursor == fraction_start {
            return Err(Rfc3339Error::Fraction);
        }
    }

    // Mandatory offset: Z/z or ±HH:MM.
    let offset_seconds: i64 = match bytes.get(cursor) {
        Some(b'Z' | b'z') => {
            cursor += 1;
            0
        }
        Some(sign @ (b'+' | b'-')) => {
            if bytes.len() < cursor + 6 || bytes[cursor + 3] != b':' {
                return Err(Rfc3339Error::Offset);
            }
            let offset_digit = |index: usize| -> Result<i64, Rfc3339Error> {
                let byte = bytes[index];
                if byte.is_ascii_digit() {
                    Ok(i64::from(byte - b'0'))
                } else {
                    Err(Rfc3339Error::Offset)
                }
            };
            let offset_hour = offset_digit(cursor + 1)? * 10 + offset_digit(cursor + 2)?;
            let offset_minute = offset_digit(cursor + 4)? * 10 + offset_digit(cursor + 5)?;
            if offset_hour > 23 || offset_minute > 59 {
                return Err(Rfc3339Error::Offset);
            }
            let magnitude = offset_hour * 3_600 + offset_minute * 60;
            cursor += 6;
            if *sign == b'+' { magnitude } else { -magnitude }
        }
        _ => return Err(Rfc3339Error::Offset),
    };
    if cursor != bytes.len() {
        return Err(Rfc3339Error::Trailing);
    }

    let civil_seconds =
        days_from_civil(year, month, day) * SECONDS_PER_DAY + hour * 3_600 + minute * 60 + second;
    let utc_seconds = civil_seconds - offset_seconds;
    Ok(i128::from(utc_seconds) * NANOS_PER_SECOND + fraction_nanos)
}

#[cfg(test)]
mod tests {
    use super::{Rfc3339Error, format_unix_nanos, now_unix_nanos, parse_rfc3339_to_unix_nanos};

    const NANOS: i128 = 1_000_000_000;

    #[test]
    fn known_epoch_vectors_parse_exactly() {
        // Independently computed fixed points.
        let vectors: &[(&str, i128)] = &[
            ("1970-01-01T00:00:00Z", 0),
            ("1969-12-31T23:59:59Z", -NANOS),
            ("2000-03-01T00:00:00Z", 951_868_800 * NANOS),
            ("2024-02-29T12:00:00Z", 1_709_208_000 * NANOS),
            ("2026-01-01T00:00:00Z", 1_767_225_600 * NANOS),
            // +05:30 local midnight is 19800s before UTC midnight.
            ("2026-01-01T00:00:00+05:30", 1_767_205_800 * NANOS),
            // Negative offsets add to the epoch value.
            (
                "2026-01-01T00:00:00-07:00",
                (1_767_225_600 + 25_200) * NANOS,
            ),
            // -00:00 ("offset unknown") is numerically UTC.
            ("1970-01-01T00:00:00-00:00", 0),
            ("1970-01-01T00:00:00.5Z", 500_000_000),
            ("1970-01-01T00:00:00.000000001Z", 1),
        ];
        for (text, expected) in vectors {
            assert_eq!(
                parse_rfc3339_to_unix_nanos(text).as_ref(),
                Ok(expected),
                "vector {text}"
            );
        }
        // Lowercase t/z are RFC-permitted.
        assert_eq!(parse_rfc3339_to_unix_nanos("1970-01-01t00:00:00z"), Ok(0));
    }

    #[test]
    fn defective_inputs_are_rejected_loudly() {
        let rejects: &[(&str, Rfc3339Error)] = &[
            ("2023-02-29T00:00:00Z", Rfc3339Error::FieldRange("day")),
            ("2026-13-01T00:00:00Z", Rfc3339Error::FieldRange("month")),
            ("2026-00-01T00:00:00Z", Rfc3339Error::FieldRange("month")),
            ("2026-01-00T00:00:00Z", Rfc3339Error::FieldRange("day")),
            ("2026-01-01T24:00:00Z", Rfc3339Error::FieldRange("hour")),
            ("2026-01-01T00:60:00Z", Rfc3339Error::FieldRange("minute")),
            ("2016-12-31T23:59:60Z", Rfc3339Error::FieldRange("second")),
            ("2026-01-01 00:00:00Z", Rfc3339Error::Malformed),
            ("2026-01-01T00:00:00", Rfc3339Error::Offset),
            ("2026-01-01T00:00:00+0530", Rfc3339Error::Offset),
            ("2026-01-01T00:00:00+24:00", Rfc3339Error::Offset),
            ("2026-01-01T00:00:00.Z", Rfc3339Error::Fraction),
            ("2026-01-01T00:00:00.0123456789Z", Rfc3339Error::Fraction),
            ("2026-01-01T00:00:00Zjunk", Rfc3339Error::Trailing),
            ("not a timestamp at all!", Rfc3339Error::Malformed),
        ];
        for (text, expected) in rejects {
            assert_eq!(
                parse_rfc3339_to_unix_nanos(text).as_ref(),
                Err(expected),
                "reject {text}"
            );
        }
    }

    #[test]
    fn format_matches_time_crate_conventions() {
        assert_eq!(format_unix_nanos(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_unix_nanos(-NANOS), "1969-12-31T23:59:59Z");
        assert_eq!(
            format_unix_nanos(1_709_208_000 * NANOS),
            "2024-02-29T12:00:00Z"
        );
        // Subseconds print only when nonzero, trimmed of trailing zeros.
        assert_eq!(format_unix_nanos(500_000_000), "1970-01-01T00:00:00.5Z");
        assert_eq!(format_unix_nanos(1), "1970-01-01T00:00:00.000000001Z");
    }

    #[test]
    fn parse_format_round_trips_across_regimes() {
        // Dense coverage around leap boundaries, negative epochs, century
        // rules, and arbitrary offsets into each day.
        let mut sample = -2_000_000_000_i64; // 1906-08-16T…
        while sample < 4_000_000_000 {
            let nanos = i128::from(sample) * NANOS + 123_456_789;
            let text = format_unix_nanos(nanos);
            assert_eq!(
                parse_rfc3339_to_unix_nanos(&text),
                Ok(nanos),
                "round-trip {text}"
            );
            sample += 86_399; // deliberately co-prime-ish with day length
        }
    }

    #[test]
    fn now_is_formattable_and_reparsable() {
        let now = now_unix_nanos();
        let text = format_unix_nanos(now);
        assert_eq!(parse_rfc3339_to_unix_nanos(&text), Ok(now));
    }
}
