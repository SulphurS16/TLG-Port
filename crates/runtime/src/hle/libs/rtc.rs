//! libSceRtc: ticks are microseconds since 0001-01-01T00:00:00 UTC. The system
//! time zone reported to the title is UTC, so local time equals UTC.

use crate::hle::Registry;
use crate::{reg, sysv};

const UNIX_EPOCH_TICK: u64 = 62_135_596_800_000_000;
const ERR_INVALID: i32 = 0x8010_0002u32 as i32;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DateTime {
    year: u16,
    month: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    microsecond: u32,
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

pub fn now_tick() -> u64 {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    UNIX_EPOCH_TICK + d.as_micros() as u64
}

fn tick_to_dt(tick: u64) -> DateTime {
    let us = tick % 1_000_000;
    let secs = (tick / 1_000_000) as i64 - (UNIX_EPOCH_TICK / 1_000_000) as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    DateTime {
        year: y as u16,
        month: m as u16,
        day: d as u16,
        hour: (rem / 3600) as u16,
        minute: ((rem / 60) % 60) as u16,
        second: (rem % 60) as u16,
        microsecond: us as u32,
    }
}

fn dt_to_tick(dt: &DateTime) -> Option<u64> {
    if dt.month == 0 || dt.month > 12 || dt.day == 0 || dt.day > 31 || dt.hour > 23 || dt.minute > 59 || dt.second > 59 {
        return None;
    }
    let days = days_from_civil(dt.year as i64, dt.month as i64, dt.day as i64);
    let secs = days * 86400 + dt.hour as i64 * 3600 + dt.minute as i64 * 60 + dt.second as i64;
    let t = secs as i128 * 1_000_000 + UNIX_EPOCH_TICK as i128 + dt.microsecond as i128;
    u64::try_from(t).ok()
}

sysv! {
    fn sceRtcGetTickResolution() -> u32 { 1_000_000 }
    fn sceRtcGetCurrentNetworkTick(tick: *mut u64) -> i32 {
        if tick.is_null() { return ERR_INVALID; }
        *tick = now_tick();
        0
    }
    fn sceRtcGetCurrentTick(tick: *mut u64) -> i32 {
        if tick.is_null() { return ERR_INVALID; }
        *tick = now_tick();
        0
    }
    /// Current UTC time shifted by `tz_minutes`.
    fn sceRtcGetCurrentClock(dt: *mut DateTime, tz_minutes: i32) -> i32 {
        if dt.is_null() { return ERR_INVALID; }
        *dt = tick_to_dt((now_tick() as i64 + tz_minutes as i64 * 60_000_000) as u64);
        0
    }
    fn sceRtcGetCurrentClockLocalTime(dt: *mut DateTime) -> i32 {
        if dt.is_null() { return ERR_INVALID; }
        *dt = tick_to_dt(now_tick());
        0
    }
    fn sceRtcConvertUtcToLocalTime(utc: *const u64, local: *mut u64) -> i32 {
        if utc.is_null() || local.is_null() { return ERR_INVALID; }
        *local = *utc;
        0
    }
    fn sceRtcConvertLocalTimeToUtc(local: *const u64, utc: *mut u64) -> i32 {
        if utc.is_null() || local.is_null() { return ERR_INVALID; }
        *utc = *local;
        0
    }
    fn sceRtcGetTick(dt: *const DateTime, tick: *mut u64) -> i32 {
        if dt.is_null() || tick.is_null() { return ERR_INVALID; }
        match dt_to_tick(&*dt) { Some(t) => { *tick = t; 0 } None => ERR_INVALID }
    }
    fn sceRtcSetTick(dt: *mut DateTime, tick: *const u64) -> i32 {
        if dt.is_null() || tick.is_null() { return ERR_INVALID; }
        *dt = tick_to_dt(*tick);
        0
    }
    fn sceRtcSetTime_t(dt: *mut DateTime, t: i64) -> i32 {
        if dt.is_null() || t < 0 { return ERR_INVALID; }
        *dt = tick_to_dt(UNIX_EPOCH_TICK + t as u64 * 1_000_000);
        0
    }
    fn sceRtcGetDayOfWeek(year: i32, month: i32, day: i32) -> i32 {
        let days = days_from_civil(year as i64, month as i64, day as i64);
        (days + 4).rem_euclid(7) as i32 // 1970-01-01 was a Thursday
    }
    fn sceRtcFormatRFC2822LocalTime(buf: *mut u8, utc: *const u64) -> i32 {
        if buf.is_null() { return ERR_INVALID; }
        let t = if utc.is_null() { now_tick() } else { *utc };
        let dt = tick_to_dt(t);
        let days = days_from_civil(dt.year as i64, dt.month as i64, dt.day as i64);
        let wd = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"][days.rem_euclid(7) as usize];
        let mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][(dt.month - 1) as usize];
        let s = format!("{wd}, {:02} {mon} {:04} {:02}:{:02}:{:02} +0000", dt.day, dt.year, dt.hour, dt.minute, dt.second);
        crate::hle::write_cstr(buf, 64, &s);
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceRtc", [
        sceRtcGetTickResolution, sceRtcGetCurrentNetworkTick, sceRtcGetCurrentTick, sceRtcGetCurrentClockLocalTime,
        sceRtcConvertUtcToLocalTime, sceRtcConvertLocalTimeToUtc, sceRtcGetTick, sceRtcSetTick, sceRtcSetTime_t,
        sceRtcGetDayOfWeek, sceRtcFormatRFC2822LocalTime, sceRtcGetCurrentClock,
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let dt = DateTime { year: 2026, month: 10, day: 7, hour: 21, minute: 3, second: 4, microsecond: 5 };
        let t = dt_to_tick(&dt).unwrap();
        let back = tick_to_dt(t);
        assert_eq!((back.year, back.month, back.day, back.hour, back.minute, back.second, back.microsecond), (2026, 10, 7, 21, 3, 4, 5));
        let epoch = DateTime { year: 1970, month: 1, day: 1, ..Default::default() };
        assert_eq!(dt_to_tick(&epoch), Some(UNIX_EPOCH_TICK));
        let first = DateTime { year: 1, month: 1, day: 1, ..Default::default() };
        assert_eq!(dt_to_tick(&first), Some(0));
    }
}
