//! Minimal process-wide logger: buffered file sink plus stderr for warnings/errors.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

pub static LEVEL: AtomicU8 = AtomicU8::new(Level::Info as u8);
/// When set, every HLE call is logged with its arguments and result.
pub static TRACE_HLE: AtomicBool = AtomicBool::new(false);

static SINK: Mutex<Option<BufWriter<File>>> = Mutex::new(None);
static START: OnceLock<Instant> = OnceLock::new();

thread_local! {
    static THREAD_NAME: RefCell<String> = RefCell::new(String::new());
}

pub fn init(path: &Path) -> std::io::Result<()> {
    START.get_or_init(Instant::now);
    let f = File::create(path)?;
    *SINK.lock() = Some(BufWriter::with_capacity(1 << 20, f));
    Ok(())
}

pub fn set_thread_name(name: &str) {
    THREAD_NAME.with(|n| *n.borrow_mut() = name.to_string());
    crate::sampler::register_current_thread(name);
}

pub fn thread_name() -> String {
    THREAD_NAME.with(|n| {
        let n = n.borrow();
        if n.is_empty() {
            std::thread::current().name().unwrap_or("?").to_string()
        } else {
            n.clone()
        }
    })
}

#[inline]
pub fn enabled(level: Level) -> bool {
    level as u8 <= LEVEL.load(Ordering::Relaxed)
}

/// When logging started (the run's time origin).
pub fn start_instant() -> Instant {
    *START.get_or_init(Instant::now)
}

pub fn elapsed_ms() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

pub fn write(level: Level, args: fmt::Arguments) {
    let tag = match level {
        Level::Error => "E",
        Level::Warn => "W",
        Level::Info => "I",
        Level::Debug => "D",
        Level::Trace => "T",
    };
    let line = format!("[{:>10.3}][{}] {}: {}\n", elapsed_ms(), thread_name(), tag, args);
    if level <= Level::Warn {
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    let mut sink = SINK.lock();
    if let Some(w) = sink.as_mut() {
        let _ = w.write_all(line.as_bytes());
        if level <= Level::Warn {
            let _ = w.flush();
        }
    } else if level > Level::Warn {
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
}

pub fn flush() {
    if let Some(w) = SINK.lock().as_mut() {
        let _ = w.flush();
    }
}

/// Flush without blocking if another thread holds the sink (used from crash handlers).
pub fn try_flush() {
    if let Some(mut g) = SINK.try_lock() {
        if let Some(w) = g.as_mut() {
            let _ = w.flush();
        }
    }
}

#[macro_export]
macro_rules! error { ($($t:tt)*) => { $crate::log::write($crate::log::Level::Error, format_args!($($t)*)) } }
#[macro_export]
macro_rules! warn { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Warn) { $crate::log::write($crate::log::Level::Warn, format_args!($($t)*)) } } }
#[macro_export]
macro_rules! info { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Info) { $crate::log::write($crate::log::Level::Info, format_args!($($t)*)) } } }
#[macro_export]
macro_rules! debug { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Debug) { $crate::log::write($crate::log::Level::Debug, format_args!($($t)*)) } } }
#[macro_export]
macro_rules! trace { ($($t:tt)*) => { if $crate::log::enabled($crate::log::Level::Trace) { $crate::log::write($crate::log::Level::Trace, format_args!($($t)*)) } } }
