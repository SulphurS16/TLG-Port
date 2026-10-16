//! tlg: runs the PS4 The Last Guardian executable natively on Windows x86-64
//! with high-level emulation of the PS4 system libraries.
//!
//! A windowed (GUI subsystem) program: no console window. Development runs
//! redirect stdout/stderr (tools/run.ps1, drive.ps1); the command-line tools
//! (--decode-at9, --probe-media) attach to the starting console.

#![windows_subsystem = "windows"]

#[macro_use]
mod log;
mod audio;
mod config;
mod debug_mem;
mod sampler;
mod tune;
mod ehframe;
mod envmon;
mod exception;
mod gpu;
mod hle;
mod input;
mod loader;
mod mem;
mod pacing;
mod padhid;
mod patch;
mod sse4a;
mod thread;
mod title_patches;
mod tls;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Config {
    /// Where this build keeps settings, saves, caches and logs.
    pub paths: config::Paths,
    /// Game folders from the command line (else settings.txt, else the defaults).
    pub game: Option<PathBuf>,
    pub update: Option<PathBuf>,
    pub no_update: bool,
    pub seconds: Option<u64>,
    /// Development control channel (see `poll_control`).
    pub control: Option<PathBuf>,
    pub trace: bool,
    pub lenient: bool,
    pub debug: bool,
    pub headless: bool,
    /// Default settings and bindings, the files left alone.
    pub safe_mode: bool,
    /// Started by the launcher (`--launcher`), which reports crashes itself.
    pub launcher: bool,
    /// Player-facing message boxes (off with `--no-dialogs` and in scripted
    /// or development runs).
    pub dialogs: bool,
}

static RUN_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// This run's directory (logs, screenshots, profiles).
pub fn run_dir() -> Option<&'static Path> {
    RUN_DIR.get().map(|p| p.as_path())
}

/// The user settings directory: `input.txt`, `settings.txt`, saves.
pub fn data_dir() -> Option<&'static Path> {
    config::try_paths().map(|p| p.user.as_path())
}

fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Civil-from-days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m, d, rem / 3600, (rem / 60) % 60, rem % 60)
}

fn parse_args() -> Config {
    let mut paths = config::Paths::current();
    let mut game: Option<PathBuf> = None;
    let mut update: Option<PathBuf> = None;
    let mut no_update = false;
    let mut seconds = None;
    let mut trace = false;
    let mut lenient = false;
    let mut debug = false;
    let mut headless = false;
    let mut control = None;
    let mut safe_mode = false;
    let mut launcher = false;
    let mut no_dialogs = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--probe-media" => {
                hle::libs::avplayer::probe(Path::new(&args.next().expect("--probe-media needs a file")));
                std::process::exit(0);
            }
            "--decode-at9" => {
                let i = args.next().expect("--decode-at9 <in.at9> <out.wav>");
                let o = args.next().expect("--decode-at9 <in.at9> <out.wav>");
                match hle::libs::atrac9::decode_file(Path::new(&i), Path::new(&o)) {
                    Ok(()) => std::process::exit(0),
                    Err(e) => {
                        eprintln!("{e}");
                        std::process::exit(1)
                    }
                }
            }
            "--seconds" => seconds = args.next().and_then(|s| s.parse().ok()),
            "--trace" => trace = true,
            "--lenient" => lenient = true,
            "--debug" => debug = true,
            "--headless" => headless = true,
            // A development checkout's root (its `local\` holds everything).
            "--root" => {
                let local = PathBuf::from(args.next().expect("--root needs a path")).join("local");
                paths = config::Paths { layout: config::Layout::Dev, exe_dir: paths.exe_dir.clone(), user: local.clone(), cache: local.clone(), game_root: local.join("game") };
            }
            "--game" => game = Some(PathBuf::from(args.next().expect("--game needs a path"))),
            "--update" => update = Some(PathBuf::from(args.next().expect("--update needs a path"))),
            // The base application alone (01.00), ignoring a prepared update overlay.
            "--no-update" => no_update = true,
            "--control" => control = Some(PathBuf::from(args.next().expect("--control needs a path"))),
            "--safe-mode" => safe_mode = true,
            "--launcher" => launcher = true,
            "--no-dialogs" => no_dialogs = true,
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let dialogs = !no_dialogs && !headless && control.is_none() && seconds.is_none() && paths.layout != config::Layout::Dev;
    Config { paths, game, update, no_update, seconds, trace, lenient, debug, headless, control, safe_mode, launcher, dialogs }
}

/// The folders mounted at /app0, highest priority first: the update (01.03)
/// over the base application (01.00). Command line, then settings.txt, then
/// the layout's default folders.
fn app_dirs(cfg: &Config, s: &config::Settings) -> Vec<PathBuf> {
    let base = cfg.game.clone().or_else(|| s.game_dir.clone()).unwrap_or_else(|| cfg.paths.default_base());
    let update = match (cfg.no_update, cfg.update.clone().or_else(|| s.update_dir.clone())) {
        (true, _) => None,
        (false, Some(u)) => Some(u),
        (false, None) => Some(cfg.paths.default_update()).filter(|u| u.join("eboot.bin").is_file()),
    };
    update.into_iter().chain([base]).collect()
}

/// Only one game process at a time (both would write the same saves).
fn single_instance() -> bool {
    use windows_sys::Win32::Foundation::{ERROR_ALREADY_EXISTS, GetLastError};
    use windows_sys::Win32::System::Threading::CreateMutexW;
    let name: Vec<u16> = "Local\\tlg-the-last-guardian-runtime".encode_utf16().chain([0]).collect();
    let h = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    // (The handle stays open for the process lifetime.)
    !h.is_null() && unsafe { GetLastError() } != ERROR_ALREADY_EXISTS
}

/// Per-monitor DPI awareness (v2): the window's pixels are the screen's,
/// not scaled by Windows.
fn dpi_aware() {
    use windows_sys::Win32::UI::HiDpi::{DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext};
    unsafe {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }
}

/// Keeps the newest run directories (and any containing a KEEP marker).
fn prune_runs(runs: &Path, keep: usize) {
    let Ok(rd) = std::fs::read_dir(runs) else { return };
    let mut dirs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    dirs.sort();
    if dirs.len() <= keep {
        return;
    }
    let n = dirs.len() - keep;
    for d in dirs.into_iter().take(n) {
        if !d.join("KEEP").exists() {
            let _ = std::fs::remove_dir_all(&d);
        }
    }
}

/// Development control channel (`--control <file>`): executes complete lines
/// appended to the file since the last poll.
///   press <buttons> [hold]   e.g. `press down`, `press cross 10`, `press ls_up+r1 30`
///   shot <name>              screenshot of the next presented flip -> shot_<name>.png
///   dump                     dump all GPU images at the next flip
///   trace                    log every draw/dispatch of an upcoming flip (FRAME lines)
///   loglevel trace|debug|info  change logging at run time (trace = every HLE call)
///   tune <name> <value>      change a run-time tunable (see `tune.rs`)
///   mark <text>              log a marker (scripts/session/ab.sh)
///   quit                     orderly shutdown
fn poll_control(path: &Path, offset: &mut u64) {
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else { return };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len < *offset {
        *offset = 0; // truncated: start over
    }
    if len == *offset || f.seek(SeekFrom::Start(*offset)).is_err() {
        return;
    }
    let mut buf = String::new();
    if f.read_to_string(&mut buf).is_err() {
        return;
    }
    let Some(end) = buf.rfind('\n') else { return };
    *offset += end as u64 + 1;
    for line in buf[..end].lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        let flip = gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let result = match cmd {
            "press" => {
                let names = parts.next().unwrap_or("");
                let hold = parts.next().and_then(|h| h.parse().ok()).unwrap_or(6);
                input::press_now(names, hold)
            }
            "shot" => {
                let name: String = parts.next().unwrap_or("x").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                *hle::libs::videoout::SHOT_REQUEST.lock() = Some(name);
                Ok(())
            }
            "dump" => {
                gpu::DUMP_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "report" => {
                gpu::REPORT_REQUEST.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "trace" => {
                gpu::render::TRACE_REQUEST.store(flip + 2, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "traceshot" => {
                // Traces the frame built after flip+2 and screenshots the image
                // that frame displays (its flip is tagged flip+3).
                let name: String = parts.next().unwrap_or("ts").chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
                gpu::render::TRACE_REQUEST.store(flip + 2, std::sync::atomic::Ordering::Relaxed);
                *hle::libs::videoout::SHOT_FRAME.lock() = Some((flip + 3, name.clone()));
                info!("traceshot {name}: tracing frame {} (displayed at flip tag {})", flip + 2, flip + 3);
                Ok(())
            }
            "break" => {
                // break <hex main-module offset>: a logging breakpoint there.
                match (u64::from_str_radix(parts.next().unwrap_or("").trim_start_matches("0x"), 16), loader::main_module()) {
                    (Ok(off), Some(m)) => exception::install_breakpoint(m.base + off, m.base).map(|()| info!("breakpoint at {}", loader::describe_addr(m.base + off))),
                    (Err(_), _) => Err("break: expected a hex offset".into()),
                    (_, None) => Err("break: no main module".into()),
                }
            }
            "breakpeek" => {
                // breakpeek <expr,...>: values logged with each breakpoint hit
                // (TLG_BREAK_PEEK syntax: rbx, rsi+c24, rdi*+10).
                *exception::BREAK_PEEK.lock() = Some(parts.collect::<Vec<_>>().join(","));
                Ok(())
            }
            "mouse" => {
                // mouse <dx> <dy> [ms]: raw counts as if the captured mouse moved.
                let mut n = || parts.next().and_then(|v| v.parse::<i64>().ok());
                match (n(), n(), n()) {
                    (Some(dx), Some(dy), ms) => {
                        gpu::vk::window::inject_mouse(dx, dy, ms.unwrap_or(0).max(0) as u64);
                        Ok(())
                    }
                    _ => Err("mouse: expected <dx> <dy> [ms]".into()),
                }
            }
            "hwwatch" => {
                // hwwatch <addr> [len] [w] | hwwatch off (see exception.rs).
                match parts.next().unwrap_or("") {
                    "off" => {
                        exception::hw_watch_off();
                        Ok(())
                    }
                    a => {
                        let len = parts.next().and_then(|v| v.parse().ok()).unwrap_or(4);
                        let write_only = parts.next() == Some("w");
                        debug_mem::parse_addr(a).and_then(|addr| exception::hw_watch(addr, len, write_only))
                    }
                }
            }
            "breaklog" => {
                let n = parts.next().and_then(|v| v.parse().ok()).unwrap_or(1000);
                exception::BREAK_VERBOSE.store(n, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "loglevel" => {
                let (lvl, hle) = match parts.next().unwrap_or("") {
                    "trace" => (log::Level::Debug, true),
                    "debug" => (log::Level::Debug, false),
                    "info" => (log::Level::Info, false),
                    other => {
                        warn!("control: unknown log level '{other}'");
                        (log::Level::Info, false)
                    }
                };
                log::LEVEL.store(lvl as u8, std::sync::atomic::Ordering::Relaxed);
                log::TRACE_HLE.store(hle, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "ime" => {
                // Text for a pending system IME dialog (e.g. the character name).
                let text = line.trim_start().strip_prefix("ime").unwrap_or("").trim().to_string();
                *hle::libs::ime_dialog::CONTROL_TEXT.lock() = Some(text);
                Ok(())
            }
            "pressms" => {
                let names = parts.next().unwrap_or("");
                let ms = parts.next().and_then(|h| h.parse().ok()).unwrap_or(100);
                input::press_ms(names, ms)
            }
            "peek" => {
                // Debug: `peek <addr> [bytes]`, addr like 0x10a0, main+55404f8
                // or [main+55404f8]+8 ([..] reads a pointer).
                let a = parts.next().unwrap_or("");
                let n = parts.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(64).min(4096);
                debug_mem::peek(a, n)
            }
            "poke" => {
                // Debug: `poke <addr> <hex u32>...` (code pages included).
                let a = parts.next().unwrap_or("").to_string();
                let words: Vec<String> = parts.map(|s| s.to_string()).collect();
                debug_mem::poke(&a, &words)
            }
            "vmas" => {
                // Debug: guest address space layout.
                let m = crate::mem::MEM.lock();
                let mut s = format!("{} guest mappings:", m.vmas.len());
                for v in m.vmas.values() {
                    s.push_str(&format!("\n  {:#014x}-{:#014x} {:>10} KiB {:?} prot {:#x} phys {:#x} '{}'", v.start, v.end(), v.size >> 10, v.kind, v.prot, v.phys, v.name));
                }
                drop(m);
                info!("{s}");
                Ok(())
            }
            "mark" => {
                pacing::mark(parts.next().unwrap_or("?"));
                Ok(())
            }
            "threads" => {
                thread::dump_threads("control");
                Ok(())
            }
            "cpu" => {
                thread::cpu_report();
                Ok(())
            }
            "camlog" => {
                // camlog <flips>: the game camera at each displayed flip.
                title_patches::camlog(parts.next().and_then(|v| v.parse().ok()).unwrap_or(60));
                Ok(())
            }
            "display" => {
                // display windowed|borderless|fullscreen|toggle: as Alt+Enter / settings.
                let m = match parts.next().unwrap_or("") {
                    "windowed" => Some(Some(config::DisplayMode::Windowed)),
                    "borderless" => Some(Some(config::DisplayMode::Borderless)),
                    "fullscreen" => Some(Some(config::DisplayMode::Fullscreen)),
                    "toggle" => Some(None),
                    _ => None,
                };
                match m {
                    Some(m) => gpu::vk::window::request_mode(m),
                    None => Err("display: windowed, borderless, fullscreen or toggle".into()),
                }
            }
            "sample" => {
                let ms = parts.next().and_then(|v| v.parse().ok()).unwrap_or(3000);
                thread::sample_threads(ms);
                Ok(())
            }
            "tune" => {
                let name = parts.next().unwrap_or("");
                let r = tune::set(name, parts.next().unwrap_or(""));
                if r.is_ok() && name == "guest_prio" {
                    thread::reapply_guest_priorities();
                }
                r
            }
            "dumpafter" => {
                // dumpafter <ps,...> [max]: in traced frames, dump the targets
                // of draws with these pixel shaders right after them.
                let ps = gpu::render::parse_hex_list(parts.next().unwrap_or(""));
                let max = parts.next().and_then(|v| v.parse().ok()).unwrap_or(12);
                *gpu::render::DUMP_AFTER.lock() = Some((ps, max, 0));
                Ok(())
            }
            "skipps" => {
                // skipps <ps,...>: skip draws with these pixel shaders ("skipps -" clears).
                *gpu::render::SKIP_PS.lock() = gpu::render::parse_hex_list(parts.next().unwrap_or(""));
                Ok(())
            }
            "skipcs" => {
                // skipcs <cs,...>: skip dispatches of these compute shaders ("skipcs -" clears).
                *gpu::render::SKIP_CS.lock() = gpu::render::parse_hex_list(parts.next().unwrap_or(""));
                Ok(())
            }
            "tracefull" => {
                // tracefull <shader,...>: traces show these shaders' buffers whole.
                *gpu::render::TRACE_FULL_ADDRS.lock() = Some(gpu::render::parse_hex_list(parts.next().unwrap_or("")));
                Ok(())
            }
            "dumpraw" => {
                // dumpraw 1|0: image dumps also write the raw texels (.bin).
                gpu::render::DUMP_RAW.store(parts.next() == Some("1"), std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            "psdebug" => {
                // psdebug <ps>:<pc>:v<N>[:<pc>:v<M>][;...]: that pixel shader exports
                // v[N..N+2] as captured after pc instead of its colors ("psdebug -" clears).
                *gpu::shaders::DEBUG_CAPTURES.lock() = Some(gpu::shaders::parse_debug_capture(parts.next().unwrap_or("")));
                Ok(())
            }
            "quit" => {
                info!("control: quit at flip {flip}");
                shutdown(0);
            }
            _ => Err(format!("unknown command '{cmd}'")),
        };
        match result {
            Ok(()) => info!("control: '{line}' at flip {flip}"),
            Err(e) => warn!("control: '{line}': {e}"),
        }
    }
}

pub fn fatal_report(text: &str) {
    crate::log::write(crate::log::Level::Error, format_args!("{text}"));
    if let Some(dir) = RUN_DIR.get() {
        let _ = std::fs::write(dir.join("crash.txt"), crate::log::redact(text).as_bytes());
    }
}

fn thread_summary() -> String {
    let mut s = String::new();
    if let Some(threads) = thread::THREADS.try_lock() {
        for t in threads.iter() {
            let state = match t.exited.try_lock() {
                Some(g) => match *g {
                    Some(v) => format!("exited({v:#x})"),
                    None => "running".into(),
                },
                None => "?".into(),
            };
            s.push_str(&format!("    thread {} '{}' host-tid {} {}\n", t.id, t.name(), t.host_tid.load(std::sync::atomic::Ordering::Relaxed), state));
        }
    }
    s
}

pub fn fatal_exit(code: u32, reason: &str) -> ! {
    crate::log::write(crate::log::Level::Error, format_args!("terminating: {reason}\n{}", thread_summary()));
    crate::log::try_flush();
    // (A display mode switched for exclusive fullscreen goes back.)
    gpu::vk::window::restore_display();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

/// Player-facing startup failure: logged, shown (with dialogs on), exit.
fn startup_failure(title: &str, text: &str, code: u32) -> ! {
    config::user_error(title, text);
    fatal_exit(code, title);
}

pub fn shutdown(code: i32) -> ! {
    // A save being written finishes first.
    hle::libs::savedata::wait_writes(Duration::from_secs(3));
    gpu::vk::window::restore_display();
    if code == 0 {
        gpu::render::save_pipeline_cache(Duration::ZERO);
        if let Some(p) = config::try_paths() {
            let _ = std::fs::remove_file(p.running_marker());
        }
    }
    pacing::finish();
    info!("shutdown with status {code}\n{}", thread_summary());
    crate::log::flush();
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code as u32);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn read_proc_param(m: &loader::LoadedModule) -> (u64, String, i32) {
    let mut stack = 0u64;
    let mut name = "GAME_MainThread".to_string();
    let mut prio = hle::kernel::pthread::PRIO_DEFAULT;
    if let Some(pp) = m.proc_param {
        unsafe {
            let size = *(pp as *const u64);
            let magic = *((pp + 8) as *const u32);
            let sdk = *((pp + 0x10) as *const u64);
            hle::kernel::misc::SDK_VERSION.store(sdk, std::sync::atomic::Ordering::Relaxed);
            info!("proc param at {pp:#x}: size {size:#x} magic {magic:#x} sdk {sdk:#x}");
            let field = |off: u64| if off + 8 <= size { *((pp + off) as *const u64) } else { 0 };
            let name_pp = field(0x20);
            if name_pp != 0 && *(name_pp as *const u64) != 0 {
                name = hle::cstr(*(name_pp as *const u64) as *const u8).to_string();
            }
            let prio_p = field(0x28);
            if prio_p != 0 {
                prio = *(prio_p as *const i32);
            }
            let stack_p = field(0x30);
            if stack_p != 0 {
                stack = *(stack_p as *const u32) as u64;
            }
        }
    }
    (stack, name, prio)
}

fn main() {
    // The guest address range must be reserved before the process grows.
    if let Err(e) = mem::reserve_guest_range() {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let slot = match tls::init_slot() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if std::env::args().any(|a| a == "--decode-at9" || a == "--probe-media") {
        unsafe { windows_sys::Win32::System::Console::AttachConsole(windows_sys::Win32::System::Console::ATTACH_PARENT_PROCESS) };
    }
    dpi_aware();
    let cfg = parse_args();
    let setting_errors = config::init(cfg.paths.clone(), cfg.safe_mode, cfg.dialogs);
    if !single_instance() {
        config::user_error("The Last Guardian (tlg)", "The game is already running.");
        std::process::exit(1);
    }
    let paths = config::paths();
    let runs = paths.logs();
    prune_runs(&runs, paths.logs_kept());
    let run_dir = runs.join(utc_stamp());
    if let Err(e) = std::fs::create_dir_all(&run_dir).and_then(|_| log::init(&run_dir.join("tlg.log"))) {
        config::user_error("The Last Guardian (tlg)", &format!("Cannot write the log folder {}: {e}", run_dir.display()));
        std::process::exit(1);
    }
    RUN_DIR.set(run_dir.clone()).ok();
    log::set_thread_name("host");
    // (Left behind if this run does not exit cleanly: the launcher offers safe mode.)
    let _ = std::fs::write(paths.running_marker(), format!("pid {}\nrun {}\n", std::process::id(), run_dir.display()));
    let app_dirs = app_dirs(&cfg, config::settings());
    if cfg.debug || cfg.trace {
        log::LEVEL.store(log::Level::Debug as u8, std::sync::atomic::Ordering::Relaxed);
    }
    log::TRACE_HLE.store(cfg.trace, std::sync::atomic::Ordering::Relaxed);
    if let Some(ms) = std::env::var("TLG_SLOW_HLE_MS").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
        hle::SLOW_NS.store((ms * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
        info!("logging HLE calls of {ms} ms or more");
    }
    hle::LENIENT_IMPORTS.store(cfg.lenient, std::sync::atomic::Ordering::Relaxed);
    info!("tlg {} | run dir {} | TLS slot {}", env!("CARGO_PKG_VERSION"), run_dir.display(), slot);
    info!(
        "layout {:?}: settings and saves {}, caches {}{}{}",
        paths.layout,
        paths.user.display(),
        paths.cache.display(),
        if cfg.safe_mode { " | SAFE MODE (default settings)" } else { "" },
        if cfg.launcher { " | started by the launcher" } else { "" }
    );
    for e in &setting_errors {
        warn!("{e} (default used)");
    }
    {
        let s = config::settings();
        info!(
            "settings: {:?} on {} (window {}x{}, fullscreen {:?} @ {} Hz), vsync {:?}, scaling {:?} {:?}, frame cap {:?}, gpu {}, volume {}",
            s.display_mode, s.monitor, s.window_width, s.window_height, s.resolution, s.refresh_rate, s.vsync, s.scaling, s.filter, s.frame_cap, s.gpu, s.volume
        );
    }
    exception::install();
    gpu::shader_dump::DIR.set(run_dir.join("shaders")).ok();
    if std::env::var_os("TLG_TEST_FAULT").is_some() {
        info!("TLG_TEST_FAULT: writing to an invalid address from host code");
        unsafe { std::ptr::write_volatile(0x32c3be8 as *mut u64, 1) };
    }
    hle::kernel::time::init();
    // TLG_CPU_AFFINITY=<hex mask>: run on those logical CPUs only (e.g. 0xfff:
    // six cores with SMT, to measure a smaller CPU than the host's).
    if let Some(mask) = std::env::var("TLG_CPU_AFFINITY").ok().and_then(|v| usize::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessAffinityMask};
        if unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask) } != 0 {
            info!("CPU affinity {mask:#x} ({} logical CPUs)", mask.count_ones());
        } else {
            warn!("TLG_CPU_AFFINITY {mask:#x}: SetProcessAffinityMask failed");
        }
    }
    // The renderer runs whether or not frames reach a window: headless runs (and a
    // window that could not be shown) render offscreen; screenshots still work.
    // The game files first: nothing else is worth starting without them.
    info!("game folders (highest priority first): {:?}", app_dirs);
    let Some(eboot_path) = app_dirs.iter().map(|d| d.join("eboot.bin")).find(|p| p.exists()) else {
        let dirs: Vec<String> = app_dirs.iter().map(|d| format!("  {}", d.display())).collect();
        startup_failure(
            "Game files not found",
            &format!("No prepared game files (eboot.bin) were found in:\n{}\n\nStart the launcher (tlg-launcher.exe) and run its game-file setup, or point settings.txt (game_dir, update_dir) at your prepared copy.", dirs.join("\n")),
            1,
        );
    };
    // The renderer runs whether or not frames reach a window: headless runs (and a
    // window that could not be shown) render offscreen; screenshots still work.
    if !cfg.headless {
        let title = format!("The Last Guardian - tlg {}{}", env!("CARGO_PKG_VERSION"), if cfg.safe_mode { " (safe mode)" } else { "" });
        if let Err(e) = gpu::vk::present::init(&title) {
            warn!("no presentation window ({e}): rendering offscreen");
        }
    }
    if let Err(e) = gpu::vk::init().and_then(|_| {
        // (TLG_PIPELINE_CACHE=0: none - pipelines compile afresh, e.g. to
        // measure compile times.)
        if std::env::var("TLG_PIPELINE_CACHE").map_or(true, |v| v != "0") {
            gpu::render::set_pipeline_cache_path(paths.pipeline_cache());
        }
        gpu::render::init()
    }) {
        startup_failure(
            "Graphics unavailable",
            &format!("The Vulkan renderer could not start:\n{e}\n\ntlg needs a graphics card and driver with Vulkan 1.3 (NVIDIA, AMD or Intel; update the graphics driver)."),
            1,
        );
    }

    // Guest file system.
    let rt = paths.runtime();
    for d in ["temp0", "download0", "data", "savedata"] {
        hle::kernel::fs::ensure_dir(&rt.join(d));
    }
    hle::kernel::fs::mount("/app0", app_dirs.clone(), false);
    hle::kernel::fs::mount("/hostapp", app_dirs.clone(), false);
    hle::kernel::fs::mount("/temp0", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/temp", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/download0", vec![rt.join("download0")], true);
    hle::kernel::fs::mount("/data", vec![rt.join("data")], true);
    hle::libs::savedata::init(paths.saves(), paths.save_backups());

    // Modules.
    let t0 = Instant::now();
    let eboot = loader::load(&eboot_path, true).unwrap_or_else(|e| {
        startup_failure("Game files unusable", &format!("Loading {} failed: {e}\n\nCheck the game files with the launcher's Verify.", eboot_path.display()), 1);
    });
    let mut libs: Vec<Arc<loader::LoadedModule>> = Vec::new();
    for name in ["libc.prx", "libSceFios2.prx"] {
        let p = app_dirs.iter().map(|d| d.join("sce_module").join(name)).find(|p| p.exists());
        match p {
            Some(p) => match loader::load(&p, false) {
                Ok(m) => libs.push(m),
                Err(e) => startup_failure("Game files unusable", &format!("Loading {} failed: {e}\n\nCheck the game files with the launcher's Verify.", p.display()), 1),
            },
            None => warn!("{name} not found in sce_module"),
        }
    }
    let _ = hle::registry();
    let modules: Vec<Arc<loader::LoadedModule>> = loader::MODULES.read().clone();
    for m in &modules {
        match loader::relocate(m) {
            Ok(s) => info!("{}: linked (lle {}, hle {}, unresolved {}, weak-null {})", m.name, s.lle, s.hle, s.unresolved, s.weak_null),
            Err(e) => {
                error!("relocating {} failed: {e}", m.name);
                fatal_exit(1, "link failure");
            }
        }
    }
    for m in &modules {
        match loader::finalize(m) {
            Ok(s) => loader::report_patch(m, &s),
            Err(e) => {
                error!("finalizing {} failed: {e}", m.name);
                fatal_exit(1, "patch failure");
            }
        }
    }
    let unresolved = hle::UNRESOLVED.read().len();
    info!("load+link+patch took {:.1} ms; {} distinct unresolved function imports", t0.elapsed().as_secs_f64() * 1000.0, unresolved);
    {
        // The full list (library, name or NID, importer) for triage: unresolved.txt.
        let mut lines: Vec<String> = hle::UNRESOLVED
            .read()
            .iter()
            .map(|u| format!("{:<28} {:<48} {}{}", u.lib, hle::describe_nid(&u.nid), u.importer, if u.is_data { " (data)" } else { "" }))
            .collect();
        lines.sort();
        lines.dedup();
        let _ = std::fs::write(run_dir.join("unresolved.txt"), lines.join("\n") + "\n");
    }

    // Guest main thread.
    let (stack, name, prio) = read_proc_param(&eboot);
    let stack = stack.max(8 << 20);
    let main_t = thread::new_thread(&name, eboot.entry, 0, stack, prio);
    let eboot2 = eboot.clone();
    thread::spawn_host(
        &name,
        stack as usize + (1 << 20),
        Box::new(move || {
            thread::enter(main_t);
            for m in &libs {
                if let Some(init) = m.init {
                    info!("starting module {} at {}", m.name, loader::describe_addr(init));
                    let r = thread::run_entry(main_t, init, 0, 0);
                    info!("module {} start returned {:#x}", m.name, r);
                }
            }
            let params = mem::arena_alloc(40, 8);
            unsafe {
                *(params as *mut u32) = 1;
                *((params + 8) as *mut u64) = mem::arena_cstr("eboot.bin");
            }
            if let Ok(list) = std::env::var("TLG_BREAK") {
                for tok in list.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                    let Ok(off) = u64::from_str_radix(tok.trim_start_matches("0x"), 16) else {
                        warn!("TLG_BREAK: '{tok}' is not a hex offset: ignored");
                        continue;
                    };
                    match exception::install_breakpoint(eboot2.base + off, eboot2.base) {
                        Ok(()) => info!("breakpoint at {}", loader::describe_addr(eboot2.base + off)),
                        Err(e) => warn!("breakpoint {off:#x}: {e}"),
                    }
                }
            }
            info!("entering {} at {}", eboot2.name, loader::describe_addr(eboot2.entry));
            let r = thread::run_entry(main_t, eboot2.entry, params, hle::kernel::misc::rtld_exit_fn());
            info!("eboot entry returned {r:#x}");
            shutdown(r as i32);
        }),
    )
    .expect("spawn main thread");

    let start = Instant::now();
    let mut control = cfg.control.clone().map(|p| (p, 0u64));
    let mut ticks = 0u64;
    // Hang report: once flips have started, a 10 s gap logs what every thread is in.
    let (mut last_flip, mut last_flip_at, mut hang_reported) = (0u64, Instant::now(), false);
    loop {
        std::thread::sleep(Duration::from_millis(250));
        ticks += 1;
        if ticks % 40 == 1 {
            envmon::report();
        }
        let flip = gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        if flip != last_flip {
            (last_flip, last_flip_at, hang_reported) = (flip, Instant::now(), false);
        } else if flip > 0 && !hang_reported && last_flip_at.elapsed() > Duration::from_secs(10) {
            hang_reported = true;
            warn!("no flip for 10 s (last flip {flip})");
            thread::dump_threads("no flips");
        }
        if ticks % 8 == 0 && exception::break_stats_enabled() {
            exception::dump_break_stats();
        }
        crate::log::try_flush();
        if let Some((path, offset)) = control.as_mut() {
            poll_control(path, offset);
        }
        if gpu::vk::window::CLOSED.load(std::sync::atomic::Ordering::SeqCst) {
            info!("window closed by user");
            shutdown(0);
        }
        if let Some(s) = cfg.seconds {
            if start.elapsed() >= Duration::from_secs(s) {
                info!("time limit of {s}s reached");
                shutdown(0);
            }
        }
    }
}
