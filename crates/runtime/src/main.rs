//! tlg: runs the PS4 The Last Guardian executable natively on Windows x86-64
//! with high-level emulation of the PS4 system libraries.

#[macro_use]
mod log;
mod audio;
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
    pub root: PathBuf,
    /// Machine-local data directory (`<root>/local`): runs, saves, caches, runtime dirs.
    pub data: PathBuf,
    /// Application directories mounted at /app0, highest priority first
    /// (an optional update overlay, then the prepared base game).
    pub app_dirs: Vec<PathBuf>,
    pub run_dir: PathBuf,
    pub seconds: Option<u64>,
    /// Development control channel (see `poll_control`).
    pub control: Option<PathBuf>,
    pub trace: bool,
    pub lenient: bool,
    pub debug: bool,
    pub headless: bool,
}

static RUN_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
static DATA_DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// This run's directory (logs, screenshots, profiles).
pub fn run_dir() -> Option<&'static Path> {
    RUN_DIR.get().map(|p| p.as_path())
}

/// The machine-local data directory (`Config::data`): user settings such as
/// `input.txt` live there.
pub fn data_dir() -> Option<&'static Path> {
    DATA_DIR.get().map(|p| p.as_path())
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

/// The project root: the nearest ancestor of the executable holding `local/` or the
/// workspace `Cargo.toml` (development builds live in target/<triple>/<profile>/);
/// otherwise the executable's own directory (a distributed build).
fn default_root() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut p = Some(dir.as_path());
    while let Some(d) = p {
        if d.join("local").is_dir() || d.join("Cargo.toml").is_file() {
            return d.to_path_buf();
        }
        p = d.parent();
    }
    dir
}

fn parse_args() -> Config {
    let mut root = default_root();
    let mut game: Option<PathBuf> = None;
    let mut update: Option<PathBuf> = None;
    let mut no_update = false;
    let mut seconds = None;
    let mut trace = false;
    let mut lenient = false;
    let mut debug = false;
    let mut headless = false;
    let mut control = None;
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
            "--root" => root = PathBuf::from(args.next().expect("--root needs a path")),
            "--game" => game = Some(PathBuf::from(args.next().expect("--game needs a path"))),
            "--update" => update = Some(PathBuf::from(args.next().expect("--update needs a path"))),
            // The base application alone (01.00), ignoring a prepared update overlay.
            "--no-update" => no_update = true,
            "--control" => control = Some(PathBuf::from(args.next().expect("--control needs a path"))),
            other => {
                eprintln!("unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let data = root.join("local");
    // Default baseline: the prepared update (01.03) over the base application (01.00).
    let default_update = data.join("game").join("CUSA03627-patch");
    if update.is_none() && !no_update && default_update.join("eboot.bin").is_file() {
        update = Some(default_update);
    }
    if no_update {
        update = None;
    }
    let mut app_dirs: Vec<PathBuf> = update.into_iter().collect();
    app_dirs.push(game.unwrap_or_else(|| data.join("game").join("CUSA03627")));
    Config {
        app_dirs,
        run_dir: data.join("runs").join(utc_stamp()),
        data,
        root,
        seconds,
        trace,
        lenient,
        debug,
        headless,
        control,
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
            "sample" => {
                let ms = parts.next().and_then(|v| v.parse().ok()).unwrap_or(3000);
                thread::sample_threads(ms);
                Ok(())
            }
            "tune" => {
                let name = parts.next().unwrap_or("");
                tune::set(name, parts.next().unwrap_or(""))
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
        let _ = std::fs::write(dir.join("crash.txt"), text);
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
    unsafe {
        windows_sys::Win32::System::Threading::TerminateProcess(windows_sys::Win32::System::Threading::GetCurrentProcess(), code);
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

pub fn shutdown(code: i32) -> ! {
    if code == 0 {
        gpu::render::save_pipeline_cache(Duration::ZERO);
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
    let cfg = parse_args();
    let runs = cfg.data.join("runs");
    prune_runs(&runs, 30);
    std::fs::create_dir_all(&cfg.run_dir).expect("create run dir");
    RUN_DIR.set(cfg.run_dir.clone()).ok();
    DATA_DIR.set(cfg.data.clone()).ok();
    log::init(&cfg.run_dir.join("tlg.log")).expect("open log");
    log::set_thread_name("host");
    if cfg.debug || cfg.trace {
        log::LEVEL.store(log::Level::Debug as u8, std::sync::atomic::Ordering::Relaxed);
    }
    log::TRACE_HLE.store(cfg.trace, std::sync::atomic::Ordering::Relaxed);
    if let Some(ms) = std::env::var("TLG_SLOW_HLE_MS").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
        hle::SLOW_NS.store((ms * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
        info!("logging HLE calls of {ms} ms or more");
    }
    hle::LENIENT_IMPORTS.store(cfg.lenient, std::sync::atomic::Ordering::Relaxed);
    info!("tlg {} | run dir {} | TLS slot {}", env!("CARGO_PKG_VERSION"), cfg.run_dir.display(), slot);
    exception::install();
    gpu::shader_dump::DIR.set(cfg.run_dir.join("shaders")).ok();
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
    if !cfg.headless {
        if let Err(e) = gpu::vk::present::init("The Last Guardian (tlg)") {
            warn!("no presentation window ({e}): rendering offscreen");
        }
    }
    if let Err(e) = gpu::vk::init().and_then(|_| {
        gpu::render::set_pipeline_cache_path(cfg.data.join("pipeline-cache.bin"));
        gpu::render::init()
    }) {
        error!("renderer unavailable: {e}");
        fatal_exit(1, "no Vulkan renderer");
    }

    // Guest file system.
    let rt = cfg.data.join("runtime");
    for d in ["temp0", "download0", "data", "savedata"] {
        hle::kernel::fs::ensure_dir(&rt.join(d));
    }
    hle::kernel::fs::mount("/app0", cfg.app_dirs.clone(), false);
    hle::kernel::fs::mount("/hostapp", cfg.app_dirs.clone(), false);
    hle::kernel::fs::mount("/temp0", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/temp", vec![rt.join("temp0")], true);
    hle::kernel::fs::mount("/download0", vec![rt.join("download0")], true);
    hle::kernel::fs::mount("/data", vec![rt.join("data")], true);
    hle::libs::savedata::init(cfg.data.join("saves"));

    // Modules.
    let t0 = Instant::now();
    let Some(eboot_path) = cfg.app_dirs.iter().map(|d| d.join("eboot.bin")).find(|p| p.exists()) else {
        error!("no eboot.bin in {:?}", cfg.app_dirs);
        fatal_exit(1, "game files missing");
    };
    let eboot = loader::load(&eboot_path, true).unwrap_or_else(|e| {
        error!("loading eboot failed: {e}");
        fatal_exit(1, "load failure");
    });
    let mut libs: Vec<Arc<loader::LoadedModule>> = Vec::new();
    for name in ["libc.prx", "libSceFios2.prx"] {
        let p = cfg.app_dirs.iter().map(|d| d.join("sce_module").join(name)).find(|p| p.exists());
        match p {
            Some(p) => match loader::load(&p, false) {
                Ok(m) => libs.push(m),
                Err(e) => {
                    error!("loading {name} failed: {e}");
                    fatal_exit(1, "load failure");
                }
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
        let _ = std::fs::write(cfg.run_dir.join("unresolved.txt"), lines.join("\n") + "\n");
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
