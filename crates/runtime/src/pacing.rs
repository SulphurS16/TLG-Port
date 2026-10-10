//! Delivered frame pacing, measured on the host: the interval between
//! consecutive presents of new guest frames (taken on the vblank thread once
//! each present has been queued). Reported per phase (`mark <name>` control
//! commands start phases) and for the whole run at exit: frame-time
//! percentiles, 1% and 0.1% lows, hitch counts and the worst frames. With
//! `SOTC_PERF`, a `pacing/s` line per second. Each hitch (an interval over
//! `SOTC_HITCH_MS`, default 50, or 1.5 frames of a lower `SOTC_FPS` cap) is
//! logged as it happens with what the frame
//! spent its time on along its path: when the title submitted the frame's
//! command buffers (relative to the previous frame's), its wait in the GPU
//! queue and the GPU thread's processing up to the flip, the flip's wait for
//! execution (vblank ticks), the wait for the renderer lock and the present
//! itself; plus the title's own measured frame time (its main thread) and
//! the GPU thread's work since the previous present (shader preparation,
//! submissions, ring slot waits, texture hashing). `SOTC_FRAMETIMES=<file>`
//! writes every frame as CSV.
//!
//! These are host clock times when presents are queued: the display adds its
//! own quantization to its refresh (without variable refresh, 6.06 ms steps
//! at 165 Hz).

use parking_lot::Mutex;
use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// The renderer's cumulative time counters (`render::PROF_NAMES` order:
/// submit+wait, writeback, buffer upload, texture hash, texture read,
/// shader prep, slot wait).
pub type Prof = [Duration; 7];

/// A GPU flip's path before it was queued for display: when the title
/// submitted its command buffers, their wait in the GPU queue and the GPU
/// thread's processing up to the flip.
#[derive(Clone, Copy, Debug)]
pub struct FlipTiming {
    pub submitted: Instant,
    pub queue: Duration,
    pub process: Duration,
}

const WORST: usize = 5;

struct Phase {
    name: String,
    start: Instant,
    intervals: Vec<f32>,
    /// The worst intervals: (ms, seconds since the run started).
    worst: Vec<(f32, f64)>,
}

impl Phase {
    fn new(name: &str, start: Instant) -> Phase {
        Phase { name: name.to_string(), start, intervals: Vec::new(), worst: Vec::new() }
    }

    fn add(&mut self, ms: f32, t: f64) {
        self.intervals.push(ms);
        if self.worst.len() < WORST || ms > self.worst[WORST - 1].0 {
            self.worst.push((ms, t));
            self.worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            self.worst.truncate(WORST);
        }
    }
}

struct State {
    run_start: Instant,
    /// The previous present: time, the renderer's counters then, flip number.
    last: Option<(Instant, Option<Prof>, u64)>,
    /// When the title submitted the previous presented frame.
    last_submitted: Option<Instant>,
    second_start: Instant,
    second: Vec<f32>,
    phase: Phase,
    run: Phase,
    csv: Option<std::io::BufWriter<std::fs::File>>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Hitch threshold: `SOTC_HITCH_MS`, else 50 ms or 1.5 frame periods of a
/// lower frame-rate cap (`SOTC_FPS`), whichever is longer.
fn hitch_ms() -> f32 {
    static H: OnceLock<f32> = OnceLock::new();
    *H.get_or_init(|| {
        let cap = crate::title_patches::target_fps().filter(|&n| n > 0).unwrap_or(30);
        std::env::var("SOTC_HITCH_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or((1500.0 / cap as f32).max(50.0))
    })
}

fn perf_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SOTC_PERF").is_some())
}

fn ms(d: Duration) -> f32 {
    d.as_secs_f32() * 1000.0
}

impl State {
    fn new(now: Instant) -> State {
        let csv = std::env::var("SOTC_FRAMETIMES").ok().and_then(|p| match std::fs::File::create(&p) {
            Ok(f) => {
                let mut w = std::io::BufWriter::new(f);
                let _ = writeln!(
                    w,
                    "t_s,interval_ms,flip,game_ms,submit_interval_ms,gpu_queue_ms,gpu_process_ms,flip_wait_ms,lock_ms,present_ms,shader_prep_ms,submit_wait_ms,slot_wait_ms,texture_read_ms,texture_hash_ms,phase"
                );
                crate::info!("pacing: writing frame times to {p}");
                Some(w)
            }
            Err(e) => {
                crate::warn!("SOTC_FRAMETIMES {p}: {e}");
                None
            }
        });
        State { run_start: now, last: None, last_submitted: None, second_start: now, second: Vec::new(), phase: Phase::new("start", now), run: Phase::new("run", now), csv }
    }
}

/// A new guest frame was presented: `flip` is the guest flip number,
/// `lock_wait` the wait for the renderer lock, `present` the present call,
/// `prof` the renderer's counters at that point, `game_ms` the title's own
/// measured duration of its last frame, `timing` the GPU flip's path and
/// `flip_wait` the time from its queueing to its execution.
pub fn presented(flip: u64, lock_wait: Duration, present: Duration, prof: Option<Prof>, game_ms: Option<f32>, timing: Option<FlipTiming>, flip_wait: Duration) {
    let now = Instant::now();
    let mut g = STATE.lock();
    let s = g.get_or_insert_with(|| State::new(now));
    let submit_interval = match (timing, s.last_submitted) {
        (Some(t), Some(prev)) => Some(ms(t.submitted.saturating_duration_since(prev))),
        _ => None,
    };
    if let Some(t) = timing {
        s.last_submitted = Some(t.submitted);
    }
    let opt = |v: Option<f32>| v.map_or(String::from("?"), |v| format!("{v:.1}"));
    if let Some((prev, prev_prof, prev_flip)) = s.last {
        let interval = ms(now - prev);
        let t = (now - s.run_start).as_secs_f64();
        s.second.push(interval);
        s.phase.add(interval, t);
        s.run.add(interval, t);
        // The GPU thread's work since the previous present.
        let d = match (prof, prev_prof) {
            (Some(a), Some(b)) => std::array::from_fn(|i| ms(a[i].saturating_sub(b[i]))),
            _ => [0.0f32; 7],
        };
        let game = game_ms.map_or(String::from("?"), |v| format!("{v:.1}"));
        if interval > hitch_ms() {
            crate::info!(
                "hitch: {interval:.1} ms before flip {flip} at {t:.1} s (phase {}; {} flips since the previous present): title submitted it {} ms after the previous frame, GPU queue {} ms, GPU processing {} ms, flip wait {:.1} ms, renderer lock wait {:.1} ms, present {:.1} ms; game frame {game} ms; GPU thread since the previous present: shader prep {:.1} ms, submit+wait {:.1} ms, slot wait {:.1} ms, texture read {:.1} ms, texture hash {:.1} ms, buffer upload {:.1} ms, writeback {:.1} ms",
                s.phase.name,
                flip.saturating_sub(prev_flip),
                opt(submit_interval),
                opt(timing.map(|t| ms(t.queue))),
                opt(timing.map(|t| ms(t.process))),
                ms(flip_wait),
                ms(lock_wait),
                ms(present),
                d[5],
                d[0],
                d[6],
                d[4],
                d[3],
                d[2],
                d[1]
            );
        }
        if let Some(w) = s.csv.as_mut() {
            let o3 = |v: Option<f32>| v.map_or(String::new(), |v| format!("{v:.3}"));
            let _ = writeln!(
                w,
                "{t:.4},{interval:.3},{flip},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
                o3(game_ms),
                o3(submit_interval),
                o3(timing.map(|t| ms(t.queue))),
                o3(timing.map(|t| ms(t.process))),
                ms(flip_wait),
                ms(lock_wait),
                ms(present),
                d[5],
                d[0],
                d[6],
                d[4],
                d[3],
                s.phase.name
            );
        }
    }
    s.last = Some((now, prof, flip));
    if now - s.second_start >= Duration::from_secs(1) {
        if perf_on() && !s.second.is_empty() {
            let v = &s.second;
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let max = v.iter().copied().fold(0.0f32, f32::max);
            let mut sorted = v.clone();
            sorted.sort_by(f32::total_cmp);
            let median = sorted[sorted.len() / 2];
            let stutters = v.iter().filter(|&&x| x > 2.0 * median).count();
            crate::info!("pacing/s: {} frames, frame ms mean {mean:.1} median {median:.1} max {max:.1}, over 2x median {stutters}", v.len());
        }
        s.second.clear();
        s.second_start = now;
    }
}

/// Flips executed without reaching the screen (minimized window), counted
/// in the run summary.
static NOT_DISPLAYED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn not_displayed() {
    if NOT_DISPLAYED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        crate::warn!("pacing: a flip was not displayed (window minimized?); frames are still counted");
    }
}

/// Frame-time statistics of `p` (seconds from its start to `end`).
fn summary(p: &Phase, end: Instant) -> String {
    let v = &p.intervals;
    if v.is_empty() {
        return format!("pacing [{}]: no frames", p.name);
    }
    let n = v.len();
    let mut s = v.clone();
    s.sort_by(f32::total_cmp);
    let pct = |q: f64| s[((n as f64 - 1.0) * q).round() as usize];
    let median = pct(0.5);
    let mean = v.iter().sum::<f32>() / n as f32;
    // "1% low": the frame rate over the slowest 1% of frames.
    let low = |frac: usize| {
        let k = (n / frac).max(1);
        1000.0 / (s[n - k..].iter().sum::<f32>() / k as f32)
    };
    let over = |t: f32| v.iter().filter(|&&x| x > t).count();
    let secs = (end - p.start).as_secs_f64().max(1e-3);
    let worst: Vec<String> = p.worst.iter().map(|(ms, t)| format!("{ms:.1} ms at {t:.1} s")).collect();
    format!(
        "pacing [{}]: {n} frames in {secs:.1} s ({:.1} fps); frame ms mean {mean:.2} p50 {median:.2} p95 {:.2} p99 {:.2} p99.9 {:.2} max {:.1}; 1% low {:.1} fps, 0.1% low {:.1} fps; over 2x median {}, over 34 ms {}, over 50 ms {}, over 100 ms {}; worst: {}",
        p.name,
        n as f64 / secs,
        pct(0.95),
        pct(0.99),
        pct(0.999),
        s[n - 1],
        low(100),
        low(1000),
        over(2.0 * median),
        over(34.0),
        over(50.0),
        over(100.0),
        worst.join(", ")
    )
}

/// Ends the current phase (logging its summary) and starts `name`.
pub fn mark(name: &str) {
    let now = Instant::now();
    let mut g = STATE.lock();
    let s = g.get_or_insert_with(|| State::new(now));
    if !s.phase.intervals.is_empty() {
        crate::info!("{}", summary(&s.phase, now));
    }
    s.phase = Phase::new(name, now);
}

/// Logs the current phase's and the whole run's summaries (at exit).
pub fn finish() {
    let now = Instant::now();
    let Some(mut g) = STATE.try_lock_for(Duration::from_millis(200)) else { return };
    let Some(s) = g.as_mut() else { return };
    if !s.phase.intervals.is_empty() && s.phase.name != "start" {
        crate::info!("{}", summary(&s.phase, now));
    }
    crate::info!("{}", summary(&s.run, now));
    let hidden = NOT_DISPLAYED.load(std::sync::atomic::Ordering::Relaxed);
    if hidden > 0 {
        crate::info!("pacing: {hidden} flips were not displayed (minimized window)");
    }
    if let Some(w) = s.csv.as_mut() {
        let _ = w.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_percentiles_lows_and_hitches() {
        let start = Instant::now();
        let mut p = Phase::new("t", start);
        // 990 frames of 16.7 ms, 9 of 40 ms, one of 120 ms.
        for i in 0..1000 {
            let ms = if i == 500 { 120.0 } else if i % 100 == 7 && i < 900 { 40.0 } else { 16.7 };
            p.add(ms, i as f64);
        }
        let s = summary(&p, start + Duration::from_secs(20));
        assert!(s.contains("1000 frames"), "{s}");
        assert!(s.contains("p50 16.70"), "{s}");
        assert!(s.contains("max 120.0"), "{s}");
        // Slowest 10 frames: nine of 40 ms and the 120 ms one -> 48 ms -> 20.8 fps.
        assert!(s.contains("1% low 20.8 fps"), "{s}");
        assert!(s.contains("over 2x median 10, over 34 ms 10, over 50 ms 1, over 100 ms 1"), "{s}");
        assert!(s.starts_with("pacing [t]"), "{s}");
        assert_eq!(p.worst[0].0, 120.0);
        assert_eq!(p.worst.len(), WORST);
    }
}
