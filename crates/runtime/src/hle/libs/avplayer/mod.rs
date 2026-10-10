//! libSceAvPlayer: movie playback.
//!
//! Each player owns a controller thread (a guest-entered host thread, so it can
//! call the title's event callback) that opens the source with Media
//! Foundation (`mf`), reports READY, decodes ahead once started, and reports
//! STOP at the end (or loops). Video frames are delivered as NV12 in buffers
//! obtained from the title's `allocateTexture` callback (GPU-visible guest
//! memory, 16-row aligned planes with crop offsets); audio as interleaved
//! 16-bit PCM in buffers from its `allocate` callback.

mod mf;

use crate::hle::{Registry, call_guest, cstr};
use crate::{fixed, reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const AVP_MAGIC: u64 = 0x4242_4156_504C_0002;

const ERR_INVALID_PARAMS: i32 = 0x806A_0001u32 as i32;
const ERR_OPERATION_FAILED: i32 = 0x806A_0002u32 as i32;

// SceAvPlayerEvents
const EV_STATE_STOP: u64 = 0x01;
const EV_STATE_READY: u64 = 0x02;
const EV_STATE_PLAY: u64 = 0x03;
const EV_STATE_PAUSE: u64 = 0x04;
const EV_STATE_BUFFERING: u64 = 0x05;

const STREAM_VIDEO: u32 = 0;
const STREAM_AUDIO: u32 = 1;

const VIDEO_QUEUE: usize = 6;
const AUDIO_QUEUE: usize = 32;
const AUDIO_BUFFERS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Idle,
    Opening,
    Ready,
    Playing,
    Paused,
    Stopped,
}

#[derive(Clone, Copy, Default)]
struct Callbacks {
    mem_obj: u64,
    alloc: u64,
    free: u64,
    alloc_tex: u64,
    free_tex: u64,
    ev_obj: u64,
    ev_cb: u64,
}

struct Shared {
    state: State,
    source: Option<(String, std::path::PathBuf)>,
    info: Option<mf::MediaInfo>,
    /// Enabled stream indices (first enabled video and audio stream are played).
    enabled: Vec<usize>,
    /// Geometry of the decoded video (set when playback starts).
    video_info: Option<mf::VideoInfo>,
    looping: bool,
    /// Playback clock: time base and accumulated pause.
    started: Option<Instant>,
    paused_at: Option<Instant>,
    paused_total: Duration,
    /// Offset added after a loop restart.
    loop_base_ms: u64,
    video_q: VecDeque<mf::VideoFrame>,
    audio_q: VecDeque<mf::AudioFrame>,
    decode_done: bool,
    stop_requested: bool,
    quit: bool,
    /// Guest frame buffers.
    video_bufs: Vec<u64>,
    video_next: usize,
    audio_bufs: Vec<(u64, usize)>,
    audio_next: usize,
}

struct Player {
    magic: u64,
    cb: Callbacks,
    vbufs: usize,
    shared: Mutex<Shared>,
    cv: Condvar,
}

impl Player {
    fn clock_ms(s: &Shared) -> u64 {
        let Some(start) = s.started else { return 0 };
        let now = s.paused_at.unwrap_or_else(Instant::now);
        (now.saturating_duration_since(start).saturating_sub(s.paused_total)).as_millis() as u64
    }
    fn event(&self, id: u64) {
        if self.cb.ev_cb != 0 {
            crate::debug!("avplayer: event {id:#x}");
            unsafe { call_guest(self.cb.ev_cb, &[self.cb.ev_obj, id, 0, 0]) };
        }
    }
}

fn player<'a>(h: u64) -> Option<&'a Player> {
    if h == 0 {
        return None;
    }
    let p = unsafe { &*(h as *const Player) };
    (p.magic == AVP_MAGIC).then_some(p)
}

fn selected(s: &Shared, video: bool) -> Option<usize> {
    let info = s.info.as_ref()?;
    s.enabled.iter().copied().find(|&i| info.streams.get(i).is_some_and(|d| d.video == video))
}

fn aligned_dims(v: &mf::VideoInfo) -> (u32, u32) {
    (v.frame_w.div_ceil(16) * 16, v.frame_h.div_ceil(16) * 16)
}

/// Controller thread body.
fn controller(p: &'static Player) {
    let mut decoder: Option<mf::Decoder> = None;
    loop {
        // Wait for something to do.
        let (state, want_open, quit) = {
            let mut s = p.shared.lock();
            loop {
                let work = s.quit
                    || s.state == State::Opening
                    || s.stop_requested
                    || (s.state == State::Playing && !s.decode_done && (s.video_q.len() < VIDEO_QUEUE || s.audio_q.len() < AUDIO_QUEUE))
                    || (s.state == State::Playing && s.decode_done && s.video_q.is_empty() && s.audio_q.is_empty());
                if work {
                    break;
                }
                p.cv.wait_for(&mut s, Duration::from_millis(20));
            }
            (s.state, s.state == State::Opening, s.quit)
        };
        if quit {
            return;
        }
        if want_open {
            let (name, path) = p.shared.lock().source.clone().unwrap();
            p.event(EV_STATE_BUFFERING);
            match mf::probe(&path) {
                Ok(info) => {
                    crate::info!(
                        "avplayer: '{name}' ready: {} ms, streams [{}]",
                        info.duration_ms,
                        info.streams.iter().map(|d| if d.video { format!("video {}x{}", d.v.width, d.v.height) } else { format!("audio {}ch {}Hz", d.a.channels, d.a.sample_rate) }).collect::<Vec<_>>().join(", ")
                    );
                    {
                        let mut s = p.shared.lock();
                        s.info = Some(info);
                        s.state = State::Ready;
                    }
                    decoder = None;
                    p.event(EV_STATE_READY);
                }
                Err(e) => {
                    crate::error!("avplayer: cannot play '{name}': {e}");
                    p.shared.lock().state = State::Stopped;
                    p.event(EV_STATE_STOP);
                }
            }
            continue;
        }
        if p.shared.lock().stop_requested {
            {
                let mut s = p.shared.lock();
                s.stop_requested = false;
                s.state = State::Stopped;
                s.video_q.clear();
                s.audio_q.clear();
            }
            p.event(EV_STATE_STOP);
            continue;
        }
        if state != State::Playing {
            continue;
        }
        if decoder.is_none() {
            let (path, info, v, a) = {
                let s = p.shared.lock();
                (s.source.clone().unwrap().1, s.info.clone().unwrap_or_default(), selected(&s, true), selected(&s, false))
            };
            match mf::Decoder::open(&path, &info, v, a) {
                Ok(d) => {
                    crate::info!("avplayer: decoding video stream {v:?} ({:?}), audio stream {a:?}", d.video_info);
                    p.shared.lock().video_info = v.map(|_| d.video_info);
                    decoder = Some(d);
                }
                Err(e) => {
                    crate::error!("avplayer: cannot decode: {e}");
                    p.shared.lock().state = State::Stopped;
                    p.event(EV_STATE_STOP);
                    continue;
                }
            }
        }
        let d = decoder.as_mut().unwrap();
        // End of playback?
        {
            let mut s = p.shared.lock();
            if s.decode_done && s.video_q.is_empty() && s.audio_q.is_empty() {
                if s.looping {
                    drop(s);
                    if let Err(e) = d.rewind() {
                        crate::error!("avplayer: loop restart failed: {e}");
                    }
                    let mut s = p.shared.lock();
                    s.loop_base_ms = Player::clock_ms(&s);
                    s.decode_done = false;
                    continue;
                }
                s.state = State::Stopped;
                drop(s);
                crate::info!("avplayer: playback finished");
                p.event(EV_STATE_STOP);
                continue;
            }
        }
        // Decode ahead.
        let (need_video, need_audio, base) = {
            let s = p.shared.lock();
            (s.video_q.len() < VIDEO_QUEUE, s.audio_q.len() < AUDIO_QUEUE, s.loop_base_ms)
        };
        let mut produced = false;
        if need_video && !d.video_eof() {
            match d.next_video() {
                Ok(Some(mut f)) => {
                    f.ts_ms += base;
                    p.shared.lock().video_q.push_back(f);
                    produced = true;
                }
                Ok(None) => {}
                Err(e) => crate::error!("avplayer: video decode: {e}"),
            }
        }
        if need_audio && !d.audio_eof() {
            match d.next_audio() {
                Ok(Some(mut f)) => {
                    f.ts_ms += base;
                    p.shared.lock().audio_q.push_back(f);
                    produced = true;
                }
                Ok(None) => {}
                Err(e) => crate::error!("avplayer: audio decode: {e}"),
            }
        }
        if d.video_eof() && d.audio_eof() {
            p.shared.lock().decode_done = true;
        } else if !produced {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

fn spawn_controller(p: &'static Player) {
    let t = crate::thread::new_thread("AvPlayerCtrl", 0, 0, 1 << 20, crate::hle::kernel::pthread::PRIO_DEFAULT);
    crate::thread::spawn_host(
        "AvPlayerCtrl",
        1 << 20,
        Box::new(move || {
            crate::thread::enter(t);
            controller(p);
        }),
    )
    .expect("avplayer controller thread");
}

unsafe fn write_stream_info(out: *mut u8, st: &mf::StreamDesc, duration_ms: u64) {
    unsafe {
        std::ptr::write_bytes(out, 0, 40);
        *(out as *mut u32) = if st.video { STREAM_VIDEO } else { STREAM_AUDIO };
        let d = out.add(8);
        if st.video {
            *(d as *mut u32) = st.v.width;
            *(d.add(4) as *mut u32) = st.v.height;
            *(d.add(8) as *mut f32) = if st.v.height != 0 { st.v.width as f32 / st.v.height as f32 } else { 16.0 / 9.0 };
        } else {
            *(d as *mut u16) = st.a.channels as u16;
            *(d.add(4) as *mut u32) = st.a.sample_rate;
            *(d.add(8) as *mut u32) = 0;
        }
        std::ptr::copy_nonoverlapping(b"und\0".as_ptr(), d.add(12), 4);
        *(out.add(24) as *mut u64) = duration_ms;
        *(out.add(32) as *mut u64) = 0;
    }
}

sysv! {
    fn sceAvPlayerInit(init: *const u8) -> u64 {
        if init.is_null() { return 0; }
        let q = |o: usize| *(init.add(o) as *const u64);
        let cb = Callbacks { mem_obj: q(0), alloc: q(8), free: q(16), alloc_tex: q(24), free_tex: q(32), ev_obj: q(80), ev_cb: q(88) };
        if q(48) != 0 {
            crate::warn!("avplayer: file replacement callbacks supplied; reading sources from the host file system instead");
        }
        let vbufs = (*(init.add(0x68) as *const i32)).max(2) as usize;
        let p: &'static Player = Box::leak(Box::new(Player {
            magic: AVP_MAGIC,
            cb,
            vbufs,
            shared: Mutex::new(Shared {
                state: State::Idle, source: None, info: None, enabled: Vec::new(), video_info: None, looping: false,
                started: None, paused_at: None, paused_total: Duration::ZERO, loop_base_ms: 0,
                video_q: VecDeque::new(), audio_q: VecDeque::new(), decode_done: false, stop_requested: false, quit: false,
                video_bufs: Vec::new(), video_next: 0, audio_bufs: Vec::new(), audio_next: 0,
            }),
            cv: Condvar::new(),
        }));
        spawn_controller(p);
        crate::info!("sceAvPlayerInit -> {:#x} (event cb {:#x}, {} output video buffers)", p as *const Player as u64, cb.ev_cb, vbufs);
        p as *const Player as u64
    }
    fn sceAvPlayerAddSource(h: u64, path: *const u8) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let name = cstr(path).to_string();
        let Some(host) = crate::hle::kernel::fs::host_path_existing(&name) else {
            crate::error!("avplayer: source '{name}' not found");
            return ERR_OPERATION_FAILED;
        };
        let mut s = p.shared.lock();
        s.source = Some((name.clone(), host));
        s.state = State::Opening;
        drop(s);
        p.cv.notify_all();
        crate::info!("avplayer: AddSource '{name}'");
        0
    }
    fn sceAvPlayerSetLooping(h: u64, looping: bool) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        p.shared.lock().looping = looping;
        0
    }
    fn sceAvPlayerStreamCount(h: u64) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let r = match &p.shared.lock().info { Some(i) => i.streams.len() as i32, None => ERR_OPERATION_FAILED };
        crate::info!("avplayer: StreamCount -> {r:#x}");
        r
    }
    fn sceAvPlayerGetStreamInfo(h: u64, idx: u32, out: *mut u8) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        if out.is_null() { return ERR_INVALID_PARAMS; }
        let s = p.shared.lock();
        let Some(info) = &s.info else { return ERR_OPERATION_FAILED };
        let Some(st) = info.streams.get(idx as usize) else { return ERR_INVALID_PARAMS };
        write_stream_info(out, st, info.duration_ms);
        let w = std::slice::from_raw_parts(out as *const u32, 10);
        crate::info!("avplayer: GetStreamInfo({idx}) -> {:08x?}", w);
        0
    }
    fn sceAvPlayerEnableStream(h: u64, idx: u32) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let mut s = p.shared.lock();
        let Some(info) = &s.info else { return ERR_OPERATION_FAILED };
        let Some(st) = info.streams.get(idx as usize) else { return ERR_INVALID_PARAMS };
        let video = st.video;
        if !s.enabled.contains(&(idx as usize)) { s.enabled.push(idx as usize); }
        crate::info!("avplayer: EnableStream {idx} ({})", if video { "video" } else { "audio" });
        0
    }
    fn sceAvPlayerStart(h: u64) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        {
            let mut s = p.shared.lock();
            let Some(info) = &s.info else { return ERR_OPERATION_FAILED };
            if s.enabled.is_empty() {
                // Titles that never enable streams get the first video and audio stream.
                let first = |v: bool| info.streams.iter().position(|d| d.video == v);
                s.enabled = [first(true), first(false)].into_iter().flatten().collect();
            }
            s.state = State::Playing;
            s.started = Some(Instant::now());
            s.paused_at = None;
            s.paused_total = Duration::ZERO;
        }
        p.cv.notify_all();
        crate::info!("avplayer: Start");
        p.event(EV_STATE_PLAY);
        0
    }
    fn sceAvPlayerStop(h: u64) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let mut s = p.shared.lock();
        if matches!(s.state, State::Playing | State::Paused | State::Ready) {
            s.stop_requested = true;
        }
        drop(s);
        p.cv.notify_all();
        crate::info!("avplayer: Stop");
        0
    }
    fn sceAvPlayerPause(h: u64) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let mut s = p.shared.lock();
        if s.state == State::Playing {
            s.state = State::Paused;
            s.paused_at = Some(Instant::now());
            drop(s);
            p.event(EV_STATE_PAUSE);
        }
        0
    }
    fn sceAvPlayerResume(h: u64) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let mut s = p.shared.lock();
        if s.state == State::Paused {
            if let Some(at) = s.paused_at.take() { s.paused_total += at.elapsed(); }
            s.state = State::Playing;
            drop(s);
            p.cv.notify_all();
            p.event(EV_STATE_PLAY);
        }
        0
    }
    fn sceAvPlayerIsActive(h: u64) -> bool {
        let Some(p) = player(h) else { return false };
        !matches!(p.shared.lock().state, State::Idle | State::Stopped)
    }
    fn sceAvPlayerCurrentTime(h: u64) -> u64 {
        let Some(p) = player(h) else { return 0 };
        Player::clock_ms(&p.shared.lock())
    }
    fn sceAvPlayerGetVideoDataEx(h: u64, out: *mut u8) -> bool {
        let Some(p) = player(h) else { return false };
        if out.is_null() { return false; }
        let (frame, vi, need_alloc) = {
            let mut s = p.shared.lock();
            if s.state != State::Playing { return false; }
            let now = Player::clock_ms(&s);
            // Newest frame that is due; older due frames are dropped.
            let mut pick = None;
            while s.video_q.front().is_some_and(|f| f.ts_ms <= now) {
                pick = s.video_q.pop_front();
            }
            let Some(f) = pick else { return false };
            let vi = s.video_info.unwrap_or_default();
            (f, vi, s.video_bufs.is_empty())
        };
        p.cv.notify_all();
        let (aw, ah) = aligned_dims(&vi);
        let size = (aw * ah * 3 / 2) as u64;
        if need_alloc {
            let n = p.vbufs.max(3);
            let mut bufs = Vec::with_capacity(n);
            for _ in 0..n {
                let b = call_guest(p.cb.alloc_tex, &[p.cb.mem_obj, 0x10000, size]);
                if b == 0 {
                    crate::error!("avplayer: allocateTexture({size:#x}) failed");
                    return false;
                }
                bufs.push(b);
            }
            crate::info!("avplayer: {} video buffers of {size:#x} bytes ({aw}x{ah} NV12)", bufs.len());
            p.shared.lock().video_bufs = bufs;
        }
        let dst = {
            let mut s = p.shared.lock();
            let i = s.video_next % s.video_bufs.len();
            s.video_next += 1;
            s.video_bufs[i]
        };
        // Repack to aligned planes: luma rows then interleaved chroma rows.
        let (fw, fh) = (vi.frame_w as usize, vi.frame_h as usize);
        let (aw_, ah_) = (aw as usize, ah as usize);
        let d = std::slice::from_raw_parts_mut(dst as *mut u8, size as usize);
        for y in 0..ah_ {
            let row = &mut d[y * aw_..(y + 1) * aw_];
            if y < fh { row[..fw].copy_from_slice(&frame.data[y * fw..(y + 1) * fw]); row[fw..].fill(16); } else { row.fill(16); }
        }
        let c0 = ah_ * aw_;
        for y in 0..ah_ / 2 {
            let row = &mut d[c0 + y * aw_..c0 + (y + 1) * aw_];
            if y < fh / 2 { row[..fw].copy_from_slice(&frame.data[(fh + y) * fw..(fh + y + 1) * fw]); row[fw..].fill(128); } else { row.fill(128); }
        }
        std::ptr::write_bytes(out, 0, 104);
        *(out as *mut u64) = dst;
        *(out.add(16) as *mut u64) = frame.ts_ms;
        let v = out.add(24);
        *(v as *mut u32) = aw;
        *(v.add(4) as *mut u32) = ah;
        *(v.add(8) as *mut f32) = vi.width as f32 / vi.height.max(1) as f32;
        std::ptr::copy_nonoverlapping(b"eng\0".as_ptr(), v.add(12), 4);
        *(v.add(16) as *mut u32) = vi.fps_num / vi.fps_den.max(1);
        *(v.add(20) as *mut u32) = 0; // crop left
        *(v.add(24) as *mut u32) = aw - vi.width.min(aw); // crop right
        *(v.add(28) as *mut u32) = 0; // crop top
        *(v.add(32) as *mut u32) = ah - vi.height.min(ah); // crop bottom
        *(v.add(36) as *mut u32) = aw; // pitch
        *v.add(40) = 8; // luma bit depth
        *v.add(41) = 8; // chroma bit depth
        *v.add(42) = 0; // limited range
        true
    }
    fn sceAvPlayerGetAudioData(h: u64, out: *mut u8) -> bool {
        let Some(p) = player(h) else { return false };
        if out.is_null() { return false; }
        let (frame, ai) = {
            let mut s = p.shared.lock();
            if s.state != State::Playing { return false; }
            let Some(f) = s.audio_q.pop_front() else { return false };
            let a = selected(&s, false).and_then(|i| s.info.as_ref().map(|inf| inf.streams[i].a)).unwrap_or_default();
            (f, a)
        };
        p.cv.notify_all();
        let need = frame.pcm.len();
        let slot = {
            let mut s = p.shared.lock();
            let i = s.audio_next % AUDIO_BUFFERS;
            s.audio_next += 1;
            if s.audio_bufs.len() < AUDIO_BUFFERS { s.audio_bufs.resize(AUDIO_BUFFERS, (0, 0)); }
            (i, s.audio_bufs[i])
        };
        let (i, (mut buf, cap)) = slot;
        if buf == 0 || cap < need {
            if buf != 0 { call_guest(p.cb.free, &[p.cb.mem_obj, buf]); }
            buf = call_guest(p.cb.alloc, &[p.cb.mem_obj, 64, need as u64]);
            if buf == 0 {
                crate::error!("avplayer: allocate({need:#x}) failed");
                return false;
            }
            p.shared.lock().audio_bufs[i] = (buf, need);
        }
        std::ptr::copy_nonoverlapping(frame.pcm.as_ptr(), buf as *mut u8, need);
        std::ptr::write_bytes(out, 0, 40);
        *(out as *mut u64) = buf;
        *(out.add(16) as *mut u64) = frame.ts_ms;
        *(out.add(24) as *mut u16) = ai.channels as u16;
        *(out.add(28) as *mut u32) = ai.sample_rate;
        *(out.add(32) as *mut u32) = need as u32;
        std::ptr::copy_nonoverlapping(b"eng\0".as_ptr(), out.add(36), 4);
        true
    }
    fn sceAvPlayerClose(h: u64) -> i32 {
        let Some(p) = player(h) else { return ERR_INVALID_PARAMS };
        let (vbufs, abufs) = {
            let mut s = p.shared.lock();
            s.quit = true;
            s.state = State::Stopped;
            (std::mem::take(&mut s.video_bufs), std::mem::take(&mut s.audio_bufs))
        };
        p.cv.notify_all();
        for b in vbufs { call_guest(p.cb.free_tex, &[p.cb.mem_obj, b]); }
        for (b, _) in abufs { if b != 0 { call_guest(p.cb.free, &[p.cb.mem_obj, b]); } }
        crate::info!("avplayer: Close");
        if let Some(n) = std::env::var("SOTC_BREAKLOG_ON_CLOSE").ok().and_then(|v| v.parse().ok()) {
            crate::exception::BREAK_VERBOSE.store(n, std::sync::atomic::Ordering::Relaxed);
        }
        // The player record stays allocated (the controller thread may still be
        // finishing); it is no longer valid as a handle.
        (*(h as *mut Player)).magic = 0;
        0
    }
}

fixed! {
    sceAvPlayerPostInit => 0,
}

/// Development probe (`sotc --probe-media <file>`): decodes the start of a
/// movie and reports stream parameters and decode speed.
pub fn probe(path: &std::path::Path) {
    let info = match mf::probe(path) {
        Ok(i) => i,
        Err(e) => {
            println!("probe failed: {e}");
            return;
        }
    };
    println!("duration {} ms", info.duration_ms);
    for (i, d) in info.streams.iter().enumerate() {
        println!("stream {i}: {:?}", d);
    }
    let v = info.streams.iter().position(|d| d.video);
    let a = info.streams.iter().position(|d| !d.video);
    let mut d = match mf::Decoder::open(path, &info, v, a) {
        Ok(d) => d,
        Err(e) => {
            println!("open failed: {e}");
            return;
        }
    };
    println!("decoded geometry {:?}", d.video_info);
    let t1 = std::time::Instant::now();
    let mut n = 0;
    let mut last = 0;
    while n < 60 {
        match d.next_video() {
            Ok(Some(f)) => {
                n += 1;
                last = f.ts_ms;
            }
            Ok(None) => break,
            Err(e) => {
                println!("video error: {e}");
                break;
            }
        }
    }
    let el = t1.elapsed();
    println!("decoded {n} video frames in {el:?} ({:.1} ms/frame), last ts {last} ms", el.as_secs_f64() * 1000.0 / n.max(1) as f64);
    let mut an = 0;
    let mut bytes = 0;
    let mut alast = 0;
    while an < 100 {
        match d.next_audio() {
            Ok(Some(f)) => {
                an += 1;
                bytes += f.pcm.len();
                alast = f.ts_ms;
            }
            Ok(None) => break,
            Err(e) => {
                println!("audio error: {e}");
                break;
            }
        }
    }
    println!("decoded {an} audio frames ({bytes} bytes), last ts {alast} ms");
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAvPlayer", [
        sceAvPlayerInit, sceAvPlayerAddSource, sceAvPlayerSetLooping, sceAvPlayerIsActive, sceAvPlayerGetVideoDataEx,
        sceAvPlayerGetAudioData, sceAvPlayerCurrentTime, sceAvPlayerClose, sceAvPlayerPostInit, sceAvPlayerStart,
        sceAvPlayerStop, sceAvPlayerPause, sceAvPlayerResume, sceAvPlayerStreamCount, sceAvPlayerGetStreamInfo,
        sceAvPlayerEnableStream,
    ]);
}
