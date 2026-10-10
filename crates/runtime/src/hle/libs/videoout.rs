//! libSceVideoOut: display buffers, flip queue and flip/vblank events.
//! A vblank thread executes each flip as soon as it is queued: the title's
//! own frame limiter paces its frames and the host swapchain synchronizes
//! with the display. (Executing them at the next 60 Hz tick of a clock
//! unrelated to the title's limiter and to the display, `TLG_FLIP=vblank`,
//! made frames that arrived around a tick alternate between 16.7 and 50 ms
//! at 30 fps: in a 2-minute fight, 117 frames over 45 ms against 13, and
//! 8 ms more latency.) Vblanks tick at 60 Hz either way.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const ERR_INVALID_HANDLE: i32 = 0x8029_000Bu32 as i32;
const ERR_INVALID_INDEX: i32 = 0x8029_0004u32 as i32;
const ERR_FLIP_QUEUE_FULL: i32 = 0x8029_0012u32 as i32;
/// Internal event id of flip events (sceVideoOutGetEventId maps it to 0, Flip).
const EVENT_FLIP: u64 = 6;
const ERR_INVALID_ADDRESS: i32 = 0x8029_0001u32 as i32;
const ERR_INVALID_VALUE: i32 = 0x8029_0002u32 as i32;
const ERR_INVALID_EVENT: i32 = 0x8029_0005u32 as i32;

#[derive(Clone, Copy, Debug, Default)]
pub struct BufferAttr {
    pub pixel_format: u32,
    pub tiling: u32,
    pub aspect: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub option: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplayBuffer {
    pub addr: u64,
    pub attr: BufferAttr,
    /// Attribute index (group) returned by sceVideoOutRegisterBuffers.
    pub group: i32,
}

struct Flip {
    index: i32,
    arg: i64,
    submit_tsc: u64,
    /// Requested through sceGnmSubmitAndFlipCommandBuffers.
    gpu: bool,
    /// GPU frame number (`gpu::FRAMES` after the flip packet; 0 for CPU flips).
    frame: u64,
    /// When it was queued, and (GPU flips) its submission's timing.
    queued: Instant,
    timing: Option<crate::pacing::FlipTiming>,
}

pub struct Port {
    pub buffers: [Option<DisplayBuffer>; 16],
    groups: u32,
    flip_rate: u32,
    flip_events: Vec<(u64, u64)>,
    pending: VecDeque<Flip>,
    pending_per_buffer: [u32; 16],
    /// GPU flips submitted with their command buffers but not yet reached by the GPU.
    gc_unreached: u32,
    count: u64,
    process_time: u64,
    tsc: u64,
    flip_arg: i64,
    submit_tsc: u64,
    current: i32,
    last_flip_vblank: u64,
    vblanks: u64,
}

static PORT: Mutex<Option<Port>> = Mutex::new(None);
static FLIP_CV: Condvar = Condvar::new();
/// Signalled when a flip is queued (wakes the vblank thread in immediate mode).
static QUEUED_CV: Condvar = Condvar::new();
/// Signalled at every vblank tick (sceVideoOutWaitVblank).
static VBLANK_CV: Condvar = Condvar::new();

/// Flips execute as soon as they are queued, unless `TLG_FLIP=vblank` (not
/// above 60 fps or uncapped, `TLG_FPS`, which 60 Hz vblanks would cap).
fn immediate_flips() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let fast = crate::title_patches::target_fps().is_some_and(|n| n == 0 || n > 60);
        let on = fast || !std::env::var("TLG_FLIP").is_ok_and(|v| v == "vblank");
        if !on {
            crate::info!("TLG_FLIP=vblank: flips execute at 60 Hz vblanks");
        }
        on
    })
}

fn vblank_thread() {
    crate::log::set_thread_name("vblank");
    let period = Duration::from_nanos(16_666_667);
    let mut next = Instant::now() + period;
    let immediate = immediate_flips();
    loop {
        // The next vblank, or (immediate mode) a queued flip before it.
        {
            let mut g = PORT.lock();
            while Instant::now() < next && !(immediate && g.as_ref().is_some_and(|p| !p.pending.is_empty())) {
                QUEUED_CV.wait_until(&mut g, next);
            }
        }
        let tick = Instant::now() >= next;
        if tick {
            next += period;
        }
        let mut fired: Vec<(u64, u64, i64)> = Vec::new();
        let mut show: Option<(DisplayBuffer, u64, Option<crate::pacing::FlipTiming>, Duration)> = None;
        {
            let mut g = PORT.lock();
            let Some(p) = g.as_mut() else { continue };
            if tick {
                p.vblanks += 1;
                VBLANK_CV.notify_all();
            }
            let interval = p.flip_rate as u64 + 1;
            if !p.pending.is_empty() && (immediate || p.vblanks - p.last_flip_vblank >= interval) {
                let f = p.pending.pop_front().unwrap();
                if f.index >= 0 {
                    p.pending_per_buffer[f.index as usize] -= 1;
                }
                p.current = f.index;
                p.count += 1;
                p.flip_arg = f.arg;
                p.submit_tsc = f.submit_tsc;
                p.process_time = crate::hle::kernel::time::process_time_us();
                p.tsc = unsafe { core::arch::x86_64::_rdtsc() };
                p.last_flip_vblank = p.vblanks;
                if p.count <= 3 || p.count % 600 == 0 {
                    crate::info!("flip #{} -> buffer {} (arg {:#x})", p.count, f.index, f.arg);
                }
                if f.index >= 0 {
                    show = p.buffers[f.index as usize].map(|b| (b, f.frame, f.timing, f.queued.elapsed()));
                }
                for &(eq, ud) in &p.flip_events {
                    fired.push((eq, ud, f.arg));
                }
                FLIP_CV.notify_all();
            }
        }
        if let Some((b, frame, timing, flip_wait)) = show {
            let n = PORT.lock().as_ref().map(|p| p.count).unwrap_or(0);
            maybe_screenshot(&b, n, frame);
            if let Some((lock_wait, present, prof)) = present_buffer(&b) {
                crate::pacing::presented(n, lock_wait, present, prof, crate::title_patches::guest_frame_ms(), timing, flip_wait);
            }
        }
        for (eq, _ud, arg) in fired {
            if let Some(q) = unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
                // Hardware layout: the internal id in the low 16 bits, the flip argument above.
                q.trigger(EVENT_FLIP, crate::hle::kernel::equeue::EVFILT_DISPLAY, Some(EVENT_FLIP | ((arg as u64) << 16)), None);
            }
        }
    }
}

fn submit_flip(handle: i32, index: i32, arg: i64, gpu: bool, timing: Option<crate::pacing::FlipTiming>) -> i32 {
    if handle != 1 {
        return ERR_INVALID_HANDLE;
    }
    let mut g = PORT.lock();
    let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
    if gpu {
        p.gc_unreached = p.gc_unreached.saturating_sub(1);
    }
    if index >= 16 || (index >= 0 && p.buffers[index as usize].is_none()) {
        return ERR_INVALID_INDEX;
    }
    if p.pending.len() >= 16 {
        return ERR_FLIP_QUEUE_FULL;
    }
    if index >= 0 {
        p.pending_per_buffer[index as usize] += 1;
    }
    let frame = if gpu { crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) } else { 0 };
    p.pending.push_back(Flip { index, arg, submit_tsc: unsafe { core::arch::x86_64::_rdtsc() }, gpu, frame, queued: Instant::now(), timing });
    QUEUED_CV.notify_all();
    0
}

/// Registers a flip submitted together with command buffers; it stays
/// pending (gcQueueNum/flipPendingNum) until the GPU reaches it and the
/// vblank thread executes it.
pub fn gpu_flip_requested(handle: i32) {
    if handle != 1 {
        return;
    }
    if let Some(p) = PORT.lock().as_mut() {
        p.gc_unreached += 1;
    }
}

/// Flip requested by the GPU after its command buffers completed.
pub fn gpu_flip(handle: i32, index: i32, _mode: u32, arg: i64, timing: crate::pacing::FlipTiming) {
    let r = submit_flip(handle, index, arg, true, Some(timing));
    if r != 0 {
        crate::error!("GPU flip of buffer {index} failed: {r:#x}");
    }
}

/// Blocks (GPU side) until no flip of `index` is pending.
pub fn wait_flip_done(handle: i32, index: i32) {
    crate::gpu::set_phase(crate::gpu::phase::FLIP_WAIT, index as u64);
    if handle != 1 || !(0..16).contains(&index) {
        return;
    }
    let mut g = PORT.lock();
    let start = Instant::now();
    let deadline = start + Duration::from_secs(5);
    let mut warned = false;
    while g.as_ref().is_some_and(|p| p.pending_per_buffer[index as usize] > 0) {
        if !warned && start.elapsed() > Duration::from_millis(250) {
            warned = true;
            crate::warn!("GPU wait for flip of buffer {index} blocked for 250 ms (pending {:?})", g.as_ref().map(|p| p.pending.len()));
        }
        if FLIP_CV.wait_until(&mut g, deadline.min(Instant::now() + Duration::from_millis(100))).timed_out() && Instant::now() >= deadline {
            crate::warn!("wait for flip of buffer {index} timed out");
            return;
        }
    }
}

pub fn display_buffer(index: i32) -> Option<DisplayBuffer> {
    PORT.lock().as_ref().and_then(|p| p.buffers.get(index as usize).copied().flatten())
}

sysv! {
    fn sceVideoOutOpen(user: i32, bus: i32, index: i32, param: *const u8) -> i32 {
        if bus != 0 { return 0x8029_0002u32 as i32; }
        let mut g = PORT.lock();
        if g.is_some() { return 0x8029_0009u32 as i32; } // resource busy
        *g = Some(Port {
            buffers: [None; 16],
            groups: 0,
            flip_rate: 0,
            flip_events: Vec::new(),
            pending: VecDeque::new(),
            pending_per_buffer: [0; 16],
            gc_unreached: 0,
            count: 0,
            process_time: 0,
            tsc: 0,
            flip_arg: 0,
            submit_tsc: 0,
            current: -1,
            last_flip_vblank: 0,
            vblanks: 0,
        });
        drop(g);
        crate::thread::spawn_host("vblank", 1 << 20, Box::new(vblank_thread)).expect("vblank thread");
        crate::info!("sceVideoOutOpen -> 1");
        1
    }
    fn sceVideoOutClose(h: i32) -> i32 {
        if h != 1 { return ERR_INVALID_HANDLE; }
        0
    }
    fn sceVideoOutSetFlipRate(h: i32, rate: i32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        p.flip_rate = rate.clamp(0, 2) as u32;
        crate::info!("sceVideoOutSetFlipRate({rate})");
        0
    }
    fn sceVideoOutAddFlipEvent(eq: u64, h: i32, udata: u64) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        let Some(q) = crate::hle::kernel::equeue::from_handle(eq) else { return 0x8029_0001u32 as i32 };
        q.add(EVENT_FLIP, crate::hle::kernel::equeue::EVFILT_DISPLAY, 0, udata);
        p.flip_events.push((eq, udata));
        0
    }
    fn sceVideoOutSetBufferAttribute(attr: *mut u32, fmt: u32, tiling: u32, aspect: u32, w: u32, h: u32, pitch: u32) {
        let v = [fmt, tiling, aspect, w, h, pitch, 0, 0, 0, 0];
        for (i, x) in v.iter().enumerate() { *attr.add(i) = *x; }
    }
    fn sceVideoOutRegisterBuffers(h: i32, start: i32, addrs: *const u64, num: i32, attr: *const u32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        if start < 0 || num <= 0 || start + num > 16 || addrs.is_null() || attr.is_null() { return ERR_INVALID_INDEX; }
        let a = BufferAttr {
            pixel_format: *attr, tiling: *attr.add(1), aspect: *attr.add(2), width: *attr.add(3),
            height: *attr.add(4), pitch: *attr.add(5), option: *attr.add(6),
        };
        for i in 0..num as usize {
            p.buffers[start as usize + i] = Some(DisplayBuffer { addr: *addrs.add(i), attr: a, group: p.groups as i32 });
        }
        let group = p.groups;
        p.groups += 1;
        crate::info!(
            "sceVideoOutRegisterBuffers [{}..{}) {}x{} pitch {} format {:#x} tiling {} -> group {} addrs {:x?}",
            start, start + num, a.width, a.height, a.pitch, a.pixel_format, a.tiling, group,
            (0..num as usize).map(|i| *addrs.add(i)).collect::<Vec<_>>()
        );
        group as i32
    }
    fn sceVideoOutSubmitFlip(h: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit_flip(h, index, arg, false, None)
    }
    fn sceVideoOutGetFlipStatus(h: i32, st: *mut u64) -> i32 {
        let g = PORT.lock();
        let Some(p) = g.as_ref() else { return ERR_INVALID_HANDLE };
        if st.is_null() { return 0x8029_0001u32 as i32; }
        *st = p.count;
        *st.add(1) = p.process_time;
        *st.add(2) = p.tsc;
        *st.add(3) = p.flip_arg as u64;
        *st.add(4) = p.submit_tsc;
        *st.add(5) = 0;
        let i = st.add(6) as *mut i32;
        let gpu_queued = p.pending.iter().filter(|f| f.gpu).count() as u32;
        *i = (p.gc_unreached + gpu_queued) as i32; // gcQueueNum
        *i.add(1) = (p.gc_unreached as usize + p.pending.len()) as i32; // flipPendingNum
        *i.add(2) = p.current;
        *i.add(3) = 0;
        0
    }
    fn sceVideoOutDeleteFlipEvent(eq: u64, h: i32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        let Some(q) = crate::hle::kernel::equeue::from_handle(eq) else { return ERR_INVALID_ADDRESS };
        q.remove(EVENT_FLIP, crate::hle::kernel::equeue::EVFILT_DISPLAY);
        p.flip_events.retain(|&(e, _)| e != eq);
        0
    }
    fn sceVideoOutGetEventId(ev: *const crate::hle::kernel::equeue::Kevent) -> i32 {
        if ev.is_null() { return ERR_INVALID_ADDRESS; }
        if (*ev).filter != crate::hle::kernel::equeue::EVFILT_DISPLAY { return ERR_INVALID_EVENT; }
        match (*ev).ident {
            6 => 0,          // flip
            7 | 0x63 => 1,   // vblank
            0x59 => 2,       // pre-vblank start
            0x51 => 8,       // set mode
            0x58 => 12,      // position
            _ => ERR_INVALID_EVENT,
        }
    }
    fn sceVideoOutGetEventData(ev: *const crate::hle::kernel::equeue::Kevent, data: *mut i64) -> i32 {
        if ev.is_null() || data.is_null() { return ERR_INVALID_ADDRESS; }
        if (*ev).filter != crate::hle::kernel::equeue::EVFILT_DISPLAY { return ERR_INVALID_EVENT; }
        // Flip arguments are 48-bit signed above the 16-bit id.
        *data = ((*ev).data as i64) >> 16;
        0
    }
    fn sceVideoOutIsFlipPending(h: i32) -> i32 {
        let g = PORT.lock();
        let Some(p) = g.as_ref() else { return ERR_INVALID_HANDLE };
        (p.gc_unreached as usize + p.pending.len()) as i32
    }
    fn sceVideoOutWaitVblank(h: i32) -> i32 {
        let mut g = PORT.lock();
        let Some(start) = g.as_ref().map(|p| p.vblanks) else { return ERR_INVALID_HANDLE };
        while g.as_ref().is_some_and(|p| p.vblanks == start) {
            VBLANK_CV.wait_for(&mut g, Duration::from_millis(50));
        }
        0
    }
    fn sceVideoOutGetDeviceCapabilityInfo_(h: i32, info: *mut u64) -> i32 {
        if info.is_null() { return ERR_INVALID_ADDRESS; }
        *info = 0; // no BT.2020 PQ (HDR) output: the title renders its SDR path
        0
    }
    fn sceVideoOutModeSetAny_(mode: *mut u8, size: u32) {
        if mode.is_null() { return; }
        std::ptr::write_bytes(mode, 0xFF, size as usize);
        *(mode as *mut u32) = size;
    }
    fn sceVideoOutConfigureOutputMode_(h: i32, reserved: u32, mode: *const u8, options: *const u8) -> i32 {
        if PORT.lock().is_none() { return ERR_INVALID_HANDLE; }
        if reserved != 0 || mode.is_null() { return ERR_INVALID_VALUE; }
        // Mode: {u32 size; u8 encoding, range, colorimetry, depth; u64 refresh; u64 resolution}.
        let colorimetry = *mode.add(6);
        crate::info!("sceVideoOutConfigureOutputMode_: colorimetry {colorimetry:#x} (SDR output kept)");
        if colorimetry != 0xFF && colorimetry != 0 { return ERR_INVALID_VALUE; }
        0
    }
    fn sceVideoOutColorSettingsSetGamma_(settings: *mut f32, gamma: f32) -> i32 {
        if settings.is_null() { return ERR_INVALID_ADDRESS; }
        if !(0.1..=2.0).contains(&gamma) { return ERR_INVALID_VALUE; }
        *settings = gamma;
        0
    }
    fn sceVideoOutAdjustColor_(h: i32, settings: *const f32) -> i32 {
        if settings.is_null() { return ERR_INVALID_ADDRESS; }
        if PORT.lock().is_none() { return ERR_INVALID_HANDLE; }
        let gamma = *settings;
        if (gamma - 1.0).abs() > 1e-3 {
            crate::warn!("sceVideoOutAdjustColor_: display gamma {gamma} requested; output gamma adjustment is not applied");
        }
        0
    }
    fn sceVideoOutSetWindowModeMargins(h: i32, top: i32, bottom: i32) -> i32 { 0 }
    fn sceVideoOutSubmitChangeBufferAttribute(h: i32, group: i32, attr: *const u32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        if attr.is_null() { return ERR_INVALID_ADDRESS; }
        let a = BufferAttr {
            pixel_format: *attr, tiling: *attr.add(1), aspect: *attr.add(2), width: *attr.add(3),
            height: *attr.add(4), pitch: *attr.add(5), option: *attr.add(6),
        };
        let mut n = 0;
        for b in p.buffers.iter_mut().flatten().filter(|b| b.group == group) {
            b.attr = a;
            n += 1;
        }
        crate::info!("sceVideoOutSubmitChangeBufferAttribute(group {group}): {}x{} pitch {} format {:#x} -> {n} buffers", a.width, a.height, a.pitch, a.pixel_format);
        if n == 0 { ERR_INVALID_VALUE } else { 0 }
    }
    fn sceVideoOutUnregisterBuffers(h: i32, group: i32) -> i32 {
        let mut g = PORT.lock();
        let Some(p) = g.as_mut() else { return ERR_INVALID_HANDLE };
        let mut n = 0;
        for b in p.buffers.iter_mut() {
            if b.is_some_and(|b| b.group == group) { *b = None; n += 1; }
        }
        if n == 0 { ERR_INVALID_VALUE } else { 0 }
    }
    fn sceVideoOutGetResolutionStatus(h: i32, st: *mut u32) -> i32 {
        if st.is_null() { return 0x8029_0001u32 as i32; }
        *st = 1920; *st.add(1) = 1080; *st.add(2) = 1920; *st.add(3) = 1080;
        *(st.add(4) as *mut u64) = 3; // 59.94 Hz class
        *(st.add(6) as *mut f32) = 50.0;
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceVideoOut", [
        sceVideoOutOpen, sceVideoOutClose, sceVideoOutSetFlipRate, sceVideoOutAddFlipEvent,
        sceVideoOutSetBufferAttribute, sceVideoOutRegisterBuffers, sceVideoOutSubmitFlip, sceVideoOutGetFlipStatus,
        sceVideoOutGetResolutionStatus, sceVideoOutDeleteFlipEvent, sceVideoOutGetEventId, sceVideoOutGetEventData,
        sceVideoOutIsFlipPending, sceVideoOutWaitVblank, sceVideoOutGetDeviceCapabilityInfo_, sceVideoOutModeSetAny_,
        sceVideoOutConfigureOutputMode_, sceVideoOutColorSettingsSetGamma_, sceVideoOutAdjustColor_,
        sceVideoOutSetWindowModeMargins, sceVideoOutSubmitChangeBufferAttribute, sceVideoOutUnregisterBuffers,
    ]);
}

/// Presents a display buffer. Until the GPU renders into Vulkan images, the
/// guest memory contents are shown directly (linear interpretation).
/// Presents a display buffer; when a frame was presented, returns the wait
/// for the renderer lock, the present call's duration and the renderer's
/// time counters (for `pacing`).
fn present_buffer(b: &DisplayBuffer) -> Option<(Duration, Duration, Option<crate::pacing::Prof>)> {
    crate::gpu::vk::try_ctx()?;
    let (w, h, pitch) = (b.attr.width, b.attr.height, b.attr.pitch.max(b.attr.width));
    let t0 = Instant::now();
    let mut g = crate::gpu::render::RENDERER.lock();
    let lock_wait = t0.elapsed();
    let prof = g.as_ref().map(|r| r.prof);
    let rendered = g.as_mut().and_then(|r| r.display_image(b.addr, w, h));
    drop(g);
    let t1 = Instant::now();
    // (Counted for pacing whether or not it reached the screen: a minimized
    // window presents nothing while the title keeps flipping.)
    if let Some((image, iw, ih)) = rendered {
        if !crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Image { image, layout: ash::vk::ImageLayout::GENERAL, width: iw, height: ih }) {
            crate::pacing::not_displayed();
        }
        return Some((lock_wait, t1.elapsed(), prof));
    }
    if w == 0 || h == 0 || w > 8192 || h > 8192 {
        return None;
    }
    let mut data = vec![0u8; (w * h * 4) as usize];
    unsafe {
        for y in 0..h as usize {
            let src = (b.addr + (y as u64) * pitch as u64 * 4) as *const u8;
            std::ptr::copy_nonoverlapping(src, data.as_mut_ptr().add(y * w as usize * 4), w as usize * 4);
        }
    }
    if !crate::gpu::vk::present::present(crate::gpu::vk::present::Source::Pixels { data, width: w, height: h }) {
        crate::pacing::not_displayed();
    }
    Some((lock_wait, t1.elapsed(), prof))
}

/// TLG_SCREENSHOT=n1,n2,... writes the displayed image of those flips as PNG.
/// Screenshot requested through the control channel (file name stem).
pub static SHOT_REQUEST: Mutex<Option<String>> = Mutex::new(None);
/// Screenshot of the image a given GPU frame displays (`traceshot`): frame
/// number (as tagged on its flip) and file name stem.
pub static SHOT_FRAME: Mutex<Option<(u64, String)>> = Mutex::new(None);

fn maybe_screenshot(b: &DisplayBuffer, flip: u64, frame: u64) {
    let requested = {
        let mut g = SHOT_FRAME.lock();
        match g.as_ref() {
            Some((f, _)) if *f == frame => g.take().map(|(_, n)| n),
            // (Missed: that frame was never displayed.)
            Some((f, _)) if *f < frame && frame != 0 => {
                crate::warn!("traceshot of frame {f} missed (now displaying frame {frame})");
                g.take();
                None
            }
            _ => None,
        }
    }
    .or_else(|| SHOT_REQUEST.lock().take());
    let listed = std::env::var("TLG_SCREENSHOT").is_ok_and(|list| list.split(',').any(|s| s.trim().parse::<u64>().ok() == Some(flip)));
    if requested.is_none() && !listed {
        return;
    }
    let (w, h) = (b.attr.width, b.attr.height);
    let px = crate::gpu::render::RENDERER.lock().as_mut().and_then(|r| r.capture(b.addr, w, h));
    let Some(px) = px else {
        crate::warn!("screenshot of flip {flip}: no rendered image for buffer {:#x}", b.addr);
        return;
    };
    if let Some(dir) = crate::gpu::shader_dump::DIR.get().and_then(|d| d.parent()) {
        let path = match &requested {
            Some(name) => dir.join(format!("shot_{name}.png")),
            None => dir.join(format!("flip{flip:05}.png")),
        };
        // Encoded and written off the vblank thread (an encode takes ~0.2 s
        // and would hold up presentation); renamed into place when complete.
        let frames = crate::gpu::recent_frames();
        let spawned = std::thread::Builder::new().name("screenshot".into()).spawn(move || {
            let tmp = path.with_extension("png.tmp");
            if std::fs::write(&tmp, crate::gpu::texdump::png(w, h, &px)).and_then(|_| std::fs::rename(&tmp, &path)).is_ok() {
                crate::info!("screenshot written: {} (flip:draws/dispatches/skipped of the last frames: {frames})", path.display());
            }
        });
        if let Err(e) = spawned {
            crate::warn!("screenshot of flip {flip}: {e}");
        }
    }
}
