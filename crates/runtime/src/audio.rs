//! Host audio output for the title's AudioOut ports.
//!
//! Each port queues its blocks (downmixed to stereo, port volume applied); a
//! device thread mixes the ports and plays them through WASAPI (shared mode,
//! 48 kHz float stereo; the system converts to the device format). An output
//! call blocks while its port has `MAX_QUEUED` frames queued, so the title's
//! mixer is paced by the audio device, as it is by the hardware.
//!
//! Without a device (none present, initialization failed, or TLG_AUDIO=0) the
//! caller paces output in real time and the audio is discarded. When the
//! device goes away (unplugged, default device changed) the device thread
//! opens the default device again (every 2 s until one works). The mix is
//! scaled by the master volume of settings.txt (`volume`, 0-100: gain
//! (v/100)^2).
//! TLG_AUDIO_DUMP=<dir>: each port's stereo stream is also written to
//! `<dir>/port<N>.f32` (interleaved little-endian float32, 48 kHz).

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::io::Write;

pub const RATE: u32 = 48000;
/// Frames a port may queue before its output call blocks (~43 ms).
const MAX_QUEUED: usize = 2048;

struct Port {
    fifo: VecDeque<[f32; 2]>,
    /// Per-channel gain (sceAudioOutSetVolume, 32768 = 0 dB).
    volume: [f32; 8],
    dump: Option<std::fs::File>,
}

struct Mixer {
    ports: Vec<Option<Port>>,
    /// The device thread is playing.
    playing: bool,
    started: bool,
}

static MIXER: Mutex<Mixer> = Mutex::new(Mixer { ports: Vec::new(), playing: false, started: false });
static CV: Condvar = Condvar::new();

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_AUDIO").map(|v| v != "0").unwrap_or(true))
}

pub fn open_port(idx: usize) {
    let dump = std::env::var_os("TLG_AUDIO_DUMP").and_then(|d| {
        let dir = std::path::PathBuf::from(d);
        let _ = std::fs::create_dir_all(&dir);
        std::fs::File::create(dir.join(format!("port{}.f32", idx + 1))).ok()
    });
    let mut m = MIXER.lock();
    if m.ports.len() <= idx {
        m.ports.resize_with(idx + 1, || None);
    }
    m.ports[idx] = Some(Port { fifo: VecDeque::new(), volume: [1.0; 8], dump });
    if !m.started && enabled() {
        m.started = true;
        std::thread::Builder::new().name("audio-out".into()).spawn(device_thread).ok();
    }
}

pub fn close_port(idx: usize) {
    if let Some(p) = MIXER.lock().ports.get_mut(idx) {
        *p = None;
    }
    CV.notify_all();
}

pub fn set_volume(idx: usize, channel: usize, vol: i32) {
    if let Some(Some(p)) = MIXER.lock().ports.get_mut(idx) {
        if channel < 8 {
            p.volume[channel] = (vol.max(0) as f32 / 32768.0).min(4.0);
        }
    }
}

/// Downmixes one frame of `ch` channels (PS4 order: FL FR FC LFE, then the
/// surround pairs) to stereo.
fn downmix(s: &[f32], vol: &[f32; 8]) -> [f32; 2] {
    let g = |i: usize| s[i] * vol[i];
    match s.len() {
        1 => [g(0), g(0)],
        2 => [g(0), g(1)],
        _ => {
            const C: f32 = 0.7071;
            let (l, r, c) = (g(0), g(1), g(2));
            let (sl, sr) = (g(4) + g(6), g(5) + g(7));
            [0.8 * (l + C * c + C * sl), 0.8 * (r + C * c + C * sr)]
        }
    }
}

/// Queues one block of `frames` frames with `channels` channels (16-bit or
/// float samples) for port `idx`. Returns false when no device plays the
/// audio (the caller then paces in real time).
///
/// # Safety
/// `data` must point to `frames * channels` samples.
pub unsafe fn output(idx: usize, data: *const u8, frames: usize, channels: usize, float: bool) -> bool {
    let ch = channels.clamp(1, 8);
    let mut m = MIXER.lock();
    let Some(Some(port)) = m.ports.get(idx) else { return false };
    let vol = port.volume;
    let mut block = Vec::with_capacity(frames);
    if !data.is_null() {
        let mut frame = [0f32; 8];
        for f in 0..frames {
            for (c, v) in frame.iter_mut().enumerate().take(ch) {
                let i = f * ch + c;
                *v = if float { std::ptr::read_unaligned((data as *const f32).add(i)) } else { std::ptr::read_unaligned((data as *const i16).add(i)) as f32 / 32768.0 };
            }
            block.push(downmix(&frame[..ch], &vol));
        }
    }
    if let Some(Some(port)) = m.ports.get_mut(idx) {
        if let Some(f) = &mut port.dump {
            let bytes: Vec<u8> = block.iter().flat_map(|s| s.iter().flat_map(|v| v.to_le_bytes())).collect();
            let _ = f.write_all(&bytes);
        }
    }
    if !m.playing {
        return false;
    }
    // Paced by the device: wait while this port has enough queued.
    loop {
        let queued = match m.ports.get(idx) {
            Some(Some(p)) => p.fifo.len(),
            _ => return true,
        };
        if queued < MAX_QUEUED || !m.playing {
            break;
        }
        CV.wait_for(&mut m, std::time::Duration::from_millis(100));
    }
    if let Some(Some(p)) = m.ports.get_mut(idx) {
        p.fifo.extend(block);
    }
    m.playing
}

/// Mixes the ports into `out` (interleaved stereo frames).
fn mix(out: &mut [f32]) {
    out.fill(0.0);
    let mut m = MIXER.lock();
    for p in m.ports.iter_mut().flatten() {
        for frame in out.chunks_exact_mut(2) {
            let Some(s) = p.fifo.pop_front() else { break };
            frame[0] += s[0];
            frame[1] += s[1];
        }
    }
    let gain = master_gain();
    for v in out.iter_mut() {
        *v = (*v * gain).clamp(-1.0, 1.0);
    }
    drop(m);
    CV.notify_all();
}

fn master_gain() -> f32 {
    static G: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        let v = crate::config::settings().volume.min(100) as f32 / 100.0;
        v * v
    })
}

fn device_thread() {
    crate::log::set_thread_name("audio-out");
    let mut failures = 0u32;
    loop {
        let r = unsafe { run_device() };
        MIXER.lock().playing = false;
        CV.notify_all();
        if let Err(e) = r {
            if failures == 0 {
                crate::warn!("audio: host output unavailable or lost ({e}); audio is paced in real time without sound until a device works");
            }
            failures += 1;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

unsafe fn run_device() -> windows::core::Result<()> {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::*;
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
    let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
    let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
    let wf = WAVEFORMATEX {
        wFormatTag: 3, // WAVE_FORMAT_IEEE_FLOAT
        nChannels: 2,
        nSamplesPerSec: RATE,
        nAvgBytesPerSec: RATE * 8,
        nBlockAlign: 8,
        wBitsPerSample: 32,
        cbSize: 0,
    };
    // 30 ms shared-mode buffer; rate/format conversion by the audio engine.
    client.Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
        300_000,
        0,
        &wf,
        None,
    )?;
    let event = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
    client.SetEventHandle(HANDLE(event as _))?;
    let render: IAudioRenderClient = client.GetService()?;
    let total = client.GetBufferSize()?;
    client.Start()?;
    crate::info!("audio: WASAPI shared-mode output, 48 kHz float stereo, buffer {total} frames, volume {}%", crate::config::settings().volume);
    MIXER.lock().playing = true;
    CV.notify_all();
    let mut buf = vec![0f32; total as usize * 2];
    loop {
        WaitForSingleObject(event, 100);
        let padding = client.GetCurrentPadding()?;
        let n = total.saturating_sub(padding);
        if n == 0 {
            continue;
        }
        let out = &mut buf[..n as usize * 2];
        mix(out);
        let p = render.GetBuffer(n)? as *mut f32;
        std::ptr::copy_nonoverlapping(out.as_ptr(), p, out.len());
        render.ReleaseBuffer(n, 0)?;
    }
}
