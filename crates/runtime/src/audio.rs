//! Host audio output for the title's AudioOut ports.
//!
//! Each port queues its blocks as 8-channel frames in the PS4's FLOAT_8CH
//! order (L R C LFE Ls Rs Lb Rb; port volume applied). The title's two ports
//! are 8-channel and the port state tells it 8 speakers, so its own mix is
//! 7.1. A device thread mixes the ports into the speaker layout (PC Settings
//! `speakers`: stereo, 5.1, 7.1, headphones - the 7.1 channels as static
//! objects of a Windows spatial audio stream, rendered by Windows Sonic or
//! Dolby Atmos for Headphones -, or automatic: that stream where the device
//! has spatial audio on, else the device's own channels) and
//! plays them through WASAPI (shared mode, 48 kHz float; the system converts
//! to the device format). An output call blocks while its port has
//! `MAX_QUEUED` frames queued, so the title's mixer is paced by the audio
//! device, as it is by the hardware.
//!
//! Without a device (none present, initialization failed, or TLG_AUDIO=0) the
//! caller paces output in real time and the audio is discarded. When the
//! device goes away (unplugged) the device thread opens the device again
//! (every 2 s until one works); playing on Windows' default device, it
//! follows a change of the default. The mix is scaled by the master volume
//! of settings.txt (`volume`, 0-100: gain (v/100)^2).
//! TLG_AUDIO_DUMP=<dir>: each port's stereo downmix is also written to
//! `<dir>/port<N>.f32` (interleaved little-endian float32, 48 kHz).

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::io::Write;

pub const RATE: u32 = 48000;
/// Frames a port may queue before its output call blocks (~43 ms).
const MAX_QUEUED: usize = 2048;

/// One frame of a port: the PS4's FLOAT_8CH order, L R C LFE Ls Rs Lb Rb
/// (mono and stereo ports fill L and R).
type Frame = [f32; 8];

struct Port {
    fifo: VecDeque<Frame>,
    /// The port's channels (1, 2 or 8).
    channels: usize,
    /// Has had audio (underruns count only after that).
    started: bool,
    /// Ran dry in the last mix: its next frames fade in (`FADE`).
    dry: bool,
    /// Per-channel gain (sceAudioOutSetVolume, 32768 = 0 dB).
    volume: [f32; 8],
    dump: Option<std::fs::File>,
}

struct Mixer {
    ports: Vec<Option<Port>>,
    /// The device thread is playing.
    playing: bool,
    started: bool,
    /// The output limiter's peak envelope (`mix`).
    limit_env: f32,
}

static MIXER: Mutex<Mixer> = Mutex::new(Mixer { ports: Vec::new(), playing: false, started: false, limit_env: 0.0 });
static CV: Condvar = Condvar::new();
/// Frames written to each port's TLG_AUDIO_DUMP file (for placing logged
/// events in the dump).
static DUMPED: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// Frames written so far to port `idx`'s dump file.
pub fn dumped_frames(idx: usize) -> u64 {
    DUMPED.get(idx).map_or(0, |d| d.load(std::sync::atomic::Ordering::Relaxed))
}

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_AUDIO").map(|v| v != "0").unwrap_or(true))
}

pub fn open_port(idx: usize, channels: usize) {
    let dump = std::env::var_os("TLG_AUDIO_DUMP").and_then(|d| {
        let dir = std::path::PathBuf::from(d);
        let _ = std::fs::create_dir_all(&dir);
        std::fs::File::create(dir.join(format!("port{}.f32", idx + 1))).ok()
    });
    let mut m = MIXER.lock();
    if m.ports.len() <= idx {
        m.ports.resize_with(idx + 1, || None);
    }
    m.ports[idx] = Some(Port { fifo: VecDeque::new(), channels: channels.clamp(1, 8), started: false, dry: false, volume: [1.0; 8], dump });
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

/// The stereo downmix of a port frame (`ch`: the port's channels): stereo
/// as it is, 8 channels with the centre and both surround pairs at -3 dB,
/// the LFE left out, all at 0.8.
fn downmix(s: &Frame, ch: usize) -> [f32; 2] {
    if ch <= 2 {
        return [s[0], s[1]];
    }
    const C: f32 = 0.7071;
    let (l, r, c) = (s[0], s[1], s[2]);
    let (sl, sr) = (s[4] + s[6], s[5] + s[7]);
    [0.8 * (l + C * c + C * sl), 0.8 * (r + C * c + C * sr)]
}

/// The speaker layouts played (WAVEFORMATEXTENSIBLE channel masks, the
/// channels in Windows' order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Stereo,
    /// FL FR FC LFE SL SR.
    Surround51,
    /// FL FR FC LFE BL BR SL SR (also the spatial stream's static objects).
    Surround71,
}

impl Layout {
    fn channels(self) -> usize {
        match self {
            Layout::Stereo => 2,
            Layout::Surround51 => 6,
            Layout::Surround71 => 8,
        }
    }

    fn mask(self) -> u32 {
        match self {
            Layout::Stereo => 0x3,
            Layout::Surround51 => 0x60F,
            Layout::Surround71 => 0x63F,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Layout::Stereo => "stereo",
            Layout::Surround51 => "5.1",
            Layout::Surround71 => "7.1",
        }
    }
}

/// Adds a port frame (`ch` channels, at gain `g`) to an output frame of the
/// layout. 5.1 takes the back pair into the sides at -3 dB.
fn render(layout: Layout, s: &Frame, ch: usize, g: f32, out: &mut [f32]) {
    if layout == Layout::Stereo || ch <= 2 {
        let [l, r] = downmix(s, ch);
        out[0] += l * g;
        out[1] += r * g;
        return;
    }
    for i in 0..4 {
        out[i] += s[i] * g;
    }
    const B: f32 = 0.7071;
    match layout {
        Layout::Surround51 => {
            out[4] += (s[4] + B * s[6]) * g;
            out[5] += (s[5] + B * s[7]) * g;
        }
        _ => {
            out[4] += s[6] * g;
            out[5] += s[7] * g;
            out[6] += s[4] * g;
            out[7] += s[5] * g;
        }
    }
}

/// Queues one block of `frames` frames with `channels` channels (16-bit or
/// float samples; `std`: the 8CH_STD order, L R C LFE Lb Rb Ls Rs) for port
/// `idx`. Returns false when no device plays the audio (the caller then
/// paces in real time).
///
/// # Safety
/// `data` must point to `frames * channels` samples.
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn output(idx: usize, data: *const u8, frames: usize, channels: usize, float: bool, std: bool) -> bool {
    let ch = channels.clamp(1, 8);
    let mut m = MIXER.lock();
    let Some(Some(port)) = m.ports.get(idx) else { return false };
    let vol = port.volume;
    let mut block: Vec<Frame> = Vec::with_capacity(frames);
    if !data.is_null() {
        for f in 0..frames {
            let mut frame = [0f32; 8];
            for (c, v) in frame.iter_mut().enumerate().take(ch) {
                let i = f * ch + c;
                let x = if float { std::ptr::read_unaligned((data as *const f32).add(i)) } else { std::ptr::read_unaligned((data as *const i16).add(i)) as f32 / 32768.0 };
                *v = x * vol[c];
            }
            if ch == 1 {
                frame[1] = frame[0];
            }
            if std && ch == 8 {
                frame.swap(4, 6);
                frame.swap(5, 7);
            }
            block.push(frame);
        }
    }
    if let Some(Some(port)) = m.ports.get_mut(idx) {
        if let Some(f) = &mut port.dump {
            let bytes: Vec<u8> = block.iter().flat_map(|s| downmix(s, ch).into_iter().flat_map(|v| v.to_le_bytes())).collect();
            let _ = f.write_all(&bytes);
            if let Some(d) = DUMPED.get(idx) {
                d.fetch_add(block.len() as u64, std::sync::atomic::Ordering::Relaxed);
            }
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

/// Mixes the ports into `out` (interleaved frames of the layout).
fn mix(out: &mut [f32], layout: Layout) {
    let nch = layout.channels();
    out.fill(0.0);
    let mut m = MIXER.lock();
    let mut short = 0usize;
    for p in m.ports.iter_mut().flatten() {
        let want = out.len() / nch;
        // (A port that has started and then runs dry: an underrun, heard as a gap.)
        if p.fifo.len() < want && p.started {
            short += want - p.fifo.len();
        }
        if !p.fifo.is_empty() {
            p.started = true;
        }
        // Around a gap the audio fades out and back in (`FADE` frames)
        // instead of stepping to silence and back, which clicked.
        let n = p.fifo.len().min(want);
        let gap = n < want && p.started;
        let ch = p.channels;
        for (k, frame) in out.chunks_exact_mut(nch).take(n).enumerate() {
            let s = p.fifo.pop_front().unwrap_or([0.0; 8]);
            let mut g = 1.0;
            if p.dry && k < FADE {
                g = (k + 1) as f32 / FADE as f32;
            }
            if gap && n - k <= FADE {
                g *= (n - k) as f32 / (FADE + 1) as f32;
            }
            render(layout, &s, ch, g, frame);
        }
        p.dry = gap;
    }
    note_underrun(short);
    let gain = master_gain();
    // (The downmix of 8 channels each limited at full scale can exceed it,
    // as can two ports' sum: gain reduction - a peak envelope over all
    // channels that jumps to each new peak and decays in 100 ms - instead
    // of clipping; the clamp only as a safety.)
    let mut env = m.limit_env;
    for frame in out.chunks_exact_mut(nch) {
        let mut peak = 0f32;
        for v in frame.iter_mut() {
            *v *= gain;
            peak = peak.max(v.abs());
        }
        env = peak.max(env * LIMIT_DECAY);
        let g = if env > LIMIT { LIMIT / env } else { 1.0 };
        for v in frame.iter_mut() {
            *v = (*v * g).clamp(-1.0, 1.0);
        }
    }
    m.limit_env = if env < 1e-20 { 0.0 } else { env };
    drop(m);
    CV.notify_all();
}

/// The output limiter's level and its envelope's per-frame decay (100 ms).
const LIMIT: f32 = 0.98;
const LIMIT_DECAY: f32 = 0.999_791_7;
/// Frames of the fades around an underrun's gap (~2.7 ms).
const FADE: usize = 128;

/// Underrun statistics: frames the device asked for that a playing port did
/// not have, logged every 10 s while there are any.
fn note_underrun(frames: usize) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    static EVENTS: AtomicU64 = AtomicU64::new(0);
    static LAST: parking_lot::Mutex<Option<std::time::Instant>> = parking_lot::Mutex::new(None);
    if frames > 0 {
        FRAMES.fetch_add(frames as u64, Ordering::Relaxed);
        EVENTS.fetch_add(1, Ordering::Relaxed);
    }
    let mut last = LAST.lock();
    let now = std::time::Instant::now();
    let due = last.is_none_or(|t| now.duration_since(t).as_secs() >= 10);
    if due {
        *last = Some(now);
        let (f, e) = (FRAMES.swap(0, Ordering::Relaxed), EVENTS.swap(0, Ordering::Relaxed));
        if e > 0 {
            crate::info!("audio: {e} underruns in the last 10 s ({f} frames = {:.1} ms of gaps)", f as f64 / RATE as f64 * 1000.0);
        }
    }
}

/// The master volume in force (0-100; u32::MAX: settings.txt's).
static VOLUME: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
/// The device thread opens the device again (another device or speaker set-up).
static REOPEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn volume() -> u32 {
    match VOLUME.load(std::sync::atomic::Ordering::Relaxed) {
        u32::MAX => crate::config::settings().volume.min(100),
        v => v.min(100),
    }
}

fn master_gain() -> f32 {
    let v = volume() as f32 / 100.0;
    v * v
}

/// The master volume changed while the game runs (0-100).
pub fn set_master_volume(v: u32) {
    VOLUME.store(v.min(100), std::sync::atomic::Ordering::Relaxed);
}

/// The sound device or speaker set-up changed: the device thread opens the
/// output again.
pub fn reopen() {
    REOPEN.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The names of the active output devices.
pub fn devices() -> Vec<String> {
    unsafe { device_list() }.unwrap_or_default().into_iter().map(|(n, _)| n).collect()
}

/// The active output devices: friendly name and device.
unsafe fn device_list() -> windows::core::Result<Vec<(String, windows::Win32::Media::Audio::IMMDevice)>> {
    use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::*;
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let enumerator: IMMDeviceEnumerator = unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }?;
    let list = unsafe { enumerator.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE) }?;
    let mut out = Vec::new();
    for i in 0..unsafe { list.GetCount() }? {
        let dev = unsafe { list.Item(i) }?;
        let store = unsafe { dev.OpenPropertyStore(STGM_READ) }?;
        let v = unsafe { store.GetValue(&PKEY_Device_FriendlyName) }?;
        let name = v.to_string();
        if !name.is_empty() {
            out.push((name, dev));
        }
    }
    Ok(out)
}

/// The output device: the PC Settings choice when it is present, else
/// Windows' default.
unsafe fn chosen_device(enumerator: &windows::Win32::Media::Audio::IMMDeviceEnumerator) -> windows::core::Result<windows::Win32::Media::Audio::IMMDevice> {
    use windows::Win32::Media::Audio::*;
    let want = crate::pc_settings::get("audio_device");
    if !want.is_empty() && want != "default" {
        match unsafe { device_list() } {
            Ok(list) => match list.into_iter().find(|(n, _)| *n == want) {
                Some((_, d)) => return Ok(d),
                None => crate::warn!("audio: the sound device '{want}' is not present: Windows' default instead"),
            },
            Err(e) => crate::warn!("audio: listing devices: {e}"),
        }
    }
    unsafe { enumerator.GetDefaultAudioEndpoint(eRender, eConsole) }
}

fn device_thread() {
    crate::log::set_thread_name("audio-out");
    // (It waits on the device's event and mixes a buffer's worth per wake:
    // time-critical, so a busy CPU does not leave the device without data.)
    {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL};
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) };
    }
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
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
}

/// A device's endpoint ID (to notice a change of Windows' default device).
unsafe fn device_id(d: &windows::Win32::Media::Audio::IMMDevice) -> String {
    match unsafe { d.GetId() } {
        Ok(p) => {
            let s = unsafe { p.to_string() }.unwrap_or_default();
            unsafe { windows::Win32::System::Com::CoTaskMemFree(Some(p.0 as *const _)) };
            s
        }
        Err(_) => String::new(),
    }
}

/// Every ~2 s while playing: whether the device thread should open the
/// output again - PC Settings asked, or Windows' default device changed
/// while it plays on the default.
struct Watch {
    enumerator: windows::Win32::Media::Audio::IMMDeviceEnumerator,
    /// The default device's ID when playing on the default.
    default_id: Option<String>,
    last: std::time::Instant,
}

impl Watch {
    unsafe fn new(enumerator: &windows::Win32::Media::Audio::IMMDeviceEnumerator, device: &windows::Win32::Media::Audio::IMMDevice) -> Watch {
        let want = crate::pc_settings::get("audio_device");
        let on_default = want.is_empty() || want == "default";
        Watch { enumerator: enumerator.clone(), default_id: on_default.then(|| unsafe { device_id(device) }), last: std::time::Instant::now() }
    }

    unsafe fn reopen(&mut self) -> bool {
        use windows::Win32::Media::Audio::{eConsole, eRender};
        if REOPEN.swap(false, std::sync::atomic::Ordering::Relaxed) {
            crate::info!("audio: opening the output again (PC Settings)");
            return true;
        }
        if self.last.elapsed().as_secs() < 2 {
            return false;
        }
        self.last = std::time::Instant::now();
        let Some(id) = &self.default_id else { return false };
        match unsafe { self.enumerator.GetDefaultAudioEndpoint(eRender, eConsole) } {
            Ok(d) if unsafe { device_id(&d) } != *id => {
                crate::info!("audio: Windows' default sound device changed: opening it");
                true
            }
            _ => false,
        }
    }
}

/// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT.
const SUBTYPE_FLOAT: windows::core::GUID = windows::core::GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);

#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn run_device() -> windows::core::Result<()> {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::*;
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    let enumerator: IMMDeviceEnumerator = CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
    let device = chosen_device(&enumerator)?;
    let speakers = crate::pc_settings::get("speakers");
    // (Headphones, or automatic with Windows' spatial audio on for the
    // device: the spatial stream; otherwise the device's own channels.)
    if speakers == "headphones" || speakers == "auto" {
        match run_spatial(&enumerator, &device) {
            Ok(()) => return Ok(()),
            Err(e) if speakers == "headphones" => crate::warn!("audio: headphone surround unavailable on this device ({e}): playing the device's own channels instead"),
            Err(_) => {}
        }
    }
    let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
    // (Automatic: the device's own channels, as Windows' speaker set-up has them.)
    let mix_channels = match client.GetMixFormat() {
        Ok(f) => {
            let n = (*f).nChannels;
            CoTaskMemFree(Some(f as *const _));
            n
        }
        Err(_) => 2,
    };
    let layout = match speakers.as_str() {
        "stereo" => Layout::Stereo,
        "5.1" => Layout::Surround51,
        "7.1" => Layout::Surround71,
        _ if mix_channels >= 8 => Layout::Surround71,
        _ if mix_channels >= 6 => Layout::Surround51,
        _ => Layout::Stereo,
    };
    let nch = layout.channels() as u16;
    let wf = WAVEFORMATEXTENSIBLE {
        Format: WAVEFORMATEX {
            wFormatTag: 0xFFFE, // WAVE_FORMAT_EXTENSIBLE
            nChannels: nch,
            nSamplesPerSec: RATE,
            nAvgBytesPerSec: RATE * 4 * nch as u32,
            nBlockAlign: 4 * nch,
            wBitsPerSample: 32,
            cbSize: 22,
        },
        Samples: WAVEFORMATEXTENSIBLE_0 { wValidBitsPerSample: 32 },
        dwChannelMask: layout.mask(),
        SubFormat: SUBTYPE_FLOAT,
    };
    // 30 ms shared-mode buffer; rate/format/channel conversion by the audio engine.
    client.Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY,
        300_000,
        0,
        &wf.Format,
        None,
    )?;
    let event = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
    client.SetEventHandle(HANDLE(event as _))?;
    let render: IAudioRenderClient = client.GetService()?;
    let total = client.GetBufferSize()?;
    client.Start()?;
    crate::info!(
        "audio: WASAPI shared-mode output on '{}', 48 kHz float {} (speakers {speakers}; the device mixes {mix_channels} channels), buffer {total} frames, volume {}%",
        device_name(&device),
        layout.name(),
        volume()
    );
    MIXER.lock().playing = true;
    CV.notify_all();
    let mut watch = Watch::new(&enumerator, &device);
    let mut buf = vec![0f32; total as usize * nch as usize];
    loop {
        WaitForSingleObject(event, 100);
        if watch.reopen() {
            client.Stop()?;
            return Ok(());
        }
        let padding = client.GetCurrentPadding()?;
        let n = total.saturating_sub(padding);
        if n == 0 {
            continue;
        }
        let out = &mut buf[..n as usize * nch as usize];
        mix(out, layout);
        let p = render.GetBuffer(n)? as *mut f32;
        std::ptr::copy_nonoverlapping(out.as_ptr(), p, out.len());
        render.ReleaseBuffer(n, 0)?;
    }
}

/// A device's friendly name (for the log).
unsafe fn device_name(d: &windows::Win32::Media::Audio::IMMDevice) -> String {
    use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
    use windows::Win32::System::Com::STGM_READ;
    unsafe { d.OpenPropertyStore(STGM_READ) }.and_then(|s| unsafe { s.GetValue(&PKEY_Device_FriendlyName) }).map(|v| v.to_string()).unwrap_or_default()
}

/// Headphone surround: the 7.1 mix as the static objects (bed channels) of
/// a Windows spatial audio stream, which Windows Sonic for Headphones or
/// Dolby Atmos for Headphones render for headphones. Fails when the device
/// has no spatial audio (Windows' Sound settings: Spatial audio).
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn run_spatial(enumerator: &windows::Win32::Media::Audio::IMMDeviceEnumerator, device: &windows::Win32::Media::Audio::IMMDevice) -> windows::core::Result<()> {
    use windows::Win32::Foundation::{E_NOTIMPL, HANDLE};
    use windows::Win32::Media::Audio::*;
    use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
    use windows::Win32::System::Com::{BLOB, CLSCTX_INPROC_SERVER};
    use windows::Win32::System::Variant::VT_BLOB;
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    let sac: ISpatialAudioClient = device.Activate(CLSCTX_INPROC_SERVER, None)?;
    // (Spatial audio off: no renderer, no dynamic objects.)
    if sac.GetMaxDynamicObjectCount()? == 0 {
        return Err(windows::core::Error::new(E_NOTIMPL, "spatial audio is off for this device (Windows Sound settings: Spatial audio)"));
    }
    // The layout's channels in Windows' 7.1 order (`render`).
    const TYPES: [AudioObjectType; 8] = [
        AudioObjectType_FrontLeft,
        AudioObjectType_FrontRight,
        AudioObjectType_FrontCenter,
        AudioObjectType_LowFrequency,
        AudioObjectType_BackLeft,
        AudioObjectType_BackRight,
        AudioObjectType_SideLeft,
        AudioObjectType_SideRight,
    ];
    let mask = AudioObjectType(TYPES.iter().fold(0, |a, t| a | t.0));
    let fmt = WAVEFORMATEX { wFormatTag: 3, nChannels: 1, nSamplesPerSec: RATE, nAvgBytesPerSec: RATE * 4, nBlockAlign: 4, wBitsPerSample: 32, cbSize: 0 };
    sac.IsAudioObjectFormatSupported(&fmt)?;
    let event = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
    let params = SpatialAudioObjectRenderStreamActivationParams {
        ObjectFormat: &fmt,
        StaticObjectTypeMask: mask,
        MinDynamicObjectCount: 0,
        MaxDynamicObjectCount: 0,
        Category: AudioCategory_GameEffects,
        EventHandle: HANDLE(event as _),
        NotifyObject: std::mem::ManuallyDrop::new(None),
    };
    let mut pv = PROPVARIANT::default();
    {
        let v = &mut *pv.Anonymous.Anonymous;
        v.vt = VT_BLOB;
        v.Anonymous.blob = BLOB { cbSize: std::mem::size_of_val(&params) as u32, pBlobData: &params as *const _ as *mut u8 };
    }
    let stream: ISpatialAudioObjectRenderStream = sac.ActivateSpatialAudioStream(&pv)?;
    let objects = TYPES.iter().map(|&t| stream.ActivateSpatialAudioObject(t)).collect::<windows::core::Result<Vec<ISpatialAudioObject>>>()?;
    stream.Start()?;
    crate::info!(
        "audio: headphone surround on '{}': the 7.1 mix as a spatial audio stream's static objects (48 kHz float; the device's renderer takes up to {} objects), volume {}%",
        device_name(device),
        sac.GetMaxDynamicObjectCount().unwrap_or(0),
        volume()
    );
    MIXER.lock().playing = true;
    CV.notify_all();
    let mut watch = Watch::new(enumerator, device);
    let mut buf: Vec<f32> = Vec::new();
    loop {
        WaitForSingleObject(event, 100);
        if watch.reopen() {
            stream.Stop()?;
            return Ok(());
        }
        let (mut avail, mut frames) = (0u32, 0u32);
        stream.BeginUpdatingAudioObjects(&mut avail, &mut frames)?;
        let n = frames as usize;
        buf.resize(n * 8, 0.0);
        mix(&mut buf, Layout::Surround71);
        for (c, o) in objects.iter().enumerate() {
            let (mut p, mut len) = (std::ptr::null_mut(), 0u32);
            o.GetBuffer(&mut p, &mut len)?;
            let dst = p as *mut f32;
            for f in 0..n.min(len as usize / 4) {
                *dst.add(f) = buf[f * 8 + c];
            }
        }
        stream.EndUpdatingAudioObjects()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        // PS4 order: L R C LFE Ls Rs Lb Rb.
        let s: Frame = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut st = [0f32; 2];
        render(Layout::Stereo, &s, 8, 1.0, &mut st);
        // (The stereo downmix the measurements were made with.)
        let c = 0.7071;
        assert_eq!(st, [0.8 * (1.0 + c * 3.0 + c * (5.0 + 7.0)), 0.8 * (2.0 + c * 3.0 + c * (6.0 + 8.0))]);
        let mut s71 = [0f32; 8];
        render(Layout::Surround71, &s, 8, 0.5, &mut s71);
        // Windows' order: FL FR FC LFE BL BR SL SR.
        assert_eq!(s71, [0.5, 1.0, 1.5, 2.0, 3.5, 4.0, 2.5, 3.0]);
        let mut s51 = [0f32; 6];
        render(Layout::Surround51, &s, 8, 1.0, &mut s51);
        assert_eq!(&s51[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert!((s51[4] - (5.0 + 0.7071 * 7.0)).abs() < 1e-5 && (s51[5] - (6.0 + 0.7071 * 8.0)).abs() < 1e-5);
        // Stereo ports stay stereo in every layout.
        let p: Frame = [0.25, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for layout in [Layout::Stereo, Layout::Surround51, Layout::Surround71] {
            let mut out = vec![0f32; layout.channels()];
            render(layout, &p, 2, 1.0, &mut out);
            assert_eq!(&out[..2], &[0.25, -0.5]);
            assert!(out[2..].iter().all(|&v| v == 0.0));
        }
        assert_eq!([Layout::Stereo.mask(), Layout::Surround51.mask(), Layout::Surround71.mask()].map(u32::count_ones), [2, 6, 8]);
    }
}
