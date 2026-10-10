//! Media Foundation (Windows) demux/decode for AvPlayer sources: H.264 video
//! to NV12 and audio (AAC) to 16-bit PCM, through independent source readers
//! so video and audio are pulled at their own pace.
//!
//! Streams are reported in file track order (as AvPlayer numbers them);
//! Media Foundation's own stream numbering differs (e.g. it lists all audio
//! tracks before the video track), so tracks are matched per media type.

use std::path::Path;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx};
use windows::core::HSTRING;

#[derive(Clone, Copy, Debug, Default)]
pub struct VideoInfo {
    /// Decoded frame size (may include alignment rows).
    pub frame_w: u32,
    pub frame_h: u32,
    /// Visible size (e.g. 1920x1080).
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AudioInfo {
    pub channels: u32,
    pub sample_rate: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct StreamDesc {
    pub video: bool,
    /// Media Foundation stream index.
    pub mf_index: u32,
    pub v: VideoInfo,
    pub a: AudioInfo,
}

#[derive(Clone, Debug, Default)]
pub struct MediaInfo {
    pub duration_ms: u64,
    /// In file track order.
    pub streams: Vec<StreamDesc>,
}

pub struct VideoFrame {
    pub ts_ms: u64,
    /// NV12: `frame_h` luma rows of `frame_w` bytes followed by `frame_h / 2`
    /// interleaved chroma rows.
    pub data: Vec<u8>,
}

pub struct AudioFrame {
    pub ts_ms: u64,
    /// Interleaved signed 16-bit PCM.
    pub pcm: Vec<u8>,
}

const ALL_STREAMS: u32 = MF_SOURCE_READER_ALL_STREAMS.0 as u32;
const MEDIA_SOURCE: u32 = MF_SOURCE_READER_MEDIASOURCE.0 as u32;

fn startup() -> Result<(), String> {
    static STARTED: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
    }
    STARTED.get_or_init(|| unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(|e| format!("MFStartup: {e}")) }).clone()
}

/// Media types of the file's tracks in `moov` order (true = video), from the
/// ISO BMFF boxes (`moov/trak/mdia/hdlr`). Non-audio/video tracks are skipped.
fn track_order(path: &Path) -> Result<Vec<bool>, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let header = |f: &mut std::fs::File, pos: u64| -> Option<(u64, [u8; 4], u64)> {
        f.seek(SeekFrom::Start(pos)).ok()?;
        let mut h = [0u8; 16];
        f.read_exact(&mut h[..8]).ok()?;
        let mut size = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as u64;
        let ty = [h[4], h[5], h[6], h[7]];
        let mut hdr = 8;
        if size == 1 {
            f.read_exact(&mut h[8..16]).ok()?;
            size = u64::from_be_bytes(h[8..16].try_into().unwrap());
            hdr = 16;
        } else if size == 0 {
            size = len - pos;
        }
        (size >= hdr).then_some((size, ty, hdr))
    };
    // Locate moov at the top level and read it whole.
    let mut pos = 0;
    let mut moov = None;
    while pos + 8 <= len {
        let Some((size, ty, hdr)) = header(&mut f, pos) else { break };
        if &ty == b"moov" {
            moov = Some((pos + hdr, size - hdr));
            break;
        }
        pos += size;
    }
    let (start, size) = moov.ok_or("no moov box")?;
    let mut buf = vec![0u8; size as usize];
    f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    fn children(b: &[u8]) -> Vec<([u8; 4], &[u8])> {
        let mut out = Vec::new();
        let mut p = 0;
        while p + 8 <= b.len() {
            let size = u32::from_be_bytes(b[p..p + 4].try_into().unwrap()) as usize;
            let ty: [u8; 4] = b[p + 4..p + 8].try_into().unwrap();
            if size < 8 || p + size > b.len() {
                break;
            }
            out.push((ty, &b[p + 8..p + size]));
            p += size;
        }
        out
    }
    let mut kinds = Vec::new();
    for (ty, trak) in children(&buf) {
        if &ty != b"trak" {
            continue;
        }
        for (ty, mdia) in children(trak) {
            if &ty != b"mdia" {
                continue;
            }
            for (ty, hdlr) in children(mdia) {
                if &ty == b"hdlr" && hdlr.len() >= 12 {
                    match &hdlr[8..12] {
                        b"vide" => kinds.push(true),
                        b"soun" => kinds.push(false),
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(kinds)
}

/// Enumerates the streams of `path`.
pub fn probe(path: &Path) -> Result<MediaInfo, String> {
    startup()?;
    unsafe {
        let r = MFCreateSourceReaderFromURL(&HSTRING::from(path.as_os_str()), None).map_err(|e| format!("MFCreateSourceReaderFromURL: {e}"))?;
        let mut info = MediaInfo::default();
        if let Ok(v) = r.GetPresentationAttribute(MEDIA_SOURCE, &MF_PD_DURATION) {
            // PROPVARIANT VT_UI8 in 100 ns units.
            info.duration_ms = v.Anonymous.Anonymous.Anonymous.uhVal / 10_000;
        }
        let mut videos = Vec::new();
        let mut audios = Vec::new();
        for i in 0..64u32 {
            let Ok(mt) = r.GetNativeMediaType(i, 0) else { break };
            let major = mt.GetGUID(&MF_MT_MAJOR_TYPE).map_err(|e| e.to_string())?;
            if major == MFMediaType_Video {
                let size = mt.GetUINT64(&MF_MT_FRAME_SIZE).unwrap_or(0);
                let rate = mt.GetUINT64(&MF_MT_FRAME_RATE).unwrap_or((30u64 << 32) | 1);
                let (w, h) = ((size >> 32) as u32, size as u32);
                let v = VideoInfo { frame_w: w, frame_h: h, width: w, height: h, fps_num: (rate >> 32) as u32, fps_den: (rate as u32).max(1) };
                videos.push(StreamDesc { video: true, mf_index: i, v, a: AudioInfo::default() });
            } else if major == MFMediaType_Audio {
                let a = AudioInfo { channels: mt.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS).unwrap_or(2), sample_rate: mt.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND).unwrap_or(48000) };
                audios.push(StreamDesc { video: false, mf_index: i, v: VideoInfo::default(), a });
            }
        }
        // File order; fall back to Media Foundation order if the box walk fails.
        let order = track_order(path).unwrap_or_else(|_| videos.iter().map(|_| true).chain(audios.iter().map(|_| false)).collect());
        let (mut vi, mut ai) = (videos.into_iter(), audios.into_iter());
        for is_video in order {
            if let Some(s) = if is_video { vi.next() } else { ai.next() } {
                info.streams.push(s);
            }
        }
        info.streams.extend(vi);
        info.streams.extend(ai);
        Ok(info)
    }
}

fn reader(path: &Path, stream: u32, video: bool) -> Result<IMFSourceReader, String> {
    unsafe {
        let mut attrs: Option<IMFAttributes> = None;
        MFCreateAttributes(&mut attrs, 2).map_err(|e| format!("MFCreateAttributes: {e}"))?;
        let attrs = attrs.ok_or("no attributes")?;
        if video {
            // Lets the reader insert a converter when the decoder cannot emit NV12 directly.
            attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1).map_err(|e| e.to_string())?;
        }
        let r = MFCreateSourceReaderFromURL(&HSTRING::from(path.as_os_str()), &attrs).map_err(|e| format!("MFCreateSourceReaderFromURL: {e}"))?;
        r.SetStreamSelection(ALL_STREAMS, false).map_err(|e| e.to_string())?;
        r.SetStreamSelection(stream, true).map_err(|e| format!("select stream: {e}"))?;
        let mt = MFCreateMediaType().map_err(|e| e.to_string())?;
        if video {
            mt.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).map_err(|e| e.to_string())?;
            mt.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12).map_err(|e| e.to_string())?;
        } else {
            mt.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio).map_err(|e| e.to_string())?;
            mt.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM).map_err(|e| e.to_string())?;
            mt.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16).map_err(|e| e.to_string())?;
        }
        r.SetCurrentMediaType(stream, None, &mt).map_err(|e| format!("output type: {e}"))?;
        Ok(r)
    }
}

pub struct Decoder {
    video: Option<(IMFSourceReader, u32)>,
    audio: Option<(IMFSourceReader, u32)>,
    /// Decoded video geometry.
    pub video_info: VideoInfo,
    stride: u32,
    video_eof: bool,
    audio_eof: bool,
}

impl Decoder {
    /// Opens decoders for the selected streams (indices into `info.streams`).
    pub fn open(path: &Path, info: &MediaInfo, video: Option<usize>, audio: Option<usize>) -> Result<Decoder, String> {
        startup()?;
        let mut d = Decoder { video: None, audio: None, video_info: VideoInfo::default(), stride: 0, video_eof: false, audio_eof: false };
        if let Some(s) = video.and_then(|i| info.streams.get(i)) {
            let r = reader(path, s.mf_index, true)?;
            unsafe {
                let mt = r.GetCurrentMediaType(s.mf_index).map_err(|e| e.to_string())?;
                let size = mt.GetUINT64(&MF_MT_FRAME_SIZE).map_err(|e| e.to_string())?;
                let (fw, fh) = ((size >> 32) as u32, size as u32);
                let (mut w, mut h) = (fw, fh);
                let mut aperture = MFVideoArea::default();
                let ap = std::slice::from_raw_parts_mut(&mut aperture as *mut MFVideoArea as *mut u8, std::mem::size_of::<MFVideoArea>());
                if mt.GetBlob(&MF_MT_MINIMUM_DISPLAY_APERTURE, ap, None).is_ok() {
                    w = aperture.Area.cx as u32;
                    h = aperture.Area.cy as u32;
                }
                d.stride = mt.GetUINT32(&MF_MT_DEFAULT_STRIDE).map(|s| s as i32).unwrap_or(fw as i32).unsigned_abs();
                d.video_info = VideoInfo { frame_w: fw, frame_h: fh, width: w, height: h, fps_num: s.v.fps_num, fps_den: s.v.fps_den };
            }
            d.video = Some((r, s.mf_index));
        }
        if let Some(s) = audio.and_then(|i| info.streams.get(i)) {
            d.audio = Some((reader(path, s.mf_index, false)?, s.mf_index));
        }
        Ok(d)
    }

    pub fn video_eof(&self) -> bool {
        self.video.is_none() || self.video_eof
    }
    pub fn audio_eof(&self) -> bool {
        self.audio.is_none() || self.audio_eof
    }

    /// Reads one sample; returns (timestamp in ms, contiguous bytes), or None at end of stream.
    fn read(r: &IMFSourceReader, stream: u32) -> Result<Option<(u64, Vec<u8>)>, String> {
        unsafe {
            loop {
                let mut flags = 0u32;
                let mut ts = 0i64;
                let mut sample: Option<IMFSample> = None;
                r.ReadSample(stream, 0, None, Some(&mut flags), Some(&mut ts), Some(&mut sample)).map_err(|e| format!("ReadSample: {e}"))?;
                if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
                    return Ok(None);
                }
                if flags & MF_SOURCE_READERF_ERROR.0 as u32 != 0 {
                    return Err("source reader error".into());
                }
                let Some(sample) = sample else { continue }; // stream tick / format change
                let buf = sample.ConvertToContiguousBuffer().map_err(|e| e.to_string())?;
                let mut p = std::ptr::null_mut();
                let mut len = 0u32;
                buf.Lock(&mut p, None, Some(&mut len)).map_err(|e| e.to_string())?;
                let data = std::slice::from_raw_parts(p, len as usize).to_vec();
                let _ = buf.Unlock();
                return Ok(Some(((ts.max(0) as u64) / 10_000, data)));
            }
        }
    }

    /// Next decoded video frame, repacked to `frame_w` pitch NV12.
    pub fn next_video(&mut self) -> Result<Option<VideoFrame>, String> {
        let Some((r, idx)) = &self.video else { return Ok(None) };
        if self.video_eof {
            return Ok(None);
        }
        let vi = self.video_info;
        match Self::read(r, *idx)? {
            None => {
                self.video_eof = true;
                Ok(None)
            }
            Some((ts_ms, raw)) => {
                let (w, h) = (vi.frame_w as usize, vi.frame_h as usize);
                let stride = (self.stride as usize).max(w);
                let mut data = vec![0u8; w * h * 3 / 2];
                // Decoders may pad rows (stride) and the chroma plane offset.
                let rows_total = raw.len() / stride;
                let luma_rows = if rows_total >= h * 3 / 2 { rows_total * 2 / 3 } else { h };
                for y in 0..h {
                    let s = y * stride;
                    if s + w <= raw.len() {
                        data[y * w..(y + 1) * w].copy_from_slice(&raw[s..s + w]);
                    }
                }
                for y in 0..h / 2 {
                    let s = (luma_rows + y) * stride;
                    if s + w <= raw.len() {
                        data[(h + y) * w..(h + y + 1) * w].copy_from_slice(&raw[s..s + w]);
                    }
                }
                Ok(Some(VideoFrame { ts_ms, data }))
            }
        }
    }

    pub fn next_audio(&mut self) -> Result<Option<AudioFrame>, String> {
        let Some((r, idx)) = &self.audio else { return Ok(None) };
        if self.audio_eof {
            return Ok(None);
        }
        match Self::read(r, *idx)? {
            None => {
                self.audio_eof = true;
                Ok(None)
            }
            Some((ts_ms, pcm)) => Ok(Some(AudioFrame { ts_ms, pcm })),
        }
    }

    /// Seeks the readers back to the start (looping).
    pub fn rewind(&mut self) -> Result<(), String> {
        unsafe {
            use windows::Win32::System::Com::StructuredStorage::PROPVARIANT;
            let mut pos = PROPVARIANT::default();
            (*pos.Anonymous.Anonymous).vt = windows::Win32::System::Variant::VT_I8;
            (*pos.Anonymous.Anonymous).Anonymous.hVal = 0;
            for (r, _) in [&self.video, &self.audio].into_iter().flatten() {
                r.SetCurrentPosition(&windows::core::GUID::zeroed(), &pos).map_err(|e| format!("seek: {e}"))?;
            }
        }
        self.video_eof = false;
        self.audio_eof = false;
        Ok(())
    }
}
