//! ATRAC9 decoding through the vendored LibAtrac9 (third_party/libatrac9, MIT).

use std::ffi::{c_int, c_void};

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct CodecInfo {
    pub channels: c_int,
    pub channel_config_index: c_int,
    pub sampling_rate: c_int,
    pub superframe_size: c_int,
    pub frames_in_superframe: c_int,
    pub frame_samples: c_int,
    pub wlength: c_int,
    pub config_data: [u8; 4],
}

unsafe extern "C" {
    fn Atrac9GetHandle() -> *mut c_void;
    fn Atrac9ReleaseHandle(handle: *mut c_void);
    fn Atrac9InitDecoder(handle: *mut c_void, config: *mut u8) -> c_int;
    fn Atrac9Decode(handle: *mut c_void, data: *const u8, pcm: *mut i16, used: *mut c_int) -> c_int;
    fn Atrac9GetCodecInfo(handle: *mut c_void, info: *mut CodecInfo) -> c_int;
}

/// One decoder instance (keeps the overlap state between frames).
pub struct Decoder {
    handle: *mut c_void,
    pub info: CodecInfo,
}

// The handle is owned by one voice at a time (accessed under the NGS2 lock).
unsafe impl Send for Decoder {}

impl Decoder {
    /// `config`: the 4 configuration bytes (first byte 0xFE).
    pub fn new(config: [u8; 4]) -> Result<Decoder, String> {
        let handle = unsafe { Atrac9GetHandle() };
        if handle.is_null() {
            return Err("Atrac9GetHandle failed".into());
        }
        let mut cfg = config;
        let r = unsafe { Atrac9InitDecoder(handle, cfg.as_mut_ptr()) };
        if r != 0 {
            unsafe { Atrac9ReleaseHandle(handle) };
            return Err(format!("ATRAC9 config {:02x?} rejected ({r:#x})", config));
        }
        let mut info = CodecInfo::default();
        unsafe { Atrac9GetCodecInfo(handle, &mut info) };
        Ok(Decoder { handle, info })
    }

    /// Decodes one frame at `data` (`avail` bytes readable) into `pcm`
    /// (interleaved, `frame_samples * channels`); returns the bytes consumed.
    pub fn decode_frame(&mut self, data: *const u8, avail: usize, pcm: &mut [i16]) -> Result<usize, String> {
        let need = (self.info.frame_samples * self.info.channels) as usize;
        if pcm.len() < need {
            return Err("ATRAC9 output buffer too small".into());
        }
        // A frame never exceeds the superframe; refuse reads past the block.
        if avail == 0 {
            return Err("ATRAC9 frame beyond the block".into());
        }
        let mut used: c_int = 0;
        let r = unsafe { Atrac9Decode(self.handle, data, pcm.as_mut_ptr(), &mut used) };
        if r != 0 {
            return Err(format!("ATRAC9 decode error {r:#x}"));
        }
        if used as usize > avail {
            return Err(format!("ATRAC9 frame used {used} bytes, {avail} available"));
        }
        Ok(used as usize)
    }
}

/// Development check (`tlg --decode-at9 <file.at9> <out.wav>`): decodes a RIFF
/// ATRAC9 file (WAVE_FORMAT_EXTENSIBLE: block align = superframe size, config
/// bytes at fmt+44) to 16-bit PCM WAV and prints its level statistics.
pub fn decode_file(input: &std::path::Path, output: &std::path::Path) -> Result<(), String> {
    let b = std::fs::read(input).map_err(|e| format!("{}: {e}", input.display()))?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a RIFF WAVE file".into());
    }
    let (mut fmt, mut data) = (None, None);
    let mut p = 12;
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let n = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
        let body = &b[p + 8..(p + 8 + n).min(b.len())];
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        p += 8 + n + (n & 1);
    }
    let fmt = fmt.ok_or("no fmt chunk")?;
    let data = data.ok_or("no data chunk")?;
    if fmt.len() < 48 {
        return Err(format!("fmt chunk of {} bytes is not ATRAC9 extensible", fmt.len()));
    }
    let block_align = u16::from_le_bytes([fmt[12], fmt[13]]) as usize;
    let config: [u8; 4] = fmt[44..48].try_into().unwrap();
    let mut d = Decoder::new(config)?;
    let i = d.info;
    println!("{}: {} ch, {} Hz, superframe {} bytes ({} frames of {} samples), block align {block_align}, config {:02x?}", input.display(), i.channels, i.sampling_rate, i.superframe_size, i.frames_in_superframe, i.frame_samples, config);
    let mut pcm = vec![0i16; (i.frame_samples * i.channels) as usize];
    let mut out: Vec<i16> = Vec::new();
    let sf = i.superframe_size.max(1) as usize;
    let mut buf = vec![0u8; sf + 64];
    for chunk in data.chunks(sf) {
        buf.fill(0);
        buf[..chunk.len()].copy_from_slice(chunk);
        let mut off = 0;
        for _ in 0..i.frames_in_superframe {
            let used = d.decode_frame(buf[off..].as_ptr(), sf - off.min(sf), &mut pcm)?;
            off += used;
            out.extend_from_slice(&pcm);
        }
    }
    let peak = out.iter().map(|&x| (x as i32).abs()).max().unwrap_or(0);
    let rms = (out.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / out.len().max(1) as f64).sqrt();
    println!("decoded {} frames ({:.1} s): peak {peak}, rms {rms:.0}", out.len() / i.channels.max(1) as usize, out.len() as f64 / i.channels.max(1) as f64 / i.sampling_rate.max(1) as f64);
    let mut w = Vec::with_capacity(44 + out.len() * 2);
    let bytes = (out.len() * 2) as u32;
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + bytes).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&(i.channels as u16).to_le_bytes());
    w.extend_from_slice(&(i.sampling_rate as u32).to_le_bytes());
    w.extend_from_slice(&(i.sampling_rate as u32 * i.channels as u32 * 2).to_le_bytes());
    w.extend_from_slice(&(i.channels as u16 * 2).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&bytes.to_le_bytes());
    for x in &out {
        w.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(output, w).map_err(|e| format!("{}: {e}", output.display()))
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { Atrac9ReleaseHandle(self.handle) };
    }
}
