//! `.eh_frame_hdr` / `.eh_frame` parsing to enumerate function ranges of a loaded
//! module. Used to decode code precisely (never data) when patching.

/// Reads guest memory of a loaded module: `base + vaddr`.
pub struct Image<'a> {
    pub base: u64,
    pub bytes: &'a [u8],
}

impl<'a> Image<'a> {
    fn slice(&self, vaddr: u64, len: usize) -> Option<&'a [u8]> {
        let off = vaddr as usize;
        self.bytes.get(off..off.checked_add(len)?)
    }
}

struct Cursor<'a> {
    img: &'a Image<'a>,
    pos: u64, // vaddr
}

impl<'a> Cursor<'a> {
    fn u8(&mut self) -> Option<u8> {
        let b = self.img.slice(self.pos, 1)?[0];
        self.pos += 1;
        Some(b)
    }
    fn u16(&mut self) -> Option<u16> {
        let s = self.img.slice(self.pos, 2)?;
        self.pos += 2;
        Some(u16::from_le_bytes(s.try_into().ok()?))
    }
    fn u32(&mut self) -> Option<u32> {
        let s = self.img.slice(self.pos, 4)?;
        self.pos += 4;
        Some(u32::from_le_bytes(s.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        let s = self.img.slice(self.pos, 8)?;
        self.pos += 8;
        Some(u64::from_le_bytes(s.try_into().ok()?))
    }
    fn uleb(&mut self) -> Option<u64> {
        let mut r = 0u64;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            if shift < 64 {
                r |= ((b & 0x7F) as u64) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                return Some(r);
            }
        }
    }
    fn sleb(&mut self) -> Option<i64> {
        let mut r = 0i64;
        let mut shift = 0;
        let mut b;
        loop {
            b = self.u8()?;
            if shift < 64 {
                r |= ((b & 0x7F) as i64) << shift;
            }
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        if shift < 64 && b & 0x40 != 0 {
            r |= -1i64 << shift;
        }
        Some(r)
    }
    fn cstr(&mut self) -> Option<Vec<u8>> {
        let mut v = Vec::new();
        loop {
            let b = self.u8()?;
            if b == 0 {
                return Some(v);
            }
            v.push(b);
        }
    }
    /// Reads a DW_EH_PE-encoded value. `datarel` is the base for datarel encodings.
    /// Returned values are module-relative (vaddr) for pcrel/datarel and raw otherwise.
    fn encoded(&mut self, enc: u8, datarel: u64) -> Option<u64> {
        if enc == 0xFF {
            return Some(0);
        }
        let start = self.pos;
        let v: u64 = match enc & 0x0F {
            0x00 => self.u64()?,
            0x01 => self.uleb()?,
            0x02 => self.u16()? as u64,
            0x03 => self.u32()? as u64,
            0x04 => self.u64()?,
            0x09 => self.sleb()? as u64,
            0x0A => self.u16()? as i16 as i64 as u64,
            0x0B => self.u32()? as i32 as i64 as u64,
            0x0C => self.u64()?,
            _ => return None,
        };
        let v = match enc & 0x70 {
            0x00 => v,
            0x10 => start.wrapping_add(v),
            0x30 => datarel.wrapping_add(v),
            _ => return None,
        };
        if enc & 0x80 != 0 {
            return None; // indirect pointers are not expected in FDE ranges
        }
        Some(v)
    }
}

/// Returns sorted, module-relative function ranges described by the FDE table.
pub fn function_ranges(img: &Image, eh_frame_hdr_vaddr: u64) -> Result<Vec<(u64, u64)>, String> {
    let mut c = Cursor { img, pos: eh_frame_hdr_vaddr };
    let version = c.u8().ok_or("hdr truncated")?;
    if version != 1 {
        return Err(format!("eh_frame_hdr version {version}"));
    }
    let ptr_enc = c.u8().ok_or("hdr truncated")?;
    let count_enc = c.u8().ok_or("hdr truncated")?;
    let table_enc = c.u8().ok_or("hdr truncated")?;
    let _eh_frame = c.encoded(ptr_enc, eh_frame_hdr_vaddr).ok_or("eh_frame_ptr")?;
    let count = c.encoded(count_enc, eh_frame_hdr_vaddr).ok_or("fde_count")?;
    let mut out = Vec::with_capacity(count as usize);
    let mut cie_cache: std::collections::HashMap<u64, u8> = std::collections::HashMap::new();
    for _ in 0..count {
        let _initial = c.encoded(table_enc, eh_frame_hdr_vaddr).ok_or("table entry")?;
        let fde = c.encoded(table_enc, eh_frame_hdr_vaddr).ok_or("table entry")?;
        // Parse the FDE.
        let mut f = Cursor { img, pos: fde };
        let len = f.u32().ok_or("fde len")? as u64;
        if len == 0xFFFF_FFFF {
            f.u64().ok_or("fde len64")?;
        }
        let cie_ptr_pos = f.pos;
        let cie_off = f.u32().ok_or("cie ptr")? as u64;
        let cie = cie_ptr_pos.wrapping_sub(cie_off);
        let enc = match cie_cache.get(&cie) {
            Some(e) => *e,
            None => {
                let e = parse_cie_fde_encoding(img, cie).ok_or_else(|| format!("bad CIE at {cie:#x}"))?;
                cie_cache.insert(cie, e);
                e
            }
        };
        let begin = f.encoded(enc, 0).ok_or("pc_begin")?;
        let range = f.encoded(enc & 0x0F, 0).ok_or("pc_range")?;
        if range > 0 {
            out.push((begin, begin + range));
        }
    }
    out.sort_unstable();
    Ok(out)
}

fn parse_cie_fde_encoding(img: &Image, cie: u64) -> Option<u8> {
    let mut c = Cursor { img, pos: cie };
    let len = c.u32()? as u64;
    if len == 0xFFFF_FFFF {
        c.u64()?;
    }
    let id = c.u32()?;
    if id != 0 {
        return None;
    }
    let version = c.u8()?;
    let aug = c.cstr()?;
    if version >= 4 {
        c.u8()?;
        c.u8()?;
    }
    c.uleb()?; // code alignment
    c.sleb()?; // data alignment
    if version == 1 {
        c.u8()?;
    } else {
        c.uleb()?;
    }
    let mut enc = 0u8; // absptr default
    if aug.first() == Some(&b'z') {
        c.uleb()?;
        for &ch in &aug[1..] {
            match ch {
                b'R' => enc = c.u8()?,
                b'P' => {
                    let penc = c.u8()?;
                    c.encoded(penc & 0x7F, 0)?;
                }
                b'L' => {
                    c.u8()?;
                }
                b'S' | b'B' => {}
                _ => return None,
            }
        }
    }
    Some(enc)
}
