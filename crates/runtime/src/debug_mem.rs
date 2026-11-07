//! Debug access to guest memory from the control channel (`peek`, `poke`):
//! reading engine state and changing it while the title runs.
//!
//! Addresses: hex numbers (`0x10a0` or `10a0`), `main` (the main module's
//! base) plus hex terms (`main+55404f8`), and `[expr]` for the 64-bit value
//! stored at expr (`[main+55404f8]+8`).


fn parse_term(t: &str) -> Result<u64, String> {
    let t = t.trim();
    if t == "main" {
        return crate::loader::main_module().map(|m| m.base).ok_or_else(|| "no main module".to_string());
    }
    // (The game camera object, once it has updated.)
    if t == "cam" {
        return Some(crate::title_patches::camera_object()).filter(|&c| c != 0).ok_or_else(|| "no camera object yet".to_string());
    }
    u64::from_str_radix(t.trim_start_matches("0x"), 16).map_err(|_| format!("bad address term '{t}'"))
}

/// Evaluates an address expression (see the module docs).
pub fn parse_addr(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (mut v, rest) = if let Some(inner) = s.strip_prefix('[') {
        let end = inner.rfind(']').ok_or("unbalanced [")?;
        let p = parse_addr(&inner[..end])?;
        if !readable(p, 8) && !host_readable(p, 8) {
            return Err(format!("{p:#x} is not mapped"));
        }
        (unsafe { std::ptr::read_unaligned(p as *const u64) }, &inner[end + 1..])
    } else {
        let mut it = s.splitn(2, '+');
        let first = it.next().unwrap_or("");
        (parse_term(first)?, it.next().unwrap_or(""))
    };
    for t in rest.split('+').filter(|t| !t.trim().is_empty()) {
        v = v.wrapping_add(parse_term(t)?);
    }
    Ok(v)
}

/// Committed, readable host memory outside the guest range (the title's
/// heap objects come from tlg's allocator), for reading only.
fn host_readable(addr: u64, len: u64) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    if addr < 0x10000 || len == 0 {
        return false;
    }
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe { VirtualQuery(addr as _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
    n != 0 && mbi.State == MEM_COMMIT && mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS) == 0 && mbi.Protect & 0xEE != 0 && addr + len <= mbi.BaseAddress as u64 + mbi.RegionSize as u64
}

/// Readable guest or host memory (for the port's reads of game objects).
pub fn host_or_guest_readable(addr: u64, len: u64) -> bool {
    readable(addr, len) || host_readable(addr, len)
}

fn readable(addr: u64, len: u64) -> bool {
    if !crate::mem::is_guest(addr, len) {
        return false;
    }
    let st = crate::mem::map_read();
    st.find(addr).is_some_and(|v| v.kind != crate::mem::VmaKind::Reserved && addr + len <= v.end())
}

pub fn peek(expr: &str, n: u64) -> Result<(), String> {
    let a = parse_addr(expr)?;
    if !readable(a, n.max(4)) && !host_readable(a, n.max(4)) {
        return Err(format!("peek {a:#x}+{n:#x}: not mapped"));
    }
    let mut out = format!("peek {expr} = {a:#x} ({}):", crate::loader::describe_addr(a));
    for o in (0..n).step_by(16) {
        out.push_str(&format!("\n  +{o:04x}"));
        for k in (o..(o + 16).min(n)).step_by(4) {
            let w = unsafe { std::ptr::read_unaligned((a + k) as *const u32) };
            out.push_str(&format!(" {w:08x}"));
        }
        out.push_str("  |");
        for k in (o..(o + 16).min(n)).step_by(4) {
            let w = unsafe { std::ptr::read_unaligned((a + k) as *const u32) };
            let f = f32::from_bits(w);
            if f.is_finite() && (f == 0.0 || (1e-6..1e7).contains(&f.abs())) {
                out.push_str(&format!(" {f}"));
            } else {
                out.push_str(" -");
            }
        }
    }
    crate::info!("{out}");
    Ok(())
}

pub fn poke(expr: &str, words: &[String]) -> Result<(), String> {
    let a = parse_addr(expr)?;
    let vals: Vec<u32> = words.iter().map(|w| u32::from_str_radix(w.trim_start_matches("0x"), 16).map_err(|_| format!("bad word '{w}'"))).collect::<Result<_, _>>()?;
    let len = vals.len() as u64 * 4;
    if vals.is_empty() || !readable(a, len) {
        return Err(format!("poke {a:#x}: nothing to write or not mapped"));
    }
    write_guest(a, &vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    crate::info!("poke {expr} = {a:#x} ({}): {:x?}", crate::loader::describe_addr(a), vals);
    Ok(())
}

/// Writes guest memory, code pages included (made writable for the write;
/// the instruction cache is flushed afterwards).
pub fn write_guest(a: u64, bytes: &[u8]) -> Result<(), String> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let len = bytes.len();
    let mut old = 0u32;
    unsafe {
        if VirtualProtect(a as _, len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err(format!("VirtualProtect({a:#x}) failed"));
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), a as *mut u8, len);
        let mut tmp = 0u32;
        VirtualProtect(a as _, len, old, &mut tmp);
        FlushInstructionCache(GetCurrentProcess(), a as _, len);
    }
    Ok(())
}

/// Debug (control `memsnap <0|1|2>`, then `memfind [size]`): snapshots of
/// the guest's writable memory that is not GPU memory (module data,
/// flexible memory) and of a host range (`memsnap <slot> host <addr>
/// <len>`), compared: places equal in snapshots 0 and 2 but different in 1
/// (a setting toggled and back), with their values - e.g. where the game
/// keeps an option.
static SNAPS: parking_lot::Mutex<[Vec<(u64, Vec<u8>)>; 3]> = parking_lot::Mutex::new([Vec::new(), Vec::new(), Vec::new()]);

pub fn memsnap(slot: usize, host: Option<(u64, u64)>) -> Result<(), String> {
    if slot > 2 {
        return Err("memsnap: slot 0, 1 or 2".into());
    }
    let mut regions = Vec::new();
    if let Some((a, n)) = host {
        if !host_readable(a, 1) {
            return Err(format!("{a:#x} not readable"));
        }
        let mut buf = vec![0u8; n as usize];
        unsafe { std::ptr::copy_nonoverlapping(a as *const u8, buf.as_mut_ptr(), n as usize) };
        regions.push((a, buf));
    } else {
        let st = crate::mem::map_read();
        for v in st.vmas.values() {
            let keep = matches!(v.kind, crate::mem::VmaKind::Module | crate::mem::VmaKind::Flexible) && v.prot & 2 != 0 && v.size <= 1 << 30;
            if !keep {
                continue;
            }
            let mut buf = vec![0u8; v.size as usize];
            unsafe { std::ptr::copy_nonoverlapping(v.start as *const u8, buf.as_mut_ptr(), v.size as usize) };
            regions.push((v.start, buf));
        }
    }
    let total: usize = regions.iter().map(|r| r.1.len()).sum();
    crate::info!("memsnap {slot}: {} regions, {} MiB", regions.len(), total >> 20);
    SNAPS.lock()[slot] = regions;
    Ok(())
}

pub fn memfind(width: usize, small: bool) -> Result<(), String> {
    let s = SNAPS.lock();
    let w = width.clamp(1, 4);
    // (`small`: only values under 16 in all three, e.g. a menu's cursor.)
    let v = |b: &[u8]| b.iter().rev().fold(0u32, |a, &x| a << 8 | x as u32);
    let mut found = 0;
    for (a0, b0) in &s[0] {
        let (Some((_, b1)), Some((_, b2))) = (s[1].iter().find(|r| r.0 == *a0), s[2].iter().find(|r| r.0 == *a0)) else { continue };
        let n = b0.len().min(b1.len()).min(b2.len());
        let mut i = 0;
        while i + w <= n {
            if b0[i..i + w] == b2[i..i + w] && b0[i..i + w] != b1[i..i + w] && (!small || (v(&b0[i..i + w]) < 16 && v(&b1[i..i + w]) < 16)) {
                if found < 80 {
                    crate::info!("memfind: {:#x}: {:02x?} -> {:02x?} -> back", a0 + i as u64, &b0[i..i + w], &b1[i..i + w]);
                }
                found += 1;
            }
            i += w;
        }
    }
    crate::info!("memfind: {found} places (width {w})");
    Ok(())
}

/// Debug (control `memsearch <hex bytes>`): where the byte pattern is in
/// the process's committed, writable private memory (the guest's and the
/// host heap the title's objects live in); up to 64 places logged.
pub fn memsearch(hex: &str) -> Result<(), String> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_READWRITE, VirtualQuery};
    let pat: Vec<u8> = (0..hex.len() / 2).filter_map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()).collect();
    if pat.len() < 4 {
        return Err("memsearch: at least 4 bytes of hex".into());
    }
    let mut addr = 0x10000u64;
    let mut hits = 0;
    let mut scanned = 0u64;
    loop {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(addr as _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            break;
        }
        let (base, size) = (mbi.BaseAddress as u64, mbi.RegionSize as u64);
        if mbi.State == MEM_COMMIT && (mbi.Type == MEM_PRIVATE || mbi.Type == MEM_MAPPED) && mbi.Protect == PAGE_READWRITE && mbi.Protect & PAGE_GUARD == 0 && size < (4 << 30) {
            let s = unsafe { std::slice::from_raw_parts(base as *const u8, size as usize) };
            scanned += size;
            for (i, w) in s.windows(pat.len()).enumerate() {
                if w == &pat[..] {
                    if hits < 64 {
                        crate::info!("memsearch: {:#x}", base + i as u64);
                    }
                    hits += 1;
                }
            }
        }
        addr = base + size;
        if addr >= 0x7fff_ffff_0000 {
            break;
        }
    }
    crate::info!("memsearch: {hits} places in {} MiB", scanned >> 20);
    Ok(())
}
