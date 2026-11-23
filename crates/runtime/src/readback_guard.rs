//! An opt-in experiment for the stage-load crash (`TLG_READBACK_GUARD=1`;
//! LOG: 0.8.3 rebuilt on 690b15b; a render crash in PGO training).
//!
//! The render job's readback (eboot +0xee91d3, +0xee927e) copies a render
//! object's GPU buffer into its CPU staging buffer with the title's
//! CopyResource, then sets the object's "readback pending" flag (+0x2d0)
//! whatever the copy did. After one of the title's buffer growths the
//! staging buffer (made at the asset's new size) and the object's GPU buffer
//! (the old size) differ: CopyResource refuses the copy ("ByteWidth is not
//! same") and the CPU would then read a staging buffer nobody wrote. With the
//! guard the flag stays unset while the two differ (the object keeps its last
//! data until it is rebuilt). Off by default: to be judged over many uncapped
//! First Antenna boots on a free machine.

use crate::patch::Patcher;
use crate::sysv;
use std::sync::atomic::{AtomicU32, Ordering};

/// `mov dword ptr [r15+2D0h], 1` right after each CopyResource call, with
/// that readback's staging-buffer and GPU-buffer offsets in the render object.
const SITES: [(usize, u64, u64); 2] = [(0xee91d8, 0x2a0, 0x2b0), (0xee9283, 0x2a8, 0x2b8)];
const SITE_BYTES: [u8; 11] = [0x41, 0xC7, 0x87, 0xD0, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
const FLAG: u64 = 0x2d0;
/// The resource fields CopyResource compares (dimension, buffers' ByteWidth).
const DIMENSION: u64 = 0x124;
const BYTE_WIDTH: u64 = 0x180;

static CAUGHT: AtomicU32 = AtomicU32::new(0);

pub fn apply(p: &mut Patcher, name: &str) {
    if !std::env::var("TLG_READBACK_GUARD").is_ok_and(|v| v == "1") {
        return;
    }
    match patch(p) {
        Ok(()) => crate::info!("readback guard: on ({name}+{:#x}, +{:#x}; TLG_READBACK_GUARD=1)", SITES[0].0, SITES[1].0),
        Err(e) => crate::warn!("readback guard: {name}: {e}; not applied"),
    }
}

fn patch(p: &mut Patcher) -> Result<(), String> {
    use iced_x86::code_asm::*;
    for (site, _, _) in SITES {
        if p.image.get(site..site + SITE_BYTES.len()) != Some(&SITE_BYTES[..]) {
            return Err(format!("unexpected code at +{site:#x} (not update 01.03?)"));
        }
    }
    let base = p.base;
    let f = tlg_readback_flag as *const () as usize as u64;
    for (i, (site, _, _)) in SITES.into_iter().enumerate() {
        p.detour(site, &SITE_BYTES, |a, _ins| {
            // (Right after a call: the stack is 16-aligned; 9 pushes and 8
            // bytes keep it so. The store itself is the port's.)
            for r in [rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11] {
                a.push(r)?;
            }
            a.sub(rsp, 8)?;
            a.mov(rdi, r15)?;
            a.mov(esi, i as u32)?;
            a.mov(rax, f)?;
            a.call(rax)?;
            a.add(rsp, 8)?;
            for r in [r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax] {
                a.pop(r)?;
            }
            a.jmp(base + (site + SITE_BYTES.len()) as u64)
        })?;
    }
    Ok(())
}

/// The readback's flag store, unless the staging buffer and the GPU buffer
/// differ as CopyResource checks them (then it refused the copy).
fn readback_flag(obj: u64, pair: u32) {
    let (_, staging, gpu) = SITES[pair as usize & 1];
    // SAFETY: the render job's object and the two buffers CopyResource was
    // just given (it read the same fields).
    unsafe {
        let s = *((obj + staging) as *const u64);
        let g = *((obj + gpu) as *const u64);
        let field = |b: u64, off: u64| *((b + off) as *const u32);
        let same = s != 0 && g != 0 && field(s, DIMENSION) == field(g, DIMENSION) && field(s, BYTE_WIDTH) == field(g, BYTE_WIDTH);
        if same {
            std::ptr::write_volatile((obj + FLAG) as *mut u32, 1);
            return;
        }
        let n = CAUGHT.fetch_add(1, Ordering::Relaxed);
        if n < 20 || n % 100 == 0 {
            let w = |b: u64| if b == 0 { (0, 0) } else { (field(b, DIMENSION), field(b, BYTE_WIDTH)) };
            crate::warn!("readback guard: render object {obj:#x} readback {pair}: staging {:?} against GPU buffer {:?} (dimension, ByteWidth): no readback this frame ({} so far)", w(s), w(g), n + 1);
        }
    }
}

sysv! {
    fn tlg_readback_flag(obj: u64, pair: u32) -> i32 {
        readback_flag(obj, pair);
        0
    }
}
