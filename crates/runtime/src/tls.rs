//! Guest thread-local storage (FreeBSD amd64 TLS variant II).
//!
//! Each guest thread owns an area: [static TLS blocks][TCB]. The thread pointer
//! (tp) points at the TCB whose first word is tp itself; module blocks live at
//! tp - offset. Patched guest code loads tp from a Windows TLS slot via GS.

use parking_lot::RwLock;
use std::alloc::{Layout, alloc_zeroed};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use windows_sys::Win32::System::Threading::{TlsAlloc, TlsGetValue, TlsSetValue};

pub const STATIC_TLS_SIZE: u64 = 0x8000;
pub const TCB_SIZE: u64 = 0x100;

#[derive(Clone, Copy, Debug)]
pub struct TlsModule {
    pub index: u64,
    pub init_image: u64,
    pub init_size: u64,
    pub mem_size: u64,
    pub align: u64,
    /// Offset below tp of the static block; 0 means dynamically allocated per thread.
    pub offset: u64,
}

static MODULES: RwLock<Vec<TlsModule>> = RwLock::new(Vec::new());
static SLOT: AtomicU32 = AtomicU32::new(u32::MAX);
static THREADS_STARTED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static DYN_BLOCKS: RefCell<HashMap<u64, u64>> = RefCell::new(HashMap::new());
}

pub fn init_slot() -> Result<u32, String> {
    let s = unsafe { TlsAlloc() };
    if s == u32::MAX || s >= 64 {
        return Err(format!("TlsAlloc returned {s}; need a direct TEB slot (<64)"));
    }
    SLOT.store(s, Ordering::SeqCst);
    Ok(s)
}

pub fn slot() -> u32 {
    SLOT.load(Ordering::Relaxed)
}

/// Registers a module's PT_TLS segment. Returns the module's TLS index.
pub fn register_module(init_image: u64, init_size: u64, mem_size: u64, align: u64, vaddr: u64) -> TlsModule {
    let mut mods = MODULES.write();
    let index = mods.len() as u64 + 1;
    let align = align.max(1);
    let offset = if THREADS_STARTED.load(Ordering::SeqCst) {
        0
    } else {
        let prev = mods.iter().map(|m| m.offset).max().unwrap_or(0);
        let mut off = if prev == 0 {
            // calculate_first_tls_offset
            let mut res = (mem_size + align - 1) & !(align - 1);
            let mis = vaddr & (align - 1);
            if mis != 0 {
                res += align - mis;
            }
            res
        } else {
            (prev + mem_size + align - 1) & !(align - 1)
        };
        if off > STATIC_TLS_SIZE {
            off = 0;
        }
        off
    };
    let m = TlsModule { index, init_image, init_size, mem_size, align, offset };
    mods.push(m);
    m
}

pub fn module(index: u64) -> Option<TlsModule> {
    MODULES.read().get(index.wrapping_sub(1) as usize).copied()
}

/// Allocates and initializes a guest TLS area; returns tp.
pub fn alloc_thread_area(thread_ptr: u64) -> u64 {
    THREADS_STARTED.store(true, Ordering::SeqCst);
    let layout = Layout::from_size_align((STATIC_TLS_SIZE + TCB_SIZE) as usize, 64).unwrap();
    let base = unsafe { alloc_zeroed(layout) } as u64;
    assert!(base != 0, "TLS area allocation failed");
    let tp = base + STATIC_TLS_SIZE;
    for m in MODULES.read().iter() {
        if m.offset == 0 || m.init_size == 0 {
            continue;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(m.init_image as *const u8, (tp - m.offset) as *mut u8, m.init_size as usize);
        }
    }
    unsafe {
        *(tp as *mut u64) = tp;
        *((tp + 8) as *mut u64) = 0;
        *((tp + 0x10) as *mut u64) = thread_ptr;
    }
    tp
}

pub fn set_current_tp(tp: u64) {
    unsafe {
        TlsSetValue(slot(), tp as *const _);
    }
}

pub fn current_tp() -> u64 {
    unsafe { TlsGetValue(slot()) as u64 }
}

/// Address of a TLS variable: module `index`, `offset` inside its block.
pub fn get_addr(index: u64, offset: u64) -> u64 {
    let Some(m) = module(index) else {
        crate::error!("__tls_get_addr: unknown module index {index}");
        return 0;
    };
    let tp = current_tp();
    if m.offset != 0 && tp != 0 {
        return tp - m.offset + offset;
    }
    let block = DYN_BLOCKS.with(|b| {
        *b.borrow_mut().entry(index).or_insert_with(|| {
            let layout = Layout::from_size_align(m.mem_size.max(16) as usize, m.align.max(16) as usize).unwrap();
            let p = unsafe { alloc_zeroed(layout) } as u64;
            unsafe {
                std::ptr::copy_nonoverlapping(m.init_image as *const u8, p as *mut u8, m.init_size as usize);
            }
            p
        })
    });
    block + offset
}
