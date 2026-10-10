//! PM4 packet encoding/decoding (AMD GCN gfx7 "Liverpool" command processor).

pub mod op {
    pub const NOP: u32 = 0x10;
    pub const SET_BASE: u32 = 0x11;
    pub const CLEAR_STATE: u32 = 0x12;
    pub const INDEX_BUFFER_SIZE: u32 = 0x13;
    pub const DISPATCH_DIRECT: u32 = 0x15;
    pub const DISPATCH_INDIRECT: u32 = 0x16;
    pub const ATOMIC_GDS: u32 = 0x1D;
    pub const ATOMIC_MEM: u32 = 0x1E;
    pub const OCCLUSION_QUERY: u32 = 0x1F;
    pub const SET_PREDICATION: u32 = 0x20;
    pub const REG_RMW: u32 = 0x21;
    pub const COND_EXEC: u32 = 0x22;
    pub const PRED_EXEC: u32 = 0x23;
    pub const DRAW_INDIRECT: u32 = 0x24;
    pub const DRAW_INDEX_INDIRECT: u32 = 0x25;
    pub const INDEX_BASE: u32 = 0x26;
    pub const DRAW_INDEX_2: u32 = 0x27;
    pub const CONTEXT_CONTROL: u32 = 0x28;
    pub const INDEX_TYPE: u32 = 0x2A;
    pub const DRAW_INDIRECT_MULTI: u32 = 0x2C;
    pub const DRAW_INDEX_AUTO: u32 = 0x2D;
    pub const NUM_INSTANCES: u32 = 0x2F;
    pub const DRAW_INDEX_MULTI_AUTO: u32 = 0x30;
    pub const INDIRECT_BUFFER_CONST: u32 = 0x33;
    pub const STRMOUT_BUFFER_UPDATE: u32 = 0x34;
    pub const DRAW_INDEX_OFFSET_2: u32 = 0x35;
    pub const DRAW_PREAMBLE: u32 = 0x36;
    pub const WRITE_DATA: u32 = 0x37;
    pub const DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
    pub const MEM_SEMAPHORE: u32 = 0x39;
    pub const COPY_DW: u32 = 0x3B;
    pub const WAIT_REG_MEM: u32 = 0x3C;
    pub const INDIRECT_BUFFER: u32 = 0x3F;
    pub const COPY_DATA: u32 = 0x40;
    pub const PFP_SYNC_ME: u32 = 0x42;
    pub const SURFACE_SYNC: u32 = 0x43;
    pub const COND_WRITE: u32 = 0x45;
    pub const EVENT_WRITE: u32 = 0x46;
    pub const EVENT_WRITE_EOP: u32 = 0x47;
    pub const EVENT_WRITE_EOS: u32 = 0x48;
    pub const RELEASE_MEM: u32 = 0x49;
    pub const PREAMBLE_CNTL: u32 = 0x4A;
    pub const DMA_DATA: u32 = 0x50;
    pub const ACQUIRE_MEM: u32 = 0x58;
    pub const REWIND: u32 = 0x59;
    pub const LOAD_UCONFIG_REG: u32 = 0x5E;
    pub const LOAD_SH_REG: u32 = 0x5F;
    pub const LOAD_CONFIG_REG: u32 = 0x60;
    pub const LOAD_CONTEXT_REG: u32 = 0x61;
    pub const SET_CONFIG_REG: u32 = 0x68;
    pub const SET_CONTEXT_REG: u32 = 0x69;
    pub const SET_CONTEXT_REG_INDIRECT: u32 = 0x73;
    pub const SET_SH_REG: u32 = 0x76;
    pub const SET_SH_REG_OFFSET: u32 = 0x77;
    pub const SET_QUEUE_REG: u32 = 0x78;
    pub const SET_UCONFIG_REG: u32 = 0x79;
    pub const SCRATCH_RAM_WRITE: u32 = 0x7D;
    pub const SCRATCH_RAM_READ: u32 = 0x7E;
    pub const LOAD_CONST_RAM: u32 = 0x80;
    pub const WRITE_CONST_RAM: u32 = 0x81;
    pub const DUMP_CONST_RAM: u32 = 0x83;
    pub const INCREMENT_CE_COUNTER: u32 = 0x84;
    pub const INCREMENT_DE_COUNTER: u32 = 0x85;
    pub const WAIT_ON_CE_COUNTER: u32 = 0x86;
    pub const WAIT_ON_DE_COUNTER_DIFF: u32 = 0x88;
    pub const SWITCH_BUFFER: u32 = 0x8B;
    pub const GET_LOD_STATS: u32 = 0x8E;
    pub const DRAW_INDEX_INDIRECT_COUNT_MULTI: u32 = 0x9D;
}

pub const CONTEXT_REG_BASE: u32 = 0xA000;
pub const SH_REG_BASE: u32 = 0x2C00;
pub const UCONFIG_REG_BASE: u32 = 0xC000;
pub const CONFIG_REG_BASE: u32 = 0x2000;

pub fn type3(opcode: u32, payload_dwords: u32) -> u32 {
    (3 << 30) | ((payload_dwords.saturating_sub(1) & 0x3FFF) << 16) | (opcode << 8)
}

/// Writes one NOP packet covering exactly `n` dwords (n >= 1).
pub fn write_nop(buf: &mut [u32], n: usize) {
    if n == 0 {
        return;
    }
    if n == 1 {
        buf[0] = 0x8000_0000; // type-2 filler
        return;
    }
    buf[0] = type3(op::NOP, (n - 1) as u32);
    for d in &mut buf[1..n] {
        *d = 0;
    }
}

/// HLE marker NOP payload tags (first payload dword).
pub mod marker {
    pub const PUSH: u32 = 0x4242_0001;
    pub const POP: u32 = 0x4242_0002;
    pub const SET: u32 = 0x4242_0003;
    pub const WAIT_FLIP: u32 = 0x4242_0010;
    pub const INIT_DEFAULT: u32 = 0x4242_0020;
}

/// Command-buffer writer used by HLE builder functions; pads with NOP on finish.
pub struct Writer<'a> {
    pub buf: &'a mut [u32],
    pub pos: usize,
}

impl<'a> Writer<'a> {
    pub unsafe fn new(ptr: *mut u32, dwords: u32) -> Self {
        Writer { buf: unsafe { std::slice::from_raw_parts_mut(ptr, dwords as usize) }, pos: 0 }
    }
    pub fn fits(&self, n: usize) -> bool {
        self.pos + n <= self.buf.len()
    }
    pub fn packet(&mut self, opcode: u32, payload: &[u32]) -> bool {
        if !self.fits(1 + payload.len()) {
            return false;
        }
        self.buf[self.pos] = type3(opcode, payload.len() as u32);
        self.buf[self.pos + 1..self.pos + 1 + payload.len()].copy_from_slice(payload);
        self.pos += 1 + payload.len();
        true
    }
    pub fn set_sh(&mut self, reg: u32, vals: &[u32]) -> bool {
        let mut p = vec![reg - SH_REG_BASE];
        p.extend_from_slice(vals);
        self.packet(op::SET_SH_REG, &p)
    }
    pub fn set_ctx(&mut self, reg: u32, vals: &[u32]) -> bool {
        let mut p = vec![reg - CONTEXT_REG_BASE];
        p.extend_from_slice(vals);
        self.packet(op::SET_CONTEXT_REG, &p)
    }
    pub fn set_uconfig(&mut self, reg: u32, vals: &[u32]) -> bool {
        let mut p = vec![reg - UCONFIG_REG_BASE];
        p.extend_from_slice(vals);
        self.packet(op::SET_UCONFIG_REG, &p)
    }
    /// NOP packet carrying a tag and a string (debug markers).
    pub fn marker(&mut self, tag: u32, text: &str) -> bool {
        let mut p = vec![tag];
        let bytes = text.as_bytes();
        let words = (bytes.len() + 4) / 4;
        for i in 0..words {
            let mut w = [0u8; 4];
            for (j, b) in w.iter_mut().enumerate() {
                if let Some(&c) = bytes.get(i * 4 + j) {
                    *b = c;
                }
            }
            p.push(u32::from_le_bytes(w));
        }
        self.packet(op::NOP, &p)
    }
    /// Pads the rest of the buffer with one NOP packet; returns dwords used (all).
    pub fn finish(mut self) -> u32 {
        let rest = self.buf.len() - self.pos;
        write_nop(&mut self.buf[self.pos..], rest);
        self.pos = self.buf.len();
        self.buf.len() as u32
    }
}

pub fn decode_string(words: &[u32]) -> String {
    let mut s = Vec::new();
    for w in words {
        for b in w.to_le_bytes() {
            if b == 0 {
                return String::from_utf8_lossy(&s).into_owned();
            }
            s.push(b);
        }
    }
    String::from_utf8_lossy(&s).into_owned()
}
