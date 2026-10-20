//! Image memory in blocks (`tune image_blocks`): images take ranges of
//! `BLOCK`-sized device allocations instead of an allocation each. Drivers
//! cap the allocations alive at once - AMD's on Windows at 4,096 - and First
//! Antenna kept ~1,800 images then, ~2,150 allocations in all. Images over
//! `DEDICATED` keep an allocation of their own, as does any image when no
//! new block can be had (then `Ctx::alloc` and its fallback to system memory
//! apply). Every image is optimal-tiled and nothing else lives in the
//! blocks, so neighbours need no buffer-image granularity. A range is freed
//! once the GPU is done with its image (the renderer's retirement); a block
//! with no range in use is released, one spare kept per memory type, and
//! spares go under VRAM pressure (`trim`).

use super::vk::Ctx;
use ash::vk;
use parking_lot::Mutex;
use std::collections::BTreeMap;

/// Block size.
const BLOCK: u64 = 64 << 20;
/// Images larger than this get an allocation of their own.
const DEDICATED: u64 = 16 << 20;
/// `ImageMem::block` of an allocation of its own.
const OWN: u32 = u32::MAX;

/// An image's memory: a range of a block, or an allocation of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMem {
    pub memory: vk::DeviceMemory,
    pub offset: u64,
    pub size: u64,
    block: u32,
}

impl Default for ImageMem {
    fn default() -> Self {
        ImageMem { memory: vk::DeviceMemory::null(), offset: 0, size: 0, block: OWN }
    }
}

/// Free ranges of a block (offset -> length), neighbours merged.
#[derive(Default)]
struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    /// The first free range that holds `size` bytes at a multiple of `align`.
    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        let align = align.max(1);
        let (off, len, at) = self.0.iter().find_map(|(&off, &len)| {
            let at = off.next_multiple_of(align);
            (at + size <= off + len).then_some((off, len, at))
        })?;
        self.0.remove(&off);
        if at > off {
            self.0.insert(off, at - off);
        }
        if at + size < off + len {
            self.0.insert(at + size, off + len - at - size);
        }
        Some(at)
    }

    fn give(&mut self, off: u64, size: u64) {
        let (mut lo, mut hi) = (off, off + size);
        if let Some((&p, &plen)) = self.0.range(..off).next_back() {
            if p + plen == off {
                lo = p;
                self.0.remove(&p);
            }
        }
        if let Some(&nlen) = self.0.get(&hi) {
            self.0.remove(&hi);
            hi += nlen;
        }
        self.0.insert(lo, hi - lo);
    }
}

struct Block {
    memory: vk::DeviceMemory,
    ty: u32,
    free: Ranges,
    /// Ranges in use and their bytes.
    ranges: u32,
    used: u64,
}

#[derive(Default)]
struct Blocks {
    blocks: BTreeMap<u32, Block>,
    next: u32,
    /// Images in allocations of their own (`OWN`), and why: too large, no
    /// block to be had, the driver's wish.
    own: [u64; 3],
}

static BLOCKS: Mutex<Option<Blocks>> = Mutex::new(None);

/// Whether images take ranges of blocks (`tune image_blocks`).
pub fn on() -> bool {
    crate::tune::get(crate::tune::IMAGE_BLOCKS) != 0
}

/// Memory for `image` with requirements `req`, device-local where it can be;
/// `dedicated`: the driver prefers or requires an allocation of its own for
/// it (VkMemoryDedicatedRequirements - render targets and depth buffers on
/// some drivers, for their compression).
pub fn alloc(c: &Ctx, image: vk::Image, req: vk::MemoryRequirements, dedicated: bool) -> Result<ImageMem, String> {
    // (An allocation of its own; `why`: 0 too large, 1 no block.)
    let own = |why: Option<usize>| -> Result<ImageMem, String> {
        if let Some(why) = why {
            BLOCKS.lock().get_or_insert_with(Default::default).own[why] += 1;
        }
        let memory = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        Ok(ImageMem { memory, offset: 0, size: req.size, block: OWN })
    };
    if !on() {
        return own(None);
    }
    if dedicated {
        BLOCKS.lock().get_or_insert_with(Default::default).own[2] += 1;
        let memory = c.alloc_dedicated(req, image)?;
        return Ok(ImageMem { memory, offset: 0, size: req.size, block: OWN });
    }
    if req.size > DEDICATED || req.alignment > BLOCK {
        return own(Some(0));
    }
    let Some(ty) = c.memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL) else { return own(Some(1)) };
    {
        let mut g = BLOCKS.lock();
        let b = g.get_or_insert_with(Default::default);
        for (&id, blk) in b.blocks.iter_mut().filter(|(_, blk)| blk.ty == ty) {
            if let Some(offset) = blk.free.take(req.size, req.alignment) {
                blk.ranges += 1;
                blk.used += req.size;
                return Ok(ImageMem { memory: blk.memory, offset, size: req.size, block: id });
            }
        }
        if let Some(memory) = c.alloc_type(BLOCK, ty) {
            let id = b.next;
            b.next += 1;
            let mut free = Ranges::default();
            free.give(0, BLOCK);
            let offset = free.take(req.size, req.alignment).expect("a new block holds the image");
            b.blocks.insert(id, Block { memory, ty, free, ranges: 1, used: req.size });
            return Ok(ImageMem { memory, offset, size: req.size, block: id });
        }
    }
    own(Some(1))
}

/// Releases an image's memory (the GPU no longer uses the image).
pub fn free(c: &Ctx, m: ImageMem) {
    if m.block == OWN {
        if m.memory != vk::DeviceMemory::null() {
            c.free(m.memory);
        }
        return;
    }
    let mut g = BLOCKS.lock();
    let Some(b) = g.as_mut() else { return };
    let Some(blk) = b.blocks.get_mut(&m.block) else { return };
    blk.free.give(m.offset, m.size);
    blk.ranges -= 1;
    blk.used -= m.size;
    if blk.ranges == 0 {
        // (One empty block per memory type stays for the next images.)
        let ty = blk.ty;
        let spare = b.blocks.iter().any(|(&id, o)| id != m.block && o.ty == ty && o.ranges == 0);
        if spare {
            if let Some(blk) = b.blocks.remove(&m.block) {
                c.free(blk.memory);
            }
        }
    }
}

/// Releases the empty blocks (VRAM pressure); returns the bytes freed.
pub fn trim(c: &Ctx) -> u64 {
    let mut g = BLOCKS.lock();
    let Some(b) = g.as_mut() else { return 0 };
    let empty: Vec<u32> = b.blocks.iter().filter(|(_, blk)| blk.ranges == 0).map(|(&id, _)| id).collect();
    for id in &empty {
        if let Some(blk) = b.blocks.remove(id) {
            c.free(blk.memory);
        }
    }
    empty.len() as u64 * BLOCK
}

/// Report text: blocks, the images in them and their bytes, images with an
/// allocation of their own.
pub fn report() -> String {
    let g = BLOCKS.lock();
    let Some(b) = g.as_ref() else { return "image blocks: none".into() };
    let ranges: u64 = b.blocks.values().map(|blk| blk.ranges as u64).sum();
    let used: u64 = b.blocks.values().map(|blk| blk.used).sum();
    let empty = b.blocks.values().filter(|blk| blk.ranges == 0).count();
    format!(
        "image blocks: {} of {} MiB ({} empty), {} images in them ({} MiB); images with an allocation of their own since the start: {} large, {} without a block, {} the driver asked for",
        b.blocks.len(),
        BLOCK >> 20,
        empty,
        ranges,
        used >> 20,
        b.own[0],
        b.own[1],
        b.own[2]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_take_align_and_merge() {
        let mut r = Ranges::default();
        r.give(0, 1 << 20);
        let a = r.take(1000, 256).unwrap();
        assert_eq!(a, 0);
        let b = r.take(4096, 4096).unwrap();
        assert_eq!(b, 4096);
        // The gap before `b` (1000..4096) holds a small range.
        let c = r.take(512, 256).unwrap();
        assert_eq!(c, 1024);
        r.give(a, 1000);
        r.give(c, 512);
        r.give(b, 4096);
        // All merged back into one range.
        assert_eq!(r.0.len(), 1);
        assert_eq!(r.0.get(&0), Some(&(1 << 20)));
        assert!(r.take(1 << 20, 1).is_some());
        assert!(r.take(1, 1).is_none());
    }

    #[test]
    fn ranges_first_fit_reuses_freed_space() {
        let mut r = Ranges::default();
        r.give(0, 64 << 10);
        let xs: Vec<u64> = (0..8).map(|_| r.take(8 << 10, 8 << 10).unwrap()).collect();
        assert_eq!(xs, (0..8).map(|i| i * (8 << 10)).collect::<Vec<_>>());
        assert!(r.take(1, 1).is_none());
        r.give(xs[3], 8 << 10);
        r.give(xs[4], 8 << 10);
        assert_eq!(r.take(16 << 10, 8 << 10), Some(xs[3]));
    }
}
