//! Swapchain presentation: copies a rendered image (or uploaded pixels) on
//! the game's queue and scales it into a swapchain image on the present
//! queue (`Ctx::present_queue`), so the game's GPU work never waits for the
//! window to hand out an image (a minimized or hidden window, a throttled
//! compositor). A frame whose slot is still being presented is not shown.

use super::{ctx, window};
use ash::vk;
use parking_lot::Mutex;

const FRAMES: usize = 3;

/// How long a present may wait for a swapchain image before the frame is
/// left out (ns).
const ACQUIRE_TIMEOUT_NS: u64 = 2_000_000;

/// One present: the frame copied into `image` on the game's queue, then
/// scaled into a swapchain image on the present queue.
struct Slot {
    copy: vk::CommandBuffer,
    blit: vk::CommandBuffer,
    /// Copy done (game queue) -> blit (present queue); unused with one queue.
    copied: vk::Semaphore,
    acquire: vk::Semaphore,
    /// The slot's last blit done (created signalled).
    fence: vk::Fence,
    /// The copy of the frame: image, memory, size, format.
    image: Option<(vk::Image, vk::DeviceMemory, u32, u32, vk::Format)>,
    /// Staging buffer of `Source::Pixels`.
    staging: Option<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)>,
}

struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    done: Vec<vk::Semaphore>,
    frame: usize,
    /// Frames not shown because the window was not taking images.
    skipped: u64,
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

pub enum Source {
    /// An image the GPU wrote; `layout` is its current layout (kept).
    Image { image: vk::Image, layout: vk::ImageLayout, width: u32, height: u32, format: vk::Format },
    /// Linear BGRA8 pixels.
    Pixels { data: Vec<u8>, width: u32, height: u32 },
}

unsafe fn create_swapchain(p: &mut Presenter) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let caps = c.surface_i.get_physical_device_surface_capabilities(c.pdev, p.surface).map_err(|e| format!("surface caps: {e}"))?;
        let formats = c.surface_i.get_physical_device_surface_formats(c.pdev, p.surface).map_err(|e| format!("surface formats: {e}"))?;
        // UNORM: display images hold sRGB-encoded values in UNORM storage
        // (8-bit sRGB targets are stored as UNORM, A2R10G10B10 sRGB has no
        // sRGB format), which a blit must pass through unchanged; an sRGB
        // swapchain would encode them a second time.
        let fmt = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
            .or_else(|| formats.first())
            .copied()
            .ok_or("no surface format")?;
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX {
            extent = vk::Extent2D { width: window::CLIENT_W.load(std::sync::atomic::Ordering::Relaxed), height: window::CLIENT_H.load(std::sync::atomic::Ordering::Relaxed) };
        }
        if extent.width == 0 || extent.height == 0 {
            return Err("window minimized".into());
        }
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count });
        let old = p.swapchain;
        let sci = vk::SwapchainCreateInfoKHR::default()
            .surface(p.surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode(p.surface))
            .clipped(true)
            .old_swapchain(old);
        let sc = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        if old != vk::SwapchainKHR::null() {
            c.device.device_wait_idle().ok();
            c.swapchain_d.destroy_swapchain(old, None);
            for s in p.done.drain(..) {
                c.device.destroy_semaphore(s, None);
            }
        }
        p.swapchain = sc;
        p.images = c.swapchain_d.get_swapchain_images(sc).map_err(|e| format!("swapchain images: {e}"))?;
        p.format = fmt.format;
        p.extent = extent;
        for _ in 0..p.images.len() {
            p.done.push(c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        crate::info!("swapchain {}x{} {:?} ({} images, {:?})", extent.width, extent.height, fmt.format, p.images.len(), sci.present_mode);
    }
    Ok(())
}

/// The swapchain's present mode: FIFO (vsync), or `TLG_PRESENT=mailbox`
/// (vsync without back-pressure: the newest frame is shown, for uncapped
/// rates above the display's) or `TLG_PRESENT=immediate` (no vsync, tears)
/// when the surface supports them; else settings.txt `vsync`.
fn present_mode(surface: vk::SurfaceKHR) -> vk::PresentModeKHR {
    use crate::config::VSync;
    let want = match std::env::var("TLG_PRESENT").as_deref() {
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("fifo") => vk::PresentModeKHR::FIFO,
        _ => match crate::config::settings().vsync {
            VSync::On => vk::PresentModeKHR::FIFO,
            VSync::Off => vk::PresentModeKHR::IMMEDIATE,
            VSync::Fast => vk::PresentModeKHR::MAILBOX,
        },
    };
    if want == vk::PresentModeKHR::FIFO {
        return want;
    }
    let c = ctx();
    let modes = unsafe { c.surface_i.get_physical_device_surface_present_modes(c.pdev, surface) }.unwrap_or_default();
    if modes.contains(&want) {
        want
    } else {
        crate::warn!("present mode {want:?} unsupported (supported: {modes:?}): FIFO");
        vk::PresentModeKHR::FIFO
    }
}

pub fn init(title: &str) -> Result<(), String> {
    super::init()?;
    let w = window::create(title)?;
    let c = ctx();
    unsafe {
        let sci = vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance).hwnd(w.hwnd);
        let surface = c.win32_i.create_win32_surface(&sci, None).map_err(|e| format!("create surface: {e}"))?;
        let supported = c.surface_i.get_physical_device_surface_support(c.pdev, c.queue_family, surface).unwrap_or(false);
        if !supported {
            return Err("queue family cannot present".into());
        }
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        let cmds = c
            .device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2 * FRAMES as u32))
            .map_err(|e| e.to_string())?;
        let sem = || c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string());
        let mut slots = Vec::new();
        for k in 0..FRAMES {
            slots.push(Slot {
                copy: cmds[2 * k],
                blit: cmds[2 * k + 1],
                copied: sem()?,
                acquire: sem()?,
                fence: c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).map_err(|e| e.to_string())?,
                image: None,
                staging: None,
            });
        }
        let mut p = Presenter {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            pool,
            slots,
            done: Vec::new(),
            frame: 0,
            skipped: 0,
        };
        create_swapchain(&mut p)?;
        *PRESENTER.lock() = Some(p);
    }
    Ok(())
}

fn barrier(c: &super::Ctx, cmd: vk::CommandBuffer, img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
    let b = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(img)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
    }
}

/// Where a `sw` x `sh` frame goes in a `dw` x `dh` image: all of it
/// (`scaling = stretch`), or the largest centred area of the frame's shape
/// (`fit`; the rest black).
fn dst_rect(sw: u32, sh: u32, dw: u32, dh: u32) -> (i32, i32, i32, i32) {
    if crate::config::settings().scaling == crate::config::Scaling::Stretch || sw == 0 || sh == 0 {
        return (0, 0, dw as i32, dh as i32);
    }
    fit_rect(sw, sh, dw, dh)
}

/// A frame up to 1% smaller than the target in each direction is shown
/// pixel for pixel, centred (e.g. 1360x764 rendered for a 1366x768
/// screen); otherwise it is scaled to fit.
fn fit_rect(sw: u32, sh: u32, dw: u32, dh: u32) -> (i32, i32, i32, i32) {
    if sw <= dw && sh <= dh && (dw - sw) * 100 <= dw && (dh - sh) * 100 <= dh {
        let (x, y) = ((dw - sw) / 2, (dh - sh) / 2);
        return (x as i32, y as i32, (x + sw) as i32, (y + sh) as i32);
    }
    let (sw, sh, dw, dh) = (sw as u64, sh as u64, dw as u64, dh as u64);
    // Width-limited if the target is relatively narrower than the frame.
    let (w, h) = if dw * sh <= dh * sw { (dw, (dw * sh + sw / 2) / sw) } else { ((dh * sw + sh / 2) / sh, dh) };
    let (x, y) = ((dw - w) / 2, (dh - h) / 2);
    (x as i32, y as i32, (x + w) as i32, (y + h) as i32)
}

#[cfg(test)]
mod tests {
    #[test]
    fn fit() {
        assert_eq!(super::fit_rect(1920, 1080, 1920, 1080), (0, 0, 1920, 1080));
        assert_eq!(super::fit_rect(1920, 1080, 2560, 1440), (0, 0, 2560, 1440));
        // 16:10: bars above and below; 21:9: at the sides.
        assert_eq!(super::fit_rect(1920, 1080, 1920, 1200), (0, 60, 1920, 1140));
        assert_eq!(super::fit_rect(1920, 1080, 3440, 1440), (440, 0, 3000, 1440));
        assert_eq!(super::fit_rect(1920, 1080, 1280, 1024), (0, 152, 1280, 872));
        // Nearly the target's size: pixel for pixel.
        assert_eq!(super::fit_rect(1360, 764, 1366, 768), (3, 2, 1363, 766));
        assert_eq!(super::fit_rect(1280, 720, 1366, 768), (0, 0, 1365, 768));
    }
}

/// Presents one frame. Returns false if it was not shown (a minimized
/// window, or the window still holding the images of earlier frames).
pub fn present(src: Source) -> bool {
    // (The frame's batches are submitted ahead of the copy: queue order.)
    crate::gpu::rec::drain();
    let c = ctx();
    let mut g = PRESENTER.lock();
    let Some(p) = g.as_mut() else { return false };
    unsafe {
        if window::RESIZED.swap(false, std::sync::atomic::Ordering::SeqCst) {
            if let Err(e) = create_swapchain(p) {
                crate::debug!("swapchain recreate: {e}");
                return false;
            }
        }
        let k = p.frame % FRAMES;
        // The slot's previous frame not yet presented: the window is not
        // taking images (minimized, hidden, a throttled compositor). This
        // frame is left out; nothing waits for the window.
        if c.device.get_fence_status(p.slots[k].fence) != Ok(true) {
            p.skipped += 1;
            note_skipped(p.skipped);
            return false;
        }
        let (sw, sh, fmt) = match &src {
            Source::Image { width, height, format, .. } => (*width, *height, *format),
            Source::Pixels { width, height, .. } => (*width, *height, vk::Format::B8G8R8A8_UNORM),
        };
        if let Err(e) = prepare_slot(&mut p.slots[k], sw, sh, fmt, &src) {
            crate::error!("present: {e}");
            return false;
        }
        let (idx, _) = match c.swapchain_d.acquire_next_image(p.swapchain, ACQUIRE_TIMEOUT_NS, p.slots[k].acquire, vk::Fence::null()) {
            Ok(r) => r,
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                p.skipped += 1;
                note_skipped(p.skipped);
                return false;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
                return false;
            }
            Err(e) => {
                crate::error!("acquire: {e}");
                return false;
            }
        };
        p.frame += 1;
        let two = c.present_queue.is_some();
        let s = &p.slots[k];
        let (copy_img, ..) = s.image.unwrap();
        // The copy, on the game's queue after its work.
        let cmd = s.copy;
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        barrier(c, cmd, copy_img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        match &src {
            Source::Image { image, layout, .. } => {
                // (Its layout kept: the game's writes before the copy, the
                // copy before its later writes.)
                barrier(c, cmd, *image, *layout, *layout);
                let region = vk::ImageCopy { src_subresource: layers, src_offset: vk::Offset3D::default(), dst_subresource: layers, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: sw, height: sh, depth: 1 } };
                c.device.cmd_copy_image(cmd, *image, *layout, copy_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
                barrier(c, cmd, *image, *layout, *layout);
            }
            Source::Pixels { data, .. } => {
                let (sb, _, sp, _) = s.staging.unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), sp, (sw * sh * 4) as usize);
                let region = vk::BufferImageCopy { buffer_offset: 0, buffer_row_length: 0, buffer_image_height: 0, image_subresource: layers, image_offset: vk::Offset3D::default(), image_extent: vk::Extent3D { width: sw, height: sh, depth: 1 } };
                c.device.cmd_copy_buffer_to_image(cmd, sb, copy_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            }
        }
        barrier(c, cmd, copy_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        c.device.end_command_buffer(cmd).ok();
        let copied: &[vk::Semaphore] = if two { std::slice::from_ref(&s.copied) } else { &[] };
        if let Err(e) = c.submit(cmd, &[], copied, vk::Fence::null()) {
            crate::error!("{e}");
            return false;
        }
        // The blit into the swapchain image, on the present queue.
        let cmd = s.blit;
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let dst = p.images[idx as usize];
        barrier(c, cmd, dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let (dx0, dy0, dx1, dy1) = dst_rect(sw, sh, p.extent.width, p.extent.height);
        if (dx0, dy0, dx1, dy1) != (0, 0, p.extent.width as i32, p.extent.height as i32) {
            // Black bars around the frame.
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            c.device.cmd_clear_color_image(cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] }, &[range]);
            let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: sw as i32, y: sh as i32, z: 1 }],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D { x: dx0, y: dy0, z: 0 }, vk::Offset3D { x: dx1, y: dy1, z: 1 }],
        };
        let filter = match crate::config::settings().filter {
            crate::config::Filter::Smooth => vk::Filter::LINEAR,
            crate::config::Filter::Sharp => vk::Filter::NEAREST,
        };
        c.device.cmd_blit_image(cmd, copy_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], filter);
        barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
        c.device.end_command_buffer(cmd).ok();
        let done = p.done[idx as usize];
        let queue = c.present_queue.as_ref().unwrap_or(&c.queue);
        let mut waits = vec![(s.acquire, vk::PipelineStageFlags::TRANSFER)];
        if two {
            waits.push((s.copied, vk::PipelineStageFlags::TRANSFER));
        }
        c.device.reset_fences(&[s.fence]).ok();
        if let Err(e) = c.submit_to(queue, &[cmd], &waits, &[done], s.fence) {
            crate::error!("{e}");
            return false;
        }
        let scs = [p.swapchain];
        let idxs = [idx];
        let waits = [done];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idxs);
        let q = queue.lock();
        match c.swapchain_d.queue_present(*q, &pi) {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst),
            Err(e) => crate::error!("present: {e}"),
        }
    }
    true
}

/// Logs the first frames left out and then every 300th.
fn note_skipped(n: u64) {
    if n <= 3 || n % 300 == 0 {
        crate::info!("present: {n} frames not shown so far (the window was not taking images)");
    }
}

/// The slot's copy image (and staging buffer for pixels) for a `w` x `h`
/// frame of `fmt`; the slot is idle (its fence signalled).
unsafe fn prepare_slot(s: &mut Slot, w: u32, h: u32, fmt: vk::Format, src: &Source) -> Result<(), String> {
    let c = ctx();
    unsafe {
        if s.image.is_none_or(|i| (i.2, i.3, i.4) != (w, h, fmt)) {
            if let Some((img, mem, ..)) = s.image.take() {
                c.device.destroy_image(img, None);
                c.free(mem);
            }
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
            let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
            s.image = Some((img, mem, w, h, fmt));
        }
        if let Source::Pixels { data, .. } = src {
            let size = (w * h * 4) as u64;
            if data.len() < size as usize {
                return Err(format!("{} bytes of pixels for {w}x{h}", data.len()));
            }
            if s.staging.as_ref().is_none_or(|st| st.3 < size) {
                let (b, m, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
                s.staging = Some((b, m, ptr, size));
            }
        }
    }
    Ok(())
}
