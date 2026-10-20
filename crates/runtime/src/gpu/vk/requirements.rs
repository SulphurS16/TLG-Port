//! What a Vulkan device must offer for tlg: the API version, extensions and
//! features the renderer enables unconditionally (the optional ones have
//! fallbacks). Shared with the launcher, which lists the graphics cards
//! (`crates/launcher/src/sys.rs` includes this file).

use ash::vk;
use std::ffi::CStr;

/// What a device lacks of the API version, extensions and features
/// `init` enables unconditionally (empty: usable).
pub unsafe fn missing(instance: &ash::Instance, pd: vk::PhysicalDevice, props: &vk::PhysicalDeviceProperties) -> Vec<String> {
    let mut m = Vec::new();
    if props.api_version < vk::API_VERSION_1_3 {
        m.push(format!("Vulkan 1.3 (has {}.{})", vk::api_version_major(props.api_version), vk::api_version_minor(props.api_version)));
        return m;
    }
    let exts = unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    for e in [ash::khr::swapchain::NAME, ash::khr::push_descriptor::NAME] {
        if !exts.iter().any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) } == e) {
            m.push(e.to_string_lossy().into_owned());
        }
    }
    let f = unsafe { instance.get_physical_device_features(pd) };
    let base = [
        ("robustBufferAccess", f.robust_buffer_access),
        ("geometryShader", f.geometry_shader),
        ("tessellationShader", f.tessellation_shader),
        ("independentBlend", f.independent_blend),
        ("dualSrcBlend", f.dual_src_blend),
        ("depthClamp", f.depth_clamp),
        ("depthBiasClamp", f.depth_bias_clamp),
        ("samplerAnisotropy", f.sampler_anisotropy),
        ("textureCompressionBC", f.texture_compression_bc),
        ("imageCubeArray", f.image_cube_array),
        ("shaderInt64", f.shader_int64),
        ("shaderStorageImageWriteWithoutFormat", f.shader_storage_image_write_without_format),
        ("shaderStorageImageReadWithoutFormat", f.shader_storage_image_read_without_format),
        ("vertexPipelineStoresAndAtomics", f.vertex_pipeline_stores_and_atomics),
        ("fragmentStoresAndAtomics", f.fragment_stores_and_atomics),
    ];
    m.extend(base.iter().filter(|(_, v)| *v == 0).map(|(n, _)| n.to_string()));
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    {
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut f13);
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
    }
    let newer = [
        ("timelineSemaphore", f12.timeline_semaphore),
        ("dynamicRendering", f13.dynamic_rendering),
        ("synchronization2", f13.synchronization2),
        ("maintenance4", f13.maintenance4),
    ];
    m.extend(newer.iter().filter(|(_, v)| *v == 0).map(|(n, _)| n.to_string()));
    m
}
