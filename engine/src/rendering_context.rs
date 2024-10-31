pub use crate::image::{Image, ImageAttributes, ImageLayoutState};
use anyhow::Result;
use ash::vk;
use ash::vk::{DeviceQueueInfo2, SurfaceCapabilitiesKHR};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};
use std::collections::HashSet;
use std::io;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub struct RenderingContext {
    pub queues: Vec<vk::Queue>,
    pub pageable_device_local_memory_extension:
        Option<ash::ext::pageable_device_local_memory::Device>,
    pub swapchain_extension: ash::khr::swapchain::Device,
    pub device: ash::Device,
    pub queue_family_indices: HashSet<u32>,
    pub queue_families: QueueFamilies,
    pub physical_device: PhysicalDevice,
    pub surface_extension: ash::khr::surface::Instance,
    pub instance: ash::Instance,
    pub entry: ash::Entry,
}

#[derive(Debug, Clone)]
pub struct QueueFamily {
    pub index: u32,
    pub properties: vk::QueueFamilyProperties,
}

#[derive(Debug)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub vulkan12_features: vk::PhysicalDeviceVulkan12Features<'static>,
    pub vulkan13_features: vk::PhysicalDeviceVulkan13Features<'static>,
    pub pageable_device_local_memory_features:
        vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT<'static>,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<QueueFamily>,
}

type QueueFamilyPicker = fn(Vec<PhysicalDevice>) -> Result<(PhysicalDevice, QueueFamilies)>;

pub struct RenderingContextAttributes<'window> {
    pub compatibility_window: &'window Window,
    pub queue_family_picker: QueueFamilyPicker,
}

pub struct QueueFamilies {
    pub graphics: u32,
    pub present: u32,
    pub transfer: u32,
    pub compute: u32,
}

pub mod queue_family_picker {
    use crate::rendering_context::{PhysicalDevice, QueueFamilies};
    use anyhow::Context as AnyhowContext;
    use anyhow::Result;
    use ash::vk;

    pub fn single_queue_family(
        physical_devices: Vec<PhysicalDevice>,
    ) -> Result<(PhysicalDevice, QueueFamilies)> {
        let physical_device = physical_devices.into_iter().next().unwrap();
        let queue_family = physical_device
            .queue_families
            .iter()
            .find(|queue_family| {
                queue_family
                    .properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && queue_family
                        .properties
                        .queue_flags
                        .contains(vk::QueueFlags::COMPUTE)
            })
            .map(|queue_family| queue_family.index)
            .context("No suitable queue family found")?;
        Ok((
            physical_device,
            QueueFamilies {
                graphics: queue_family,
                present: queue_family,
                transfer: queue_family,
                compute: queue_family,
            },
        ))
    }
}

macro_rules! check_feature {
    ($features:expr, $feature_name:ident) => {
        if $features.$feature_name == vk::FALSE {
            return Err(anyhow::anyhow!(concat!(
                "Physical device does not support ",
                stringify!($feature_name)
            )));
        }
    };
}

impl RenderingContext {
    pub fn new(attributes: RenderingContextAttributes) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;

            let raw_display_handle = attributes.compatibility_window.display_handle()?.as_raw();
            let raw_window_handle = attributes.compatibility_window.window_handle()?.as_raw();

            let available_extensions = entry
                .enumerate_instance_extension_properties(None)?
                .into_iter()
                .map(|extension| {
                    let name = extension.extension_name;
                    std::ffi::CStr::from_ptr(name.as_ptr())
                        .to_str()
                        .unwrap()
                        .to_string()
                })
                .collect::<HashSet<_>>();

            let mut extensions =
                ash_window::enumerate_required_extensions(raw_display_handle)?.to_vec();

            if cfg!(debug_assertions) {
                if available_extensions.contains(ash::ext::debug_utils::NAME.to_str()?) {
                    extensions.push(ash::ext::debug_utils::NAME.as_ptr());
                }
            }

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_extension_names(&extensions),
                None,
            )?;

            let surface_extension = ash::khr::surface::Instance::new(&entry, &instance);

            let compatibility_surface = ash_window::create_surface(
                &entry,
                &instance,
                raw_display_handle,
                raw_window_handle,
                None,
            )?;

            let mut physical_devices = instance
                .enumerate_physical_devices()?
                .into_iter()
                .map(|handle| {
                    let properties = instance.get_physical_device_properties(handle);
                    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
                    let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default();
                    let mut pageable_device_local_memory_features =
                        vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::default();
                    let mut features = vk::PhysicalDeviceFeatures2::default()
                        .push_next(&mut vulkan12_features)
                        .push_next(&mut vulkan13_features)
                        .push_next(&mut pageable_device_local_memory_features);
                    instance.get_physical_device_features2(handle, &mut features);
                    let features = features.features;
                    let memory_properties = instance.get_physical_device_memory_properties(handle);
                    let queue_family_properties =
                        instance.get_physical_device_queue_family_properties(handle);

                    let queue_families = queue_family_properties
                        .into_iter()
                        .enumerate()
                        .map(|(index, properties)| QueueFamily {
                            index: index as u32,
                            properties,
                        })
                        .collect::<Vec<_>>();

                    PhysicalDevice {
                        handle,
                        properties,
                        features,
                        vulkan12_features,
                        vulkan13_features,
                        pageable_device_local_memory_features,
                        memory_properties,
                        queue_families,
                    }
                })
                .collect::<Vec<_>>();

            physical_devices.retain(|device| {
                surface_extension
                    .get_physical_device_surface_support(device.handle, 0, compatibility_surface)
                    .unwrap_or(false)
            });

            surface_extension.destroy_surface(compatibility_surface, None);

            let (physical_device, queue_families) =
                (attributes.queue_family_picker)(physical_devices)?;

            let features12 = physical_device.vulkan12_features;
            let features13 = physical_device.vulkan13_features;

            check_feature!(features12, buffer_device_address);
            check_feature!(features12, descriptor_indexing);
            check_feature!(features12, scalar_block_layout);
            check_feature!(features13, dynamic_rendering);
            check_feature!(features13, synchronization2);

            let queue_family_indices = HashSet::from([
                queue_families.graphics,
                queue_families.present,
                queue_families.transfer,
                queue_families.compute,
            ]);

            let queue_create_infos = queue_family_indices
                .iter()
                .copied()
                .map(|index| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(index)
                        .queue_priorities(&[1.0])
                })
                .collect::<Vec<_>>();

            let is_debug = cfg!(debug_assertions);

            let is_capture_replay_supported = physical_device
                .vulkan12_features
                .buffer_device_address_capture_replay
                == vk::TRUE;

            let is_pageable_device_local_memory_supported = physical_device
                .pageable_device_local_memory_features
                .pageable_device_local_memory
                == vk::TRUE;

            let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];

            let mut pageable_device_local_memory_extension = None;

            if is_pageable_device_local_memory_supported {
                device_extensions.push(ash::ext::memory_priority::NAME.as_ptr());
                device_extensions.push(ash::ext::pageable_device_local_memory::NAME.as_ptr());
            }

            let device = instance.create_device(
                physical_device.handle,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_create_infos)
                    .enabled_extension_names(&device_extensions)
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan12Features::default()
                            .buffer_device_address(true)
                            .buffer_device_address_capture_replay(
                                is_debug && is_capture_replay_supported,
                            )
                            .descriptor_indexing(true)
                            .scalar_block_layout(true)
                            .shader_sampled_image_array_non_uniform_indexing(true)
                            .descriptor_binding_sampled_image_update_after_bind(true)
                            .shader_uniform_buffer_array_non_uniform_indexing(true)
                            .descriptor_binding_uniform_buffer_update_after_bind(true)
                            .shader_storage_buffer_array_non_uniform_indexing(true)
                            .descriptor_binding_storage_buffer_update_after_bind(true)
                            .descriptor_binding_partially_bound(true),
                    )
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan13Features::default()
                            .dynamic_rendering(true)
                            .synchronization2(true),
                    )
                    .push_next(
                        &mut vk::PhysicalDevicePageableDeviceLocalMemoryFeaturesEXT::default()
                            .pageable_device_local_memory(
                                is_pageable_device_local_memory_supported,
                            ),
                    ),
                None,
            )?;

            if is_pageable_device_local_memory_supported {
                pageable_device_local_memory_extension = Some(
                    ash::ext::pageable_device_local_memory::Device::new(&instance, &device),
                );
            }

            let swapchain_extension = ash::khr::swapchain::Device::new(&instance, &device);

            let queues = queue_family_indices
                .iter()
                .map(|index| {
                    device
                        .get_device_queue2(&DeviceQueueInfo2::default().queue_family_index(*index))
                })
                .collect::<Vec<_>>();

            Ok(Self {
                queues,
                device,
                queue_family_indices,
                queue_families,
                physical_device,
                surface_extension,
                instance,
                entry,
                swapchain_extension,
                pageable_device_local_memory_extension,
            })
        }
    }

    // safety: The window should outlive the surface.
    pub unsafe fn create_surface(&self, window: &Window) -> Result<Surface> {
        let raw_display_handle = window.display_handle()?.as_raw();
        let raw_window_handle = window.window_handle()?.as_raw();

        let handle = ash_window::create_surface(
            &self.entry,
            &self.instance,
            raw_display_handle,
            raw_window_handle,
            None,
        )?;

        let capabilities = self
            .surface_extension
            .get_physical_device_surface_capabilities(self.physical_device.handle, handle)?;

        let formats = self
            .surface_extension
            .get_physical_device_surface_formats(self.physical_device.handle, handle)?;

        let present_modes = self
            .surface_extension
            .get_physical_device_surface_present_modes(self.physical_device.handle, handle)?;

        Ok(Surface {
            handle,
            capabilities,
            formats,
            present_modes,
        })
    }

    pub fn create_shader_module(&self, code: &[u8]) -> Result<vk::ShaderModule> {
        let mut code = io::Cursor::new(code);
        let code = ash::util::read_spv(&mut code)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader_module = unsafe { self.device.create_shader_module(&create_info, None) }?;
        Ok(shader_module)
    }

    pub fn create_graphics_pipeline(
        &self,
        vertex_shader: vk::ShaderModule,
        fragment_shader: vk::ShaderModule,
        image_extent: vk::Extent2D,
        image_format: vk::Format,
        depth_format: vk::Format,
        pipeline_layout: vk::PipelineLayout,
        pipeline_cache: vk::PipelineCache,
    ) -> Result<vk::Pipeline> {
        let entry_point = std::ffi::CString::new("main")?;

        unsafe {
            Ok(self
                .device
                .create_graphics_pipelines(
                    pipeline_cache,
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(vertex_shader)
                                .name(&entry_point),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .module(fragment_shader)
                                .name(&entry_point),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewports(&[vk::Viewport::default()
                                    .width(image_extent.width as f32)
                                    .height(image_extent.height as f32)
                                    .max_depth(1.0)])
                                .scissors(&[vk::Rect2D::default().extent(image_extent)]),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::NONE)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default()
                                .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(true)
                                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_4),
                        )
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[image_format])
                                .depth_attachment_format(depth_format),
                        )],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap())
        }
    }

    pub fn create_allocator(
        &self,
        debug_settings: AllocatorDebugSettings,
        allocation_sizes: AllocationSizes,
    ) -> Result<Allocator> {
        Ok(Allocator::new(&AllocatorCreateDesc {
            instance: self.instance.clone(),
            device: self.device.clone(),
            physical_device: self.physical_device.handle,
            debug_settings,
            buffer_device_address: true,
            allocation_sizes,
        })?)
    }
}

pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl Drop for RenderingContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
