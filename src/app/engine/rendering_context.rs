/* Assumptions:
- The Vulkan instance is created with the Vulkan 1.3 API version.
- The Vulkan instance is created only with the required extensions for the window system.
- The Vulkan instance is created with the required extensions for the dynamic rendering and buffer device address features.
 */

use anyhow::Result;
use ash::vk;
use ash::vk::{DeviceQueueInfo2, SurfaceCapabilitiesKHR};
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings, MemoryLocation};
use std::collections::HashSet;
use std::io;
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub struct RenderingContext {
    pub queues: Vec<vk::Queue>,
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
    use crate::app::engine::rendering_context::{PhysicalDevice, QueueFamilies};
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

impl RenderingContext {
    pub fn new(attributes: RenderingContextAttributes) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()?;

            let raw_display_handle = attributes.compatibility_window.display_handle()?.as_raw();
            let raw_window_handle = attributes.compatibility_window.window_handle()?.as_raw();

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_extension_names(ash_window::enumerate_required_extensions(
                        raw_display_handle,
                    )?),
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
                    let features = instance.get_physical_device_features(handle);
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

            let device = instance.create_device(
                physical_device.handle,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_create_infos)
                    .enabled_extension_names(&[ash::khr::swapchain::NAME.as_ptr()])
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan12Features::default()
                            .buffer_device_address(true)
                            .descriptor_indexing(true),
                    )
                    .push_next(
                        &mut vk::PhysicalDeviceVulkan13Features::default()
                            .dynamic_rendering(true)
                            .synchronization2(true),
                    ),
                None,
            )?;

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

    pub fn create_image_view(
        &self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let image_view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspect_flags)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
        }?;
        Ok(image_view)
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
                            &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                                vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(false),
                            ]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(&[image_format]),
                        )],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap())
        }
    }

    pub fn transition_image_layout(
        &self,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_state: ImageLayoutState,
        new_state: ImageLayoutState,
    ) {
        unsafe {
            let aspect_mask =
                if new_state.layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                } else {
                    vk::ImageAspectFlags::COLOR
                };

            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2KHR::default()
                        .src_stage_mask(old_state.stage_mask)
                        .dst_stage_mask(new_state.stage_mask)
                        .src_access_mask(old_state.access_mask)
                        .dst_access_mask(new_state.access_mask)
                        .old_layout(old_state.layout)
                        .new_layout(new_state.layout)
                        .src_queue_family_index(old_state.queue_family_index)
                        .dst_queue_family_index(new_state.queue_family_index)
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect_mask)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        ),
                ]),
            );
        }
    }

    pub fn blit_image(
        &self,
        command_buffer: vk::CommandBuffer,
        src_image: vk::Image,
        dst_image: vk::Image,
        src_extent: vk::Extent3D,
        dst_extent: vk::Extent3D,
    ) {
        let subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        unsafe {
            self.device.cmd_blit_image(
                command_buffer,
                src_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_subresource(subresource)
                    .src_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: src_extent.width as i32,
                            y: src_extent.height as i32,
                            z: src_extent.depth as i32,
                        },
                    ])
                    .dst_subresource(subresource)
                    .dst_offsets([
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: dst_extent.width as i32,
                            y: dst_extent.height as i32,
                            z: 1,
                        },
                    ])],
                vk::Filter::NEAREST,
            );
        }
    }

    pub fn begin_rendering(
        &self,
        command_buffer: vk::CommandBuffer,
        view: vk::ImageView,
        clear_color: vk::ClearColorValue,
        render_area: vk::Rect2D,
    ) {
        unsafe {
            self.device.cmd_begin_rendering(
                command_buffer,
                &vk::RenderingInfo::default()
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .clear_value(vk::ClearValue { color: clear_color })
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)])
                    .render_area(render_area),
            );
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

    pub fn create_image(
        &self,
        allocator: &mut Allocator,
        name: &str,
        attributes: ImageAttributes,
    ) -> Result<Image> {
        let image = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(attributes.format)
                    .extent(attributes.extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(attributes.usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }?;

        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: attributes.location,
            linear: attributes.linear,
            allocation_scheme: attributes.allocation_scheme,
        })?;

        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        let view = self.create_image_view(image, attributes.format, vk::ImageAspectFlags::COLOR)?;

        Ok(Image {
            handle: image,
            allocation: Some(allocation),
            view,
            layout: ImageLayoutState {
                access_mask: vk::AccessFlags2::empty(),
                layout: vk::ImageLayout::UNDEFINED,
                stage_mask: vk::PipelineStageFlags2::empty(),
                queue_family_index: 0,
            },
            attributes,
        })
    }

    pub fn destroy_image(&self, allocator: &mut Allocator, image: &mut Image) -> Result<()> {
        unsafe {
            self.device.destroy_image_view(image.view, None);
            if let Some(allocation) = image.allocation.take() {
                allocator.free(allocation)?;
            }
            self.device.destroy_image(image.handle, None);
        }
        Ok(())
    }

    pub fn clear_image(
        &self,
        command_buffer: vk::CommandBuffer,
        image: Image,
        clear_color: [f32; 4],
    ) {
        unsafe {
            self.device.cmd_clear_color_image(
                command_buffer,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: clear_color,
                },
                &[vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)],
            )
        }
    }
}

pub struct ImageAttributes {
    pub location: MemoryLocation,
    pub allocation_scheme: AllocationScheme,
    pub linear: bool,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

pub struct Image {
    pub handle: vk::Image,
    pub allocation: Option<Allocation>,
    pub view: vk::ImageView,
    pub layout: ImageLayoutState,
    pub attributes: ImageAttributes,
}

#[derive(Clone, Copy)]
pub struct ImageLayoutState {
    pub access_mask: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
    pub stage_mask: vk::PipelineStageFlags2,
    pub queue_family_index: u32,
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
