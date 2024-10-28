mod swapchain;

use crate::app::engine::renderer::swapchain::Swapchain;
use crate::app::engine::rendering_context::{
    Image, ImageAttributes, ImageLayoutState, RenderingContext,
};
use anyhow::Result;
use ash::vk;
use ash::vk::{CommandBuffer, Extent3D};
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;
use winit::window::Window;

struct Frame {
    command_buffer: CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

pub struct Renderer {
    allocator: Allocator,
    in_flight_frames_count: usize,
    frame_index: usize,
    frames: Vec<Frame>,
    command_pool: vk::CommandPool,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,
    render_target: Image,
}

const SHADERS_DIR: &str = "res/shaders/";

fn load_shader_module(context: &RenderingContext, path: &str) -> Result<vk::ShaderModule> {
    let code = std::fs::read(format!("{}{}", SHADERS_DIR, path))?;
    context.create_shader_module(&code)
}

impl Renderer {
    pub fn new(context: Arc<RenderingContext>, window: Arc<Window>) -> Result<Self> {
        let mut swapchain = Swapchain::new(context.clone(), window.clone())?;
        swapchain.resize()?;

        let vertex_shader = load_shader_module(context.as_ref(), "vert.spv")?;
        let fragment_shader = load_shader_module(context.as_ref(), "frag.spv")?;

        let mut allocator = context.create_allocator(Default::default(), Default::default())?;

        let render_target = context.create_image(
            &mut allocator,
            "render target",
            ImageAttributes {
                extent: Extent3D::default()
                    .width(swapchain.extent.width)
                    .height(swapchain.extent.height)
                    .depth(1),
                format: vk::Format::R16G16B16A16_SFLOAT,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            },
        )?;

        unsafe {
            let pipeline_layout = context
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;

            let pipeline = context.create_graphics_pipeline(
                vertex_shader,
                fragment_shader,
                vk::Extent2D {
                    width: render_target.attributes.extent.width,
                    height: render_target.attributes.extent.height,
                },
                render_target.attributes.format,
                pipeline_layout,
                Default::default(),
            )?;

            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);

            let command_pool = context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_families.graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

            let in_flight_frames_count = 2;

            let command_buffers = context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(in_flight_frames_count as u32),
            )?;

            let mut frames = Vec::with_capacity(command_buffers.len());

            for &command_buffer in command_buffers.iter() {
                let image_available_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let in_flight_fence = context.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;

                frames.push(Frame {
                    command_buffer,
                    image_available_semaphore,
                    render_finished_semaphore,
                    in_flight_fence,
                });
            }

            Ok(Self {
                allocator,
                in_flight_frames_count,
                frame_index: 0,
                frames,
                command_pool,
                pipeline,
                pipeline_layout,
                swapchain,
                context,
                render_target,
            })
        }
    }

    pub fn resize(&mut self) -> Result<()> {
        self.swapchain.is_dirty = true;
        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        let frame = &self.frames[self.frame_index];

        unsafe {
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;

            if self.swapchain.is_dirty {
                self.swapchain.resize()?;
            }

            if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
                return Ok(());
            }

            let image_index = self
                .swapchain
                .acquire_next_image(frame.image_available_semaphore)?;

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

            self.context
                .device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;

            let undefined_image_state = ImageLayoutState {
                layout: vk::ImageLayout::UNDEFINED,
                access_mask: vk::AccessFlags2::NONE,
                stage_mask: vk::PipelineStageFlags2::NONE,
                queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            };

            let renderable_image_state = ImageLayoutState {
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            };

            let present_image_state = ImageLayoutState {
                layout: vk::ImageLayout::PRESENT_SRC_KHR,
                access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            };

            self.context.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.context.transition_image_layout(
                frame.command_buffer,
                self.render_target.handle,
                undefined_image_state,
                renderable_image_state,
            );

            self.context.begin_rendering(
                frame.command_buffer,
                self.render_target.view,
                vk::ClearColorValue {
                    float32: [0.01, 0.01, 0.01, 1.0],
                },
                vk::Rect2D::default().extent(self.swapchain.extent),
            );
            self.draw(frame.command_buffer);
            self.context.device.cmd_end_rendering(frame.command_buffer);

            // Transition the swapchain image to transfer destination layout
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.images[image_index as usize],
                undefined_image_state,
                ImageLayoutState {
                    layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    stage_mask: vk::PipelineStageFlags2::TRANSFER,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            );

            // Transition the render target image to transfer source layout
            self.context.transition_image_layout(
                frame.command_buffer,
                self.render_target.handle,
                renderable_image_state,
                ImageLayoutState {
                    layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    access_mask: vk::AccessFlags2::TRANSFER_READ,
                    stage_mask: vk::PipelineStageFlags2::TRANSFER,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
            );

            // Copy the render target image to the swapchain image
            self.context.blit_image(
                frame.command_buffer,
                self.render_target.handle,
                self.swapchain.images[image_index as usize],
                self.render_target.attributes.extent,
                Extent3D::default()
                    .width(self.swapchain.extent.width)
                    .height(self.swapchain.extent.height),
            );

            // Transition the swapchain image to present layout
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.images[image_index as usize],
                ImageLayoutState {
                    layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    stage_mask: vk::PipelineStageFlags2::TRANSFER,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
                present_image_state,
            );

            self.context
                .device
                .end_command_buffer(frame.command_buffer)?;

            self.context.device.queue_submit2(
                self.context.queues[self.context.queue_families.graphics as usize],
                &[vk::SubmitInfo2KHR::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfoKHR::default()
                        .command_buffer(frame.command_buffer)
                        .device_mask(1)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(frame.image_available_semaphore)
                        .stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(frame.render_finished_semaphore)
                        .stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)])],
                frame.in_flight_fence,
            )?;

            self.swapchain
                .present(image_index, frame.render_finished_semaphore)?;

            self.frame_index = (self.frame_index + 1) % self.in_flight_frames_count;
            Ok(())
        }
    }

    pub fn draw(&self, command_buffer: CommandBuffer) {
        unsafe {
            self.context.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(self.swapchain.extent.width as f32)
                    .height(self.swapchain.extent.height as f32)],
            );

            self.context.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::default().extent(self.swapchain.extent)],
            );

            self.context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.context.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.context
                .destroy_image(&mut self.allocator, &mut self.render_target)
                .unwrap();

            self.frames.drain(..).for_each(|frame| {
                self.context
                    .device
                    .destroy_semaphore(frame.image_available_semaphore, None);
                self.context
                    .device
                    .destroy_semaphore(frame.render_finished_semaphore, None);
                self.context
                    .device
                    .destroy_fence(frame.in_flight_fence, None);
                self.context
                    .device
                    .free_command_buffers(self.command_pool, &[frame.command_buffer]);
            });
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
