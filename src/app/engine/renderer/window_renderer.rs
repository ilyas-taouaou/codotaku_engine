use crate::app::engine::renderer::swapchain::Swapchain;
use crate::app::engine::renderer::Renderer;
use crate::app::engine::rendering_context::{ImageLayoutState, RenderingContext};
use ash::vk;
use ash::vk::{CommandBuffer, Extent3D};
use std::sync::Arc;
use winit::window::Window;

use anyhow::Result;

struct Frame {
    command_buffer: CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

pub struct WindowRenderer {
    in_flight_frames_count: usize,
    frame_index: usize,
    frames: Vec<Frame>,
    command_pool: vk::CommandPool,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,

    clear_color: vk::ClearColorValue,

    pub renderer: Renderer,
    pub window: Arc<Window>,
}

impl WindowRenderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        in_flight_frames_count: usize,
        format: vk::Format,
        clear_color: vk::ClearColorValue,
    ) -> Result<Self> {
        let mut swapchain = Swapchain::new(context.clone(), window.clone())?;
        swapchain.resize()?;

        unsafe {
            let command_pool = context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_families.graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

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

            let renderer = Renderer::new(
                context.clone(),
                (swapchain.extent.width, swapchain.extent.height),
                format,
                in_flight_frames_count,
            )?;

            Ok(Self {
                in_flight_frames_count,
                frame_index: 0,
                frames,
                command_pool,
                swapchain,
                context,
                clear_color,
                renderer,
                window,
            })
        }
    }

    pub fn resize(&mut self) {
        self.swapchain.is_dirty = true;
    }

    pub fn render(&mut self) -> Result<()> {
        let frame = &self.frames[self.frame_index];

        unsafe {
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;

            if self.swapchain.is_dirty {
                self.context.device.device_wait_idle()?;
                self.swapchain.resize()?;
                if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
                    return Ok(());
                }
                self.renderer
                    .resize((self.swapchain.extent.width, self.swapchain.extent.height))?;
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

            self.renderer
                .render(frame.command_buffer, self.clear_color, self.frame_index)?;

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

            let render_target = &self.renderer.render_targets[self.frame_index];

            // Transition the render target image to transfer source layout
            self.context.transition_image_layout(
                frame.command_buffer,
                render_target.handle,
                render_target.layout,
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
                render_target.handle,
                self.swapchain.images[image_index as usize],
                render_target.attributes.extent,
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
}

impl Drop for WindowRenderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

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
        }
    }
}
