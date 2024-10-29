use crate::rendering_context::{Image, ImageLayoutState, RenderingContext};
use anyhow::Result;
use ash::vk;
use std::ops::Range;
use std::sync::Arc;

pub struct Commands {
    context: Arc<RenderingContext>,
    command_buffer: vk::CommandBuffer,
}

impl Commands {
    pub fn new(context: Arc<RenderingContext>, command_buffer: vk::CommandBuffer) -> Result<Self> {
        unsafe {
            context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            context.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        Ok(Self {
            context,
            command_buffer,
        })
    }

    pub fn transition_image_layout(&self, image: &mut Image, new_state: ImageLayoutState) -> &Self {
        unsafe {
            let old_state = image.layout;

            self.context.device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2KHR::default()
                        .src_stage_mask(old_state.stage)
                        .dst_stage_mask(new_state.stage)
                        .src_access_mask(old_state.access)
                        .dst_access_mask(new_state.access)
                        .old_layout(old_state.layout)
                        .new_layout(new_state.layout)
                        .src_queue_family_index(old_state.queue_family)
                        .dst_queue_family_index(new_state.queue_family)
                        .image(image.handle)
                        .subresource_range(image.attributes.subresource_range),
                ]),
            );

            image.layout = new_state;
        }
        self
    }

    pub fn blit_image(
        &self,
        src_image: &Image,
        dst_image: &Image,
        src_offsets: [vk::Offset3D; 2],
        dst_offsets: [vk::Offset3D; 2],
    ) -> &Self {
        unsafe {
            self.context.device.cmd_blit_image(
                self.command_buffer,
                src_image.handle,
                src_image.layout.layout,
                dst_image.handle,
                dst_image.layout.layout,
                &[vk::ImageBlit::default()
                    .src_subresource(src_image.subresource_layers())
                    .src_offsets(src_offsets)
                    .dst_subresource(dst_image.subresource_layers())
                    .dst_offsets(dst_offsets)],
                vk::Filter::NEAREST,
            );
        }

        self
    }

    pub fn blit_image_extent(
        &self,
        src_image: &Image,
        dst_image: &Image,
        src_extent: vk::Extent3D,
        dst_extent: vk::Extent3D,
    ) -> &Self {
        self.blit_image(
            src_image,
            dst_image,
            [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: src_extent.width as i32,
                    y: src_extent.height as i32,
                    z: src_extent.depth as i32,
                },
            ],
            [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: dst_extent.width as i32,
                    y: dst_extent.height as i32,
                    z: dst_extent.depth as i32,
                },
            ],
        )
    }

    pub fn blit_full_image(&self, src_image: &Image, dst_image: &Image) -> &Self {
        self.blit_image_extent(
            src_image,
            dst_image,
            src_image.attributes.extent,
            dst_image.attributes.extent,
        )
    }

    pub fn begin_rendering(
        &self,
        image: &Image,
        clear_color: vk::ClearColorValue,
        render_area: vk::Rect2D,
    ) -> &Self {
        unsafe {
            self.context.device.cmd_begin_rendering(
                self.command_buffer,
                &vk::RenderingInfo::default()
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(image.view)
                        .image_layout(image.layout.layout)
                        .clear_value(vk::ClearValue { color: clear_color })
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)])
                    .render_area(render_area),
            );
        }

        self
    }

    pub fn end_rendering(&self) -> &Self {
        unsafe {
            self.context.device.cmd_end_rendering(self.command_buffer);
        }

        self
    }

    pub fn set_viewport(&self, viewport: vk::Viewport) -> &Self {
        unsafe {
            self.context
                .device
                .cmd_set_viewport(self.command_buffer, 0, &[viewport]);
        }

        self
    }

    pub fn set_scissor(&self, scissor: vk::Rect2D) -> &Self {
        unsafe {
            self.context
                .device
                .cmd_set_scissor(self.command_buffer, 0, &[scissor]);
        }

        self
    }

    pub fn bind_pipeline(&self, pipeline: vk::Pipeline) -> &Self {
        unsafe {
            self.context.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }

        self
    }

    pub fn draw(&self, vertices: Range<u32>, instances: Range<u32>) -> &Self {
        unsafe {
            self.context.device.cmd_draw(
                self.command_buffer,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            );
        }

        self
    }

    pub fn submit(
        &self,
        queue: vk::Queue,
        wait_semaphore: (vk::Semaphore, vk::PipelineStageFlags2KHR),
        signal_semaphore: (vk::Semaphore, vk::PipelineStageFlags2KHR),
        fence: vk::Fence,
    ) -> Result<()> {
        unsafe {
            self.context
                .device
                .end_command_buffer(self.command_buffer)?;

            self.context.device.queue_submit2(
                queue,
                &[vk::SubmitInfo2KHR::default()
                    .command_buffer_infos(&[vk::CommandBufferSubmitInfoKHR::default()
                        .command_buffer(self.command_buffer)
                        .device_mask(1)])
                    .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(wait_semaphore.0)
                        .stage_mask(wait_semaphore.1)])
                    .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                        .semaphore(signal_semaphore.0)
                        .stage_mask(signal_semaphore.1)])],
                fence,
            )?;
            Ok(())
        }
    }
}
