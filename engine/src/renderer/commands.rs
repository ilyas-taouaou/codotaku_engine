use crate::buffer::Buffer;
use crate::rendering_context::{Image, ImageLayoutState, RenderingContext};
use anyhow::Result;
use ash::vk;
use ash::vk::DeviceSize;
use std::ops::Range;
use std::sync::Arc;
use tracing::trace;

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

    pub fn bind_index_buffer(&self, buffer: &Buffer) -> &Self {
        unsafe {
            self.context.device.cmd_bind_index_buffer(
                self.command_buffer,
                buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
        }

        self
    }

    pub fn copy_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        offset: DeviceSize,
    ) -> &Self {
        unsafe {
            self.context.device.cmd_copy_buffer(
                self.command_buffer,
                src_buffer.handle,
                dst_buffer.handle,
                &[vk::BufferCopy::default()
                    .size(dst_buffer.attributes.size)
                    .src_offset(offset)],
            );
        }

        self
    }

    pub fn set_push_constants<T: bytemuck::Pod>(
        &self,
        pipeline_layout: vk::PipelineLayout,
        data: T,
    ) -> &Self {
        unsafe {
            self.context.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&data),
            );
        }

        self
    }

    pub fn transition_image_layout(&self, image: &mut Image, new_state: ImageLayoutState) -> &Self {
        unsafe {
            let old_state = image.layout;

            trace!("Transitioned image layout from {old_state:#?} to {new_state:#?}");

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

    pub fn ensure_image_layout(&self, image: &mut Image, new_state: ImageLayoutState) -> &Self {
        let state = image.layout;
        if !new_state.is_subset_of(state) {
            self.transition_image_layout(image, new_state);
        }
        self
    }

    pub fn blit_image(
        &self,
        src_image: &mut Image,
        dst_image: &mut Image,
        src_offsets: [vk::Offset3D; 2],
        dst_offsets: [vk::Offset3D; 2],
        filter: vk::Filter,
    ) -> &Self {
        self.ensure_image_layout(src_image, ImageLayoutState::transfer_source())
            .ensure_image_layout(dst_image, ImageLayoutState::transfer_destination());

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
                filter,
            );
        }

        self
    }

    pub fn blit_image_extent(
        &self,
        src_image: &mut Image,
        dst_image: &mut Image,
        src_extent: vk::Extent3D,
        dst_extent: vk::Extent3D,
        filter: vk::Filter,
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
            filter,
        )
    }

    pub fn blit_full_image(
        &self,
        src_image: &mut Image,
        dst_image: &mut Image,
        filter: vk::Filter,
    ) -> &Self {
        self.blit_image_extent(
            src_image,
            dst_image,
            src_image.attributes.extent,
            dst_image.attributes.extent,
            filter,
        )
    }

    pub fn begin_rendering(
        &self,
        render_target: &mut Image,
        depth_buffer: &mut Image,
        clear_color: vk::ClearColorValue,
        render_area: vk::Rect2D,
    ) -> &Self {
        self.ensure_image_layout(render_target, ImageLayoutState::color_attachment())
            .ensure_image_layout(depth_buffer, ImageLayoutState::depth_stencil_attachment());

        unsafe {
            self.context.device.cmd_begin_rendering(
                self.command_buffer,
                &vk::RenderingInfo::default()
                    .layer_count(1)
                    .color_attachments(&[vk::RenderingAttachmentInfo::default()
                        .image_view(render_target.view)
                        .image_layout(render_target.layout.layout)
                        .clear_value(vk::ClearValue { color: clear_color })
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)])
                    .render_area(render_area)
                    .depth_attachment(
                        &vk::RenderingAttachmentInfo::default()
                            .image_view(depth_buffer.view)
                            .image_layout(depth_buffer.layout.layout)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 1.0,
                                    stencil: 0,
                                },
                            })
                            .load_op(vk::AttachmentLoadOp::CLEAR)
                            .store_op(vk::AttachmentStoreOp::STORE),
                    ),
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

    pub fn draw_indexed(&self, indices: Range<u32>, instances: Range<u32>) -> &Self {
        unsafe {
            self.context.device.cmd_draw_indexed(
                self.command_buffer,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                0,
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

            let command_buffer_submit_infos =
                &[vk::CommandBufferSubmitInfoKHR::default().command_buffer(self.command_buffer)];

            let mut submit_info =
                vk::SubmitInfo2KHR::default().command_buffer_infos(command_buffer_submit_infos);

            let wait_semaphore_submit_infos = &[vk::SemaphoreSubmitInfo::default()
                .semaphore(wait_semaphore.0)
                .stage_mask(wait_semaphore.1)];

            let signal_semaphore_submit_infos = &[vk::SemaphoreSubmitInfo::default()
                .semaphore(signal_semaphore.0)
                .stage_mask(signal_semaphore.1)];

            if wait_semaphore.0 != vk::Semaphore::null() {
                submit_info = submit_info.wait_semaphore_infos(wait_semaphore_submit_infos);
            }

            if signal_semaphore.0 != vk::Semaphore::null() {
                submit_info = submit_info.signal_semaphore_infos(signal_semaphore_submit_infos)
            }

            self.context
                .device
                .queue_submit2(queue, &[submit_info], fence)?;
            Ok(())
        }
    }
}
