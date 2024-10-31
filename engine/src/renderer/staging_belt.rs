use crate::buffer::{Buffer, BufferAttributes};
use crate::image::Image;
use crate::renderer::commands::Commands;
use crate::renderer::geometry::GPUGeometry;
use crate::rendering_context::RenderingContext;
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

pub struct StagingBelt {
    buffer: Buffer,
    write_cursor: vk::DeviceSize,
    copy_cursor: vk::DeviceSize,
}

impl StagingBelt {
    pub fn new(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        let buffer = Buffer::new(
            allocator,
            BufferAttributes {
                name: "staging_buffer".into(),
                context,
                size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::CpuToGpu,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                allocation_priority: 1.0,
            },
        )?;
        Ok(Self {
            buffer,
            write_cursor: 0,
            copy_cursor: 0,
        })
    }

    pub fn write<T: bytemuck::Pod>(&mut self, data: &[T]) -> Result<&mut Self> {
        let size = (data.len() * size_of::<T>()) as vk::DeviceSize;
        self.buffer.write(data, self.write_cursor)?;
        self.write_cursor += size;
        Ok(self)
    }

    pub fn copy_to(&mut self, buffer: &Buffer, commands: &Commands) -> &mut Self {
        commands.copy_buffer(&self.buffer, buffer, self.copy_cursor);
        self.copy_cursor += buffer.attributes.size;
        self
    }

    pub fn copy_image_to(&mut self, image: &mut Image, commands: &Commands) -> &mut Self {
        commands.copy_buffer_to_image(&self.buffer, image, self.copy_cursor);
        self.copy_cursor +=
            (image.attributes.extent.width * image.attributes.extent.height * 4) as vk::DeviceSize;
        self
    }

    pub fn stage_geometry(
        &mut self,
        gpu_geometry: &GPUGeometry,
        commands: &Commands,
    ) -> Result<&mut Self> {
        Ok(self
            .write(&gpu_geometry.geometry.vertices)?
            .copy_to(&gpu_geometry.vertex_buffer, commands)
            .write(&gpu_geometry.geometry.indices)?
            .copy_to(&gpu_geometry.index_buffer, commands))
    }

    pub fn done(&mut self) {
        self.write_cursor = 0;
        self.copy_cursor = 0;
    }

    pub fn destroy(&mut self, allocator: &mut Allocator) -> Result<()> {
        self.buffer.destroy(allocator)
    }
}
