use crate::rendering_context::RenderingContext;
use anyhow::{Context as AnyhowContext, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

pub struct BufferAttributes {
    pub name: String,
    pub context: Arc<RenderingContext>,
    pub size: vk::DeviceSize,
    pub usage: vk::BufferUsageFlags,
    pub location: MemoryLocation,
    pub allocation_scheme: AllocationScheme,
}

pub struct Buffer {
    pub handle: vk::Buffer,
    allocation: Allocation,
    pub attributes: BufferAttributes,
    requirements: vk::MemoryRequirements,
    pub address: vk::DeviceAddress,
}

impl Buffer {
    pub fn new(allocator: &mut Allocator, attributes: BufferAttributes) -> Result<Self> {
        unsafe {
            let is_debug = cfg!(debug_assertions);
            let is_capture_replay_supported = attributes
                .context
                .physical_device
                .vulkan12_features
                .buffer_device_address_capture_replay
                == vk::TRUE;

            let flags = if is_debug && is_capture_replay_supported {
                vk::BufferCreateFlags::DEVICE_ADDRESS_CAPTURE_REPLAY
            } else {
                vk::BufferCreateFlags::empty()
            };

            let handle = attributes.context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(attributes.size)
                    .usage(attributes.usage)
                    .flags(flags),
                None,
            )?;

            let requirements = attributes
                .context
                .device
                .get_buffer_memory_requirements(handle);

            let allocation = allocator.allocate(&AllocationCreateDesc {
                name: &attributes.name,
                requirements,
                location: attributes.location,
                linear: true,
                allocation_scheme: attributes.allocation_scheme,
            })?;

            attributes.context.device.bind_buffer_memory(
                handle,
                allocation.memory(),
                allocation.offset(),
            )?;

            let address = if attributes
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            {
                attributes.context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(handle),
                )
            } else {
                Default::default()
            };

            Ok(Self {
                handle,
                allocation,
                attributes,
                requirements,
                address,
            })
        }
    }

    pub fn write<T: bytemuck::Pod>(&mut self, data: &[T], offset: vk::DeviceSize) -> Result<()> {
        self.allocation
            .mapped_slice_mut()
            .context("Failed to map buffer memory")?[offset as usize..]
            [..data.len() * size_of::<T>()]
            .copy_from_slice(bytemuck::cast_slice(data));
        Ok(())
    }

    pub fn destroy(&mut self, allocator: &mut Allocator) -> Result<()> {
        unsafe {
            self.attributes
                .context
                .device
                .destroy_buffer(self.handle, None);
            allocator.free(std::mem::take(&mut self.allocation))?;
            Ok(())
        }
    }
}
