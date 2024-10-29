use ash::vk;
use ash::vk::QUEUE_FAMILY_IGNORED;
use gpu_allocator::vulkan::{Allocation, AllocationScheme};
use gpu_allocator::MemoryLocation;

pub struct ImageAttributes {
    pub location: MemoryLocation,
    pub allocation_scheme: AllocationScheme,
    pub linear: bool,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub subresource_range: vk::ImageSubresourceRange,
}

pub struct Image {
    pub handle: vk::Image,
    pub allocation: Option<Allocation>,
    pub view: vk::ImageView,
    pub layout: ImageLayoutState,
    pub attributes: ImageAttributes,
}

impl Image {
    pub fn reset_layout(&mut self) {
        self.layout = ImageLayoutState::ignored();
    }
}

#[derive(Clone, Copy)]
pub struct ImageLayoutState {
    pub access: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
    pub stage: vk::PipelineStageFlags2,
    pub queue_family: u32,
}

impl ImageLayoutState {
    pub fn ignored() -> Self {
        Self {
            access: vk::AccessFlags2::empty(),
            layout: vk::ImageLayout::UNDEFINED,
            stage: vk::PipelineStageFlags2::NONE,
            queue_family: QUEUE_FAMILY_IGNORED,
        }
    }

    pub fn color_attachment() -> Self {
        Self {
            access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            queue_family: QUEUE_FAMILY_IGNORED,
        }
    }

    pub fn present() -> Self {
        Self {
            access: vk::AccessFlags2::TRANSFER_READ,
            layout: vk::ImageLayout::PRESENT_SRC_KHR,
            stage: vk::PipelineStageFlags2::TRANSFER,
            queue_family: QUEUE_FAMILY_IGNORED,
        }
    }

    pub fn transfer_destination() -> Self {
        Self {
            access: vk::AccessFlags2::TRANSFER_WRITE,
            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            stage: vk::PipelineStageFlags2::TRANSFER,
            queue_family: QUEUE_FAMILY_IGNORED,
        }
    }

    pub fn transfer_source() -> Self {
        Self {
            access: vk::AccessFlags2::TRANSFER_READ,
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            stage: vk::PipelineStageFlags2::TRANSFER,
            queue_family: QUEUE_FAMILY_IGNORED,
        }
    }
}

impl Default for ImageLayoutState {
    fn default() -> Self {
        Self {
            access: vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            layout: vk::ImageLayout::GENERAL,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            queue_family: QUEUE_FAMILY_IGNORED,
        }
    }
}
