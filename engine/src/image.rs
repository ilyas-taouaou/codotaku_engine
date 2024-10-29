use crate::rendering_context::RenderingContext;
use anyhow::Result;
use ash::vk;
use ash::vk::QUEUE_FAMILY_IGNORED;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

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
    context: Arc<RenderingContext>,
}

fn create_image_view(
    context: &RenderingContext,
    image: vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let image_view = unsafe {
        context.device.create_image_view(
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

impl Image {
    pub fn new(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        name: &str,
        attributes: ImageAttributes,
    ) -> Result<Self> {
        let image = unsafe {
            context.device.create_image(
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

        let requirements = unsafe { context.device.get_image_memory_requirements(image) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: attributes.location,
            linear: attributes.linear,
            allocation_scheme: attributes.allocation_scheme,
        })?;

        unsafe {
            context
                .device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        let view = create_image_view(
            context.as_ref(),
            image,
            attributes.format,
            attributes.subresource_range.aspect_mask,
        )?;

        Ok(Image {
            handle: image,
            allocation: Some(allocation),
            view,
            layout: ImageLayoutState::ignored(),
            attributes,
            context,
        })
    }

    pub fn new_render_target(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        name: &str,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Image> {
        Image::new(
            context,
            allocator,
            name,
            ImageAttributes {
                extent: extent.into(),
                format,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                subresource_range: vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            },
        )
    }

    pub fn new_depth_buffer(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        name: &str,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Image> {
        Image::new(
            context,
            allocator,
            name,
            ImageAttributes {
                extent: extent.into(),
                format,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                subresource_range: vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            },
        )
    }

    pub fn wrap(
        context: Arc<RenderingContext>,
        handle: vk::Image,
        attributes: ImageAttributes,
    ) -> Result<Self> {
        let view = create_image_view(
            context.as_ref(),
            handle,
            attributes.format,
            attributes.subresource_range.aspect_mask,
        )?;

        Ok(Self {
            handle,
            allocation: None,
            view,
            layout: ImageLayoutState::ignored(),
            attributes,
            context,
        })
    }

    pub fn reset_layout(&mut self) {
        self.layout = ImageLayoutState::ignored();
    }

    pub fn destroy(&mut self, allocator: &mut Allocator) -> Result<()> {
        unsafe {
            self.context.device.destroy_image_view(self.view, None);
            if let Some(allocation) = self.allocation.take() {
                self.context.device.destroy_image(self.handle, None);
                allocator.free(allocation)?;
            }
        }
        Ok(())
    }

    pub fn subresource_layers(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(self.attributes.subresource_range.aspect_mask)
            .mip_level(self.attributes.subresource_range.base_mip_level)
            .base_array_layer(self.attributes.subresource_range.base_array_layer)
            .layer_count(self.attributes.subresource_range.layer_count)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

    pub fn depth_stencil_attachment() -> Self {
        Self {
            access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
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

    pub fn is_subset_of(&self, other: Self) -> bool {
        self.layout == other.layout
            && self.access.contains(other.access)
            && self.stage.contains(other.stage)
            && (self.queue_family == QUEUE_FAMILY_IGNORED
                || self.queue_family == other.queue_family)
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
