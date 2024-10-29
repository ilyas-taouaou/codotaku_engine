#![allow(dead_code)]
mod buffer;
mod image;
mod renderer;
mod rendering_context;

use crate::rendering_context::{queue_family_picker, RenderingContext, RenderingContextAttributes};
use anyhow::Result;
use renderer::window_renderer::WindowRenderer;
use std::collections::HashMap;
use std::sync::Arc;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

pub use crate::renderer::window_renderer::WindowRendererAttributes;
pub use anyhow;
pub use ash::vk;
pub use winit;

pub struct Engine {
    windows: HashMap<WindowId, Arc<Window>>,
    renderers: HashMap<WindowId, WindowRenderer>,
    primary_window_id: WindowId,
    rendering_context: Arc<RenderingContext>,
}

impl Engine {
    pub fn new(
        event_loop: &ActiveEventLoop,
        primary_window_attributes: WindowAttributes,
        primary_renderer_attributes: WindowRendererAttributes,
    ) -> Result<Self> {
        let primary_window = Arc::new(event_loop.create_window(primary_window_attributes)?);
        let primary_window_id = primary_window.id();

        let rendering_context = Arc::new(RenderingContext::new(RenderingContextAttributes {
            compatibility_window: primary_window.as_ref(),
            queue_family_picker: queue_family_picker::single_queue_family,
        })?);

        let windows = HashMap::from([(primary_window_id, primary_window)]);

        let renderers = windows
            .iter()
            .map(|(id, window)| {
                let renderer = WindowRenderer::new(
                    rendering_context.clone(),
                    window.clone(),
                    primary_renderer_attributes.clone(),
                )
                .unwrap();
                (*id, renderer)
            })
            .collect::<HashMap<_, _>>();

        Ok(Self {
            renderers,
            windows,
            primary_window_id,
            rendering_context,
        })
    }

    pub fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if window_id == self.primary_window_id {
                    event_loop.exit();
                } else {
                    self.windows.remove(&window_id);
                    self.renderers.remove(&window_id);
                }
            }
            WindowEvent::Resized(_) => {
                if let Some(renderer) = self.renderers.get_mut(&window_id) {
                    renderer.resize();
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(renderer) = self.renderers.get_mut(&window_id) {
                    renderer.resize();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = self.renderers.get_mut(&window_id) {
                    renderer.render().unwrap();
                }
            }
            _ => {}
        }
    }

    pub fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        attributes: WindowAttributes,
        renderer_attributes: WindowRendererAttributes,
    ) -> Result<WindowId> {
        let window = Arc::new(event_loop.create_window(attributes)?);
        let window_id = window.id();
        self.windows.insert(window_id, window.clone());

        let renderer = WindowRenderer::new(
            self.rendering_context.clone(),
            window.clone(),
            renderer_attributes,
        )?;
        self.renderers.insert(window_id, renderer);

        Ok(window_id)
    }

    pub fn request_redraw(&self) {
        for window in self.windows.values() {
            window.request_redraw();
        }
    }
}
