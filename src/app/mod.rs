use engine::winit::window::WindowAttributes;
use ::engine::Engine;
use engine::{vk, winit, WindowRendererAttributes};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;

#[derive(Default)]
pub struct App {
    engine: Option<Engine>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let primary_window_attributes = WindowAttributes::default().with_title("Primary window");
        let primary_window_renderer_attributes = WindowRendererAttributes {
            format: vk::Format::R16G16B16A16_SFLOAT,
            depth_format: vk::Format::D32_SFLOAT,
            clear_color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            ssaa: 1.0,
            ssaa_filter: vk::Filter::NEAREST,
            in_flight_frames_count: 2,
        };

        let secondary_window_attributes =
            WindowAttributes::default().with_title("Secondary window");
        let secondary_window_renderer_attributes = WindowRendererAttributes {
            format: vk::Format::R16G16B16A16_SFLOAT,
            depth_format: vk::Format::D32_SFLOAT,
            clear_color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            ssaa: 0.1,
            ssaa_filter: vk::Filter::NEAREST,
            in_flight_frames_count: 2,
        };

        let secondary_window_count = 1;

        self.engine = Some(
            Engine::new(
                event_loop,
                primary_window_attributes,
                primary_window_renderer_attributes,
            )
            .unwrap(),
        );
        if let Some(engine) = self.engine.as_mut() {
            for _ in 0..secondary_window_count {
                _ = engine
                    .create_window(
                        event_loop,
                        secondary_window_attributes.clone(),
                        secondary_window_renderer_attributes.clone(),
                    )
                    .unwrap();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(engine) = self.engine.as_mut() {
            engine.window_event(event_loop, window_id, event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(engine) = self.engine.as_mut() {
            engine.request_redraw();
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.engine = None;
    }
}
