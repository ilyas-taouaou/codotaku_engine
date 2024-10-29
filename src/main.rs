use crate::app::App;
use anyhow::Result;
use engine::anyhow;
use engine::winit;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;

fn main() -> Result<()> {
    tracing_subscriber::fmt::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    let mut app = App::default();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app)?;

    Ok(())
}
