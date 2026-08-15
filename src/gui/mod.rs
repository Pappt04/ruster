//! Interactive application: the egui-based UI (`app`), the camera model
//! pixel/complex-plane mapping is derived from (`viewport`), the
//! histogram-equalized coloring pipeline (`color`), and the background
//! render worker that decouples input handling from render latency
//! (`render`).

pub mod app;
pub mod viewport;
pub mod color;
pub mod render;