mod fractal;
mod gui;

use gui::app::FractalApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "NovaFractal",
        options,
        Box::new(|_cc| Ok(Box::new(FractalApp::default()))),
    )
}
