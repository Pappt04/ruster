pub struct FractalApp;

impl Default for FractalApp {
    fn default() -> Self {
        Self {}
    }
}

impl eframe::App for FractalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("Hello World!");
        });
    }
}
