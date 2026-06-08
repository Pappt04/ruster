use egui::{Color32, Rect, TextureHandle, TextureOptions, Vec2};
use crate::{
    fractal::fractal_type::FractalType,
    gui::{
        color::ColorScheme,
        render::{RenderRequest, RenderWorker},
        viewport::Viewport,
    },
};

pub struct FractalApp {
    // — state
    fractal: FractalType,
    viewport: Viewport,
    julia_c: [f64; 2],
    max_iter: u32,
    scheme: ColorScheme,
    show_panel: bool,

    // — rendering
    worker: RenderWorker,
    texture: Option<TextureHandle>,
    needs_render: bool,

    // — adaptive iteration depth
    auto_iter: bool,

    use_ms: bool,
}

impl Default for FractalApp {
    fn default() -> Self {
        Self {
            fractal: FractalType::Mandelbrot,
            viewport: Viewport::default(),
            julia_c: [-0.7269, 0.1889],
            max_iter: 256,
            scheme: ColorScheme::Inferno,
            show_panel: true,
            worker: RenderWorker::new(),
            texture: None,
            needs_render: true,
            auto_iter: true,
            use_ms: false,
        }
    }
}

impl FractalApp {
    fn request_render(&mut self) {
        if self.auto_iter {
            // More iterations as you zoom in (logarithmic scale)
            self.max_iter = (100.0 * (1.0 + self.viewport.zoom.ln().max(0.0))) as u32;
            self.max_iter = self.max_iter.clamp(64, 2048);
        }
        self.worker.request(RenderRequest {
            vp: self.viewport.clone(),
            fractal: self.fractal,
            julia_c: self.julia_c,
            max_iter: self.max_iter,
            scheme: self.scheme,
            use_ms: self.use_ms,
        });
        self.needs_render = false;
    }

    fn handle_mouse(&mut self, response: &egui::Response, image_rect: Rect) {
        // Scroll to zoom
        if response.hovered() {
            let scroll = response.ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                if let Some(pos) = response.ctx.input(|i| i.pointer.hover_pos()) {
                    let px = (pos.x - image_rect.min.x) as f64;
                    let py = (pos.y - image_rect.min.y) as f64;
                    let factor = if scroll > 0.0 { 1.15 } else { 1.0 / 1.15 };
                    self.viewport.zoom_at(px, py, factor);
                    self.needs_render = true;
                }
            }
        }

        // Drag to pan
        if response.dragged() {
            let d = response.drag_delta();
            self.viewport.pan(d.x as f64, d.y as f64);
            self.needs_render = true;
        }

        // Double-click to zoom in 2×
        if response.double_clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let px = (pos.x - image_rect.min.x) as f64;
                let py = (pos.y - image_rect.min.y) as f64;
                self.viewport.zoom_at(px, py, 2.0);
                self.needs_render = true;
            }
        }

        // Right-click context menu
        response.context_menu(|ui| {
            if ui.button("Reset view").clicked() {
                self.viewport.reset(self.fractal);
                self.needs_render = true;
                ui.close();
            }
            ui.separator();
            ui.label(format!("Re: {:.8}", self.viewport.center[0]));
            ui.label(format!("Im: {:.8}", self.viewport.center[1]));
            ui.label(format!("Zoom: {:.2e}×", self.viewport.zoom));
        });
    }

    fn settings_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("settings")
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.heading("NovFractal");
                ui.separator();

                // Fractal picker
                ui.label("Fractal type");
                for ft in FractalType::ALL {
                    if ui.selectable_label(self.fractal == *ft, ft.name()).clicked() {
                        self.fractal = *ft;
                        self.viewport.reset(*ft);
                        self.needs_render = true;
                    }
                }

                ui.add_space(8.0);
                ui.separator();

                // Julia c parameter (only shown when Julia is selected)
                // Nova and Newton do not use julia_c — c is the pixel coordinate for Nova
                // and Newton starts at the pixel coordinate directly.
                if self.fractal == FractalType::Julia {
                    ui.label("Julia constant c");
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Re:");
                        changed |= ui.add(
                            egui::DragValue::new(&mut self.julia_c[0])
                                .speed(0.002).range(-2.0..=2.0)
                        ).changed();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Im:");
                        changed |= ui.add(
                            egui::DragValue::new(&mut self.julia_c[1])
                                .speed(0.002).range(-2.0..=2.0)
                        ).changed();
                    });

                    ui.label("Presets");
                    for (name, c) in [
                        ("Dragon",    [-0.7269, 0.1889]),
                        ("Spiral",    [-0.8,    0.156 ]),
                        ("Dendrite",  [-0.235,  0.827 ]),
                        ("Snowflake", [-0.4,    0.6   ]),
                        ("Fire",      [ 0.285,  0.01  ]),
                    ] {
                        if ui.small_button(name).clicked() {
                            self.julia_c = c;
                            changed = true;
                        }
                    }
                    if changed { self.needs_render = true; }
                    ui.separator();
                }

                // Info label for Newton/Nova so the user knows what they're seeing
                if self.fractal == FractalType::Newton || self.fractal == FractalType::Nova {
                    egui::Frame::new()
                        .fill(ui.visuals().faint_bg_color)
                        .inner_margin(6.0)
                        .corner_radius(4.0)
                        .show(ui, |ui| {
                        match self.fractal {
                            FractalType::Newton =>
                                ui.small("Newton: f(z) = z³−1\nColors show convergence speed."),
                            FractalType::Nova =>
                                ui.small("Nova: Newton step + pixel c offset\nz starts at (1,0)."),
                            _ => unreachable!(),
                        };
                    });
                    ui.add_space(4.0);
                }

                // Color scheme
                ui.label("Color scheme");
                for cs in ColorScheme::ALL {
                    if ui.selectable_label(self.scheme == *cs, cs.name()).clicked() {
                        self.scheme = *cs;
                        self.needs_render = true;
                    }
                }

                ui.add_space(8.0);
                ui.separator();

                // Mariani-Silver subdivision
                if ui.checkbox(&mut self.use_ms, "Mariani-Silver fill").changed() {
                    self.needs_render = true;
                }

                ui.add_space(4.0);

                // Iteration depth
                ui.checkbox(&mut self.auto_iter, "Auto iterations");
                if !self.auto_iter {
                    if ui.add(
                        egui::Slider::new(&mut self.max_iter, 32..=4096)
                            .text("Max iter")
                            .logarithmic(true)
                    ).changed() {
                        self.needs_render = true;
                    }
                } else {
                    ui.label(format!("Current: {} iter", self.max_iter));
                }

                ui.add_space(8.0);
                ui.separator();

                // State readout
                egui::Grid::new("info").num_columns(2).spacing([6.0, 4.0]).show(ui, |ui| {
                    ui.label("Re:");
                    ui.monospace(format!("{:.8}", self.viewport.center[0]));
                    ui.end_row();
                    ui.label("Im:");
                    ui.monospace(format!("{:.8}", self.viewport.center[1]));
                    ui.end_row();
                    ui.label("Zoom:");
                    ui.monospace(format!("{:.3e}×", self.viewport.zoom));
                    ui.end_row();
                });

                ui.add_space(8.0);
                if ui.button("Reset view").clicked() {
                    self.viewport.reset(self.fractal);
                    self.needs_render = true;
                }

                ui.add_space(4.0);
                ui.label("Tips");
                ui.small("Scroll — zoom");
                ui.small("Drag — pan");
                ui.small("Double-click — zoom 2×");
            });
    }
}

impl eframe::App for FractalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keyboard shortcut: toggle panel with Space
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.show_panel = !self.show_panel;
        }
        // R to reset
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.viewport.reset(self.fractal);
            self.needs_render = true;
        }

        if self.show_panel {
            self.settings_panel(ctx);
        }

        // Poll for completed render
        if let Some(image) = self.worker.poll() {
            self.texture = Some(ctx.load_texture("fractal", image, TextureOptions::LINEAR));
            ctx.request_repaint(); // repaint once to show new frame
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let w = available.x as u32;
            let h = available.y as u32;

            // If window resized, re-render
            if self.viewport.width != w || self.viewport.height != h {
                self.viewport.width = w;
                self.viewport.height = h;
                self.needs_render = true;
            }

            // Kick off a render if needed and worker is free
            if self.needs_render && !self.worker.busy {
                self.request_render();
            }

            if let Some(tex) = &self.texture {
                let image_rect = Rect::from_min_size(
                    ui.next_widget_position(),
                    Vec2::new(w as f32, h as f32),
                );
                let response = ui.allocate_rect(image_rect, egui::Sense::click_and_drag());
                ui.painter().image(
                    tex.id(),
                    image_rect,
                    Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
                self.handle_mouse(&response, image_rect);

                // Spinner overlay while a new frame is rendering
                if self.worker.busy {
                    let spinner_pos = image_rect.right_bottom() - Vec2::splat(30.0);
                    ui.painter().circle_filled(spinner_pos, 8.0, Color32::from_black_alpha(120));
                    ui.ctx().request_repaint(); // keep polling until done
                }
            } else {
                // First frame — just show a spinner
                ui.centered_and_justified(|ui| { ui.spinner(); });
                ctx.request_repaint();
            }
        });

        // If worker just finished, immediately repaint so the new image shows
        if self.worker.busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}
