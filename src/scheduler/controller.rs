//! PI controller that adapts the tile-classification variance threshold
//! frame-to-frame based on observed GPU/CPU finish-time imbalance.

pub struct ThresholdController {
    pub threshold: f32,
    integral: f32,
    history: [f32; 16],
    head: usize,
}

impl ThresholdController {
    pub fn new(initial: f32) -> Self {
        Self { threshold: initial, integral: 0.0, history: [0.0; 16], head: 0 }
    }

    /// Call after each frame with observed GPU and CPU finish times in milliseconds.
    pub fn update(&mut self, gpu_ms: f32, cpu_ms: f32) {
        let error = gpu_ms - cpu_ms; // positive => GPU overloaded => lower threshold => route more to CPU
        self.history[self.head] = error;
        self.head = (self.head + 1) % self.history.len();
        self.integral = self.history.iter().sum::<f32>() / self.history.len() as f32;

        const K_P: f32 = 0.5;
        const K_I: f32 = 0.05;
        self.threshold -= K_P * error + K_I * self.integral;
        self.threshold = self.threshold.clamp(10.0, 500.0);
    }
}
