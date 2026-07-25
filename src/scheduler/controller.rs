//! PI controller that adapts the corner-spread classification threshold
//! frame-to-frame based on observed GPU/CPU finish-time imbalance.
//!
//! `threshold` gates `classifier::corner_spread` — a normalized `(max-min)/
//! max_iter` value in `~[0,1]` — not the old prepass-variance threshold
//! (which was implicitly scaled by `max_iter`). Mechanism, calling
//! convention, and sign are otherwise unchanged from the prepass-based
//! design: lower threshold => stricter "close enough for GPU" requirement =>
//! fewer tiles terminate early as GPU => more (and, under recursive
//! partitioning, smaller) CPU-routed tiles.
//!
//! Clamp bounds and initial value are starting guesses, not measured —
//! re-tune empirically via `bench_runner --scheduler-sweep` against real
//! hardware, the same way the old `[10.0, 500.0]`/`50.0` were presumably
//! derived. `K_P`/`K_I` may also need re-tuning: under recursive
//! partitioning, "route more to CPU" now compounds tile count *and* size
//! together, whereas the old fixed-grid design only changed count.

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

        // K_P/K_I rescaled ~1000x down from the old 0.5/0.05: those were
        // calibrated for a [10,500] threshold range (span ~490), and applied
        // unscaled to the new [0.001,0.5] range (span ~0.5) would pin the
        // threshold at a clamp bound on almost every frame given typical
        // multi-ms gpu_ms/cpu_ms gaps — not "adaptive" so much as "always
        // saturated." This keeps the old range-relative responsiveness
        // (fraction of the range one frame's error can move the threshold)
        // roughly intact, but — like the clamp bounds above — it's a starting
        // guess pending a real sweep, not a derived constant.
        const K_P: f32 = 0.0005;
        const K_I: f32 = 0.00005;
        self.threshold -= K_P * error + K_I * self.integral;
        self.threshold = self.threshold.clamp(0.001, 0.5);
    }
}
