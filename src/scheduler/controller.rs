//! Runtime feedback control over the classifier's GPU/CPU split.
//!
//! [`crate::scheduler::classifier`]'s corner-spread threshold is a proxy
//! for tile cost, not a direct measurement, and the right proxy value
//! depends on scene content and current GPU/CPU load in ways that are not
//! known in advance. [`ThresholdController`] instead observes each
//! frame's actual measured GPU and CPU wall-clock time and nudges the
//! threshold so the two stay balanced — if the GPU is consistently
//! finishing its tiles faster than the CPU finishes its tiles, the
//! threshold should relax to route more tiles to the GPU, and vice versa.

/// A PI (proportional-integral) controller on the classifier's threshold:
/// `threshold` is the classifier input, `error = gpu_ms - cpu_ms` is what
/// it is being driven to keep near zero.
pub struct ThresholdController {
    pub threshold: f32,
    integral: f32,
    /// Ring buffer of the last 16 frames' error, used to compute a moving
    /// average for the integral term rather than an unbounded running sum
    /// — bounds how long a past imbalance keeps influencing the threshold.
    history: [f32; 16],
    head: usize,
}

impl ThresholdController {
    pub fn new(initial: f32) -> Self {
        Self { threshold: initial, integral: 0.0, history: [0.0; 16], head: 0 }
    }

    /// Advances the controller by one frame's measured timings.
    ///
    /// `error > 0` (GPU slower than CPU) pushes the threshold down —
    /// classifying fewer tiles as GPU-uniform, since the classifier's
    /// corner-spread test uses `spread < threshold`, so a lower threshold
    /// routes more tiles to the CPU and relieves GPU load. `error < 0`
    /// does the opposite. The integral term (mean error over the last 16
    /// frames) corrects the steady-state bias a proportional-only
    /// controller would leave behind if the two backends have a
    /// persistent relative speed difference rather than just transient
    /// noise. The threshold is clamped to `[0.001, 0.5]` to keep the
    /// classifier from collapsing to "everything is one backend" even
    /// under a large sustained imbalance.
    pub fn update(&mut self, gpu_ms: f32, cpu_ms: f32) {
        let error = gpu_ms - cpu_ms;
        self.history[self.head] = error;
        self.head = (self.head + 1) % self.history.len();
        self.integral = self.history.iter().sum::<f32>() / self.history.len() as f32;

        const K_P: f32 = 0.0005;
        const K_I: f32 = 0.00005;
        self.threshold -= K_P * error + K_I * self.integral;
        self.threshold = self.threshold.clamp(0.001, 0.5);
    }
}
