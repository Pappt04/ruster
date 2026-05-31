use crate::fractal::fractal_type::FractalType;
use rayon::prelude::*;

pub const ESCAPE_RADIUS_SQ: f64 = 256.0 * 256.0;

pub type IterationBuffer= Vec<f32>;
