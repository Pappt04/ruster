use std::sync::LazyLock;
use egui::Color32;
use rayon::prelude::*;

const LUT_SIZE: usize = 4096*2;
const N_SCHEMES: usize = 6;

static PALETTES: LazyLock<[[Color32; LUT_SIZE]; N_SCHEMES]> = LazyLock::new(|| {
    std::array::from_fn(|scheme_idx| {
        let scheme = ColorScheme::ALL[scheme_idx];
        std::array::from_fn(|i| scheme.sample(i as f32 / (LUT_SIZE - 1) as f32))
    })
});

/// Flat RGBA8 bytes (`LUT_SIZE * 4` per scheme) of [`PALETTES`] — the same
/// values `colorize` looks up on the CPU, laid out for `CudaFractal::colorize_into`
/// to upload once and index identically on the GPU. Built lazily from
/// `PALETTES` itself rather than duplicating the sampling, so the two never
/// drift: whatever `colorize` would draw for a given scheme is exactly what
/// `colorize_into` draws for it too.
#[cfg(feature = "cuda")]
static PALETTE_BYTES: LazyLock<[Vec<u8>; N_SCHEMES]> = LazyLock::new(|| {
    std::array::from_fn(|i| PALETTES[i].iter().flat_map(|c| c.to_array()).collect())
});

/// GPU-side LUT bytes for `scheme` — see [`PALETTE_BYTES`]. Pair with
/// `scheme.palette_index() as u8` as `CudaFractal::colorize_into`'s cache key.
#[cfg(feature = "cuda")]
pub fn lut_bytes(scheme: ColorScheme) -> &'static [u8] {
    &PALETTE_BYTES[scheme.palette_index()]
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ColorScheme {
    #[default]
    Inferno,
    Ocean,
    Plasma,
    Grayscale,
    Electric,
    Sunset,
}

impl ColorScheme {
    pub const ALL: &'static [Self] = &[
        Self::Inferno, Self::Ocean, Self::Plasma,
        Self::Grayscale, Self::Electric, Self::Sunset,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Inferno   => "Inferno",
            Self::Ocean     => "Ocean",
            Self::Plasma    => "Plasma",
            Self::Grayscale => "Grayscale",
            Self::Electric  => "Electric",
            Self::Sunset    => "Sunset",
        }
    }

    pub const fn palette_index(self) -> usize {
        match self {
            Self::Inferno   => 0,
            Self::Ocean     => 1,
            Self::Plasma    => 2,
            Self::Grayscale => 3,
            Self::Electric  => 4,
            Self::Sunset    => 5,
        }
    }

    /// Sample the palette at t ∈ [0, 1].
    pub fn sample(self, t: f32) -> Color32 {
        use std::f32::consts::TAU;
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Inferno => {
                let r = (t * 2.0).min(1.0);
                let g = (t * 1.5 - 0.25).clamp(0.0, 1.0);
                let b = ((1.0 - t) * 0.8 * (TAU * t * 0.5).sin().abs()).clamp(0.0, 1.0);
                rgb(r, g * r.sqrt(), b)
            }
            Self::Ocean => {
                let depth = t.powf(1.5);
                let shimmer = (t * TAU * 3.0).sin() * 0.07 + 1.0;
                rgb(depth * 0.15 * shimmer,
                    (0.3 + 0.5 * depth) * shimmer,
                    (0.55 + 0.45 * depth) * shimmer)
            }
            Self::Plasma => {
                let r = (0.5 + 0.5 * (TAU * t).sin()).powf(0.8);
                let g = (0.5 + 0.5 * (TAU * t * 1.5 + 2.0).sin()).powf(0.8);
                let b = (0.5 + 0.5 * (TAU * t * 0.7 + 4.0).sin()).powf(0.6);
                rgb(r, g, b)
            }
            Self::Grayscale => {
                let v = t.powf(0.8);
                rgb(v, v, v)
            }
            Self::Electric => {
                let arc = (t * TAU * 4.0).sin().abs();
                let bolt = if (t * TAU * 12.0).sin() > 0.93 { 0.8 } else { 0.0 };
                rgb(0.4 + 0.4 * arc + bolt,
                    0.2 + 0.3 * (1.0 - t) + bolt,
                    0.7 + 0.3 * arc)
            }
            Self::Sunset => {
                if t < 0.4 {
                    let s = t / 0.4;
                    rgb(0.1 + 0.4 * s, 0.05 + 0.25 * s, 0.3 + 0.3 * s)
                } else if t < 0.75 {
                    let s = (t - 0.4) / 0.35;
                    rgb(0.5 + 0.5 * s, 0.3 + 0.4 * s, 0.6 - 0.4 * s)
                } else {
                    let s = (t - 0.75) / 0.25;
                    rgb(1.0, 0.7 + 0.3 * s, 0.2 + 0.6 * s)
                }
            }
        }
    }
}

fn rgb(r: f32, g: f32, b: f32) -> Color32 {
    Color32::from_rgb(
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

/// Maps a smooth iteration buffer to RGBA pixels using histogram equalization.
pub fn colorize(buf: &[f32], max_iter: u32, scheme: ColorScheme) -> Vec<Color32> {
    let max_f = max_iter as f32;
    let lut = &PALETTES[scheme.palette_index()];
    let bins = max_iter as usize + 1;

    // Build histogram of escaped pixels (exclude in-set)
    // Use parallel fold+reduce if the buffer is large enough to warrant the overhead
    let hist = if buf.len() >= 200_000 {
        buf.par_iter()
            .fold(
                || vec![0u32; bins],
                |mut local, &v| {
                    if v < max_f {
                        local[(v.floor() as usize).min(bins - 1)] += 1;
                    }
                    local
                }
            )
            .reduce(
                || vec![0u32; bins],
                |mut a, b| {
                    for i in 0..bins {
                        a[i] += b[i];
                    }
                    a
                }
            )
    } else {
        // Small buffers: sequential histogram
        let mut hist = vec![0u32; bins];
        for &v in buf {
            if v < max_f {
                hist[(v.floor() as usize).min(bins - 1)] += 1;
            }
        }
        hist
    };

    // Cumulative distribution → equalized [0,1] value per bin (always sequential, cheap)
    let total_escaped = hist.iter().map(|&c| c as f64).sum::<f64>();
    let mut cdf = vec![0.0f32; bins];
    let mut running = 0.0f64;
    for i in 0..bins {
        running += hist[i] as f64;
        cdf[i] = if total_escaped > 0.0 { (running / total_escaped) as f32 } else { 0.0 };
    }

    // Colorize each pixel via static LUT lookup (parallel for large buffers)
    if buf.len() >= 200_000 {
        buf.par_iter().map(|&v| {
            if v >= max_f {
                Color32::BLACK
            } else {
                let frac = v.fract();
                let lo = v.floor() as usize;
                let hi = (lo + 1).min(bins - 1);
                let t = cdf[lo] + frac * (cdf[hi] - cdf[lo]);
                lut[((t * (LUT_SIZE - 1) as f32) as usize).min(LUT_SIZE - 1)]
            }
        }).collect()
    } else {
        // Small buffers: sequential mapping
        buf.iter().map(|&v| {
            if v >= max_f {
                Color32::BLACK
            } else {
                let frac = v.fract();
                let lo = v.floor() as usize;
                let hi = (lo + 1).min(bins - 1);
                let t = cdf[lo] + frac * (cdf[hi] - cdf[lo]);
                lut[((t * (LUT_SIZE - 1) as f32) as usize).min(LUT_SIZE - 1)]
            }
        }).collect()
    }
}
