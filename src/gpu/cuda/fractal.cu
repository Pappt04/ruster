// CUDA device kernels for escape-time and perturbation-theory fractal
// rendering. These implement the same recurrences as the Rust kernels in
// src/fractal/kernels/ and src/fractal/perturbation/ (see those files for
// the underlying math); the two must stay in algebraic step, since a
// mismatch here is invisible except as a small, hard-to-spot difference
// between CPU- and GPU-rendered images. build.rs compiles this file to PTX
// via nvrtc and embeds it into the Rust binary.

#include <math.h>
#include <stdint.h>

#define ESCAPE_SQ 4.0

__device__ static const double INV_LN2 = 1.4426950408889634;

// Continuous iteration count from discrete escape count and |z|^2 at
// escape; see smooth_iter in src/fractal/fractal.rs for the derivation.
__device__ double smooth_iter(uint32_t i, double zn_sq) {
    double log_zn = log(zn_sq) * 0.5;
    double nu     = log(log_zn * INV_LN2) * INV_LN2;
    return (double)i + 1.0 - nu;
}

// Period-3 bulb membership test (see bulb_precheck.rs); the main
// cardioid/period-2 bulb tests are inlined directly into mandelbrot()
// below instead of factored out, since CUDA has no equivalent call-site
// cost concern to weigh against duplicating three lines of arithmetic.
#define PERIOD3_CENTER_RE (-0.1225611668766536)
#define PERIOD3_CENTER_IM (0.7448617666197442)
#define PERIOD3_RADIUS_SQ (0.07371484375 * 0.07371484375)

__device__ __forceinline__ bool in_period3_bulb(double cr, double ci) {
    double dr     = cr - PERIOD3_CENTER_RE;
    double di_pos = ci - PERIOD3_CENTER_IM;
    double di_neg = ci + PERIOD3_CENTER_IM;
    return (dr * dr + di_pos * di_pos < PERIOD3_RADIUS_SQ)
        || (dr * dr + di_neg * di_neg < PERIOD3_RADIUS_SQ);
}

__device__ __forceinline__ bool in_period3_bulb_f32(float cr, float ci) {
    float dr     = cr - (float)PERIOD3_CENTER_RE;
    float di_pos = ci - (float)PERIOD3_CENTER_IM;
    float di_neg = ci + (float)PERIOD3_CENTER_IM;
    return (dr * dr + di_pos * di_pos < (float)PERIOD3_RADIUS_SQ)
        || (dr * dr + di_neg * di_neg < (float)PERIOD3_RADIUS_SQ);
}

// f64 Mandelbrot escape-time iteration: cardioid/period-2/period-3
// prechecks, then Brent-style periodicity detection during iteration.
// Mirrors mandelbrot() in src/fractal/kernels/mandelbrot.rs.
__device__ float mandelbrot(double cr, double ci, uint32_t max_iter) {
    double q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if (q * (q + cr - 0.25) < 0.25 * ci * ci) return (float)max_iter;
    if ((cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625) return (float)max_iter;
    if (in_period3_bulb(cr, ci)) return (float)max_iter;

    double zr = cr, zi = ci;
    if (max_iter <= 1) return (float)max_iter;
    double zr2 = zr * zr;
    double zi2 = zi * zi;
    zi = 2.0 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    if (max_iter <= 2) return (float)max_iter;

    double zr_b = 0.0, zi_b = 0.0;
    uint32_t period = 0, check = 8;

    for (uint32_t i = 2; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        double zn_sq = zr2 + zi2;
        if (zn_sq > ESCAPE_SQ) return (float)smooth_iter(i, zn_sq);
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;

        double dr = zr - zr_b, di = zi - zi_b;
        if (dr * dr + di * di < 1e-20) return (float)max_iter;
        if (++period == check) {
            period = 0;
            check  = (check * 2 < 512) ? check * 2 : 512;
            zr_b   = zr;
            zi_b   = zi;
        }
    }
    return (float)max_iter;
}

// f64 Julia escape-time iteration with the same periodicity detection as
// mandelbrot() above. Mirrors julia() in src/fractal/kernels/julia.rs.
__device__ float julia(double zr0, double zi0, double cr, double ci, uint32_t max_iter) {
    double zr = zr0, zi = zi0;
    double zr_b = zr0, zi_b = zi0;
    uint32_t period = 0, check = 8;

    for (uint32_t i = 0; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        double zn_sq = zr2 + zi2;
        if (zn_sq > ESCAPE_SQ) return (float)smooth_iter(i, zn_sq);
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;

        double dr = zr - zr_b, di = zi - zi_b;
        if (dr * dr + di * di < 1e-20) return (float)max_iter;
        if (++period == check) {
            period = 0;
            check  = (check * 2 < 512) ? check * 2 : 512;
            zr_b   = zr;
            zi_b   = zi;
        }
    }
    return (float)max_iter;
}

// f32 kernels used below F32_PRECISION_THRESHOLD (see fractal.rs). Like
// the SIMD CPU kernels, these skip periodicity detection — not worth the
// extra branching at this granularity — relying only on the bulb
// prechecks and the escape-radius bailout.
#define ESCAPE_SQ_F32 4.0f

__device__ float smooth_iter_f32(uint32_t i, float zn_sq, uint32_t max_iter) {
    if (i >= max_iter) return (float)max_iter;
    const float inv_ln2 = 1.4426950408889634f;
    float log_zn = logf(zn_sq) * 0.5f;
    float nu     = logf(log_zn * inv_ln2) * inv_ln2;
    return (float)i + 1.0f - nu;
}

__device__ float mandelbrot_f32(float cr, float ci, uint32_t max_iter) {
    float q = (cr - 0.25f) * (cr - 0.25f) + ci * ci;
    if (q * (q + cr - 0.25f) < 0.25f * ci * ci) return (float)max_iter;
    if ((cr + 1.0f) * (cr + 1.0f) + ci * ci < 0.0625f) return (float)max_iter;
    if (in_period3_bulb_f32(cr, ci)) return (float)max_iter;

    float zr = cr, zi = ci;
    if (max_iter <= 1) return (float)max_iter;
    float zr2 = zr * zr;
    float zi2 = zi * zi;
    zi = 2.0f * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    if (max_iter <= 2) return (float)max_iter;

    for (uint32_t i = 2; i < max_iter; i++) {
        float zr2 = zr * zr, zi2 = zi * zi;
        float zn_sq = zr2 + zi2;
        if (zn_sq > ESCAPE_SQ_F32) return smooth_iter_f32(i, zn_sq, max_iter);
        zi = 2.0f * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return (float)max_iter;
}

__device__ float julia_f32(float zr0, float zi0, float cr, float ci, uint32_t max_iter) {
    float zr = zr0, zi = zi0;
    for (uint32_t i = 0; i < max_iter; i++) {
        float zr2 = zr * zr, zi2 = zi * zi;
        float zn_sq = zr2 + zi2;
        if (zn_sq > ESCAPE_SQ_F32) return smooth_iter_f32(i, zn_sq, max_iter);
        zi = 2.0f * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return (float)max_iter;
}

// Newton's method for f(z) = z^3 - 1, seeded at z0 = c. Mirrors newton()
// in src/fractal/kernels/newton.rs.
__device__ float newton(double cr, double ci, uint32_t max_iter) {
    double zr = cr, zi = ci;
    for (uint32_t i = 0; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        double z3r = zr * (zr2 - 3.0 * zi2);
        double z3i = zi * (3.0 * zr2 - zi2);
        double d_re = 3.0 * (zr2 - zi2);
        double d_im = 6.0 * zr * zi;
        double denom = d_re * d_re + d_im * d_im;
        if (denom < 1e-20) return (float)max_iter;
        double new_zr = zr - ((z3r - 1.0) * d_re + z3i * d_im) / denom;
        double new_zi = zi - (z3i * d_re - (z3r - 1.0) * d_im) / denom;
        double dr = new_zr - zr, di = new_zi - zi;
        zr = new_zr; zi = new_zi;
        if (dr * dr + di * di < 1e-12) return (float)i;
    }
    return (float)max_iter;
}

// Nova fractal: same Newton update as newton() above, with c added back
// in each step and z fixed-started at 1+0i. Mirrors nova() in
// src/fractal/kernels/nova.rs.
__device__ float nova(double cr, double ci, uint32_t max_iter) {
    double zr = 1.0, zi = 0.0;
    for (uint32_t i = 0; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        double z3r = zr * (zr2 - 3.0 * zi2);
        double z3i = zi * (3.0 * zr2 - zi2);
        double d_re = 3.0 * (zr2 - zi2);
        double d_im = 6.0 * zr * zi;
        double denom = d_re * d_re + d_im * d_im;
        if (denom < 1e-20) return (float)max_iter;
        double new_zr = zr - ((z3r - 1.0) * d_re + z3i * d_im) / denom + cr;
        double new_zi = zi - (z3i * d_re - (z3r - 1.0) * d_im) / denom + ci;
        double dr = new_zr - zr, di = new_zi - zi;
        zr = new_zr; zi = new_zi;
        if (dr * dr + di * di < 1e-12) return (float)i;
    }
    return (float)max_iter;
}

// Morton (Z-order) encode/decode via bit interleaving: spread_bits
// inserts a zero bit between each bit of a 16-bit input, so
// morton_encode's OR of spread_bits(x) with spread_bits(y) shifted left
// one interleaves x and y's bits into a single Z-order index. Used by
// fractal_kernel[_f32] below to map a linear thread index to a spatially
// local (x, y) pixel — see the module-level comment in gpu/cuda/mod.rs for
// why Morton order is used instead of row-major.
__device__ __forceinline__ uint32_t spread_bits(uint32_t x) {
    x &= 0x0000FFFFu;
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

__device__ __forceinline__ uint32_t compact_bits(uint32_t x) {
    x &= 0x55555555u;
    x = (x | (x >> 1u)) & 0x33333333u;
    x = (x | (x >> 2u)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4u)) & 0x00FF00FFu;
    x = (x | (x >> 8u)) & 0x0000FFFFu;
    return x;
}

__device__ __forceinline__ uint32_t morton_encode(uint32_t x, uint32_t y) {
    return spread_bits(x) | (spread_bits(y) << 1u);
}

__device__ __forceinline__ void morton_decode(uint32_t code, uint32_t *x, uint32_t *y) {
    *x = compact_bits(code);
    *y = compact_bits(code >> 1u);
}

// GPU perturbation rendering against a reference orbit uploaded by the
// host (orbit_re/orbit_im — see RefOrbit and the perturbation recurrence
// documented in src/fractal/perturbation/perturbation_theory.rs). The
// reference orbit's center is not passed explicitly: since the orbit was
// always computed for the pixel grid's own center, ref_re/ref_im are
// reconstructed here from re_start/im_start/re_step/im_step directly,
// exactly the coordinate the middle pixel of a width x height grid maps
// to. Falls back to full-precision mandelbrot() per pixel on a glitch or
// on reference-orbit exhaustion, the same policy as
// perturb_mandelbrot()/perturb_mandelbrot_flagged() on the CPU side (no
// rebasing or multi-reference correction on the GPU path).
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_perturb_kernel(
    float* __restrict__         buf,
    const double* __restrict__  orbit_re,
    const double* __restrict__  orbit_im,
    uint32_t        orbit_len,
    double          re_start,
    double          im_start,
    double          re_step,
    double          im_step,
    uint32_t        max_iter,
    uint32_t        width,
    uint32_t        height
) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double re    = re_start + (double)x * re_step;
    double im    = im_start + (double)y * im_step;
    double ref_re = re_start + (width  - 1.0) * 0.5 * re_step;
    double ref_im = im_start + (height - 1.0) * 0.5 * im_step;
    double dc_re = re - ref_re;
    double dc_im = im - ref_im;

    double er = 0.0, ei = 0.0;

    for (uint32_t n = 0; n < orbit_len; n++) {
        double zr = orbit_re[n];
        double zi = orbit_im[n];

        double new_er = 2.0*zr*er - 2.0*zi*ei + er*er - ei*ei + dc_re;
        double new_ei = 2.0*zr*ei + 2.0*zi*er + 2.0*er*ei    + dc_im;
        er = new_er;
        ei = new_ei;

        double zr1   = orbit_re[n + 1];
        double zi1   = orbit_im[n + 1];
        double az    = zr1 + er;
        double bz    = zi1 + ei;
        double zn_sq = az*az + bz*bz;

        if (zn_sq > ESCAPE_SQ) {
            buf[y * width + x] = (float)smooth_iter(n + 1, zn_sq);
            return;
        }

        double ref_sq = zr1*zr1 + zi1*zi1;
        if (er*er + ei*ei > ref_sq * 1e-6) {
            buf[y * width + x] = mandelbrot(re, im, max_iter);
            return;
        }
    }

    if (orbit_len < max_iter) {
        buf[y * width + x] = mandelbrot(re, im, max_iter);
    } else {
        buf[y * width + x] = (float)max_iter;
    }
}

// Full-frame f64 render, one thread per pixel, dispatched in Morton order:
// morton_idx is the thread's flat position within its 256-thread (16x16)
// block's slice of the Z-order curve over the padded dim x dim frame (see
// morton_cfg in gpu/cuda/mod.rs), decoded back to (x, y) and bounds-checked
// against the true frame size.
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_kernel(
    float* __restrict__ buf,
    double   re_start,
    double   im_start,
    double   re_step,
    double   im_step,
    double   julia_cr,
    double   julia_ci,
    uint32_t max_iter,
    uint32_t fractal,
    uint32_t width,
    uint32_t height
) {
    uint32_t morton_idx = (uint32_t)blockIdx.x * 256u
                        + (uint32_t)threadIdx.y * 16u
                        + (uint32_t)threadIdx.x;

    uint32_t x, y;
    morton_decode(morton_idx, &x, &y);
    if (x >= width || y >= height) return;

    double re = re_start + (double)x * re_step;
    double im = im_start + (double)y * im_step;

    float v;
    switch (fractal) {
        case 0: v = mandelbrot(re, im, max_iter); break;
        case 1: v = julia(re, im, julia_cr, julia_ci, max_iter); break;
        case 2: v = newton(re, im, max_iter); break;
        default: v = nova(re, im, max_iter); break;
    }

    buf[y * width + x] = v;
}

// f32 counterpart of fractal_kernel; Newton/Nova have no f32 kernel, so
// this only handles Mandelbrot (fractal == 0) and Julia.
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_kernel_f32(
    float* __restrict__ buf,
    float    re_start,
    float    im_start,
    float    re_step,
    float    im_step,
    float    julia_cr,
    float    julia_ci,
    uint32_t max_iter,
    uint32_t fractal,
    uint32_t width,
    uint32_t height
) {
    uint32_t morton_idx = (uint32_t)blockIdx.x * 256u
                        + (uint32_t)threadIdx.y * 16u
                        + (uint32_t)threadIdx.x;

    uint32_t x, y;
    morton_decode(morton_idx, &x, &y);
    if (x >= width || y >= height) return;

    float re = re_start + (float)x * re_step;
    float im = im_start + (float)y * im_step;

    float v = (fractal == 0)
        ? mandelbrot_f32(re, im, max_iter)
        : julia_f32(re, im, julia_cr, julia_ci, max_iter);

    buf[y * width + x] = v;
}

// Renders a batch of independently-sized/positioned tiles in one launch:
// blockIdx.z selects the tile, tile_descs[tile_idx] gives its
// [x0, y0, w, h] in frame coordinates, and threads outside that tile's
// own w x h are discarded (the launch grid is sized to the largest tile
// in the batch, so smaller tiles waste some threads rather than needing
// a separate launch each). Writes land row-major into the full-frame
// buf at the tile's true screen position. Used by the heterogeneous
// scheduler's GPU worker for row-major (non-Morton) tile dispatch.
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_kernel_tiled(
    float* __restrict__          buf,
    const uint32_t* __restrict__ tile_descs,
    double   re_start,
    double   im_start,
    double   re_step,
    double   im_step,
    double   julia_cr,
    double   julia_ci,
    uint32_t max_iter,
    uint32_t fractal,
    uint32_t full_width
) {
    uint32_t tile_idx = blockIdx.z;
    uint32_t tx0 = tile_descs[tile_idx * 4u + 0u];
    uint32_t ty0 = tile_descs[tile_idx * 4u + 1u];
    uint32_t tw  = tile_descs[tile_idx * 4u + 2u];
    uint32_t th  = tile_descs[tile_idx * 4u + 3u];

    uint32_t lx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= tw || ly >= th) return;

    uint32_t x = tx0 + lx;
    uint32_t y = ty0 + ly;

    double re = re_start + (double)x * re_step;
    double im = im_start + (double)y * im_step;

    float v;
    switch (fractal) {
        case 0: v = mandelbrot(re, im, max_iter); break;
        case 1: v = julia(re, im, julia_cr, julia_ci, max_iter); break;
        case 2: v = newton(re, im, max_iter); break;
        default: v = nova(re, im, max_iter); break;
    }

    buf[y * full_width + x] = v;
}

// f32 counterpart of fractal_kernel_tiled.
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_kernel_tiled_f32(
    float* __restrict__          buf,
    const uint32_t* __restrict__ tile_descs,
    float    re_start,
    float    im_start,
    float    re_step,
    float    im_step,
    float    julia_cr,
    float    julia_ci,
    uint32_t max_iter,
    uint32_t fractal,
    uint32_t full_width
) {
    uint32_t tile_idx = blockIdx.z;
    uint32_t tx0 = tile_descs[tile_idx * 4u + 0u];
    uint32_t ty0 = tile_descs[tile_idx * 4u + 1u];
    uint32_t tw  = tile_descs[tile_idx * 4u + 2u];
    uint32_t th  = tile_descs[tile_idx * 4u + 3u];

    uint32_t lx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= tw || ly >= th) return;

    uint32_t x = tx0 + lx;
    uint32_t y = ty0 + ly;

    float re = re_start + (float)x * re_step;
    float im = im_start + (float)y * im_step;

    float v = (fractal == 0)
        ? mandelbrot_f32(re, im, max_iter)
        : julia_f32(re, im, julia_cr, julia_ci, max_iter);

    buf[y * full_width + x] = v;
}

// Like fractal_kernel_tiled, but each tile descriptor carries a fifth
// field (offset) giving its position in a densely-packed output buffer
// instead of the tile's own screen offset in a full-frame buffer — used
// when the dispatched tiles are a scattered, non-contiguous subset of the
// frame rather than a full tiling of it.
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_kernel_tiled_compact(
    float* __restrict__          buf,
    const uint32_t* __restrict__ tile_descs,
    double   re_start,
    double   im_start,
    double   re_step,
    double   im_step,
    double   julia_cr,
    double   julia_ci,
    uint32_t max_iter,
    uint32_t fractal
) {
    uint32_t tile_idx = blockIdx.z;
    uint32_t tx0    = tile_descs[tile_idx * 5u + 0u];
    uint32_t ty0    = tile_descs[tile_idx * 5u + 1u];
    uint32_t tw     = tile_descs[tile_idx * 5u + 2u];
    uint32_t th     = tile_descs[tile_idx * 5u + 3u];
    uint32_t offset = tile_descs[tile_idx * 5u + 4u];

    uint32_t lx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= tw || ly >= th) return;

    double re = re_start + (double)(tx0 + lx) * re_step;
    double im = im_start + (double)(ty0 + ly) * im_step;

    float v;
    switch (fractal) {
        case 0: v = mandelbrot(re, im, max_iter); break;
        case 1: v = julia(re, im, julia_cr, julia_ci, max_iter); break;
        case 2: v = newton(re, im, max_iter); break;
        default: v = nova(re, im, max_iter); break;
    }

    buf[offset + ly * tw + lx] = v;
}

// f32 counterpart of fractal_kernel_tiled_compact.
extern "C" __global__ __launch_bounds__(256, 2)
void fractal_kernel_tiled_f32_compact(
    float* __restrict__          buf,
    const uint32_t* __restrict__ tile_descs,
    float    re_start,
    float    im_start,
    float    re_step,
    float    im_step,
    float    julia_cr,
    float    julia_ci,
    uint32_t max_iter,
    uint32_t fractal
) {
    uint32_t tile_idx = blockIdx.z;
    uint32_t tx0    = tile_descs[tile_idx * 5u + 0u];
    uint32_t ty0    = tile_descs[tile_idx * 5u + 1u];
    uint32_t tw     = tile_descs[tile_idx * 5u + 2u];
    uint32_t th     = tile_descs[tile_idx * 5u + 3u];
    uint32_t offset = tile_descs[tile_idx * 5u + 4u];

    uint32_t lx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= tw || ly >= th) return;

    float re = re_start + (float)(tx0 + lx) * re_step;
    float im = im_start + (float)(ty0 + ly) * im_step;

    float v = (fractal == 0)
        ? mandelbrot_f32(re, im, max_iter)
        : julia_f32(re, im, julia_cr, julia_ci, max_iter);

    buf[offset + ly * tw + lx] = v;
}

// Accumulates a per-integer-iteration-count histogram of buf via atomic
// increments, one bin per possible escape iteration (0..=max_iter).
// In-set pixels (v >= max_iter) are excluded, matching the CPU
// colorize pipeline's treatment of interior points as a fixed color
// rather than part of the equalized range.
extern "C" __global__
void hist_kernel(
    const float* __restrict__ buf,
    uint32_t* __restrict__    hist,
    uint32_t                  n,
    uint32_t                  max_iter
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = buf[i];
    if (v < (float)max_iter) {
        uint32_t bin = (uint32_t)floorf(v);
        if (bin > max_iter) bin = max_iter;
        atomicAdd(&hist[bin], 1u);
    }
}

// Maps each pixel's (fractional) iteration count through the CDF built
// from hist_kernel's histogram, then through the palette LUT: cdf[lo] and
// cdf[hi] (the two integer bins the fractional value falls between) are
// linearly interpolated by the fractional part, giving a smooth,
// histogram-equalized t in [0, 1], which then indexes the LUT. In-set
// pixels (v >= max_iter) bypass this and are always opaque black,
// matching the CPU colorize pipeline in src/gui/color.rs.
extern "C" __global__
void colorize_kernel(
    const float* __restrict__          buf,
    const float* __restrict__          cdf,
    const unsigned char* __restrict__  lut,
    unsigned char* __restrict__        out,
    uint32_t                           n,
    uint32_t                           max_iter,
    uint32_t                           lut_size
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t o = i * 4u;
    float v = buf[i];
    if (v >= (float)max_iter) {
        out[o + 0] = 0; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 255;
        return;
    }
    float    frac = v - floorf(v);
    uint32_t lo   = (uint32_t)floorf(v);
    uint32_t hi   = lo + 1u;
    if (hi > max_iter) hi = max_iter;
    float t = cdf[lo] + frac * (cdf[hi] - cdf[lo]);
    uint32_t idx = (uint32_t)(t * (float)(lut_size - 1u));
    if (idx >= lut_size) idx = lut_size - 1u;
    uint32_t li = idx * 4u;
    out[o + 0] = lut[li + 0];
    out[o + 1] = lut[li + 1];
    out[o + 2] = lut[li + 2];
    out[o + 3] = lut[li + 3];
}
