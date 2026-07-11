#include <math.h>
#include <stdint.h>

#define ESCAPE_SQ 65536.0

// Precomputed 1/ln(2) — avoids repeated log(2.0) calls in smooth_iter.
__device__ static const double INV_LN2 = 1.4426950408889634;

__device__ double smooth_iter(uint32_t i, double zn_sq) {
    double log_zn = log(zn_sq) * 0.5;
    double nu     = log(log_zn * INV_LN2) * INV_LN2;
    return (double)i + 1.0 - nu;
}

__device__ float mandelbrot(double cr, double ci, uint32_t max_iter) {
    double q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if (q * (q + cr - 0.25) < 0.25 * ci * ci) return (float)max_iter;
    if ((cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625) return (float)max_iter;

    double zr = 0.0, zi = 0.0;
    double zr_b = 0.0, zi_b = 0.0;
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

// ── Morton (Z-order) curve helpers ───────────────────────────────────────────
//
// Interleave the lower 16 bits of x and y into a 32-bit Morton code so that
// spatially adjacent pixels have adjacent Morton indices — improving L2/L1
// cache locality when adjacent warp threads write nearby output addresses.
//
// Bit pattern: y15 x15 y14 x14 … y1 x1 y0 x0

// Spread: insert a zero bit after each of the 16 input bits.
__device__ __forceinline__ uint32_t spread_bits(uint32_t x) {
    x &= 0x0000FFFFu;
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

// Compact: inverse of spread_bits (extract every other bit).
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

// ── Perturbation-theory kernel ────────────────────────────────────────────────
//
// orbit_re / orbit_im hold orbit_len+1 entries of the reference orbit Z_0..Z_{orbit_len}.
// Each thread iterates ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ and checks escape on
// z_{n+1} = Z_{n+1} + ε_{n+1}.  Glitched pixels fall back to the scalar kernel.
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

// ── Main fractal kernel — Z-order (Morton) dispatch ───────────────────────────
//
// Launched with a 1D grid of 256-thread blocks (16×16).  Each block handles
// 256 consecutive Morton codes so that threads in a warp read from nearby input
// coordinates and write to nearby output addresses, improving L2 hit rate.
// The output buffer remains in row-major order — only the traversal order changes.
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
    // Map the flat 1D thread index to a pixel via Morton decode.
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

    // Write in row-major order — CPU reads linearly.
    buf[y * width + x] = v;
}

// ── Tiled kernel — for the heterogeneous scheduler (Stage 3) ─────────────────
//
// Each thread block handles one sub-region (tile) of the full image.
// blockIdx.z selects the tile from the descriptor array.
// tile_descs layout: [x0, y0, w, h] per tile (4 × uint32 per entry).
// Launch with grid_dim = (ceil(max_tile_w/16), ceil(max_tile_h/16), num_tiles).
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
