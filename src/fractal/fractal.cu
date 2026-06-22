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

// Perturbation-theory kernel for Mandelbrot.
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
    // Reference center = viewport center, recoverable from grid params.
    double ref_re = re_start + (width  - 1.0) * 0.5 * re_step;
    double ref_im = im_start + (height - 1.0) * 0.5 * im_step;
    double dc_re = re - ref_re;
    double dc_im = im - ref_im;

    double er = 0.0, ei = 0.0;

    for (uint32_t n = 0; n < orbit_len; n++) {
        double zr = orbit_re[n];
        double zi = orbit_im[n];

        // ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ
        double new_er = 2.0*zr*er - 2.0*zi*ei + er*er - ei*ei + dc_re;
        double new_ei = 2.0*zr*ei + 2.0*zi*er + 2.0*er*ei    + dc_im;
        er = new_er;
        ei = new_ei;

        // z_{n+1} = Z_{n+1} + ε_{n+1}
        double zr1   = orbit_re[n + 1];
        double zi1   = orbit_im[n + 1];
        double az    = zr1 + er;
        double bz    = zi1 + ei;
        double zn_sq = az*az + bz*bz;

        if (zn_sq > ESCAPE_SQ) {
            buf[y * width + x] = (float)smooth_iter(n + 1, zn_sq);
            return;
        }

        // Glitch: |ε|² > 1e-6 · |Z|²  → scalar fallback
        double ref_sq = zr1*zr1 + zi1*zi1;
        if (er*er + ei*ei > ref_sq * 1e-6) {
            buf[y * width + x] = mandelbrot(re, im, max_iter);
            return;
        }
    }

    // Reference orbit escaped early → scalar fallback
    if (orbit_len < max_iter) {
        buf[y * width + x] = mandelbrot(re, im, max_iter);
    } else {
        buf[y * width + x] = (float)max_iter;
    }
}

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
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
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
