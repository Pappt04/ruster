#include <math.h>
#include <stdint.h>

#define ESCAPE_SQ 65536.0

__device__ double smooth_iter(uint32_t i, double zn_sq) {
    double log_zn = log(zn_sq) * 0.5;
    double nu     = log(log_zn / log(2.0)) / log(2.0);
    return (double)i + 1.0 - nu;
}

__device__ float mandelbrot(double cr, double ci, uint32_t max_iter) {
    double q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if (q * (q + cr - 0.25) < 0.25 * ci * ci) return (float)max_iter;
    if ((cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625) return (float)max_iter;

    double zr = 0.0, zi = 0.0;
    for (uint32_t i = 0; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        double zn_sq = zr2 + zi2;
        if (zn_sq > ESCAPE_SQ) return (float)smooth_iter(i, zn_sq);
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return (float)max_iter;
}

__device__ float julia(double zr0, double zi0, double cr, double ci, uint32_t max_iter) {
    double zr = zr0, zi = zi0;
    for (uint32_t i = 0; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        double zn_sq = zr2 + zi2;
        if (zn_sq > ESCAPE_SQ) return (float)smooth_iter(i, zn_sq);
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
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

extern "C" __global__ void fractal_kernel(
    float*   buf,
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
