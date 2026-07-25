// Uniforms: everything the shader needs to know about the viewport + fractal params
struct Uniforms {
    re_start : f32,
    im_start : f32,
    re_step  : f32,
    im_step  : f32,
    julia_cr : f32,
    julia_ci : f32,
    max_iter : u32,
    fractal  : u32,   // 0=mandelbrot 1=julia 2=newton 3=nova
    width    : u32,
    height   : u32,
}

@group(0) @binding(0) var<uniform>            uni  : Uniforms;
@group(0) @binding(1) var<storage, read_write> buf : array<f32>;

// Must match src/fractal/fractal.rs's ESCAPE_RADIUS_SQ (bailout radius 2).
const ESCAPE_SQ : f32 = 4.0;

// ── smooth coloring helper ──────────────────────────────────────────────────
fn smooth_iter(i: u32, zn_sq: f32) -> f32 {
    let log_zn = log(zn_sq) * 0.5;
    let nu     = log(log_zn / log(2.0)) / log(2.0);
    return f32(i) + 1.0 - nu;
}

// ── Mandelbrot ──────────────────────────────────────────────────────────────
fn mandelbrot(cr: f32, ci: f32, max_iter: u32) -> f32 {
    // cardioid / period-2 bulb check
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci { return f32(max_iter); }
    if (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625 { return f32(max_iter); }

    var zr = 0.0f;  var zi = 0.0f;
    for (var i = 0u; i < max_iter; i++) {
        let zr2 = zr * zr;   let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_SQ { return smooth_iter(i, zn_sq); }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return f32(max_iter);
}

// ── Julia ───────────────────────────────────────────────────────────────────
fn julia(zr0: f32, zi0: f32, cr: f32, ci: f32, max_iter: u32) -> f32 {
    var zr = zr0;  var zi = zi0;
    for (var i = 0u; i < max_iter; i++) {
        let zr2 = zr * zr;   let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_SQ { return smooth_iter(i, zn_sq); }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return f32(max_iter);
}

// ── Newton ──────────────────────────────────────────────────────────────────
fn newton(cr: f32, ci: f32, max_iter: u32) -> f32 {
    var zr = cr;  var zi = ci;
    for (var i = 0u; i < max_iter; i++) {
        let zr2 = zr * zr;   let zi2 = zi * zi;
        let z3r = zr * (zr2 - 3.0 * zi2);
        let z3i = zi * (3.0 * zr2 - zi2);
        let d_re = 3.0 * (zr2 - zi2);
        let d_im = 6.0 * zr * zi;
        let denom = d_re * d_re + d_im * d_im;
        if denom < 1e-20 { return f32(max_iter); }
        let new_zr = zr - (z3r - 1.0) * d_re / denom + z3i * d_im / denom;
        let new_zi = zi - z3i * d_re / denom + (z3r - 1.0) * d_im / denom;
        let dr = new_zr - zr;   let di = new_zi - zi;
        zr = new_zr;   zi = new_zi;
        if dr * dr + di * di < 1e-12 { return f32(i); }
    }
    return f32(max_iter);
}

// ── Nova ─────────────────────────────────────────────────────────────────────
fn nova(cr: f32, ci: f32, max_iter: u32) -> f32 {
    var zr = 1.0f;  var zi = 0.0f;
    for (var i = 0u; i < max_iter; i++) {
        let zr2 = zr * zr;   let zi2 = zi * zi;
        let z3r = zr * (zr2 - 3.0 * zi2);
        let z3i = zi * (3.0 * zr2 - zi2);
        let d_re = 3.0 * (zr2 - zi2);
        let d_im = 6.0 * zr * zi;
        let denom = d_re * d_re + d_im * d_im;
        if denom < 1e-20 { return f32(max_iter); }
        let new_zr = zr - (z3r - 1.0) * d_re / denom + z3i * d_im / denom + cr;
        let new_zi = zi - z3i * d_re / denom + (z3r - 1.0) * d_im / denom + ci;
        let dr = new_zr - zr;   let di = new_zi - zi;
        zr = new_zr;   zi = new_zi;
        if dr * dr + di * di < 1e-12 { return f32(i); }
    }
    return f32(max_iter);
}

// ── Entry point ──────────────────────────────────────────────────────────────
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= uni.width || y >= uni.height { return; }

    let re = uni.re_start + f32(x) * uni.re_step;
    let im = uni.im_start + f32(y) * uni.im_step;

    var v: f32;
    switch uni.fractal {
        case 0u: { v = mandelbrot(re, im, uni.max_iter); }
        case 1u: { v = julia(re, im, uni.julia_cr, uni.julia_ci, uni.max_iter); }
        case 2u: { v = newton(re, im, uni.max_iter); }
        default: { v = nova(re, im, uni.max_iter); }
    }

    buf[y * uni.width + x] = v;
}