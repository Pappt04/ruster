struct Uniforms {
    re_start  : f32,
    im_start  : f32,
    re_step   : f32,
    im_step   : f32,
    ref_re    : f32,
    ref_im    : f32,
    orbit_len : u32,
    max_iter  : u32,
    width     : u32,
    height    : u32,
}

@group(0) @binding(0) var<uniform>             uni      : Uniforms;
@group(0) @binding(1) var<storage, read_write> buf      : array<f32>;
@group(0) @binding(2) var<storage, read>       orbit_re : array<f32>;
@group(0) @binding(3) var<storage, read>       orbit_im : array<f32>;

const ESCAPE_SQ  : f32 = 4.0;
const GLITCH_SQ  : f32 = 1e-6;      

fn smooth_iter(i: u32, zn_sq: f32) -> f32 {
    let log_zn = log(zn_sq) * 0.5;
    let nu     = log(log_zn / log(2.0)) / log(2.0);
    return f32(i) + 1.0 - nu;
}

fn mandelbrot_scalar(cr: f32, ci: f32) -> f32 {
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci { return f32(uni.max_iter); }
    if (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625 { return f32(uni.max_iter); }

    var zr = 0.0f;
    var zi = 0.0f;
    for (var i = 0u; i < uni.max_iter; i++) {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_SQ { return smooth_iter(i, zn_sq); }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    return f32(uni.max_iter);
}

fn mandelbrot_perturb(re: f32, im: f32) -> f32 {
    let dc_re = re - uni.ref_re;
    let dc_im = im - uni.ref_im;

    var er = 0.0f;
    var ei = 0.0f;

    for (var n = 0u; n < uni.orbit_len; n++) {
        let zr = orbit_re[n];
        let zi = orbit_im[n];

        let new_er = 2.0 * zr * er - 2.0 * zi * ei + er * er - ei * ei + dc_re;
        let new_ei = 2.0 * zr * ei + 2.0 * zi * er + 2.0 * er * ei     + dc_im;
        er = new_er;
        ei = new_ei;

        let zr1   = orbit_re[n + 1u];
        let zi1   = orbit_im[n + 1u];
        let az    = zr1 + er;
        let bz    = zi1 + ei;
        let zn_sq = az * az + bz * bz;

        if zn_sq > ESCAPE_SQ {
            return smooth_iter(n + 1u, zn_sq);
        }

        let ref_sq = zr1 * zr1 + zi1 * zi1;
        if er * er + ei * ei > ref_sq * GLITCH_SQ {
            return mandelbrot_scalar(re, im);
        }
    }

    if uni.orbit_len < uni.max_iter {
        return mandelbrot_scalar(re, im);
    }
    return f32(uni.max_iter);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= uni.width || y >= uni.height { return; }

    let re = uni.re_start + f32(x) * uni.re_step;
    let im = uni.im_start + f32(y) * uni.im_step;

    buf[y * uni.width + x] = mandelbrot_perturb(re, im);
}
