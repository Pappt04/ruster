use crate::gpu::unifroms::{PerturbUniforms, Uniforms};

// Maximum orbit entries pre-allocated on the GPU.
// Handles max_iter up to 8192 (orbit has max_iter+1 entries).
const MAX_ORBIT: u64 = 8193;

pub struct FractalCompute {
    // ── standard pipeline ────────────────────────────────────────────────────
    pipeline:         wgpu::ComputePipeline,
    bgl:              wgpu::BindGroupLayout,
    uniform_buf:      wgpu::Buffer,
    output_buf:       wgpu::Buffer,
    readback_buf:     wgpu::Buffer,
    // ── tiled pipeline (heterogeneous scheduler) ─────────────────────────────
    tiled_pipeline:   wgpu::ComputePipeline,
    tiled_bgl:        wgpu::BindGroupLayout,
    // ── perturbation pipeline ─────────────────────────────────────────────────
    perturb_pipeline: wgpu::ComputePipeline,
    perturb_bgl:      wgpu::BindGroupLayout,
    perturb_uni_buf:  wgpu::Buffer,
    orbit_re_buf:     wgpu::Buffer,
    orbit_im_buf:     wgpu::Buffer,
    // ── shared ───────────────────────────────────────────────────────────────
    width:  u32,
    height: u32,
}

impl FractalCompute {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // ── standard pipeline ────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fractal"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../fractal/fractal.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pipeline = build_pipeline(device, &bgl, &shader, "main", "fractal");

        // Tiled bind group layout adds one read-only storage binding (tile
        // descriptors) over the standard layout's uniform + output buffers —
        // reuses the same shader module (`main_tiled` alongside `main`).
        let tiled_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let tiled_pipeline = build_pipeline(device, &tiled_bgl, &shader, "main_tiled", "fractal_tiled");

        let pixel_bytes = (width * height) as u64 * 4;

        let uniform_buf  = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"), size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output_buf   = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"), size: pixel_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"), size: pixel_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── perturbation pipeline ────────────────────────────────────────────
        let perturb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fractal_perturb"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../fractal/fractal_perturb.wgsl").into(),
            ),
        });

        let perturb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });

        let perturb_pipeline =
            build_pipeline(device, &perturb_bgl, &perturb_shader, "main", "fractal_perturb");

        let orbit_bytes = MAX_ORBIT * 4; // f32 per entry
        let perturb_uni_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perturb_uniforms"),
            size: std::mem::size_of::<PerturbUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orbit_re_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbit_re"), size: orbit_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orbit_im_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("orbit_im"), size: orbit_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline, bgl, uniform_buf, output_buf, readback_buf,
            tiled_pipeline, tiled_bgl,
            perturb_pipeline, perturb_bgl, perturb_uni_buf, orbit_re_buf, orbit_im_buf,
            width, height,
        }
    }

    /// Full-res dimensions this instance was constructed with.
    pub fn width(&self) -> u32 { self.width }

    /// Full-res dimensions this instance was constructed with.
    pub fn height(&self) -> u32 { self.height }

    pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, uniforms: Uniforms) -> Vec<f32> {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.output_buf.as_entire_binding() },
            ],
        });

        self.dispatch_and_readback(device, queue, &self.pipeline, &bg)
    }

    pub fn render_perturbation(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uni: PerturbUniforms,
        orbit_re: &[f32],
        orbit_im: &[f32],
    ) -> Vec<f32> {
        let needed = (uni.orbit_len as u64 + 1) * 4;
        assert!(
            needed <= MAX_ORBIT * 4,
            "orbit_len {} exceeds pre-allocated GPU buffer (max {})",
            uni.orbit_len, MAX_ORBIT - 1
        );

        queue.write_buffer(&self.perturb_uni_buf, 0, bytemuck::bytes_of(&uni));
        queue.write_buffer(&self.orbit_re_buf, 0, bytemuck::cast_slice(orbit_re));
        queue.write_buffer(&self.orbit_im_buf, 0, bytemuck::cast_slice(orbit_im));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.perturb_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.perturb_uni_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.orbit_re_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.orbit_im_buf.as_entire_binding() },
            ],
        });

        self.dispatch_and_readback(device, queue, &self.perturb_pipeline, &bg)
    }

    /// Launches `main_tiled` over `tiles` — dispatch only, no readback (mirrors
    /// `CudaFractal::dispatch_tiled`, split from readback for the same reason:
    /// a caller doing work-stealing may need to dispatch more than once per
    /// frame before reading anything back). No-op if `tiles` is empty.
    ///
    /// `uniforms.width`/`uniforms.height` must be the FULL frame dimensions
    /// (used for the output buffer's row stride), not any single tile's —
    /// same convention as the CUDA tiled kernel.
    pub fn dispatch_tiled(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tiles: &[[u32; 4]],
        uniforms: Uniforms,
    ) {
        if tiles.is_empty() {
            return;
        }

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let flat: Vec<u32> = tiles.iter().flat_map(|t| t.iter().copied()).collect();
        let tile_descs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_descs"),
            size: (flat.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&tile_descs_buf, 0, bytemuck::cast_slice(&flat));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.tiled_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.output_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tile_descs_buf.as_entire_binding() },
            ],
        });

        // Grid must be large enough to cover the widest/tallest tile; bounds
        // checks inside the shader discard threads beyond the actual tile
        // extent — same approach as `fractal_kernel_tiled`'s CUDA grid sizing.
        let max_tw = tiles.iter().map(|t| t[2]).max().unwrap_or(1);
        let max_th = tiles.iter().map(|t| t[3]).max().unwrap_or(1);

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            pass.set_pipeline(&self.tiled_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((max_tw + 15) / 16, (max_th + 15) / 16, tiles.len() as u32);
        }
        queue.submit(std::iter::once(enc.finish()));
    }

    /// Copies the persistent full-frame output buffer back to the host. Call
    /// once per frame, after all of that frame's `dispatch_tiled` calls.
    pub fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(
            &self.output_buf, 0, &self.readback_buf, 0,
            (self.width * self.height * 4) as u64,
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::MaintainBase::Wait);

        let data   = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&*data).to_vec();
        drop(data);
        self.readback_buf.unmap();
        result
    }

    fn dispatch_and_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
    ) -> Vec<f32> {
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((self.width + 15) / 16, (self.height + 15) / 16, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.output_buf, 0, &self.readback_buf, 0,
            (self.width * self.height * 4) as u64,
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::MaintainBase::Wait);

        let data   = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&*data).to_vec();
        drop(data);
        self.readback_buf.unmap();
        result
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: entry,
        compilation_options: Default::default(),
        cache: None,
    })
}
