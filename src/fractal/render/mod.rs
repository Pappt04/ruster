//! CPU render backends built on top of the per-pixel kernels in
//! `crate::fractal::kernels`: plain SIMD sweeps, tile-granular rendering
//! for the scheduler, incremental pan updates, Hilbert-curve tile
//! ordering, and Mariani-Silver boundary tracing.

pub mod hilbert;
pub mod mariani_silver;
pub mod pan;
pub mod simd;
pub mod tile;
