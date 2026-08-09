# Chapter 3 — Related Work

> **⚠ Verify every citation before submission.** The bibliographic details below
> (authors, venues, years) are given to the best of my knowledge, but I have not
> opened these papers from within this project. Titles, authors, venues and years
> should be checked against DOI, ACM DL, IEEE Xplore or the publisher's page, and
> page numbers added, before this goes into a thesis. Where I am less certain,
> the entry is marked ⚠. Fabricated or garbled citations are among the fastest
> ways to lose credibility with an examiner, so treat this chapter as a
> structured reading list with drafted prose — not as verified bibliography.
>
> Sources marked **[non-peer-reviewed]** are cited deliberately: for deep-zoom
> perturbation the primary literature *is* community writing, and pretending
> otherwise would misrepresent the field.

---

This work sits at the intersection of three literatures: the mathematics and
rendering of escape-time fractals, arbitrary-precision techniques for deep
zoom, and task scheduling across heterogeneous processors. Each is surveyed
below, closing with the specific gap this thesis addresses.

## 3.1 Escape-time fractal rendering

The Mandelbrot set was introduced by Mandelbrot [1] and given its rigorous
dynamical foundation by Douady and Hubbard [2], whose work established the
connectedness of the set and the structure of its boundary. Shishikura [3]
later proved that the boundary has Hausdorff dimension exactly 2 — a result
this thesis uses as the reference value when validating the box-counting
estimator in §6.x, and which explains why any finite-resolution estimate
necessarily falls short of it.

The standard rendering algorithm is *escape-time*: iterate
`z_{n+1} = z_n² + c` until `|z| > 2` or an iteration cap is reached. Two
refinements are near-universal in practice and both are used here. **Smooth
(continuous) colouring** replaces the integer escape count with a fractional
value derived from the escape magnitude, removing the visible banding of
integer iteration counts; the technique is described in the fractal-rendering
literature and is standard in the field [4] **[non-peer-reviewed]**.
**Interior detection** short-circuits points provably inside the set — the main
cardioid and the period-2 bulb admit closed-form membership tests, and further
bulbs can be tested against their known nuclei and radii. This thesis
additionally applies Brent's cycle-detection algorithm [5] to catch periodic
orbits that the closed-form tests miss.

Several algorithms attempt to *avoid* per-pixel evaluation entirely. The
**Mariani–Silver** algorithm exploits the fact that the escape-time field has no
holes: if every pixel on the border of a rectangle shares a value, the interior
must share it too, and can be flood-filled rather than computed. The algorithm
is attributed to Mariani and Silver and is described in the fractal-rendering
community rather than in a canonical peer-reviewed paper ⚠ [6]
**[non-peer-reviewed]**; §6.x measures it and finds that on a multi-core CPU
its sequential recursion costs more than the work it saves.

**Distance estimation** provides a lower bound on the distance from a point to
the set boundary, allowing large empty regions to be skipped or shaded. The
exterior distance estimator is presented in Peitgen and Saupe's *The Science of
Fractal Images* [7], and the technique was generalised to ray-traced
deterministic fractals by Hart, Sandin and Kauffman [8]. An interior
counterpart, which bounds the distance from an interior point to the boundary by
tracking the derivative of the attracting cycle, is described by
Heiland-Allen [9] **[non-peer-reviewed]**. Both are implemented and evaluated
here.

## 3.2 Deep zoom and arbitrary precision

Double-precision floating point exhausts its ability to distinguish adjacent
pixels at a zoom of roughly 10¹⁵, and — as §6.x shows — the *pixel grid itself*
begins to degrade well before that. Naïve arbitrary-precision iteration is
correct but prohibitively slow, since every pixel pays the full cost.

**Perturbation theory** avoids this. A single high-precision *reference orbit*
is computed once for the frame; every other pixel is then expressed as a small
offset `ε` from that reference and iterated in ordinary double precision using
`ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ`. Because `ε` remains small, double precision
suffices for it even when the absolute coordinates do not. The technique was
introduced to the fractal-rendering community by K. I. Martin [10]
**[non-peer-reviewed]** and has been developed extensively by Heiland-Allen
[11] **[non-peer-reviewed]**, whose analyses of glitch detection, rebasing and
series approximation are the practical reference for implementers. This
literature is not peer-reviewed, but it is the primary literature for the
technique, and the algorithms in Kalles Fraktaler and comparable renderers
derive from it.

Perturbation's principal failure mode is the **glitch**: pixels for which the
chosen reference stops being a valid linearisation, producing visibly wrong
output. Two mitigations are evaluated in this thesis. *Multiple references*
partition the frame among several reference orbits so that one poor choice
cannot corrupt the whole image; *rebasing* restarts a diverging pixel's
perturbation from the current orbit point rather than falling back to
full-precision iteration. Both are described in [11].

The high-precision reference orbit itself is computed here using
**double-double arithmetic**, which represents a number as an unevaluated sum of
two doubles and achieves roughly 32 decimal digits without arbitrary-precision
software. The underlying error-free transformations — `two-sum` and `two-prod` —
are due to Dekker [12] and Knuth [13]; the modern algorithmic treatment,
including the quad-double generalisation, is by Hida, Li and Bailey [14].
Shewchuk [15] gives a careful account of the same primitives in the context of
robust geometric predicates.

## 3.3 GPU execution and its costs

GPUs execute in a SIMT model: threads are grouped into warps that share an
instruction pointer. When threads within a warp take different branches the warp
executes both paths serially with inactive lanes masked — **branch divergence**.
The model is described by Lindholm et al. [16] and Nickolls et al. [17];
hardware mitigations such as dynamic warp formation were proposed by Fung et
al. [18]. Divergence is the central reason escape-time fractals are not a
uniformly good fit for GPUs: interior pixels iterate to the cap while adjacent
exterior pixels escape immediately, so boundary regions leave most lanes idle.
This observation motivates the scheduler evaluated in this thesis.

A second cost is more mundane and, as this thesis finds, dominant. Data must
cross PCIe to reach the host. Gregg and Hazelwood [19] argue directly that GPU
speedups reported without accounting for transfer are not meaningful, and Lee et
al. [20] re-evaluated a set of widely-cited GPU-versus-CPU results and found
reported speedups substantially reduced once both platforms were optimised
comparably. Both findings are reproduced here: §6.x measures the host readback
at 82–88% of a GPU frame, and shows that the apparent 4.4× GPU advantage falls
to 2.4× when compared against the CPU's own best vectorised path and 2.0×
end-to-end.

Memory access order also matters. This implementation dispatches CUDA work in
Morton (Z-order) sequence [21] and evaluates Hilbert-curve [22] tile traversal
on the CPU; §6.x reports that neither improves on row-major traversal at the
working-set sizes involved, and that Hilbert traversal is measurably worse.

The **roofline model** of Williams, Waterman and Patterson [23] provides the
standard framing for whether a kernel is compute- or bandwidth-bound; the
readback analysis in §6.x is essentially a roofline argument applied to the
PCIe link rather than to device memory.

## 3.4 Heterogeneous scheduling

Distributing work across processors of different speeds is a classical
scheduling problem. **HEFT** [24] is the canonical list-scheduling heuristic for
heterogeneous processors, though it assumes a task graph with known costs — an
assumption escape-time rendering violates, since a tile's cost is not known
until it has been computed.

Several runtime systems address heterogeneity directly. **StarPU** [25]
provides a unified task abstraction with pluggable scheduling policies and
automatic data movement between CPU and accelerator memories. **Qilin** [26] is
the closest antecedent to the approach taken here: it builds a performance model
from runtime profiling and uses it to decide the CPU/GPU split adaptively, rather
than fixing it ahead of time. **OmpSs** [27] extends OpenMP-style directives with
task dependencies across heterogeneous devices, and **Merge** [28] proposes a
map-reduce-style programming model for heterogeneous multicore.

The scheduler described in Chapter 5 is best understood as a domain-specialised
instance of the Qilin idea. Rather than a general task graph it uses a
domain-specific cost *predictor* — corner sampling of the escape-time field,
which is cheap because the field is continuous almost everywhere — and closes
the loop with a PI controller over observed CPU/GPU finish times, in the spirit
of Qilin's adaptive mapping. Its dynamic load balancing uses **work stealing**,
the theoretical properties of which were established by Blumofe and Leiserson
[29] and realised in the Cilk runtime [30]; the implementation here uses
Rust's Rayon library [31], which follows the same design.

Two properties distinguish this work's scheduler from the systems above. First,
stealing is *bidirectional across device boundaries*: idle CPU workers claim
tiles reserved from the GPU's queue, and the GPU issues a second dispatch for
tiles the CPU never reached. General runtimes typically steal within a device
class. Second, the classification is *free of a prepass* — corner sampling
reuses the same kernel the render itself uses, so no separate profiling pass is
required.

## 3.5 Performance modelling and the gap this thesis addresses

Amdahl's law [32] bounds the speedup available from parallelising part of a
computation; Gustafson [33] gives the complementary scaled-speedup view. Both
apply here: §6.x shows that a CPU-side colour-mapping stage, which no backend
accelerates, caps the achievable end-to-end gain at roughly 2× regardless of how
fast the iteration kernel becomes.

**The gap.** The literature establishes that heterogeneous scheduling *can*
help, and provides runtimes that implement it. What it is less explicit about is
the *precondition*: heterogeneous execution is only worthwhile when the
processors' throughputs are within a small factor of one another, since the
achievable combined time is bounded below by the harmonic combination
`1/(1/T_cpu + 1/T_gpu)`. When one device is an order of magnitude faster, no
scheduling policy recovers a meaningful fraction of the slower device's
contribution, and the machinery costs more than it returns.

This thesis makes that precondition concrete for one workload. It shows that for
escape-time fractal rendering on consumer hardware the CPU/GPU throughput ratio
is not a fixed property of the machine but a function of **required numerical
precision**: below a zoom threshold the GPU uses single precision and is roughly
twenty times the CPU, above it the GPU falls back to double precision — which
consumer GPUs execute at a small fraction of their single-precision rate — and
the two become comparable. The scheduler's viability therefore switches at a
threshold that is a property of the *problem*, not of the hardware alone, and
that threshold is identified and measured here.

---

## References

Verify all entries before submission; see the warning at the head of this chapter.

[1] B. B. Mandelbrot, *The Fractal Geometry of Nature*. W. H. Freeman, 1982.
    (The earlier note "Fractal aspects of the iteration of z → λz(1−z) for complex λ and z",
    *Annals of the New York Academy of Sciences*, 357(1), 1980, is the primary source
    for the set itself. ⚠ check which you want to cite.)

[2] A. Douady and J. H. Hubbard, "Étude dynamique des polynômes complexes"
    (the *Orsay Notes*), Publications Mathématiques d'Orsay, 1984–1985.

[3] M. Shishikura, "The Hausdorff dimension of the boundary of the Mandelbrot set
    and Julia sets", *Annals of Mathematics*, 147(2), 1998.

[4] Í. Quílez, "Smooth iteration count for generalized Mandelbrot sets".
    Available: https://iquilezles.org/articles/msetsmooth/ **[non-peer-reviewed]**
    ⚠ Alternative/older attributions exist for the normalised iteration count;
    check whether your supervisor prefers a textbook source.

[5] R. P. Brent, "An improved Monte Carlo factorization algorithm",
    *BIT Numerical Mathematics*, 20(2), 1980. (Source of the cycle-detection
    algorithm used here.)

[6] R. Mariani and B. Silver — border-tracing / rectangle-subdivision rendering.
    ⚠ **No canonical peer-reviewed citation located.** Commonly attributed in
    fractal-rendering documentation and implementations. Consider citing a
    concrete implementation's documentation, or describing it as folklore with a
    footnote. **[non-peer-reviewed]**

[7] H.-O. Peitgen and D. Saupe (eds.), *The Science of Fractal Images*.
    Springer-Verlag, 1988. (Exterior distance estimation for the Mandelbrot set.)

[8] J. C. Hart, D. J. Sandin and L. H. Kauffman, "Ray tracing deterministic 3-D
    fractals", *Computer Graphics* (SIGGRAPH '89), 23(3), 1989.

[9] C. Heiland-Allen, "Practical interior distance rendering", 2014.
    https://mathr.co.uk/blog/2014-11-02_practical_interior_distance_rendering.html
    **[non-peer-reviewed]**

[10] K. I. Martin, "Superfractalthing Maths", 2013. **[non-peer-reviewed]**
     ⚠ Circulated as an unpublished note; cite the URL you actually consulted.

[11] C. Heiland-Allen, "Deep zoom theory and practice", 2021.
     https://mathr.co.uk/blog/2021-05-14_deep_zoom_theory_and_practice.html
     **[non-peer-reviewed]**

[12] T. J. Dekker, "A floating-point technique for extending the available
     precision", *Numerische Mathematik*, 18(3), 1971.

[13] D. E. Knuth, *The Art of Computer Programming, Volume 2: Seminumerical
     Algorithms*. Addison-Wesley. (Error-free two-sum.)

[14] Y. Hida, X. S. Li and D. H. Bailey, "Algorithms for quad-double precision
     floating point arithmetic", *Proc. 15th IEEE Symposium on Computer
     Arithmetic (ARITH-15)*, 2001.

[15] J. R. Shewchuk, "Adaptive precision floating-point arithmetic and fast
     robust geometric predicates", *Discrete & Computational Geometry*, 18(3), 1997.

[16] E. Lindholm, J. Nickolls, S. Oberman and J. Montrym, "NVIDIA Tesla: A unified
     graphics and computing architecture", *IEEE Micro*, 28(2), 2008.

[17] J. Nickolls, I. Buck, M. Garland and K. Skadron, "Scalable parallel
     programming with CUDA", *ACM Queue*, 6(2), 2008.

[18] W. W. L. Fung, I. Sham, G. Yuan and T. M. Aamodt, "Dynamic warp formation and
     scheduling for efficient GPU control flow", *Proc. 40th IEEE/ACM International
     Symposium on Microarchitecture (MICRO-40)*, 2007.

[19] C. Gregg and K. Hazelwood, "Where is the data? Why you cannot debate CPU vs.
     GPU performance without the answer", *Proc. IEEE International Symposium on
     Performance Analysis of Systems and Software (ISPASS)*, 2011.

[20] V. W. Lee et al., "Debunking the 100X GPU vs. CPU myth: an evaluation of
     throughput computing on CPU and GPU", *Proc. 37th International Symposium on
     Computer Architecture (ISCA)*, 2010.

[21] G. M. Morton, "A computer oriented geodetic data base and a new technique in
     file sequencing", IBM Technical Report, 1966.

[22] D. Hilbert, "Über die stetige Abbildung einer Linie auf ein Flächenstück",
     *Mathematische Annalen*, 38(3), 1891.

[23] S. Williams, A. Waterman and D. Patterson, "Roofline: an insightful visual
     performance model for multicore architectures", *Communications of the ACM*,
     52(4), 2009.

[24] H. Topcuoglu, S. Hariri and M.-Y. Wu, "Performance-effective and
     low-complexity task scheduling for heterogeneous computing", *IEEE
     Transactions on Parallel and Distributed Systems*, 13(3), 2002.

[25] C. Augonnet, S. Thibault, R. Namyst and P.-A. Wacrenier, "StarPU: a unified
     platform for task scheduling on heterogeneous multicore architectures",
     *Concurrency and Computation: Practice and Experience*, 23(2), 2011.
     (Earlier version: Euro-Par 2009.)

[26] C.-K. Luk, S. Hong and H. Kim, "Qilin: exploiting parallelism on heterogeneous
     multiprocessors with adaptive mapping", *Proc. 42nd IEEE/ACM International
     Symposium on Microarchitecture (MICRO-42)*, 2009.

[27] A. Duran et al., "OmpSs: a proposal for programming heterogeneous multi-core
     architectures", *Parallel Processing Letters*, 21(2), 2011.

[28] M. D. Linderman, J. D. Collins, H. Wang and T. H. Meng, "Merge: a programming
     model for heterogeneous multi-core systems", *Proc. ASPLOS*, 2008.

[29] R. D. Blumofe and C. E. Leiserson, "Scheduling multithreaded computations by
     work stealing", *Journal of the ACM*, 46(5), 1999. (Earlier: FOCS 1994.)

[30] R. D. Blumofe, C. F. Joerg, B. C. Kuszmaul, C. E. Leiserson, K. H. Randall and
     Y. Zhou, "Cilk: an efficient multithreaded runtime system", *Proc. PPoPP*, 1995.

[31] N. Matsakis and J. Stone, *Rayon: a data parallelism library for Rust*.
     https://github.com/rayon-rs/rayon **[software]**

[32] G. M. Amdahl, "Validity of the single processor approach to achieving large
     scale computing capabilities", *Proc. AFIPS Spring Joint Computer Conference*, 1967.

[33] J. L. Gustafson, "Reevaluating Amdahl's law", *Communications of the ACM*,
     31(5), 1988.

### Also cited in the methodology chapter

[34] J. Heisey et al., *Criterion.rs: statistics-driven benchmarking for Rust*.
     https://github.com/bheisler/criterion.rs **[software]**
     ⚠ check the maintainer attribution you want to use.

[35] NVIDIA Corporation, *CUDA C++ Programming Guide*. **[vendor documentation]**
     Cite the version you actually used.
