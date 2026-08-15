//! Double-double ("f128") arithmetic: represents a real number as the
//! unevaluated sum of two f64 values (a high part and a low correction
//! term), giving roughly twice the mantissa precision of f64 — about 106
//! bits, versus f64's 53 — using only native f64 operations.
//!
//! Reference-orbit computation for perturbation-theory rendering needs
//! more precision than f64 offers once the zoom is deep enough that the
//! reference point's coordinates and the perturbation delta differ by more
//! bits than f64 can represent simultaneously. A true arbitrary-precision
//! library would work too but is far more expensive per operation;
//! double-double is the standard middle ground for deep-zoom fractal
//! rendering because it stays entirely in hardware floating-point.

/// A number represented as `hi + lo`, where `hi` holds the value to f64
/// precision and `lo` holds the rounding error `hi` alone would have
/// discarded. Invariant: `|lo| <= ulp(hi)/2`.
#[derive(Clone, Copy)]
pub(crate) struct DoubleDouble(f64, f64); // (hi, lo)

impl DoubleDouble {
    #[inline] pub(crate) fn from_f64(x: f64) -> Self { DoubleDouble(x, 0.0) }
    #[inline] pub(crate) fn hi(self) -> f64 { self.0 }
}

/// Knuth's `TwoSum`: computes `a + b` exactly as a double-double, i.e.
/// `s` is the f64-rounded sum and the second component recovers the exact
/// rounding error, with no branch on the relative magnitudes of `a` and `b`.
#[inline]
fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    let v = s - a;
    DoubleDouble(s, (a - (s - v)) + (b - v))
}

/// `TwoProduct` via FMA: `a * b` computed exactly as a double-double. The
/// hardware fused multiply-add computes `a*b - p` with the true
/// (unrounded) product, which is exactly the rounding error `p` lost —
/// no double-width intermediate representation needed.
#[inline]
fn two_prod(a: f64, b: f64) -> DoubleDouble {
    let p = a * b;
    DoubleDouble(p, a.mul_add(b, -p))
}

impl std::ops::Add for DoubleDouble {
    type Output = DoubleDouble;
    /// Sums the high parts exactly via [`two_sum`], folds both low parts
    /// into the resulting error term, then renormalizes with a second
    /// [`two_sum`] so the result again satisfies the `|lo| <= ulp(hi)/2`
    /// invariant.
    fn add(self, b: DoubleDouble) -> DoubleDouble {
        let s = two_sum(self.0, b.0);
        let e = s.1 + (self.1 + b.1);
        two_sum(s.0, e)
    }
}

impl std::ops::Sub for DoubleDouble {
    type Output = DoubleDouble;
    fn sub(self, b: DoubleDouble) -> DoubleDouble { self + DoubleDouble(-b.0, -b.1) }
}

impl std::ops::Mul for DoubleDouble {
    type Output = DoubleDouble;
    /// Exact high*high product via [`two_prod`], plus the cross terms
    /// `hi*lo` (each already below the precision the result can represent,
    /// so computed as plain f64 products) folded in and renormalized.
    fn mul(self, b: DoubleDouble) -> DoubleDouble {
        let p = two_prod(self.0, b.0);
        let e = p.1 + self.0 * b.1 + self.1 * b.0;
        two_sum(p.0, e)
    }
}

impl std::ops::Mul<DoubleDouble> for f64 {
    type Output = DoubleDouble;
    fn mul(self, b: DoubleDouble) -> DoubleDouble { DoubleDouble::from_f64(self) * b }
}

impl PartialEq for DoubleDouble { fn eq(&self, o: &DoubleDouble) -> bool { self.0 == o.0 && self.1 == o.1 } }
impl PartialOrd for DoubleDouble {
    fn partial_cmp(&self, o: &DoubleDouble) -> Option<std::cmp::Ordering> {
        match self.0.partial_cmp(&o.0) {
            Some(std::cmp::Ordering::Equal) => self.1.partial_cmp(&o.1),
            other => other,
        }
    }
}