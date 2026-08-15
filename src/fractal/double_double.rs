#[derive(Clone, Copy)]
pub(crate) struct DoubleDouble(f64, f64); // (hi, lo)

impl DoubleDouble {
    #[inline] pub(crate) fn from_f64(x: f64) -> Self { DoubleDouble(x, 0.0) }
    #[inline] pub(crate) fn hi(self) -> f64 { self.0 }
}

#[inline]
fn two_sum(a: f64, b: f64) -> DoubleDouble {
    let s = a + b;
    let v = s - a;
    DoubleDouble(s, (a - (s - v)) + (b - v))
}

#[inline]
fn two_prod(a: f64, b: f64) -> DoubleDouble {
    let p = a * b;
    DoubleDouble(p, a.mul_add(b, -p)) 
}

impl std::ops::Add for DoubleDouble {
    type Output = DoubleDouble;
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