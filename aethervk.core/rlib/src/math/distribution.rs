//! distribution module.

use alloc::vec::Vec;

#[derive(Debug, Clone)]
/// TODO: Document this item
pub struct Distribution1D {
  pub func: Vec<f32>,
  pub cdf: Vec<f32>,
  pub func_int: f32,
}

impl Distribution1D {
  /// TODO: Document this item
  pub fn new(f: &[f32]) -> Self {
    let n = f.len();
    let mut func = Vec::with_capacity(n);
    let mut cdf = Vec::with_capacity(n + 1);
    func.extend_from_slice(f);
    cdf.push(0.0);

    let mut sum = 0.0;
    let inv_n = 1.0 / (n as f32);
    for i in 1..=n {
      sum += func[i - 1] * inv_n;
      cdf.push(sum);
    }

    let func_int = sum;
    if func_int == 0.0 {
      for i in 1..=n {
        cdf[i] = (i as f32) * inv_n;
      }
    } else {
      for i in 1..=n {
        cdf[i] /= func_int;
      }
    }

    Self {
      func,
      cdf,
      func_int,
    }
  }

  /// TODO: Document this item
  pub fn count(&self) -> usize {
    self.func.len()
  }

  /// Sample continuous distribution, returning the sampled position in [0,1],
  /// the pdf, and the offset index.
  pub fn sample_continuous(&self, u: f32) -> (f32, f32, usize) {
    let offset = self.sample_discrete(u);

    let du = u - self.cdf[offset];
    let dt = self.cdf[offset + 1] - self.cdf[offset];
    let mut t = if dt > 0.0 { du / dt } else { 0.0 };
    t = t.clamp(0.0, 1.0);

    let pdf = if self.func_int > 0.0 {
      self.func[offset] / self.func_int
    } else {
      0.0
    };

    let x = (offset as f32 + t) / (self.count() as f32);
    (x, pdf, offset)
  }

  /// Sample discrete returns index offset
  pub fn sample_discrete(&self, u: f32) -> usize {
    let offset = self.cdf.partition_point(|&c| c <= u).saturating_sub(1);
    offset.min(self.count().saturating_sub(1))
  }
}

#[derive(Debug, Clone)]
/// TODO: Document this item
pub struct Distribution2D {
  pub conditional_v: Vec<Distribution1D>,
  pub marginal: Distribution1D,
}

impl Distribution2D {
  /// TODO: Document this item
  pub fn new(data: &[f32], nu: usize, nv: usize) -> Self {
    let mut conditional_v = Vec::with_capacity(nv);
    for v in 0..nv {
      let row = &data[v * nu..(v + 1) * nu];
      conditional_v.push(Distribution1D::new(row));
    }

    let mut marginal_func = Vec::with_capacity(nv);
    for v in 0..nv {
      marginal_func.push(conditional_v[v].func_int);
    }
    let marginal = Distribution1D::new(&marginal_func);

    Self {
      conditional_v,
      marginal,
    }
  }

  /// Samples the 2D distribution. Returns the sampled (u, v) in [0, 1]^2 and the pdf.
  pub fn sample_continuous(&self, u: &[f32; 2]) -> (f32, f32, f32) {
    let (v_sampled, pdf_v, v_offset) = self.marginal.sample_continuous(u[1]);
    let (u_sampled, pdf_u, _) = self.conditional_v[v_offset].sample_continuous(u[0]);
    let pdf = pdf_v * pdf_u;
    (u_sampled, v_sampled, pdf)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_distribution_1d() {
    let weights = [1.0, 2.0, 1.0];
    let dist = Distribution1D::new(&weights);

    assert_eq!(dist.count(), 3);

    // func_int should be (1/3 + 2/3 + 1/3) = 4/3
    let expected_int = 4.0 / 3.0;
    assert!((dist.func_int - expected_int).abs() < 1e-5);

    // CDF
    // 0: 0.0
    // 1: (1/3) / (4/3) = 0.25
    // 2: (1/3 + 2/3) / (4/3) = 0.75
    // 3: 1.0
    assert!((dist.cdf[0] - 0.0).abs() < 1e-5);
    assert!((dist.cdf[1] - 0.25).abs() < 1e-5);
    assert!((dist.cdf[2] - 0.75).abs() < 1e-5);
    assert!((dist.cdf[3] - 1.0).abs() < 1e-5);

    // Sampling
    let (x, pdf, offset) = dist.sample_continuous(0.5);
    assert_eq!(offset, 1); // 0.5 is between 0.25 and 0.75
    assert!((pdf - 2.0 / expected_int).abs() < 1e-5);

    // t = (0.5 - 0.25) / (0.75 - 0.25) = 0.25 / 0.5 = 0.5
    // x = (1 + 0.5) / 3 = 0.5
    assert!((x - 0.5).abs() < 1e-5);
  }

  #[test]
  fn test_distribution_2d() {
    let weights = [1.0, 0.0, 0.0, 1.0];
    let dist = Distribution2D::new(&weights, 2, 2);

    let (u, v, pdf) = dist.sample_continuous(&[0.2, 0.2]);
    assert!(pdf > 0.0);
    assert!(u <= 0.5);
    assert!(v <= 0.5);

    let (u2, v2, pdf2) = dist.sample_continuous(&[0.8, 0.8]);
    assert!(pdf2 > 0.0);
    assert!(u2 > 0.5);
    assert!(v2 > 0.5);
  }
}
