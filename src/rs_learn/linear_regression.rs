use crate::errors::common_error::CommonError;
use crate::matrix::Matrix;
use num::{Float, Signed};
use rayon::prelude::*;

pub struct LinearRegression<T: Float + Copy + Send + Sync + Signed> {
    coef_: Vec<T>,
    interceptor_: T,
    n_jobs: isize,
}

impl<T: Float + Copy + Send + Sync + Signed> Default for LinearRegression<T> {
    fn default() -> Self {
        Self {
            coef_: Vec::new(),
            interceptor_: T::zero(),
            n_jobs: 1,
        }
    }
}

impl<T: Float + Copy + Send + Sync + Signed> LinearRegression<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn n_jobs(mut self, n_jobs: isize) -> Self {
        self.n_jobs = match n_jobs {
            -1 => num_cpus::get() as isize,
            n if n > 0 => n,
            _ => 1,
        };
        self
    }

    pub fn fit(&mut self, x: &Matrix<T>, y: &Vec<T>) -> Result<(), CommonError> {
        let n = x.rows;
        let d = x.cols;

        if y.len() != n {
            return Err(CommonError::DimensionMismatch);
        }

        let mut x_bias = Matrix::new(n, d + 1);

        for i in 0..n {
            for j in 0..d {
                x_bias.matrix[i * (d + 1) + j] = x.matrix[i * d + j];
            }
            x_bias.matrix[i * (d + 1) + d] = T::one();
        }

        let xt = x_bias.transposed_matrix();
        let xtx = xt.mul_matrix(&x_bias).unwrap();

        let y_mat = Matrix::from_vec(n, 1, y.clone()).unwrap();
        let xty = xt.mul_matrix(&y_mat).unwrap();

        let xtx_inv = xtx.inverse_matrix().unwrap();
        let w = xtx_inv.mul_matrix(&xty).unwrap();

        self.coef_ = w.matrix[..d].to_vec();
        self.interceptor_ = w.matrix[d];

        Ok(())
    }

    pub fn predict(&self, x: &[T]) -> T {
        let mut sum = self.interceptor_;

        for (w, xi) in self.coef_.iter().zip(x.iter()) {
            sum = sum + (*w * *xi);
        }

        sum
    }

    pub fn predict_multiple(&self, x: &Vec<Vec<T>>) -> Vec<T> {
        let mut res = vec![T::zero(); x.len()];

        if self.n_jobs > 1 {
            res.par_iter_mut().zip(x.par_iter()).for_each(|(r, x)| {
                *r = self.predict(x);
            });
        } else {
            for (i, x) in x.iter().enumerate() {
                res[i] = self.predict(x);
            }
        }

        res
    }

    fn mean(&self, data: &Vec<T>) -> T {
        let sum: T = data.iter().copied().fold(T::zero(), |a, b| a + b);
        sum / T::from(data.len()).unwrap()
    }


    pub fn score(&self, x: &Matrix<T>, y: &Vec<T>) -> Result<f32, CommonError> {
        if x.rows != y.len() {
            return Err(CommonError::DimensionMismatch);
        }

        let y_mean = self.mean(y);

        let mut ss_res = T::zero();
        let mut ss_tot = T::zero();

        for i in 0..x.rows {
            let mut x_row_sum = self.interceptor_;

            for j in 0..x.cols {
                x_row_sum = x_row_sum + self.coef_[j] * x.matrix[i * x.cols + j];
            }

            let y_pred = x_row_sum;
            let y_true = y[i];

            ss_res = ss_res + (y_true - y_pred).powi(2);
            ss_tot = ss_tot + (y_true - y_mean).powi(2);
        }

        let r2 = T::one() - ss_res / ss_tot;

        Ok(r2.to_f32().unwrap())
    }
}
