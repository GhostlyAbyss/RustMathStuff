use rayon::prelude::*;
use num::Float;
use crate::errors::CommonError::CommonError;

pub struct LinearRegression<T: Float + Copy + Send + Sync>{
    coef_: T,
    interceptor_: T,
    n_jobs: isize,
}

impl<T: Float + Copy + Send + Sync> LinearRegression<T>{

    pub fn new(n_jobs: Option<isize>) -> Self {
        let jobs = match n_jobs {
            Some(-1) => num_cpus::get() as isize,
            Some(n) if n > 0 => n,
            _ => 1,
        };

        Self {
            coef_: T::zero(),
            interceptor_: T::zero(),
            n_jobs: jobs,
        }
    }

    pub fn fit(&mut self, X: Vec<T>, y: Vec<T>) -> Result<(), CommonError>{
        if self.n_jobs > 1 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.n_jobs as usize)
                .build_global()
                .ok();
        }

        if (X.len() != y.len()){
            return Err(CommonError::DimensionMismatch);
        }
        let x_mean = self.mean(&X);
        let y_mean = self.mean(&y);

        let numerator: T = if self.n_jobs > 1 {
            X.par_iter()
                .zip(y.par_iter())
                .map(|(x, y)| (*x - x_mean) * (*y - y_mean))
                .reduce(|| T::zero(), |a, b| a + b)
        } else {
            X.iter()
                .zip(y.iter())
                .map(|(x, y)| (*x - x_mean) * (*y - y_mean))
                .fold(T::zero(), |a, b| a + b)
        };

        let denominator: T = if self.n_jobs > 1 {
            X.par_iter()
                .map(|x| (*x - x_mean).powi(2))
                .reduce(|| T::zero(), |a, b| a + b)
        } else {
            X.iter()
                .map(|x| (*x - x_mean).powi(2))
                .fold(T::zero(), |a, b| a + b)
        };

        self.coef_ = numerator / denominator;
        self.interceptor_ = y_mean - self.coef_ * x_mean;

        Ok(())
    }

    pub fn predict(&self, x: T) -> T{
        self.coef_ * x + self.interceptor_
    }

    pub fn predict_multiple(&self, x: Vec<T>) -> Vec<T>{
        let mut res = vec![T::zero(); x.len()];

        if self.n_jobs > 1 {
            res.par_iter_mut()
                .zip(x.par_iter())
                .for_each(|(r, xi)| {
                    *r = self.coef_ * *xi + self.interceptor_;
                });
        } else {
            for i in 0..x.len() {
                res[i] = self.coef_ * x[i] + self.interceptor_;
            }
        }

        res
    }

    fn mean(&self, data: &Vec<T>) -> T{
        let mut sum = T::zero();

        for x in data.iter() {
            sum = sum + *x;
        }

        sum / T::from(data.iter().len()).unwrap()
    }

    pub fn score(&self, X: Vec<T>, y: Vec<T>) -> Result<f32, CommonError> {

        if X.len() != y.len() {
            return Err(CommonError::DimensionMismatch);
        }

        let y_mean = self.mean(&y);

        let (ss_res, ss_tot) = if self.n_jobs > 1 {

            use rayon::prelude::*;

            X.par_iter()
                .zip(y.par_iter())
                .map(|(x, y)| {
                    let y_pred = self.predict(*x);

                    let res = (*y - y_pred).powi(2);
                    let tot = (*y - y_mean).powi(2);

                    (res, tot)
                })
                .reduce(
                    || (T::zero(), T::zero()),
                    |a, b| (a.0 + b.0, a.1 + b.1),
                )

        } else {

            let mut ss_res = T::zero();
            let mut ss_tot = T::zero();

            for (x, y) in X.iter().zip(y.iter()) {

                let y_pred = self.predict(*x);

                ss_res = ss_res + (*y - y_pred).powi(2);
                ss_tot = ss_tot + (*y - y_mean).powi(2);
            }

            (ss_res, ss_tot)
        };

        let r2 = T::one() - ss_res / ss_tot;

        Ok(r2.to_f32().unwrap())
    }
}