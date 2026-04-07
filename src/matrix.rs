use num::{Num, Signed};
use crate::errors::matrix_error::MatrixError;

#[derive(Clone)]
pub struct Matrix<T: Num + Copy + Signed>{
    pub matrix: Vec<T>,
    pub rows: usize,
    pub cols: usize
}

impl<T: Num + Copy + Signed> Matrix<T>{

    pub fn new(rows: usize, cols: usize) -> Self {
        Self { matrix: vec![T::zero();rows*cols] , rows, cols }
    }


    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self {
            matrix: data,
            rows,
            cols,
        })
    }

    pub fn determinant(&self) -> Result<T, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquaredMatrix);
        }

        let n = self.rows;
        let mut mat = self.matrix.clone();
        let mut det: T = T::one();

        for i in 0..n {
            let mut pivot = i;
            while pivot < n && mat[pivot * n + i] == T::zero() {
                pivot += 1;
            }

            if pivot == n {
                return Ok(T::zero());
            }

            if pivot != i {
                for j in 0..n {
                    mat.swap(i * n + j, pivot * n + j);
                }
                det = det * T::one().neg();
            }

            det = det * mat[i * n + i];

            for k in i + 1..n {
                let factor = mat[k * n + i] / mat[i * n + i];
                for j in i..n {
                    mat[k * n + j] = mat[k * n + j] - factor * mat[i * n + j];
                }
            }
        }

        Ok(det)
    }

    pub fn inverse_matrix(&self) -> Result<Matrix<T>, MatrixError> {
        let det = self.determinant()?;
        if det == T::zero() {
            return Err(MatrixError::NotInverseable);
        }

        let n = self.rows;
        let mut mat = self.matrix.clone();
        let mut inv = vec![T::zero(); n * n];
        for i in 0..n {
            inv[i * n + i] = T::one();
        }

        for i in 0..n {
            let mut pivot = i;
            while pivot < n && mat[pivot * n + i] == T::zero() {
                pivot += 1;
            }

            if pivot != i {
                for j in 0..n {
                    mat.swap(i * n + j, pivot * n + j);
                    inv.swap(i * n + j, pivot * n + j);
                }
            }

            let pivot_val = mat[i * n + i];
            for j in 0..n {
                mat[i * n + j] = mat[i*n+j] / pivot_val;
                inv[i * n + j] = inv[i * n + j] / pivot_val;
            }

            for k in 0..n {
                if k != i {
                    let factor = mat[k * n + i];
                    for j in 0..n {
                        mat[k * n + j] = mat[k * n + j] - factor * mat[i * n + j];
                        inv[k * n + j] = inv[k * n + j] - factor * inv[i * n + j];
                    }
                }
            }
        }

        Ok(Matrix {
            matrix: inv,
            rows: n,
            cols: n,
        })
    }

    pub fn fill_matrix(&mut self, value: T){
        self.matrix.fill(value);
    }

    pub fn clear_matrix(&mut self){
        for x in 0..self.matrix.iter().len() {
            self.matrix[x] = T::zero();
        }
    }

    pub fn scale_matrix(&mut self, scale: T){
        for x in 0..self.matrix.iter().len() {
            self.matrix[x] = self.matrix[x] * scale
        }
    }

    pub fn sub_matrix(&mut self, other: &Matrix<T>) -> Result<(), MatrixError>{
        if other.rows != self.rows || other.cols != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        for x in 0..self.matrix.len(){
            self.matrix[x] = self.matrix[x] - other.matrix[x]
        }

        Ok(())
    }

    pub fn transposed_matrix(&mut self) -> Matrix<T>{
        let mut field = vec![T::zero(); self.cols * self.rows];
        let mut field_count = 0;
        for i in 0..self.cols{
            for j in (i..self.rows).step_by(3){
                field[field_count] = self.matrix[j];
                field_count+=1;
            }
        }

        Matrix{
            matrix: field,
            rows: self.cols,
            cols: self.rows
        }

    }

    pub fn sum_matrix(&self) -> T{
        let mut sum = T::zero();
        for x in &self.matrix {
            sum = sum + *x
        }
        sum
    }

    pub fn mul_matrix(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch)
        }

        let mut res: Vec<T> = vec![T::zero(); self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::zero();

                for k in 0..self.cols {
                    sum = sum + self.matrix[i * self.cols + k]
                        * other.matrix[k * other.cols + j];
                }

                res[i * other.cols + j] = sum;
            }
        }

        Ok(Matrix {
            matrix: res,
            rows: self.rows,
            cols: other.cols,
        })
    }

    pub fn add_matrix(&mut self, other: &Matrix<T>) -> Result<(), MatrixError> {

        if other.rows != self.rows || other.cols != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        for x in 0.. self.matrix.len() {
            self.matrix[x] = self.matrix[x] + other.matrix[x]
        }
        Ok(())
    }


}