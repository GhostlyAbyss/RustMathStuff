use crate::matrix_error::MatrixError;

#[derive(Clone)]
pub struct Matrix{
    pub matrix: Vec<f64>,
    pub rows: usize,
    pub cols: usize
}

impl Matrix{

    pub fn new(rows: usize, cols: usize) -> Self {
        Self { matrix: vec![0f64;rows*cols] , rows, cols }
    }


    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, MatrixError> {
        if data.len() != rows * cols {
            return Err(MatrixError::DimensionMismatch);
        }

        Ok(Self {
            matrix: data,
            rows,
            cols,
        })
    }

    pub fn determinant(&self) -> Result<f64, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::NotSquaredMatrix);
        }

        let n = self.rows;
        let mut mat = self.matrix.clone();
        let mut det = 1.0;

        for i in 0..n {
            let mut pivot = i;
            while pivot < n && mat[pivot * n + i] == 0.0 {
                pivot += 1;
            }

            if pivot == n {
                return Ok(0.0);
            }

            if pivot != i {
                for j in 0..n {
                    mat.swap(i * n + j, pivot * n + j);
                }
                det *= -1.0;
            }

            det *= mat[i * n + i];

            for k in i + 1..n {
                let factor = mat[k * n + i] / mat[i * n + i];
                for j in i..n {
                    mat[k * n + j] -= factor * mat[i * n + j];
                }
            }
        }

        Ok(det)
    }

    pub fn inverse_matrix(&self) -> Result<Matrix, MatrixError> {
        let det = self.determinant()?;
        if det == 0.0 {
            return Err(MatrixError::NotInverseable);
        }

        let n = self.rows;
        let mut mat = self.matrix.clone();
        let mut inv = vec![0.0; n * n];
        for i in 0..n {
            inv[i * n + i] = 1.0;
        }

        for i in 0..n {
            let mut pivot = i;
            while pivot < n && mat[pivot * n + i] == 0.0 {
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
                mat[i * n + j] /= pivot_val;
                inv[i * n + j] /= pivot_val;
            }

            for k in 0..n {
                if k != i {
                    let factor = mat[k * n + i];
                    for j in 0..n {
                        mat[k * n + j] -= factor * mat[i * n + j];
                        inv[k * n + j] -= factor * inv[i * n + j];
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

    pub fn fill_matrix(&mut self, value: f64){
        self.matrix.fill(value);
    }

    pub fn clear_matrix(&mut self){
        for x in 0..self.matrix.iter().len() {
            self.matrix[x] = 0f64;
        }
    }

    pub fn scale_matrix(&mut self, scale: f64){
        for x in 0..self.matrix.iter().len() {
            self.matrix[x] *= scale
        }
    }

    pub fn sub_matrix(&mut self, other: Matrix) -> Result<(), MatrixError>{
        if other.rows != self.rows || other.cols != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        for x in 0..self.matrix.len(){
            self.matrix[x] -= other.matrix[x]
        }

        Ok(())
    }

    pub fn transposed_matrix(&mut self) -> Matrix{
        let mut field = vec![0f64; self.cols * self.rows];
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

    pub fn sum_matrix(&self) -> f64{
        self.matrix.iter().sum()
    }

    pub fn mul_matrix(&self, other: &Matrix) -> Result<Matrix, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch)
        }

        let mut res: Vec<f64> = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.matrix[i * self.cols + k]
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

    pub fn add_matrix(&mut self, other: Matrix) -> Result<(), MatrixError> {

        if other.rows != self.rows || other.cols != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        for x in 0.. self.matrix.len() {
            self.matrix[x] += other.matrix[x]
        }
        Ok(())
    }


}