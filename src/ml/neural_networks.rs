use num::{Num, Signed};
use crate::ml::activation::Activation;
use crate::matrix::Matrix;

struct Network<T: Num + Copy + Signed>{
    layers: Vec<usize>,
    weigts: Vec<Matrix<f64>>,
    biases: Vec<Matrix<f64>>,
    data: Vec<Matrix<T>>,
    activation: Activation<T>,
    learning_rate: f64
}

impl<T: Copy + Signed + Num> Network<T>{
}