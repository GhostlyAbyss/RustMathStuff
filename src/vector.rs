use num::{pow, Num, Signed};
use crate::errors::CommonError::CommonError::DimensionMismatch;
use crate::errors::vector_error::*;
use crate::errors::vector_error::VectorError::{CommonError, LenIsZero};

#[derive(Clone)]
struct Vector<T: Num + Copy + Signed> {
    data: Vec<T>,
}

impl<T: Num + Copy + Signed> Vector<T> {
    pub fn new(data: Vec<T>) -> Self {
        Vector { data }
    }

    pub fn new_from_points(p1: &[T], p2: &[T]) -> Self {
        let data = p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| *b - *a)
            .collect();
        Vector{
            data
        }
    }

    pub fn add_mut(&mut self, other: &Vector<T>) -> Result<(), VectorError> {
        if self.data.len() != other.data.len() {
            return Err(CommonError(DimensionMismatch));
        }

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a + *b;
        }

        Ok(())
    }

    pub fn add(&self, other: &Vector<T>) -> Result<Self, VectorError> {
        let mut result = self.clone();
        result.add_mut(other)?;
        Ok(result)
    }

    pub fn sub_mut(&mut self, other: &Vector<T>) -> Result<(), VectorError> {
        if self.data.len() != other.data.len() {
            return Err(CommonError(DimensionMismatch));
        }

        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = *a - *b;
        }

        Ok(())
    }

    pub fn sub(&self, other: &Vector<T>) -> Result<Self, VectorError> {
        let mut result = self.clone();
        result.sub_mut(other)?;
        Ok(result)
    }

    pub fn dot_mut(&mut self, other: &T){
        for a in self.data.iter_mut() {
            *a = *a * *other;
        }
    }

    pub fn dot(&self, other: &T) -> Self {
        let mut result = self.clone();
        result.dot_mut(other);
        result
    }

    pub fn div_const_mut(&mut self, other: &T){
        for a in self.data.iter_mut() {
            *a = *a / *other
        }
    }

    pub fn div_const(&self, other: &T) -> Self{
        let mut res = self.clone();
        res.div_const_mut(other);
        res
    }

    pub fn scalar(&self, other: &Vector<T>) -> Result<T, VectorError>{
        if self.data.len() != other.data.len() {
            return Err(CommonError(DimensionMismatch));
        }

        let mut sum = T::zero();

        for (&a,&b) in self.data.iter().zip(other.data.iter()){
            sum = sum + a * b
        }

        Ok(sum)
    }

    pub fn magnitude(&self) -> T{
        let mut sum = T::zero();

        for x in &self.data {
            sum = sum + (*x * *x);
        }

        pow(sum, 1/2)
    }

    pub fn angle_between(&self, other: &Vector<T>) -> Result<T, VectorError>{
        if (self.magnitude() == T::zero() || other.magnitude() == T::zero()) {
            return Err(LenIsZero)
        }
        let dividend = self.scalar(other);
        if (dividend.is_err()){
            return Err(dividend.err().unwrap())
        }
        Ok(dividend? / (self.magnitude() * other.magnitude()))
    }

    pub fn unit_vector(&self) -> Result<Vector<T>, VectorError>{
        if self.magnitude() == T::zero(){
            return Err(LenIsZero)
        }
        Ok(self.div_const(&self.magnitude()))
    }

    pub fn check_dim(&self, expected: usize) -> Result<(), VectorError> {
        if self.data.len() != expected {
            Err(CommonError(DimensionMismatch))
        } else {
            Ok(())
        }
    }

    pub fn cross_mut(&mut self, other: &Vector<T>) -> Result<(), VectorError> {
        self.check_dim(3)?;
        other.check_dim(3)?;

        let x = self.data[1] * other.data[2] - self.data[2] * other.data[1];
        let y = self.data[2] * other.data[0] - self.data[0] * other.data[2];
        let z = self.data[0] * other.data[1] - self.data[1] * other.data[0];

        self.data[0] = x;
        self.data[1] = y;
        self.data[2] = z;

        Ok(())
    }

    pub fn cross(&self, other: &Vector<T>) -> Result<Self, VectorError>{
        let mut res = self.clone();
        res.cross_mut(other)?;
        Ok(res)
    }

}
