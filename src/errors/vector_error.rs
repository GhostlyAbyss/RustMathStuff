use crate::errors::CommonError::CommonError;

#[derive(Debug)]
pub enum VectorError {
    CommonError(CommonError),
    LenIsZero
}