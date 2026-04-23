use std::fmt;
use crate::errors::CommonError::CommonError;

#[derive(Debug)]
pub enum VectorError {
    CommonError(CommonError),
    LenIsZero,
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::CommonError(e) => write!(f, "{}", e),
            VectorError::LenIsZero => write!(f, "Vector length cannot be zero"),
        }
    }
}

impl std::error::Error for VectorError {}

impl From<CommonError> for VectorError {
    fn from(err: CommonError) -> Self {
        VectorError::CommonError(err)
    }
}