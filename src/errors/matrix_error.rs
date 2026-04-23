use std::fmt;
use crate::errors::CommonError::CommonError;

#[derive(Debug)]
pub enum MatrixError {
    CommonError(CommonError),
    NotSquaredMatrix,
    NotInverseable,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::CommonError(e) => write!(f, "{}", e),
            MatrixError::NotSquaredMatrix => write!(f, "Matrix must be square"),
            MatrixError::NotInverseable => write!(f, "Matrix is not inversable"),
        }
    }
}

impl std::error::Error for MatrixError {}

impl From<CommonError> for MatrixError {
    fn from(err: CommonError) -> Self {
        MatrixError::CommonError(err)
    }
}