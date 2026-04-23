use std::fmt;

#[derive(Debug)]
pub enum CommonError {
    DimensionMismatch,
}

impl fmt::Display for CommonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommonError::DimensionMismatch => {
                write!(f, "Matrix or vector dimensions do not match")
            }
        }
    }
}

impl std::error::Error for CommonError {}