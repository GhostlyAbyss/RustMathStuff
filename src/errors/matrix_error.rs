use crate::errors::CommonError::CommonError;

#[derive(Debug)]
pub enum MatrixError{
    CommonError(CommonError),
    NotSquaredMatrix,
    NotInverseable
}