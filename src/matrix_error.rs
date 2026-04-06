#[derive(Debug)]
pub enum MatrixError{
    DimensionMismatch,
    NotSquaredMatrix,
    NotInverseable
}