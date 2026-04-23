use std::fmt;

#[derive(Debug)]
pub enum SplitError {
    StratifyNotImplementedForShuffle,
    InvalidInput(String),
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplitError::StratifyNotImplementedForShuffle => {
                write!(f, "Stratified splitting requires shuffle=true")
            }
            SplitError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
        }
    }
}

impl std::error::Error for SplitError {}