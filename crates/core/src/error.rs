use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum StructuraError {
    InvalidInput(String),
    Unsupported(&'static str),
    External(String),
}

impl Display for StructuraError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported operation: {message}"),
            Self::External(message) => write!(f, "external error: {message}"),
        }
    }
}

impl Error for StructuraError {}
