use thiserror::Error;

/// Result type used throughout the crate.
pub type Result<T> = std::result::Result<T, SQuaJLError>;

/// Error type for `clark-hash`.
#[derive(Debug, Error)]
pub enum SQuaJLError {
    /// The provided configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The input vector has the wrong dimension.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// The vector to encode or sketch was empty or had zero norm.
    #[error("vector norm must be positive")]
    ZeroNorm,
    /// A quantized code does not match the codec configuration.
    #[error("incompatible quantized vector: {0}")]
    IncompatibleCode(String),
    /// Backend integration failed.
    #[cfg(feature = "fastembed")]
    #[error("backend error: {0}")]
    Backend(String),
}
