#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Clark Hash: stateless sparse-JL quantization for neural embeddings.
//!
//! The library centers around [`ClarkHash`], a deterministic codec that projects an
//! input vector into a low-dimensional sparse signed sketch and then applies a fixed
//! scalar quantizer. The resulting code can be scored asymmetrically against
//! floating-point queries while staying fully online and fully stateless.
//!
//! The original codec and configuration names, [`SQuaJL`] and [`SQuaJLConfig`], remain
//! public for compatibility with earlier experiments and papers.
//!
//! See the crate-level `README.md` for motivation, design notes, and usage examples.

mod bitpack;
mod hash;

/// Configuration types and similarity-mode selection.
pub mod config;
/// Error types returned by the crate.
pub mod error;
/// A simple exact-scan index over quantized vectors.
pub mod index;
/// Encoded database vectors and prepared query sketches.
pub mod quantized;
/// The core stateless codec implementation.
pub mod squajl;

#[cfg(feature = "fastembed")]
/// Optional local text-embedding integration powered by `fastembed`.
pub mod fastembed_integration;

pub use config::{SQuaJLConfig, SimilarityMetric};
pub use error::{Result, SQuaJLError};
#[cfg(feature = "fastembed")]
pub use fastembed_integration::FastEmbedQuantizer;
pub use index::{FlatIndex, ScoredIndex};
pub use quantized::{QuantizedVector, QuerySketch};
pub use squajl::SQuaJL;

/// Package-level name for the stateless sparse-JL codec.
pub type ClarkHash = SQuaJL;

/// Package-level name for the codec configuration.
pub type ClarkHashConfig = SQuaJLConfig;

/// Package-level name for errors returned by this crate.
pub type ClarkHashError = SQuaJLError;
