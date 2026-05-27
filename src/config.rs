use crate::error::{Result, SQuaJLError};

/// Similarity objective used by the codec.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimilarityMetric {
    /// Approximate cosine similarity.
    ///
    /// The codec stores only the quantized sketch of the normalized direction.
    /// This is usually the right choice for modern semantic embeddings.
    #[default]
    Cosine,
    /// Approximate raw inner product.
    ///
    /// The codec stores the normalized direction sketch plus a tiny quantized norm
    /// channel so the final score can recover scale information.
    Dot,
}

/// Configuration for the stateless sparse-JL quantizer.
///
/// The defaults are chosen to be a practical starting point for 384-dimensional
/// sentence embeddings such as `all-MiniLM-L6-v2`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct SQuaJLConfig {
    /// Input embedding dimension.
    pub input_dim: usize,
    /// Output sketch dimension.
    ///
    /// Larger values usually improve recall at the cost of more memory.
    pub sketch_dim: usize,
    /// Number of bits per quantized coordinate.
    ///
    /// Supported range: `1..=8`.
    pub bits: u8,
    /// Number of non-zero sketch updates per input coordinate.
    ///
    /// Larger values reduce projection noise but cost more CPU at encode time.
    pub hashes_per_input: u8,
    /// Symmetric clip range for the scaled sketch.
    ///
    /// Coordinates are clipped to `[-clip, clip]` before scalar quantization.
    pub clip: f32,
    /// Global seed used to derive sparse bucket locations and signs.
    pub seed: u64,
    /// Similarity objective for scoring.
    pub metric: SimilarityMetric,
    /// Lower bound for `log2(norm)` when the norm channel is enabled.
    pub norm_log2_min: f32,
    /// Upper bound for `log2(norm)` when the norm channel is enabled.
    pub norm_log2_max: f32,
}

impl Default for SQuaJLConfig {
    fn default() -> Self {
        Self {
            input_dim: 384,
            sketch_dim: 96,
            bits: 4,
            hashes_per_input: 4,
            clip: 3.0,
            seed: 0x5EED_CAFE_1234_5678,
            metric: SimilarityMetric::Cosine,
            norm_log2_min: -16.0,
            norm_log2_max: 16.0,
        }
    }
}

impl SQuaJLConfig {
    /// Creates a configuration with the provided input dimension and sensible defaults
    /// for the remaining fields.
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            ..Self::default()
        }
    }

    /// Sets the output sketch dimension.
    pub fn with_sketch_dim(mut self, sketch_dim: usize) -> Self {
        self.sketch_dim = sketch_dim;
        self
    }

    /// Sets the number of bits per quantized coordinate.
    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Sets the number of sparse hash updates performed for each input dimension.
    pub fn with_hashes_per_input(mut self, hashes_per_input: u8) -> Self {
        self.hashes_per_input = hashes_per_input;
        self
    }

    /// Sets the symmetric clip range used by the scalar quantizer.
    pub fn with_clip(mut self, clip: f32) -> Self {
        self.clip = clip;
        self
    }

    /// Sets the deterministic seed used by the sparse signed projection.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the similarity objective.
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the `log2(norm)` range for the optional norm channel.
    pub fn with_norm_log2_range(mut self, min: f32, max: f32) -> Self {
        self.norm_log2_min = min;
        self.norm_log2_max = max;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.input_dim == 0 {
            return Err(SQuaJLError::InvalidConfig(
                "input_dim must be greater than zero".to_owned(),
            ));
        }
        if self.sketch_dim == 0 {
            return Err(SQuaJLError::InvalidConfig(
                "sketch_dim must be greater than zero".to_owned(),
            ));
        }
        if !(1..=8).contains(&self.bits) {
            return Err(SQuaJLError::InvalidConfig(
                "bits must be between 1 and 8".to_owned(),
            ));
        }
        if self.hashes_per_input == 0 {
            return Err(SQuaJLError::InvalidConfig(
                "hashes_per_input must be greater than zero".to_owned(),
            ));
        }
        if !self.clip.is_finite() || self.clip <= 0.0 {
            return Err(SQuaJLError::InvalidConfig(
                "clip must be finite and greater than zero".to_owned(),
            ));
        }
        if !self.norm_log2_min.is_finite()
            || !self.norm_log2_max.is_finite()
            || self.norm_log2_min >= self.norm_log2_max
        {
            return Err(SQuaJLError::InvalidConfig(
                "norm_log2_min must be smaller than norm_log2_max".to_owned(),
            ));
        }
        Ok(())
    }
}
