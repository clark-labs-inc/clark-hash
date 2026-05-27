use crate::bitpack::PackedCodes;
use crate::config::{SQuaJLConfig, SimilarityMetric};
use crate::error::{Result, SQuaJLError};
use crate::hash::splitmix64;
use crate::quantized::{QuantizedVector, QuerySketch};

/// Stateless sparse signed JL quantizer.
///
/// `SQuaJL` never learns codebooks or corpus-dependent parameters. Every vector is
/// encoded independently using the same seeded sparse projection and the same fixed
/// scalar quantizer.
#[derive(Debug, Clone)]
pub struct SQuaJL {
    config: SQuaJLConfig,
    levels_minus_one: u16,
}

impl SQuaJL {
    /// Creates a new codec from the provided configuration.
    pub fn new(config: SQuaJLConfig) -> Result<Self> {
        config.validate()?;
        let levels_minus_one = (1_u16 << config.bits) - 1;
        Ok(Self {
            config,
            levels_minus_one,
        })
    }

    /// Returns the immutable configuration.
    pub fn config(&self) -> &SQuaJLConfig {
        &self.config
    }

    /// Returns the expected input dimension.
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Returns the sketch dimension.
    pub fn sketch_dim(&self) -> usize {
        self.config.sketch_dim
    }

    /// Returns the number of bits per quantized coordinate.
    pub fn bits(&self) -> u8 {
        self.config.bits
    }

    /// Returns the similarity metric configured for scoring.
    pub fn metric(&self) -> SimilarityMetric {
        self.config.metric
    }

    /// Returns the number of bytes required to store one quantized vector.
    pub fn storage_bytes_per_vector(&self) -> usize {
        let packed = (self.config.sketch_dim * self.config.bits as usize).div_ceil(8);
        packed
            + match self.config.metric {
                SimilarityMetric::Cosine => 0,
                SimilarityMetric::Dot => 2,
            }
    }

    /// Returns the compressed size divided by the original dense `f32` size.
    pub fn compression_ratio_vs_f32(&self) -> f32 {
        let original = (self.config.input_dim * std::mem::size_of::<f32>()) as f32;
        self.storage_bytes_per_vector() as f32 / original
    }

    /// Encodes a single dense embedding.
    pub fn encode(&self, input: &[f32]) -> Result<QuantizedVector> {
        self.validate_input(input)?;
        let mut sketch = vec![0.0_f32; self.config.sketch_dim];
        let norm = self.project_direction(input, &mut sketch)?;
        let mut codes = PackedCodes::new(self.config.bits, self.config.sketch_dim);

        for (index, &value) in sketch.iter().enumerate() {
            codes.set(index, self.quantize_scalar(value));
        }

        let encoded_norm = match self.config.metric {
            SimilarityMetric::Cosine => None,
            SimilarityMetric::Dot => Some(self.encode_norm(norm)),
        };

        Ok(QuantizedVector {
            sketch_dim: self.config.sketch_dim,
            bits: self.config.bits,
            codes,
            encoded_norm,
        })
    }

    /// Encodes a batch of dense embeddings.
    pub fn encode_batch<'a, I>(&self, inputs: I) -> Result<Vec<QuantizedVector>>
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        inputs.into_iter().map(|input| self.encode(input)).collect()
    }

    /// Converts a raw query embedding into a floating-point sketch for asymmetric
    /// scoring.
    pub fn sketch_query(&self, input: &[f32]) -> Result<QuerySketch> {
        self.validate_input(input)?;
        let mut sketch = vec![0.0_f32; self.config.sketch_dim];
        let input_norm = self.project_direction(input, &mut sketch)?;
        Ok(QuerySketch {
            values: sketch,
            input_norm,
        })
    }

    /// Scores a prepared query sketch against a quantized vector.
    pub fn score(&self, query: &QuerySketch, code: &QuantizedVector) -> Result<f32> {
        self.validate_code(code)?;
        if query.values.len() != self.config.sketch_dim {
            return Err(SQuaJLError::DimensionMismatch {
                expected: self.config.sketch_dim,
                actual: query.values.len(),
            });
        }

        let mut acc = 0.0_f32;
        for index in 0..self.config.sketch_dim {
            let value = self.dequantize_scalar(code.codes.get(index));
            acc += query.values[index] * value;
        }

        let approx_cosine = acc / self.config.sketch_dim as f32;
        match self.config.metric {
            SimilarityMetric::Cosine => Ok(approx_cosine),
            SimilarityMetric::Dot => {
                let encoded_norm = code.encoded_norm.ok_or_else(|| {
                    SQuaJLError::IncompatibleCode(
                        "dot-product scoring requires the norm channel".to_owned(),
                    )
                })?;
                Ok(approx_cosine * query.input_norm * self.decode_norm(encoded_norm))
            }
        }
    }

    /// Convenience method that sketches the query internally and then scores.
    pub fn score_from_input(&self, query: &[f32], code: &QuantizedVector) -> Result<f32> {
        let query = self.sketch_query(query)?;
        self.score(&query, code)
    }

    /// Returns the dequantized sketch-space vector for inspection or debugging.
    ///
    /// This does **not** reconstruct the original embedding. It only returns the
    /// dequantized low-dimensional sketch used during scoring.
    pub fn decode_sketch(&self, code: &QuantizedVector) -> Result<Vec<f32>> {
        self.validate_code(code)?;
        let mut out = vec![0.0_f32; self.config.sketch_dim];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = self.dequantize_scalar(code.codes.get(index));
        }
        Ok(out)
    }

    /// Validates that the provided quantized code matches this codec.
    pub fn validate_code(&self, code: &QuantizedVector) -> Result<()> {
        if code.sketch_dim != self.config.sketch_dim {
            return Err(SQuaJLError::IncompatibleCode(format!(
                "sketch dimension mismatch: expected {}, got {}",
                self.config.sketch_dim, code.sketch_dim
            )));
        }
        if code.bits != self.config.bits {
            return Err(SQuaJLError::IncompatibleCode(format!(
                "bit width mismatch: expected {}, got {}",
                self.config.bits, code.bits
            )));
        }
        if code.codes.len() != self.config.sketch_dim || code.codes.bits() != self.config.bits {
            return Err(SQuaJLError::IncompatibleCode(
                "packed code layout does not match codec".to_owned(),
            ));
        }
        if matches!(self.config.metric, SimilarityMetric::Dot) && code.encoded_norm.is_none() {
            return Err(SQuaJLError::IncompatibleCode(
                "dot-product codec expects an encoded norm".to_owned(),
            ));
        }
        Ok(())
    }

    fn validate_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.config.input_dim {
            return Err(SQuaJLError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: input.len(),
            });
        }
        Ok(())
    }

    fn project_direction(&self, input: &[f32], out: &mut [f32]) -> Result<f32> {
        debug_assert_eq!(out.len(), self.config.sketch_dim);
        out.fill(0.0);

        let sparse_scale = 1.0_f32 / (self.config.hashes_per_input as f32).sqrt();
        let mut norm_sq = 0.0_f32;

        for (dimension, &value) in input.iter().enumerate() {
            norm_sq += value * value;

            for repetition in 0..self.config.hashes_per_input {
                let (bucket, sign) = self.bucket_and_sign(dimension, repetition);
                out[bucket] += sign * value * sparse_scale;
            }
        }

        let norm = norm_sq.sqrt();
        if norm <= f32::EPSILON {
            return Err(SQuaJLError::ZeroNorm);
        }

        let post_scale = (self.config.sketch_dim as f32).sqrt() / norm;
        for coordinate in out.iter_mut() {
            *coordinate *= post_scale;
        }

        Ok(norm)
    }

    fn bucket_and_sign(&self, dimension: usize, repetition: u8) -> (usize, f32) {
        let key = self.config.seed
            ^ (dimension as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (repetition as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let hash = splitmix64(key);
        let bucket = (hash % self.config.sketch_dim as u64) as usize;
        let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
        (bucket, sign)
    }

    fn quantize_scalar(&self, value: f32) -> u8 {
        let clipped = value.clamp(-self.config.clip, self.config.clip);
        let normalized = (clipped + self.config.clip) / (2.0 * self.config.clip);
        let level = (normalized * self.levels_minus_one as f32).round();
        level.clamp(0.0, self.levels_minus_one as f32) as u8
    }

    fn dequantize_scalar(&self, level: u8) -> f32 {
        let normalized = level as f32 / self.levels_minus_one as f32;
        normalized * (2.0 * self.config.clip) - self.config.clip
    }

    fn encode_norm(&self, norm: f32) -> u16 {
        let min = self.config.norm_log2_min;
        let max = self.config.norm_log2_max;
        let log2_norm = norm.max(f32::MIN_POSITIVE).log2().clamp(min, max);
        let t = (log2_norm - min) / (max - min);
        (t * u16::MAX as f32).round() as u16
    }

    fn decode_norm(&self, code: u16) -> f32 {
        let min = self.config.norm_log2_min;
        let max = self.config.norm_log2_max;
        let t = code as f32 / u16::MAX as f32;
        let log2_norm = min + t * (max - min);
        2.0_f32.powf(log2_norm)
    }
}
