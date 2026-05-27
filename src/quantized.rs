use crate::bitpack::PackedCodes;

/// A quantized database-side sketch.
///
/// The codes are bit-packed and can be scored asymmetrically against a
/// [`QuerySketch`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedVector {
    pub(crate) sketch_dim: usize,
    pub(crate) bits: u8,
    pub(crate) codes: PackedCodes,
    pub(crate) encoded_norm: Option<u16>,
}

impl QuantizedVector {
    /// Returns the number of sketch coordinates.
    pub fn sketch_dim(&self) -> usize {
        self.sketch_dim
    }

    /// Returns the number of bits used per coordinate.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Returns the packed bytes that store the quantized sketch.
    pub fn packed_bytes(&self) -> &[u8] {
        self.codes.bytes()
    }

    /// Returns the optional encoded norm channel.
    pub fn encoded_norm(&self) -> Option<u16> {
        self.encoded_norm
    }

    /// Returns the total number of bytes used by this code.
    pub fn storage_bytes(&self) -> usize {
        self.codes.bytes().len() + usize::from(self.encoded_norm.is_some()) * 2
    }
}

/// A floating-point query-side sketch used for asymmetric scoring.
#[derive(Debug, Clone, PartialEq)]
pub struct QuerySketch {
    pub(crate) values: Vec<f32>,
    pub(crate) input_norm: f32,
}

impl QuerySketch {
    /// Returns the floating-point sketch coordinates.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns the L2 norm of the original unnormalized query embedding.
    pub fn input_norm(&self) -> f32 {
        self.input_norm
    }

    /// Returns the sketch dimension.
    pub fn sketch_dim(&self) -> usize {
        self.values.len()
    }
}
