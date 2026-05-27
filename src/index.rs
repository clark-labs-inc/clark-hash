use crate::error::Result;
use crate::quantized::{QuantizedVector, QuerySketch};
use crate::SQuaJL;

/// Result item returned by [`FlatIndex::search`] and [`FlatIndex::search_prepared`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredIndex {
    /// Index of the matched vector inside the flat index.
    pub index: usize,
    /// Approximate similarity score.
    pub score: f32,
}

/// Reference flat index over quantized vectors.
///
/// This type performs an exact scan over compressed vectors. It is intended for
/// evaluation, benchmarking, and simple deployments rather than large-scale ANN use.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    codec: SQuaJL,
    items: Vec<QuantizedVector>,
}

impl FlatIndex {
    /// Creates an empty flat index.
    pub fn new(codec: SQuaJL) -> Self {
        Self {
            codec,
            items: Vec::new(),
        }
    }

    /// Creates an index from already encoded vectors.
    pub fn from_encoded(codec: SQuaJL, items: Vec<QuantizedVector>) -> Result<Self> {
        for item in &items {
            codec.validate_code(item)?;
        }
        Ok(Self { codec, items })
    }

    /// Returns the codec used by the index.
    pub fn codec(&self) -> &SQuaJL {
        &self.codec
    }

    /// Returns the number of stored items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` when the index is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns the encoded vector at `index`, if present.
    pub fn get(&self, index: usize) -> Option<&QuantizedVector> {
        self.items.get(index)
    }

    /// Adds an already encoded vector.
    pub fn add_encoded(&mut self, item: QuantizedVector) -> Result<usize> {
        self.codec.validate_code(&item)?;
        self.items.push(item);
        Ok(self.items.len() - 1)
    }

    /// Encodes and adds a dense vector.
    pub fn add_vector(&mut self, vector: &[f32]) -> Result<usize> {
        let item = self.codec.encode(vector)?;
        self.add_encoded(item)
    }

    /// Encodes and appends a batch of vectors.
    pub fn extend_vectors<'a, I>(&mut self, vectors: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a [f32]>,
    {
        for vector in vectors {
            self.add_vector(vector)?;
        }
        Ok(())
    }

    /// Searches the index by first preparing a query sketch.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<ScoredIndex>> {
        let prepared = self.codec.sketch_query(query)?;
        self.search_prepared(&prepared, k)
    }

    /// Searches the index using a precomputed query sketch.
    pub fn search_prepared(&self, query: &QuerySketch, k: usize) -> Result<Vec<ScoredIndex>> {
        let mut hits = Vec::with_capacity(self.items.len());
        for (index, item) in self.items.iter().enumerate() {
            hits.push(ScoredIndex {
                index,
                score: self.codec.score(query, item)?,
            });
        }
        hits.sort_by(|left, right| right.score.total_cmp(&left.score));
        hits.truncate(k.min(hits.len()));
        Ok(hits)
    }
}
