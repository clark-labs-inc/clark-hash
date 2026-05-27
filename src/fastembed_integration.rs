use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::{Result, SQuaJLError};
use crate::quantized::{QuantizedVector, QuerySketch};
use crate::{SQuaJL, SQuaJLConfig};

/// Local text embedding + stateless quantization pipeline backed by `fastembed`.
///
/// This wrapper keeps the quantization logic separate from the embedding model while
/// making it easy to build an end-to-end local text embedding path.
pub struct FastEmbedQuantizer {
    model: TextEmbedding,
    codec: SQuaJL,
}

impl FastEmbedQuantizer {
    /// Creates a new pipeline for a given `fastembed` model.
    ///
    /// `config.input_dim` should match the output dimension of the selected model.
    pub fn new(model_name: EmbeddingModel, codec: SQuaJL) -> Result<Self> {
        Self::with_options(model_name, codec, None, None, false)
    }

    /// Creates a new pipeline with explicit cache and initialization settings.
    pub fn with_options(
        model_name: EmbeddingModel,
        codec: SQuaJL,
        cache_dir: Option<PathBuf>,
        max_length: Option<usize>,
        show_download_progress: bool,
    ) -> Result<Self> {
        let mut options = InitOptions::new(model_name);
        options.show_download_progress = show_download_progress;
        if let Some(cache_dir) = cache_dir {
            options.cache_dir = cache_dir;
        }
        if let Some(max_length) = max_length {
            options.max_length = max_length;
        }

        let model = TextEmbedding::try_new(options)
            .map_err(|error| SQuaJLError::Backend(error.to_string()))?;
        Ok(Self { model, codec })
    }

    /// Creates an `all-MiniLM-L6-v2` pipeline with a matching default input dimension.
    pub fn all_minilm_l6_v2(config: SQuaJLConfig) -> Result<Self> {
        if config.input_dim != 384 {
            return Err(SQuaJLError::InvalidConfig(
                "all-MiniLM-L6-v2 emits 384-dimensional embeddings".to_owned(),
            ));
        }
        let codec = SQuaJL::new(config)?;
        Self::new(EmbeddingModel::AllMiniLML6V2, codec)
    }

    /// Returns the underlying codec.
    pub fn codec(&self) -> &SQuaJL {
        &self.codec
    }

    /// Returns the underlying `fastembed` text model.
    pub fn model(&self) -> &TextEmbedding {
        &self.model
    }

    /// Embeds raw text into dense vectors.
    pub fn embed_texts<S>(
        &mut self,
        texts: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>>
    where
        S: AsRef<str> + Send + Sync,
    {
        self.model
            .embed(texts, batch_size)
            .map_err(|error| SQuaJLError::Backend(error.to_string()))
    }

    /// Embeds text and quantizes the resulting dense vectors.
    pub fn quantize_texts<S>(
        &mut self,
        texts: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<QuantizedVector>>
    where
        S: AsRef<str> + Send + Sync,
    {
        let embeddings = self.embed_texts(texts, batch_size)?;
        embeddings
            .iter()
            .map(|embedding| self.codec.encode(embedding))
            .collect()
    }

    /// Embeds a single query string and prepares a floating-point query sketch.
    pub fn embed_query(&mut self, text: &str) -> Result<QuerySketch> {
        let embeddings = self.embed_texts(&[text], Some(1))?;
        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| SQuaJLError::Backend("fastembed returned no embedding".to_owned()))?;
        self.codec.sketch_query(&embedding)
    }
}
