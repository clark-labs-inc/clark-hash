#[cfg(feature = "fastembed")]
use clark_hash::{ClarkHash, ClarkHashConfig, FastEmbedQuantizer, FlatIndex};
#[cfg(feature = "fastembed")]
use fastembed::EmbeddingModel;

#[cfg(feature = "fastembed")]
fn main() -> clark_hash::Result<()> {
    let codec = ClarkHash::new(
        ClarkHashConfig::new(384)
            .with_sketch_dim(96)
            .with_bits(4)
            .with_hashes_per_input(4),
    )?;

    let mut pipeline = FastEmbedQuantizer::new(EmbeddingModel::AllMiniLML6V2, codec)?;

    let texts = vec![
        "passage: Rust gives you control over performance and memory.",
        "passage: Sentence embeddings are useful for semantic retrieval.",
        "passage: Stateless quantization helps online indexing pipelines.",
        "passage: Cats and dogs are common household pets.",
    ];

    let codes = pipeline.quantize_texts(&texts, Some(32))?;
    let query = pipeline.embed_query("query: online semantic vector compression")?;

    let index = FlatIndex::from_encoded(pipeline.codec().clone(), codes)?;
    let hits = index.search_prepared(&query, 3)?;

    println!("top_hits = {hits:#?}");
    Ok(())
}

#[cfg(not(feature = "fastembed"))]
fn main() {
    eprintln!("This example requires the `fastembed` feature.");
    eprintln!("Run with: cargo run --features fastembed --example fastembed_quantize");
}
