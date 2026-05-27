#![cfg(feature = "fastembed")]

use std::hint::black_box;

use clark_hash::{ClarkHash, ClarkHashConfig, FastEmbedQuantizer};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use fastembed::EmbeddingModel;

fn sample_documents(count: usize) -> Vec<String> {
    let base = [
        "passage: Rust is memory-safe and fast.",
        "passage: Embeddings can encode semantic similarity.",
        "passage: Quantization reduces storage costs.",
        "passage: Sparse JL sketches support stateless compression.",
        "passage: Vector search is useful for retrieval systems.",
        "passage: ONNX models can run locally on commodity hardware.",
        "passage: Semantic search compares text by meaning instead of keywords.",
        "passage: Streaming systems prefer online and deterministic transforms.",
    ];

    (0..count)
        .map(|index| format!("{} example {}", base[index % base.len()], index))
        .collect()
}

fn bench_fastembed_quantize(c: &mut Criterion) {
    let codec = ClarkHash::new(
        ClarkHashConfig::new(384)
            .with_sketch_dim(96)
            .with_bits(4)
            .with_hashes_per_input(4),
    )
    .unwrap();

    let mut pipeline = FastEmbedQuantizer::new(EmbeddingModel::AllMiniLML6V2, codec).unwrap();
    let docs = sample_documents(128);

    let mut group = c.benchmark_group("fastembed_pipeline");
    group.throughput(Throughput::Elements(docs.len() as u64));
    group.bench_function("all_minilm_l6_v2_embed_and_quantize_128", |b| {
        b.iter(|| pipeline.quantize_texts(black_box(&docs), Some(64)).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_fastembed_quantize);
criterion_main!(benches);
