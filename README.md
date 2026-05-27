# Clark Hash

Clark Hash is a Rust package for compact, searchable sketches of neural embeddings.
It packages a stateless sparse Johnson-Lindenstrauss projection with fixed scalar
quantization, so each database vector can be encoded independently and searched later
with an asymmetric floating-point query sketch.

The core codec was originally developed under the internal name `SQuaJL`. The Rust API
keeps the `SQuaJL` and `SQuaJLConfig` names for compatibility, and also exports
`ClarkHash` and `ClarkHashConfig` aliases for new code.

## Repository Scope

This repository is now focused on the Clark Hash embedding codec:

- Stateless sparse-JL sketching and scalar quantization for dense embeddings.
- Bit-packed database-side vectors and floating-point query sketches.
- A simple flat compressed-scan index for evaluation and small deployments.
- Optional `fastembed` integration for local text-embedding examples.
- Reproducible sentence-similarity benchmarks and paper sources.

Model-runtime compression experiments are intentionally outside this package. The
library surface here is the embedding sketch codec and its benchmark harnesses.

## Why Use It

A common 384-dimensional `f32` sentence embedding costs 1,536 bytes per vector. The
default Clark Hash profile stores the same vector as a 48-byte cosine sketch:

| Representation | Bytes per vector | Storage ratio |
| --- | ---: | ---: |
| Dense `f32`, 384 dimensions | 1,536 | 1.0000 |
| Clark Hash, `m = 96`, `b = 4` | 48 | 0.03125 |

That is 32x smaller, or 96.875% less vector memory, for this configuration. The quality
tradeoff depends on the embedding model, sketch dimension, bit width, hash count, and
retrieval workload; the benchmark section below shows measured results rather than a
universal guarantee.

Clark Hash is useful when embeddings arrive continuously and you do not want a training
or calibration pass before storing each vector:

- Encode one vector at a time with a deterministic seed.
- Store compact bit-packed sketches for hot memory, local cache, disk, or object storage.
- Keep query vectors in floating point for asymmetric scoring.
- Avoid corpus-specific codebooks, centroids, rotations, or learned quantization tables.
- Use the same codec in simple flat scans, evaluation harnesses, and larger retrieval systems.

## Install

From GitHub:

```toml
[dependencies]
clark-hash = { git = "ssh://git@github.com/clark-labs-inc/clark-hash.git" }
```

With local text embedding support through `fastembed`:

```toml
[dependencies]
clark-hash = { git = "ssh://git@github.com/clark-labs-inc/clark-hash.git", features = ["fastembed"] }
```

With serialization support for quantized codes:

```toml
[dependencies]
clark-hash = { git = "ssh://git@github.com/clark-labs-inc/clark-hash.git", features = ["serde"] }
```

In Rust code, the crate is imported as `clark_hash`.

## Quick Start

```rust
use clark_hash::{ClarkHash, ClarkHashConfig, FlatIndex, SimilarityMetric};

fn main() -> clark_hash::Result<()> {
    let codec = ClarkHash::new(
        ClarkHashConfig::new(384)
            .with_sketch_dim(96)
            .with_bits(4)
            .with_hashes_per_input(4)
            .with_metric(SimilarityMetric::Cosine),
    )?;

    let doc_a = vec![0.1_f32; 384];
    let doc_b = vec![0.2_f32; 384];
    let query = vec![0.15_f32; 384];

    let mut index = FlatIndex::new(codec);
    index.add_vector(&doc_a)?;
    index.add_vector(&doc_b)?;

    let hits = index.search(&query, 2)?;
    println!("{hits:#?}");

    Ok(())
}
```

## Text Embedding Pipeline

Enable the `fastembed` feature when you want local text embeddings and immediate
quantization in one pipeline.

```rust
use clark_hash::{ClarkHash, ClarkHashConfig, FastEmbedQuantizer, FlatIndex};
use fastembed::EmbeddingModel;

fn main() -> clark_hash::Result<()> {
    let codec = ClarkHash::new(
        ClarkHashConfig::new(384)
            .with_sketch_dim(96)
            .with_bits(4)
            .with_hashes_per_input(4),
    )?;

    let mut pipeline = FastEmbedQuantizer::new(EmbeddingModel::AllMiniLML6V2, codec)?;

    let documents = vec![
        "passage: Rust is a systems programming language.",
        "passage: Embeddings can preserve semantic similarity.",
        "passage: Quantization reduces memory usage.",
    ];

    let codes = pipeline.quantize_texts(&documents, Some(32))?;
    let query = pipeline.embed_query("query: semantic vector compression")?;
    let index = FlatIndex::from_encoded(pipeline.codec().clone(), codes)?;

    println!("{:#?}", index.search_prepared(&query, 3)?);
    Ok(())
}
```

Run the example:

```bash
cargo run --release --features fastembed --example fastembed_quantize
```

## How It Works

For an input vector `x in R^d`, the codec:

1. Computes the input norm.
2. Projects the normalized vector into a lower-dimensional sparse signed JL sketch.
3. Rescales the projected coordinates by `sqrt(sketch_dim)`.
4. Clips and uniformly quantizes every sketch coordinate into `1..=8` bits.
5. Optionally stores a two-byte norm channel for raw dot-product scoring.

The database side stores a `QuantizedVector`. The query side uses a floating-point
`QuerySketch`. Scoring happens in sketch space, which is a natural fit for cosine
similarity over normalized sentence embeddings.

For the compact mathematical note and paper, see:

- [Clark Hash paper PDF](docs/Clark_Hash_Paper.pdf)
- [Typst paper source](docs/CLARK_HASH_PAPER.typ)
- [Editable paper note](docs/CLARK_HASH_PAPER.md)
- [arXiv LaTeX source package](arxiv_submission/)

Regenerate the PDF with:

```bash
typst compile docs/CLARK_HASH_PAPER.typ docs/Clark_Hash_Paper.pdf
```

## Configuration Guide

For common 384-dimensional sentence embeddings, start here:

```rust
ClarkHashConfig::new(384)
    .with_sketch_dim(96)
    .with_bits(4)
    .with_hashes_per_input(4)
    .with_metric(SimilarityMetric::Cosine)
```

Useful tuning directions:

- `sketch_dim = 64` with `bits = 2` or `3` gives more aggressive compression.
- `sketch_dim = 128` with `bits = 4` or `6` gives better quality.
- `SimilarityMetric::Cosine` is best for normalized semantic embeddings.
- `SimilarityMetric::Dot` stores a small norm channel and is better when raw inner product matters.
- `seed` controls the deterministic projection, so keep it stable across indexed data.

## Benchmarks

Run the core encode and scan Criterion benchmark:

```bash
cargo bench --bench throughput
```

Run the local text embedding plus quantization benchmark:

```bash
cargo bench --features fastembed --bench fastembed_pipeline
```

Run the synthetic retrieval sanity check:

```bash
cargo run --release --example quality_report
```

## Hugging Face Sentence Similarity Benchmark

The real-text benchmark downloads multilingual sentence-similarity corpora from Hugging
Face, embeds each unique sentence once, quantizes the embeddings, and compares score
correlations.

Default `all-MiniLM-L6-v2` run:

```bash
cargo run --release --features fastembed --example hf_sentence_similarity
```

Multilingual model run:

```bash
cargo run --release --features fastembed --example hf_sentence_similarity -- \
  --model ParaphraseMLMiniLML12V2 \
  --report target/hf-sts-report-paraphrase-multilingual-minilm-l12-v2.json
```

Fast smoke run:

```bash
cargo run --release --features fastembed --example hf_sentence_similarity -- \
  --max-pairs-per-subset 200
```

The benchmark currently uses:

- `mteb/sts17-crosslingual-sts`
- `mteb/sts22-crosslingual-sts`

It reports:

- Dense cosine score vs. human similarity correlation.
- Clark Hash approximate score vs. human similarity correlation.
- Quantized score vs. dense score correlation.
- Macro averages across language-pair subsets.

## Benchmark Results

These results were produced locally on April 23, 2026 with:

- `sketch_dim = 96`
- `bits = 4`
- `hashes_per_input = 4`
- cosine scoring
- 48 bytes per stored vector
- 0.03125 compression ratio vs. dense `f32`

The full benchmark used 9,304 labeled sentence pairs across 29 multilingual subsets and
17,000 unique sentences.

| Model | Dataset | Dense Spearman | Sketch Spearman | Sketch Loss | Sketch vs Dense Pearson |
| --- | --- | ---: | ---: | ---: | ---: |
| `all-MiniLM-L6-v2` | `mteb/sts17-crosslingual-sts` | 0.3644 | 0.2719 | -0.0926 | 0.7242 |
| `all-MiniLM-L6-v2` | `mteb/sts22-crosslingual-sts` | 0.4168 | 0.2876 | -0.1292 | 0.8531 |
| `paraphrase-multilingual-MiniLM-L12-v2` | `mteb/sts17-crosslingual-sts` | 0.8144 | 0.7460 | -0.0684 | 0.9099 |
| `paraphrase-multilingual-MiniLM-L12-v2` | `mteb/sts22-crosslingual-sts` | 0.2973 | 0.2472 | -0.0501 | 0.9460 |

The main readout is that model fit matters more than quantization in this test. The
English-centric `all-MiniLM-L6-v2` model is weak on many cross-lingual subsets. The
multilingual MiniLM backbone is much stronger on STS17, and the sketch preserves a large
part of that ranking signal while storing each vector in 48 bytes.

STS22 is a harder and more mixed corpus. The multilingual model is not universally better
there, but the quantized sketches still track dense scores more closely than they did
with the English MiniLM baseline.

Full JSON reports from the local run:

- `target/hf-sts-report.json`
- `target/hf-sts-report-paraphrase-multilingual-minilm-l12-v2.json`

## API Overview

Core types:

- `ClarkHash` / `SQuaJL`: stateless codec used to encode vectors, sketch queries, and score codes.
- `ClarkHashConfig` / `SQuaJLConfig`: sketch size, bit width, hash count, clip range, seed, and metric.
- `QuantizedVector`: bit-packed database-side sketch.
- `QuerySketch`: floating-point query-side sketch.
- `FlatIndex`: reference exact scan over compressed vectors.
- `FastEmbedQuantizer`: optional text embedding and quantization pipeline.

## Limitations

- Clark Hash is a quantization library, not a full approximate-nearest-neighbor engine.
- `FlatIndex` scans compressed vectors exactly and is meant for evaluation and simple deployments.
- Quality depends on the embedding model, sketch dimension, bit width, and workload.
- No fixed sketch dimension can preserve every future pair in an adversarial unbounded stream.
- This package does not claim that Johnson-Lindenstrauss transforms, feature hashing,
  scalar quantization, or compressed retrieval are new. It documents and implements one
  practical stateless combination for Clark's embedding and memory workloads.

## Citation

MLA:

> Clark Labs Inc., Autoresearch, and Stanislav Kirdey. "Clark Hash: Stateless Sparse
> Johnson-Lindenstrauss Quantization for Neural Embeddings." Clark Labs Inc., 2026,
> GitHub, https://github.com/clark-labs-inc/clark-hash.

BibTeX:

```bibtex
@misc{clark_hash_2026,
  author = {{Clark Labs Inc.} and {Autoresearch} and {Stanislav Kirdey}},
  title = {Clark Hash: Stateless Sparse Johnson-Lindenstrauss Quantization for Neural Embeddings},
  year = {2026},
  publisher = {Clark Labs Inc.},
  url = {https://github.com/clark-labs-inc/clark-hash}
}
```

## Development

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo bench --bench throughput --no-run
```

The `fastembed` benchmark and examples may download models on first use.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
