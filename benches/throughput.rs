use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use clark_hash::{ClarkHash, ClarkHashConfig, FlatIndex, SimilarityMetric};

fn random_unit_vector(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(dim);
    let mut norm_sq = 0.0_f32;

    for _ in 0..dim {
        let value = rng.gen_range(-1.0_f32..1.0_f32);
        norm_sq += value * value;
        values.push(value);
    }

    let norm = norm_sq.sqrt().max(f32::EPSILON);
    for value in &mut values {
        *value /= norm;
    }
    values
}

fn build_random_corpus(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| random_unit_vector(&mut rng, dim))
        .collect()
}

fn bench_encode(c: &mut Criterion) {
    let input = build_random_corpus(1, 384, 1).pop().unwrap();
    let mut group = c.benchmark_group("encode");

    for &sketch_dim in &[64_usize, 96, 128] {
        for &bits in &[2_u8, 4, 8] {
            let codec = ClarkHash::new(
                ClarkHashConfig::new(384)
                    .with_sketch_dim(sketch_dim)
                    .with_bits(bits)
                    .with_hashes_per_input(4)
                    .with_metric(SimilarityMetric::Cosine),
            )
            .unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("m{sketch_dim}"), bits),
                &codec,
                |b, codec| {
                    b.iter(|| codec.encode(black_box(&input)).unwrap());
                },
            );
        }
    }

    group.finish();
}

fn bench_scan(c: &mut Criterion) {
    let corpus = build_random_corpus(10_000, 384, 2);
    let query = build_random_corpus(1, 384, 3).pop().unwrap();

    let mut group = c.benchmark_group("scan_top10");
    group.throughput(Throughput::Elements(corpus.len() as u64));

    for &sketch_dim in &[64_usize, 96, 128] {
        for &bits in &[2_u8, 4, 8] {
            let codec = ClarkHash::new(
                ClarkHashConfig::new(384)
                    .with_sketch_dim(sketch_dim)
                    .with_bits(bits)
                    .with_hashes_per_input(4)
                    .with_metric(SimilarityMetric::Cosine),
            )
            .unwrap();

            let encoded: Vec<_> = corpus
                .iter()
                .map(|vector| codec.encode(vector).unwrap())
                .collect();
            let index = FlatIndex::from_encoded(codec, encoded).unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("m{sketch_dim}"), bits),
                &index,
                |b, index| {
                    b.iter(|| index.search(black_box(&query), black_box(10)).unwrap());
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_encode, bench_scan);
criterion_main!(benches);
