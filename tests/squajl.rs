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

fn noisy(center: &[f32], rng: &mut StdRng, noise: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(center.len());
    let mut norm_sq = 0.0_f32;

    for &base in center {
        let value = base + rng.gen_range(-noise..noise);
        norm_sq += value * value;
        values.push(value);
    }

    let norm = norm_sq.sqrt().max(f32::EPSILON);
    for value in &mut values {
        *value /= norm;
    }
    values
}

#[test]
fn encoding_is_deterministic_for_same_seed() {
    let codec = ClarkHash::new(
        ClarkHashConfig::new(32)
            .with_sketch_dim(16)
            .with_bits(3)
            .with_hashes_per_input(3)
            .with_seed(7),
    )
    .unwrap();

    let vector = vec![0.25_f32; 32];
    let left = codec.encode(&vector).unwrap();
    let right = codec.encode(&vector).unwrap();

    assert_eq!(left.packed_bytes(), right.packed_bytes());
    assert_eq!(left.encoded_norm(), right.encoded_norm());
}

#[test]
fn approximate_search_preserves_cluster_identity() {
    let mut rng = StdRng::seed_from_u64(42);
    let dim = 64;
    let clusters = 4;
    let docs_per_cluster = 32;
    let queries_per_cluster = 16;

    let codec = ClarkHash::new(
        ClarkHashConfig::new(dim)
            .with_sketch_dim(48)
            .with_bits(4)
            .with_hashes_per_input(4)
            .with_seed(1234)
            .with_metric(SimilarityMetric::Cosine),
    )
    .unwrap();

    let mut centers = Vec::new();
    for _ in 0..clusters {
        centers.push(random_unit_vector(&mut rng, dim));
    }

    let mut labels = Vec::new();
    let mut index = FlatIndex::new(codec);

    for (cluster_id, center) in centers.iter().enumerate() {
        for _ in 0..docs_per_cluster {
            let doc = noisy(center, &mut rng, 0.03);
            index.add_vector(&doc).unwrap();
            labels.push(cluster_id);
        }
    }

    let mut correct = 0_usize;
    let mut total = 0_usize;

    for (cluster_id, center) in centers.iter().enumerate() {
        for _ in 0..queries_per_cluster {
            let query = noisy(center, &mut rng, 0.03);
            let hit = index.search(&query, 1).unwrap();
            let predicted = labels[hit[0].index];
            correct += usize::from(predicted == cluster_id);
            total += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    assert!(accuracy > 0.90, "accuracy too low: {accuracy}");
}

#[test]
fn dot_metric_preserves_scale_information() {
    let codec = ClarkHash::new(
        ClarkHashConfig::new(16)
            .with_sketch_dim(12)
            .with_bits(4)
            .with_hashes_per_input(4)
            .with_metric(SimilarityMetric::Dot),
    )
    .unwrap();

    let unit = vec![0.25_f32; 16];
    let double = vec![0.50_f32; 16];
    let query = codec.sketch_query(&unit).unwrap();

    let score_unit = codec.score(&query, &codec.encode(&unit).unwrap()).unwrap();
    let score_double = codec
        .score(&query, &codec.encode(&double).unwrap())
        .unwrap();

    assert!(score_double > score_unit);
}
