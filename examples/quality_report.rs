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

fn dense_dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn main() -> clark_hash::Result<()> {
    let mut rng = StdRng::seed_from_u64(7);

    let dim = 384;
    let clusters = 10;
    let docs_per_cluster = 128;
    let queries_per_cluster = 32;

    let codec = ClarkHash::new(
        ClarkHashConfig::new(dim)
            .with_sketch_dim(96)
            .with_bits(4)
            .with_hashes_per_input(4)
            .with_seed(12345)
            .with_metric(SimilarityMetric::Cosine),
    )?;

    let mut centers = Vec::new();
    for _ in 0..clusters {
        centers.push(random_unit_vector(&mut rng, dim));
    }

    let mut docs = Vec::new();
    let mut labels = Vec::new();
    let mut index = FlatIndex::new(codec.clone());

    for (cluster_id, center) in centers.iter().enumerate() {
        for _ in 0..docs_per_cluster {
            let vector = noisy(center, &mut rng, 0.05);
            labels.push(cluster_id);
            index.add_vector(&vector)?;
            docs.push(vector);
        }
    }

    let mut approx_correct = 0_usize;
    let mut exact_correct = 0_usize;
    let mut total = 0_usize;

    for (cluster_id, center) in centers.iter().enumerate() {
        for _ in 0..queries_per_cluster {
            let query = noisy(center, &mut rng, 0.05);

            let approx_hit = index.search(&query, 1)?[0];
            approx_correct += usize::from(labels[approx_hit.index] == cluster_id);

            let mut best_index = 0_usize;
            let mut best_score = f32::NEG_INFINITY;
            for (index, doc) in docs.iter().enumerate() {
                let score = dense_dot(&query, doc);
                if score > best_score {
                    best_score = score;
                    best_index = index;
                }
            }
            exact_correct += usize::from(labels[best_index] == cluster_id);
            total += 1;
        }
    }

    println!("codec: {:?}", codec.config());
    println!(
        "storage_bytes_per_vector: {}",
        codec.storage_bytes_per_vector()
    );
    println!(
        "compression_ratio_vs_f32: {:.4}",
        codec.compression_ratio_vs_f32()
    );
    println!(
        "approx_cluster_accuracy: {:.4}",
        approx_correct as f32 / total as f32
    );
    println!(
        "exact_cluster_accuracy: {:.4}",
        exact_correct as f32 / total as f32
    );

    Ok(())
}
