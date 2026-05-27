#[cfg(feature = "fastembed")]
mod app {
    use std::collections::{BTreeMap, HashMap};
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use fastembed::{EmbeddingModel, TextEmbedding};
    use flate2::read::GzDecoder;
    use reqwest::blocking::Client;
    use serde::{Deserialize, Serialize};

    use clark_hash::{
        ClarkHash, ClarkHashConfig, FastEmbedQuantizer, QuantizedVector, QuerySketch,
        SimilarityMetric,
    };

    const STS17_DATASET: &str = "mteb/sts17-crosslingual-sts";
    const STS22_DATASET: &str = "mteb/sts22-crosslingual-sts";

    const STS17_FILES: &[(&str, &str)] = &[
        (
            "ar-ar",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/ar-ar.jsonl.gz",
        ),
        (
            "en-ar",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/en-ar.jsonl.gz",
        ),
        (
            "en-de",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/en-de.jsonl.gz",
        ),
        (
            "en-en",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/en-en.jsonl.gz",
        ),
        (
            "en-tr",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/en-tr.jsonl.gz",
        ),
        (
            "es-en",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/es-en.jsonl.gz",
        ),
        (
            "es-es",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/es-es.jsonl.gz",
        ),
        (
            "fr-en",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/fr-en.jsonl.gz",
        ),
        (
            "it-en",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/it-en.jsonl.gz",
        ),
        (
            "ko-ko",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/ko-ko.jsonl.gz",
        ),
        (
            "nl-en",
            "https://huggingface.co/datasets/mteb/sts17-crosslingual-sts/resolve/main/test/nl-en.jsonl.gz",
        ),
    ];

    const STS22_TEST_URL: &str =
        "https://huggingface.co/datasets/mteb/sts22-crosslingual-sts/resolve/main/data/test.jsonl.gz";

    #[derive(Debug, Clone)]
    struct Cli {
        batch_size: usize,
        bits: u8,
        cache_dir: PathBuf,
        hashes_per_input: u8,
        max_pairs_per_subset: Option<usize>,
        model: EmbeddingModel,
        report_path: PathBuf,
        seed: u64,
        sketch_dim: usize,
    }

    impl Default for Cli {
        fn default() -> Self {
            Self {
                batch_size: 64,
                bits: 4,
                cache_dir: PathBuf::from("target/hf-sts-cache"),
                hashes_per_input: 4,
                max_pairs_per_subset: None,
                model: EmbeddingModel::AllMiniLML6V2,
                report_path: PathBuf::from("target/hf-sts-report.json"),
                seed: 12_345,
                sketch_dim: 96,
            }
        }
    }

    #[derive(Debug, Clone, Deserialize)]
    struct JsonPair {
        sentence1: String,
        sentence2: String,
        score: f32,
        #[serde(default)]
        lang: String,
    }

    #[derive(Debug, Clone)]
    struct PairExample {
        dataset: &'static str,
        subset: String,
        sentence1: String,
        sentence2: String,
        score: f64,
    }

    #[derive(Debug, Clone)]
    struct IndexedPair {
        dataset: &'static str,
        subset: String,
        left: usize,
        right: usize,
        score: f64,
    }

    #[derive(Debug, Default)]
    struct MetricAccumulator {
        human: Vec<f64>,
        dense: Vec<f64>,
        quantized: Vec<f64>,
    }

    impl MetricAccumulator {
        fn push(&mut self, human: f64, dense: f64, quantized: f64) {
            self.human.push(human);
            self.dense.push(dense);
            self.quantized.push(quantized);
        }

        fn len(&self) -> usize {
            self.human.len()
        }
    }

    #[derive(Debug, Serialize)]
    struct Report {
        model: String,
        note: String,
        config: ReportConfig,
        corpus: CorpusSummary,
        runtime_seconds: RuntimeSummary,
        codec: CodecSummary,
        datasets: Vec<DatasetSummary>,
        subsets: Vec<SubsetSummary>,
    }

    #[derive(Debug, Serialize)]
    struct ReportConfig {
        batch_size: usize,
        bits: u8,
        cache_dir: String,
        hashes_per_input: u8,
        max_pairs_per_subset: Option<usize>,
        model: String,
        report_path: String,
        seed: u64,
        sketch_dim: usize,
    }

    #[derive(Debug, Serialize)]
    struct CorpusSummary {
        total_pairs: usize,
        total_subsets: usize,
        unique_sentences: usize,
    }

    #[derive(Debug, Serialize)]
    struct RuntimeSummary {
        download: f64,
        embed: f64,
        quantize: f64,
        query_prepare: f64,
        score: f64,
        total: f64,
    }

    #[derive(Debug, Serialize)]
    struct CodecSummary {
        compression_ratio_vs_f32: f32,
        storage_bytes_per_vector: usize,
    }

    #[derive(Debug, Clone, Serialize)]
    struct DatasetSummary {
        dataset: String,
        subset_count: usize,
        pair_count: usize,
        dense_human_pearson_macro: f64,
        dense_human_spearman_macro: f64,
        quantized_human_pearson_macro: f64,
        quantized_human_spearman_macro: f64,
        quantized_vs_dense_pearson_macro: f64,
        quantized_minus_dense_spearman_macro: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SubsetSummary {
        dataset: String,
        subset: String,
        pair_count: usize,
        dense_human_pearson: f64,
        dense_human_spearman: f64,
        quantized_human_pearson: f64,
        quantized_human_spearman: f64,
        quantized_vs_dense_pearson: f64,
        quantized_minus_dense_spearman: f64,
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let cli = parse_cli()?;
        let run_started = Instant::now();
        let client = Client::builder()
            .user_agent("clark-hash-hf-sts-benchmark/0.1")
            .build()?;

        let download_started = Instant::now();
        let pairs = load_examples(&cli, &client)?;
        let download_seconds = download_started.elapsed().as_secs_f64();

        let (texts, indexed_pairs) = index_examples(&pairs);

        let model_info = TextEmbedding::get_model_info(&cli.model)?;
        let config = ClarkHashConfig::new(model_info.dim)
            .with_sketch_dim(cli.sketch_dim)
            .with_bits(cli.bits)
            .with_hashes_per_input(cli.hashes_per_input)
            .with_seed(cli.seed)
            .with_metric(SimilarityMetric::Cosine);
        let codec = ClarkHash::new(config)?;
        let mut pipeline = FastEmbedQuantizer::new(cli.model.clone(), codec.clone())?;

        let embed_started = Instant::now();
        let embeddings = pipeline.embed_texts(&texts, Some(cli.batch_size))?;
        let embed_seconds = embed_started.elapsed().as_secs_f64();

        let quantize_started = Instant::now();
        let codes = codec.encode_batch(embeddings.iter().map(|embedding| embedding.as_slice()))?;
        let quantize_seconds = quantize_started.elapsed().as_secs_f64();

        let query_started = Instant::now();
        let queries = prepare_queries(&codec, &embeddings)?;
        let query_prepare_seconds = query_started.elapsed().as_secs_f64();

        let score_started = Instant::now();
        let subset_summaries =
            summarize_subsets(&indexed_pairs, &embeddings, &queries, &codes, &codec)?;
        let score_seconds = score_started.elapsed().as_secs_f64();

        let dataset_summaries = summarize_datasets(&subset_summaries);
        let total_seconds = run_started.elapsed().as_secs_f64();

        let report = Report {
            model: model_info.model_code.clone(),
            note: benchmark_note(&cli.model).to_owned(),
            config: ReportConfig {
                batch_size: cli.batch_size,
                bits: cli.bits,
                cache_dir: cli.cache_dir.display().to_string(),
                hashes_per_input: cli.hashes_per_input,
                max_pairs_per_subset: cli.max_pairs_per_subset,
                model: cli.model.to_string(),
                report_path: cli.report_path.display().to_string(),
                seed: cli.seed,
                sketch_dim: cli.sketch_dim,
            },
            corpus: CorpusSummary {
                total_pairs: indexed_pairs.len(),
                total_subsets: subset_summaries.len(),
                unique_sentences: texts.len(),
            },
            runtime_seconds: RuntimeSummary {
                download: download_seconds,
                embed: embed_seconds,
                quantize: quantize_seconds,
                query_prepare: query_prepare_seconds,
                score: score_seconds,
                total: total_seconds,
            },
            codec: CodecSummary {
                compression_ratio_vs_f32: codec.compression_ratio_vs_f32(),
                storage_bytes_per_vector: codec.storage_bytes_per_vector(),
            },
            datasets: dataset_summaries.clone(),
            subsets: subset_summaries.clone(),
        };

        write_report(&cli.report_path, &report)?;
        print_report(&report);

        Ok(())
    }

    fn parse_cli() -> Result<Cli, Box<dyn Error>> {
        let mut cli = Cli::default();
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--batch-size" => cli.batch_size = parse_value(&arg, args.next())?,
                "--bits" => cli.bits = parse_value(&arg, args.next())?,
                "--cache-dir" => cli.cache_dir = PathBuf::from(require_value(&arg, args.next())?),
                "--hashes" => cli.hashes_per_input = parse_value(&arg, args.next())?,
                "--max-pairs-per-subset" => {
                    cli.max_pairs_per_subset = Some(parse_value(&arg, args.next())?)
                }
                "--model" => cli.model = parse_value(&arg, args.next())?,
                "--report" => cli.report_path = PathBuf::from(require_value(&arg, args.next())?),
                "--seed" => cli.seed = parse_value(&arg, args.next())?,
                "--sketch-dim" => cli.sketch_dim = parse_value(&arg, args.next())?,
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                unknown => {
                    return Err(format!("unknown argument: {unknown}").into());
                }
            }
        }

        Ok(cli)
    }

    fn print_usage() {
        println!(
            "Usage: cargo run --release --features fastembed --example hf_sentence_similarity [options]\n\
             \n\
             Options:\n\
               --batch-size <N>            Embedding batch size (default: 64)\n\
               --bits <N>                  Quantizer bits per coordinate (default: 4)\n\
               --cache-dir <PATH>          Dataset download cache (default: target/hf-sts-cache)\n\
               --hashes <N>                Sparse JL hashes per input dim (default: 4)\n\
               --max-pairs-per-subset <N>  Limit rows per dataset subset for faster iteration\n\
               --model <NAME>              fastembed model enum, e.g. AllMiniLML6V2 or ParaphraseMLMiniLML12V2\n\
               --report <PATH>             JSON report output path (default: target/hf-sts-report.json)\n\
               --seed <N>                  Codec seed (default: 12345)\n\
               --sketch-dim <N>            Sketch dimension (default: 96)"
        );
    }

    fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, Box<dyn Error>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let raw = require_value(flag, value)?;
        raw.parse::<T>()
            .map_err(|error| format!("invalid value for {flag}: {error}").into())
    }

    fn require_value(flag: &str, value: Option<String>) -> Result<String, Box<dyn Error>> {
        value.ok_or_else(|| format!("missing value for {flag}").into())
    }

    fn load_examples(cli: &Cli, client: &Client) -> Result<Vec<PairExample>, Box<dyn Error>> {
        let mut examples = Vec::new();

        for (subset, url) in STS17_FILES {
            let path = cli
                .cache_dir
                .join("sts17")
                .join(format!("{subset}.jsonl.gz"));
            let rows = read_jsonl_gz(client, url, &path)?;
            for row in rows
                .into_iter()
                .take(cli.max_pairs_per_subset.unwrap_or(usize::MAX))
            {
                if let Some(example) = into_example(STS17_DATASET, subset.to_string(), row) {
                    examples.push(example);
                }
            }
        }

        let sts22_path = cli.cache_dir.join("sts22").join("test.jsonl.gz");
        let sts22_rows = read_jsonl_gz(client, STS22_TEST_URL, &sts22_path)?;
        let mut by_lang: BTreeMap<String, Vec<JsonPair>> = BTreeMap::new();
        for row in sts22_rows {
            let subset = if row.lang.trim().is_empty() {
                "unknown".to_owned()
            } else {
                row.lang.clone()
            };
            by_lang.entry(subset).or_default().push(row);
        }

        for (subset, rows) in by_lang {
            for row in rows
                .into_iter()
                .take(cli.max_pairs_per_subset.unwrap_or(usize::MAX))
            {
                if let Some(example) = into_example(STS22_DATASET, subset.clone(), row) {
                    examples.push(example);
                }
            }
        }

        Ok(examples)
    }

    fn into_example(dataset: &'static str, subset: String, row: JsonPair) -> Option<PairExample> {
        let sentence1 = normalize_text(&row.sentence1);
        let sentence2 = normalize_text(&row.sentence2);
        if sentence1.is_empty() || sentence2.is_empty() {
            return None;
        }

        Some(PairExample {
            dataset,
            subset,
            sentence1,
            sentence2,
            score: row.score as f64,
        })
    }

    fn normalize_text(input: &str) -> String {
        input.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn read_jsonl_gz(
        client: &Client,
        url: &str,
        cache_path: &Path,
    ) -> Result<Vec<JsonPair>, Box<dyn Error>> {
        download_if_missing(client, url, cache_path)?;

        let file = File::open(cache_path)?;
        let reader = BufReader::new(GzDecoder::new(file));
        let mut rows = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            rows.push(serde_json::from_str::<JsonPair>(&line)?);
        }

        Ok(rows)
    }

    fn download_if_missing(
        client: &Client,
        url: &str,
        cache_path: &Path,
    ) -> Result<(), Box<dyn Error>> {
        if cache_path.exists() {
            return Ok(());
        }

        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let response = client.get(url).send()?.error_for_status()?;
        let bytes = response.bytes()?;
        fs::write(cache_path, &bytes)?;
        Ok(())
    }

    fn index_examples(examples: &[PairExample]) -> (Vec<String>, Vec<IndexedPair>) {
        let mut sentence_ids = HashMap::<String, usize>::new();
        let mut sentences = Vec::<String>::new();
        let mut indexed = Vec::with_capacity(examples.len());

        for example in examples {
            let left = intern(&mut sentence_ids, &mut sentences, &example.sentence1);
            let right = intern(&mut sentence_ids, &mut sentences, &example.sentence2);
            indexed.push(IndexedPair {
                dataset: example.dataset,
                subset: example.subset.clone(),
                left,
                right,
                score: example.score,
            });
        }

        (sentences, indexed)
    }

    fn intern(
        sentence_ids: &mut HashMap<String, usize>,
        sentences: &mut Vec<String>,
        text: &str,
    ) -> usize {
        if let Some(&existing) = sentence_ids.get(text) {
            return existing;
        }

        let index = sentences.len();
        let owned = text.to_owned();
        sentence_ids.insert(owned.clone(), index);
        sentences.push(owned);
        index
    }

    fn prepare_queries(
        codec: &clark_hash::ClarkHash,
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<QuerySketch>, Box<dyn Error>> {
        embeddings
            .iter()
            .map(|embedding| codec.sketch_query(embedding).map_err(Into::into))
            .collect()
    }

    fn summarize_subsets(
        pairs: &[IndexedPair],
        embeddings: &[Vec<f32>],
        queries: &[QuerySketch],
        codes: &[QuantizedVector],
        codec: &clark_hash::ClarkHash,
    ) -> Result<Vec<SubsetSummary>, Box<dyn Error>> {
        let mut grouped = BTreeMap::<(String, String), MetricAccumulator>::new();

        for pair in pairs {
            let dense = cosine(&embeddings[pair.left], &embeddings[pair.right]);
            let quantized_left = codec.score(&queries[pair.left], &codes[pair.right])? as f64;
            let quantized_right = codec.score(&queries[pair.right], &codes[pair.left])? as f64;
            let quantized = 0.5 * (quantized_left + quantized_right);

            grouped
                .entry((pair.dataset.to_owned(), pair.subset.clone()))
                .or_default()
                .push(pair.score, dense, quantized);
        }

        let mut summaries = Vec::with_capacity(grouped.len());
        for ((dataset, subset), acc) in grouped {
            let dense_human_pearson = pearson(&acc.human, &acc.dense);
            let dense_human_spearman = spearman(&acc.human, &acc.dense);
            let quantized_human_pearson = pearson(&acc.human, &acc.quantized);
            let quantized_human_spearman = spearman(&acc.human, &acc.quantized);
            let quantized_vs_dense_pearson = pearson(&acc.dense, &acc.quantized);

            summaries.push(SubsetSummary {
                dataset,
                subset,
                pair_count: acc.len(),
                dense_human_pearson,
                dense_human_spearman,
                quantized_human_pearson,
                quantized_human_spearman,
                quantized_vs_dense_pearson,
                quantized_minus_dense_spearman: quantized_human_spearman - dense_human_spearman,
            });
        }

        Ok(summaries)
    }

    fn summarize_datasets(subsets: &[SubsetSummary]) -> Vec<DatasetSummary> {
        let mut grouped = BTreeMap::<String, Vec<&SubsetSummary>>::new();
        for subset in subsets {
            grouped
                .entry(subset.dataset.clone())
                .or_default()
                .push(subset);
        }

        grouped
            .into_iter()
            .map(|(dataset, items)| {
                let subset_count = items.len();
                let pair_count = items.iter().map(|item| item.pair_count).sum();
                DatasetSummary {
                    dataset,
                    subset_count,
                    pair_count,
                    dense_human_pearson_macro: mean(
                        items.iter().map(|item| item.dense_human_pearson),
                    ),
                    dense_human_spearman_macro: mean(
                        items.iter().map(|item| item.dense_human_spearman),
                    ),
                    quantized_human_pearson_macro: mean(
                        items.iter().map(|item| item.quantized_human_pearson),
                    ),
                    quantized_human_spearman_macro: mean(
                        items.iter().map(|item| item.quantized_human_spearman),
                    ),
                    quantized_vs_dense_pearson_macro: mean(
                        items.iter().map(|item| item.quantized_vs_dense_pearson),
                    ),
                    quantized_minus_dense_spearman_macro: mean(
                        items.iter().map(|item| item.quantized_minus_dense_spearman),
                    ),
                }
            })
            .collect()
    }

    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        let mut dot = 0.0_f64;
        let mut left_norm = 0.0_f64;
        let mut right_norm = 0.0_f64;

        for (&a, &b) in left.iter().zip(right.iter()) {
            let a = a as f64;
            let b = b as f64;
            dot += a * b;
            left_norm += a * a;
            right_norm += b * b;
        }

        let denom = left_norm.sqrt() * right_norm.sqrt();
        if denom <= f64::EPSILON {
            0.0
        } else {
            dot / denom
        }
    }

    fn pearson(left: &[f64], right: &[f64]) -> f64 {
        if left.len() != right.len() || left.is_empty() {
            return f64::NAN;
        }

        let mean_left = mean(left.iter().copied());
        let mean_right = mean(right.iter().copied());

        let mut numerator = 0.0_f64;
        let mut left_sq = 0.0_f64;
        let mut right_sq = 0.0_f64;

        for (&x, &y) in left.iter().zip(right.iter()) {
            let dx = x - mean_left;
            let dy = y - mean_right;
            numerator += dx * dy;
            left_sq += dx * dx;
            right_sq += dy * dy;
        }

        let denom = left_sq.sqrt() * right_sq.sqrt();
        if denom <= f64::EPSILON {
            0.0
        } else {
            numerator / denom
        }
    }

    fn spearman(left: &[f64], right: &[f64]) -> f64 {
        let left_ranks = average_tied_ranks(left);
        let right_ranks = average_tied_ranks(right);
        pearson(&left_ranks, &right_ranks)
    }

    fn average_tied_ranks(values: &[f64]) -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

        let mut ranks = vec![0.0_f64; values.len()];
        let mut start = 0_usize;
        while start < order.len() {
            let mut end = start + 1;
            while end < order.len() && values[order[start]] == values[order[end]] {
                end += 1;
            }

            let average_rank = (start + end - 1) as f64 / 2.0 + 1.0;
            for &index in &order[start..end] {
                ranks[index] = average_rank;
            }

            start = end;
        }

        ranks
    }

    fn mean<I>(values: I) -> f64
    where
        I: IntoIterator<Item = f64>,
    {
        let mut total = 0.0_f64;
        let mut count = 0_usize;
        for value in values {
            total += value;
            count += 1;
        }

        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    fn write_report(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    }

    fn print_report(report: &Report) {
        println!("model: {}", report.model);
        println!("note: {}", report.note);
        println!(
            "pairs: {} across {} subsets; unique_sentences: {}",
            report.corpus.total_pairs, report.corpus.total_subsets, report.corpus.unique_sentences
        );
        println!(
            "codec: sketch_dim={} bits={} hashes={} compression_ratio_vs_f32={:.4} storage_bytes_per_vector={}",
            report.config.sketch_dim,
            report.config.bits,
            report.config.hashes_per_input,
            report.codec.compression_ratio_vs_f32,
            report.codec.storage_bytes_per_vector
        );
        println!(
            "timings_seconds: download={:.2} embed={:.2} quantize={:.2} query_prepare={:.2} score={:.2} total={:.2}",
            report.runtime_seconds.download,
            report.runtime_seconds.embed,
            report.runtime_seconds.quantize,
            report.runtime_seconds.query_prepare,
            report.runtime_seconds.score,
            report.runtime_seconds.total
        );
        println!();
        println!("dataset_macro_summary:");
        for dataset in &report.datasets {
            println!(
                "  {:30} subsets={:2} pairs={:5} dense_spear={:.4} sketch_spear={:.4} delta={:+.4} dense_pearson={:.4} sketch_pearson={:.4} sketch_vs_dense={:.4}",
                dataset.dataset,
                dataset.subset_count,
                dataset.pair_count,
                dataset.dense_human_spearman_macro,
                dataset.quantized_human_spearman_macro,
                dataset.quantized_minus_dense_spearman_macro,
                dataset.dense_human_pearson_macro,
                dataset.quantized_human_pearson_macro,
                dataset.quantized_vs_dense_pearson_macro,
            );
        }
        println!();
        println!("subset_summary:");
        for subset in &report.subsets {
            println!(
                "  {:30} {:8} pairs={:4} dense_spear={:.4} sketch_spear={:.4} delta={:+.4} dense_pearson={:.4} sketch_pearson={:.4} sketch_vs_dense={:.4}",
                subset.dataset,
                subset.subset,
                subset.pair_count,
                subset.dense_human_spearman,
                subset.quantized_human_spearman,
                subset.quantized_minus_dense_spearman,
                subset.dense_human_pearson,
                subset.quantized_human_pearson,
                subset.quantized_vs_dense_pearson,
            );
        }
        println!();
        println!("json_report: {}", report.config.report_path);
    }

    fn benchmark_note(model: &EmbeddingModel) -> &'static str {
        match model {
            EmbeddingModel::AllMiniLML6V2 | EmbeddingModel::AllMiniLML6V2Q => {
                "all-MiniLM-L6-v2 is primarily English-centric, so multilingual scores here are a stress test for Clark Hash on top of a non-multilingual embedding backbone."
            }
            EmbeddingModel::ParaphraseMLMiniLML12V2
            | EmbeddingModel::ParaphraseMLMiniLML12V2Q
            | EmbeddingModel::ParaphraseMLMpnetBaseV2
            | EmbeddingModel::MultilingualE5Small
            | EmbeddingModel::MultilingualE5Base
            | EmbeddingModel::MultilingualE5Large
            | EmbeddingModel::BGEM3
            | EmbeddingModel::BGESmallZHV15
            | EmbeddingModel::BGELargeZHV15 => {
                "This model is multilingual, so the benchmark is a closer read on quantization loss than the earlier English-only MiniLM run."
            }
            _ => {
                "This benchmark compares dense cosine scores against Clark Hash approximate scores on multilingual Hugging Face sentence-similarity corpora."
            }
        }
    }
}

#[cfg(feature = "fastembed")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::run()
}

#[cfg(not(feature = "fastembed"))]
fn main() {
    eprintln!(
        "Run with `cargo run --release --features fastembed --example hf_sentence_similarity`."
    );
}
