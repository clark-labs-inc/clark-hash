# Algorithm notes

`clark-hash` implements a deliberately simple stateless quantization pipeline.
The core codec still carries the earlier `SQuaJL` API name for compatibility:

1. **Sparse signed JL projection**
2. **Direction normalization**
3. **`sqrt(m)` rescaling**
4. **Fixed-range scalar quantization**
5. **Optional norm side channel**

## Projection

For an input vector `x ∈ R^d`, each input coordinate contributes to
`hashes_per_input = s` output buckets. Bucket locations and signs are derived
from a seed, the input dimension index, and the repetition index.

Conceptually, the sketch corresponds to a sparse matrix `R ∈ R^{m×d}` where
each column has `s` non-zeros with value `± 1 / sqrt(s)`.

## Direction handling

The codec always sketches the normalized direction:

```text
u = x / ||x||_2
```

This matches cosine retrieval well and allows the same sketch to be used for
cosine similarity or raw dot product.

## Why rescale by sqrt(m)?

With a JL-style sketch, each coordinate of `R u` has variance about `1 / m`
for a unit vector. Multiplying by `sqrt(m)` makes the per-coordinate variance
roughly one, which makes a fixed clip range practical across multiple sketch
sizes.

```text
z = sqrt(m) * R u
```

The asymmetric score divides by `m` so the estimate maps back to cosine scale.

## Quantization

Each scaled coordinate is clipped to `[-clip, clip]` and uniformly quantized
into `2^b` levels. No learned or per-block scale is stored.

## Dot-product mode

When `SimilarityMetric::Dot` is enabled, the codec stores a tiny `u16`
quantization of `log2(||x||_2)`. The final score is then

```text
approx_dot(q, x) ≈ approx_cosine(q, x) * ||q||_2 * ||x||_2
```

## Tuning

A good starting point for 384-dimensional sentence embeddings is:

- `sketch_dim = 96`
- `bits = 4`
- `hashes_per_input = 4`
- `clip = 3.0`

Increase `sketch_dim` first when quality matters more than memory.
Increase `bits` when clipping / scalar distortion appears to dominate.
Increase `hashes_per_input` when projection noise dominates and encode CPU is cheap.
