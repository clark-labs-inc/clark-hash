# Clark Hash: Algorithm and Mathematical Notes

Authors: Clark Labs Inc., Autoresearch, and Stanislav Kirdey

Clark Hash is a stateless sparse Johnson-Lindenstrauss sketch and scalar
quantization package for neural embeddings. It encodes each dense database
vector independently, without learned codebooks, fitted rotations, corpus
statistics, or calibration tables. The original codec name in the codebase is
SQuaJL; the package name is Clark Hash.

This note documents the engineering and math behind the implementation. It does
not claim that Johnson-Lindenstrauss transforms, feature hashing, scalar
quantization, or compressed vector retrieval are new. The contribution here is a
specific online packaging of those ideas for compact embedding storage and
asymmetric sketch-space scoring.

## Symbols

- `x in R^d`: input database embedding.
- `r in R^d`: input query embedding.
- `d`: input dimension.
- `m`: sketch dimension.
- `s`: hashes per input coordinate.
- `b`: bits per quantized sketch coordinate.
- `c`: symmetric clipping range.
- `seed`: global deterministic seed.
- `h_j(i)`: bucket hash for input coordinate `i` and repetition `j`.
- `sigma_j(i)`: sign hash in `{-1, +1}`.

## Sparse Signed JL Projection

Clark Hash uses a sparse random matrix `R in R^{m x d}`. Each input coordinate
has `s` non-zero contributions:

```text
R_{k,i} = (1 / sqrt(s)) * sum_{j=1..s} sigma_j(i) * 1[h_j(i) = k]
```

Equivalently, the projected coordinate is:

```text
y_k = sum_{i=1..d} sum_{j=1..s}
      1[h_j(i) = k] * sigma_j(i) * x_i / sqrt(s)
```

The projection is data-oblivious. The bucket hashes and sign hashes are derived
only from the seed, the input coordinate, and the repetition index.

For fixed vectors `u` and `v`, the signed-hash estimator is centered around the
original inner product:

```text
E[<R u, R v>] = <u, v>
```

The variance depends on the sketch dimension, sparsity, and collision pattern.
Increasing `m` reduces projection noise at the cost of storage; increasing `s`
usually reduces sparse projection noise at the cost of encode CPU.

## Direction Normalization

For cosine search, Clark Hash sketches the unit direction:

```text
||x||_2 = sqrt(sum_i x_i^2)
u = x / ||x||_2
```

The raw sketch is:

```text
y = R u
```

For a unit vector, each coordinate of `R u` has scale roughly `1 / sqrt(m)`.
Clark Hash rescales by `sqrt(m)` before quantization:

```text
z = sqrt(m) * R u
```

This makes sketch coordinates roughly unit scale, so a fixed clipping range such
as `[-3, 3]` is practical across sketch sizes.

## Fixed Scalar Quantization

Let:

```text
L = 2^b - 1
```

Each scaled coordinate is clipped and uniformly quantized:

```text
z'_k = clamp(z_k, -c, c)
q_k = round(L * (z'_k + c) / (2c))
```

The database-side dequantizer is:

```text
zhat_k = (2c * q_k / L) - c
```

The integer code stores exactly `b` bits per coordinate. The quantization step is:

```text
Delta = 2c / L
```

Without clipping, scalar quantization error per coordinate is bounded by:

```text
|zhat_k - z_k| <= Delta / 2
```

Clipping adds the residual term:

```text
clip_error_k = z_k - clamp(z_k, -c, c)
```

The implementation exposes `clip` so workloads with heavier-tailed sketch
coordinates can trade quantization resolution against clipping frequency.

## Asymmetric Cosine Scoring

Queries stay in floating point. For a query `r`, Clark Hash computes:

```text
v = r / ||r||_2
a = sqrt(m) * R v
```

The database vector stores the quantized/dequantized `zhat` for `x`. The
asymmetric cosine estimate is:

```text
cos_hat(r, x) = (1 / m) * sum_{k=1..m} a_k * zhat_k
```

Without quantization, the same estimator is:

```text
(1 / m) * <sqrt(m) R v, sqrt(m) R u>
= <R v, R u>
approx <v, u>
= cos(r, x)
```

The quantized score adds scalar quantization error and clipping error on the
database side. A simple bound for the quantization-only score perturbation is:

```text
|score_error| <= (1 / m) * sum_k |a_k| * (Delta / 2)
```

plus the analogous clipping term using `|clip_error_k|`.

## Dot-Product Mode

Cosine mode stores only the normalized direction sketch. Dot-product mode also
stores a two-byte log-norm side channel:

```text
ell = clamp(log2(||x||_2), ell_min, ell_max)
n = round(65535 * (ell - ell_min) / (ell_max - ell_min))
```

Decode:

```text
ell_hat = ell_min + n * (ell_max - ell_min) / 65535
||x||_hat = 2^{ell_hat}
```

Final dot-product estimate:

```text
dot_hat(r, x) = cos_hat(r, x) * ||r||_2 * ||x||_hat
```

The norm channel costs 2 bytes per vector.

## Storage Math

For cosine mode, bytes per vector are:

```text
ceil(m * b / 8)
```

For dot-product mode:

```text
ceil(m * b / 8) + 2
```

For the default benchmark configuration:

```text
d = 384
m = 96
b = 4
s = 4
```

Dense storage:

```text
384 * 4 = 1536 bytes
```

Clark Hash cosine storage:

```text
ceil(96 * 4 / 8) = 48 bytes
```

Compression ratio:

```text
48 / 1536 = 0.03125
```

That is `32x` smaller, or `96.875%` less vector memory, for this configuration.

## Operational Contract

1. Choose `d`, `m`, `b`, `s`, `c`, and `seed`.
2. Keep the same config for a collection.
3. Encode every database vector independently.
4. Sketch each query in floating point.
5. Score compressed vectors with asymmetric dot products in sketch space.

The result is an online, deterministic, compact vector representation that can be
used directly for evaluation, flat compressed scans, or as a storage layer inside
larger retrieval systems.

## MLA Citation

Clark Labs Inc., Autoresearch, and Stanislav Kirdey. "Clark Hash: Stateless
Sparse Johnson-Lindenstrauss Quantization for Neural Embeddings." Clark Labs
Inc., 2026, GitHub, https://github.com/clark-labs-inc/clark-hash.
