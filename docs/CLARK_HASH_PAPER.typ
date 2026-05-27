#set document(
  title: "Clark Hash: Stateless Sparse Johnson-Lindenstrauss Quantization for Neural Embeddings",
  author: "Clark Labs Inc.; Autoresearch; Stanislav Kirdey",
)

#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.62in),
  numbering: "1",
)

#set text(
  size: 10pt,
  lang: "en",
)

#set heading(numbering: "1.")
#set par(justify: true, leading: 0.62em)
#set list(indent: 1.1em, body-indent: 0.45em)

#let blue = rgb("#174A7C")
#let slate = rgb("#233142")
#let pale = rgb("#EEF6FF")
#let line-color = rgb("#C8D9EA")
#let green = rgb("#1F7A4D")
#let muted = rgb("#52616F")

#show heading.where(level: 1): it => {
  v(0.62em)
  text(fill: blue, size: 16pt, weight: "bold", it)
  v(0.25em)
  line(length: 100%, stroke: 0.7pt + line-color)
  v(0.34em)
}

#show heading.where(level: 2): it => {
  v(0.55em)
  text(fill: slate, size: 11.6pt, weight: "bold", it)
  v(0.15em)
}

#let callout(title, body) = block(
  width: 100%,
  fill: pale,
  stroke: 0.7pt + line-color,
  radius: 5pt,
  inset: 10pt,
)[
  #text(fill: blue, weight: "bold", title)

  #v(0.35em)
  #body
]

#let metric(label, value, detail) = block(
  width: 100%,
  fill: rgb("#F7FAFC"),
  stroke: 0.6pt + rgb("#D8E2EC"),
  radius: 5pt,
  inset: 10pt,
)[
  #text(size: 8pt, fill: muted, label)

  #v(0.25em)
  #text(size: 17pt, weight: "bold", fill: green, value)

  #v(0.2em)
  #text(size: 7.8pt, fill: muted, detail)
]

#align(center)[
  #text(size: 25pt, weight: "bold", fill: blue)[Clark Hash]

  #v(0.14em)
  #text(size: 13pt, weight: "semibold")[
    Stateless Sparse Johnson-Lindenstrauss Quantization for Neural Embeddings
  ]

  #v(0.35em)
  #text(size: 9pt, fill: muted)[Clark Labs Inc. | Autoresearch | Stanislav Kirdey]
]

#v(0.7em)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.18in,
  metric("Vector bytes", "1536 -> 48", "384-dim f32 to default cosine code"),
  metric("Compression", "32x", "0.03125 storage ratio"),
  metric("Memory saved", "96.875%", "same vector count, smaller footprint"),
)

#v(0.65em)

#callout("Scope and claim")[
  Clark Hash packages a specific stateless engineering design: deterministic sparse
  signed JL sketches, fixed scalar quantization, and asymmetric sketch-space scoring.
  It does not claim first invention of JL transforms, feature hashing, scalar
  quantization, or compressed vector retrieval.
]

= Abstract

Clark Hash is a compact embedding-storage package for systems that need to encode
vectors online. A database embedding is normalized, projected through a deterministic
sparse signed Johnson-Lindenstrauss sketch, rescaled, clipped, and stored as a
fixed-width scalar-quantized code. Queries stay in floating point and are scored
asymmetrically against database-side sketches. The default 384-dimensional profile
stores a cosine-search vector in 48 bytes instead of 1536 bytes for dense `f32`
storage. The design is intentionally simple: no corpus fitting, learned codebooks,
rotations, or calibration tables are required before new vectors can be encoded.

= Symbols

#table(
  columns: (0.20fr, 0.80fr),
  stroke: 0.45pt + rgb("#D8E2EC"),
  inset: 5pt,
  table.header([Symbol], [Meaning]),
  [$x in RR^d$], [Input database embedding],
  [$r in RR^d$], [Input query embedding],
  [$d$], [Input dimension],
  [$m$], [Sketch dimension],
  [$s$], [Hashes per input coordinate],
  [$b$], [Bits per quantized sketch coordinate],
  [$c$], [Symmetric clipping range],
  [$h_j(i)$], [Bucket hash for coordinate $i$ and repetition $j$],
  [$sigma_j(i) in {-1,+1}$], [Sign hash],
)

= Sparse Signed JL Projection

Clark Hash uses a sparse random matrix $R in RR^(m times d)$. Each input coordinate
has $s$ non-zero contributions. Bucket locations and signs are derived
deterministically from the seed, input coordinate, and repetition index:

$ R_(k,i) = 1 / sqrt(s) sum_(j=1)^s sigma_j(i) dot 1[h_j(i) = k] $

Equivalently, the projected coordinate is:

$ y_k = sum_(i=1)^d sum_(j=1)^s 1[h_j(i)=k] dot sigma_j(i) dot x_i / sqrt(s) $

This is data-oblivious: the projection is independent of the corpus and does not
need fitting before new vectors can be encoded.

== Inner-product centering

For fixed vectors $u$ and $v$, the signed-hash estimator is centered around the
original inner product:

$ E[chevron.l R u, R v chevron.r] = chevron.l u, v chevron.r $

The variance depends on sketch dimension, sparsity, and collisions. Increasing
$m$ reduces projection noise at the cost of storage. Increasing $s$ usually reduces
sparse projection noise at the cost of encode CPU.

= Direction Normalization

For cosine search, Clark Hash sketches the unit direction:

$ norm(x)_2 = sqrt(sum_i x_i^2), quad u = x / norm(x)_2 $

The raw sketch is:

$ y = R u $

For a unit vector, each coordinate of $R u$ has scale roughly $1 / sqrt(m)$.
Clark Hash rescales by $sqrt(m)$ before quantization:

$ z = sqrt(m) R u $

That makes sketch coordinates roughly unit scale, so a fixed clipping range such
as $[-3, 3]$ is practical across sketch sizes.

= Fixed Scalar Quantization

Let $L = 2^b - 1$. Each scaled coordinate is clipped and uniformly quantized:

$ z'_k = "clamp"(z_k, -c, c) $

$ q_k = "round"(L dot (z'_k + c) / (2c)) $

The database-side dequantizer is:

$ hat(z)_k = (2c dot q_k / L) - c $

The integer code stores exactly $b$ bits per coordinate. The quantization step is:

$ Delta = 2c / L $

Without clipping, scalar quantization error per coordinate is bounded by:

$ abs(hat(z)_k - z_k) <= Delta / 2 $

Clipping adds the residual term:

$ "clip_error"_k = z_k - "clamp"(z_k, -c, c) $

The implementation exposes `clip` so workloads with heavier-tailed sketch
coordinates can trade quantization resolution against clipping frequency.

= Asymmetric Cosine Scoring

Queries stay in floating point. For query $r$, Clark Hash computes:

$ v = r / norm(r)_2, quad a = sqrt(m) R v $

The database vector stores the quantized and dequantized sketch $hat(z)$ for
the database vector $x$. The asymmetric cosine estimate is:

$ hat(cos)(r,x) = 1 / m sum_(k=1)^m a_k dot hat(z)_k $

Without quantization, this estimator maps back to cosine scale:

$ 1 / m chevron.l sqrt(m) R v, sqrt(m) R u chevron.r
  = chevron.l R v, R u chevron.r
  approx chevron.l v, u chevron.r
  = cos(r,x) $

Quantization adds scalar distortion and clipping error on the database side
while preserving the query in floating point. A simple quantization-only score
perturbation bound is:

$ abs("score_error") <= 1 / m sum_(k=1)^m abs(a_k) dot Delta / 2 $

plus the analogous clipping term using $abs("clip_error"_k)$.

= Dot-Product Mode

Cosine mode stores only the normalized direction sketch. Dot-product mode also
stores a two-byte log-norm side channel:

$ ell = "clamp"("log"_2(norm(x)_2), ell_min, ell_max) $

$ n = "round"(65535 dot (ell - ell_min) / (ell_max - ell_min)) $

Decode:

$ hat(ell) = ell_min + n dot (ell_max - ell_min) / 65535 $

$ hat(norm(x)_2) = 2^hat(ell) $

Final dot-product estimate:

$ "dot_hat"(r,x) = "cos_hat"(r,x) dot norm(r)_2 dot hat(norm(x)_2) $

The norm channel costs 2 bytes per vector.

= Storage Math

For cosine mode:

$ "bytes"_"cosine" = "ceil"(m dot b / 8) $

For dot-product mode:

$ "bytes"_"dot" = "ceil"(m dot b / 8) + 2 $

For the default benchmark configuration:

$ d = 384, quad m = 96, quad b = 4, quad s = 4 $

$ "dense_bytes" = 384 dot 4 = 1536 $

$ "clark_hash_bytes" = "ceil"(96 dot 4 / 8) = 48 $

$ "compression_ratio" = 48 / 1536 = 0.03125 $

#callout("Operational result")[
  The default profile stores each 384-dimensional vector in 48 bytes instead of
  1536 bytes. That is 32x smaller, or 96.875% less vector memory, while keeping
  vectors searchable through asymmetric sketch-space scoring.
]

= Operational Contract

1. Choose $d$, $m$, $b$, $s$, $c$, and `seed`.
2. Keep the same config for a collection.
3. Encode every database vector independently.
4. Sketch each query in floating point.
5. Score compressed vectors with asymmetric dot products in sketch space.

Clark Hash is an online, deterministic, and compact vector representation that can
be used directly for evaluation, flat compressed scans, or as a storage layer
inside larger retrieval systems.

= MLA Citation

Clark Labs Inc., Autoresearch, and Stanislav Kirdey. "Clark Hash: Stateless Sparse
Johnson-Lindenstrauss Quantization for Neural Embeddings." Clark Labs Inc., 2026,
GitHub, #link("https://github.com/clark-labs-inc/clark-hash")[https://github.com/clark-labs-inc/clark-hash].
