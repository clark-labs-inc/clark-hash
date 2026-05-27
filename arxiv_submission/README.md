# Clark Hash arXiv submission package

This directory contains a LaTeX source tree for:

> Clark Hash: Stateless Sparse Johnson-Lindenstrauss Quantization for Neural Embeddings

The arXiv source bundle should contain only:

- `main.tex`
- `references.bib`
- `main.bbl`

Do not include local outputs such as `main.pdf`, `.aux`, `.log`, `.out`, `.blg`,
or the source tarball itself.

## arXiv notes checked

Checked against arXiv help on 2026-05-27 UTC (2026-05-26 America/Los_Angeles):

- arXiv processes submitted TeX source and makes source public, so source must
  be self-contained.
- arXiv's default AutoTeX environment is TeX Live 2025, with TeX Live 2023 also
  supported.
- TeX source submissions should include necessary source files and avoid output
  or intermediate files.
- If BibTeX is used, include the generated `.bbl` file for robustness.

Relevant arXiv help pages:

- https://info.arxiv.org/help/submit_tex.html
- https://info.arxiv.org/help/faq/texlive.html

## Build locally

If a TeX distribution is installed:

```bash
cd arxiv_submission
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current machine did not have `pdflatex`, `bibtex`, `latexmk`, or `tectonic`
available when this package was generated, so local PDF compilation was not run.

## Create source tarball

From the repository root:

```bash
tar -czf dist/clark-hash-arxiv-source.tar.gz \
  -C arxiv_submission main.tex references.bib main.bbl
```
