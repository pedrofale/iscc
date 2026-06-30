# iscc manuscript

Draft of the iscc software paper (target: PLoS Computational Biology).

## Files
- `paper.tex` — the manuscript.
- `references.bib` — bibliography. Entries flagged "auto-added — verify" still need volume/DOI/author
  checks before submission.
- `arxiv.sty` — the document style (vendored so the project is self-contained), MIT-licensed
  (kourgeorge/arxiv-style).
- `figures/` — all figures. Most are produced by the `validation/validate_*.py` scripts; the schematic
  `overview.png` is a **draft** drawn by `figures/make_overview.py` (replace with a designed vector
  figure before submission).

## Build the PDF
**Overleaf (easiest for collaborators):** upload the whole `manuscript/` folder (it contains
`paper.tex`, `references.bib`, `arxiv.sty`, and `figures/`), set the main document to `paper.tex`,
and compile.

**Locally (needs a TeX distribution):**
```bash
cd manuscript
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

## Regenerate figures
```bash
# validation figures (each writes into manuscript/figures/)
python validation/validate_scrna.py        # and validate_dna.py, validate_visium.py, ...
# the schematic overview
python manuscript/figures/make_overview.py
```
