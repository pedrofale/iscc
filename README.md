<div align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pedrofale/iscc/raw/main/docs/assets/logo_text_dark.svg">
    <img src="https://github.com/pedrofale/iscc/raw/main/docs/assets/logo_text.svg" alt="in silico cancer center" width="340px">
  </picture>
</div>
<p></p>

[![PyPI](https://img.shields.io/pypi/v/insilico-cancer-center)](https://pypi.org/project/insilico-cancer-center/)
[![Tests](https://github.com/pedrofale/iscc/actions/workflows/main.yaml/badge.svg)](https://github.com/pedrofale/iscc/actions/workflows/main.yaml)
[![Docs](https://github.com/pedrofale/iscc/actions/workflows/docs.yml/badge.svg)](https://pedrofale.github.io/iscc/)

`iscc` (in silico cancer center) is a Python package that simulates a growing tumor and the
molecular data measured from it. It grows a tumor under copy-number and driver selection in a field
of glands, stroma and an optional metastatic site, where selection depends on the niche. It then
carries the same cells through biopsy and dissociation into bulk and single-cell DNA, RNA and spatial
transcriptomics data, down to sequencing reads, and therapies can be applied during growth. `iscc`
records the ground truth of every cell, clone and niche, and of where and when each clone arose, so
that methods that profile tumor heterogeneity, including those that integrate several data layers,
can be scored against it.

## Installation

```bash
pip install insilico-cancer-center
```

## Citing iscc

If you use `iscc`, please cite the preprint that describes it and the release you used. The preprint
is in preparation, and release v0.4.0 will be archived with a DOI. Both references will be listed here
when they are available. Until then, the citation of the software is in `CITATION.cff`.
