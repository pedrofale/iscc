<div align="left">
  <img src="https://github.com/pedrofale/iscc/raw/main/docs/assets/logo_text.svg" width="340px">
</div>
<p></p>

[![PyPI](https://img.shields.io/pypi/v/insilico-cancer-center)](https://pypi.org/project/insilico-cancer-center/)
[![Tests](https://github.com/pedrofale/iscc/actions/workflows/main.yaml/badge.svg)](https://github.com/pedrofale/iscc/actions/workflows/main.yaml)
[![Docs](https://github.com/pedrofale/iscc/actions/workflows/docs.yml/badge.svg)](https://pedrofale.github.io/iscc/)

`iscc` (in silico cancer center) is a multi-modal tumor-evolution data simulator. It grows one
selection-driven, spatially-structured tumor, optionally treats it, samples it, and generates
single-cell and bulk DNA, RNA, and spatial data, down to sequencing reads. Because every run knows
the true clones, mutations, copy numbers, cell states, spatial niches, and lineages, `iscc` provides
a realistic ground truth for benchmarking computational tumor evolution methods.

## Installation

```bash
pip install insilico-cancer-center
```
