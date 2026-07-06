# Installation

```bash
pip install iscc          # or: poetry install
```

`iscc` requires **Python 3.10–3.13**.

## From source

```bash
git clone https://github.com/pedrofale/tumorevo
cd tumorevo
poetry install
```

## Command-line tools

Installing `iscc` provides the pipeline entry points:

| Tool | Stage |
|---|---|
| `isccsim` | grow a tumor |
| `isccsample` | biopsy / dissociate into a sample |
| `isccdata` | generate DNA/RNA/spatial assay data |
| `isccfig`, `isccgif` | figures and animations |

Once installed, head to the [pipeline walkthrough](tutorials/01_pipeline_walkthrough.ipynb).
