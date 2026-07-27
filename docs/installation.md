# Installation

```bash
pip install insilico-cancer-center
```

Graph drawing uses `pygraphviz`, which needs the system **Graphviz** library installed first
(e.g. `brew install graphviz`, `apt-get install graphviz graphviz-dev`, or `conda install -c conda-forge graphviz`).

## From source

```bash
git clone https://github.com/pedrofale/iscc
cd iscc
poetry install
```

## Command-line tools

Installing the package provides the pipeline entry points:

| Tool | Stage |
|---|---|
| `isccsim` | grow a tumor |
| `isccsample` | biopsy / dissociate into a sample |
| `isccdata` | generate DNA/RNA/spatial assay data |
| `isccfig`, `isccgif` | figures and animations |

Once installed, see the [Overview](overview.md) for the pipeline and a quickstart.
