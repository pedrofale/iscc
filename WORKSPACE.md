# This repository is one of three

`iscc` is developed in a **workspace** holding three sibling git repositories.
If you are reading this inside a lone clone of `iscc`, two thirds of the project is missing.

| Repository | Where | Holds |
|---|---|---|
| `iscc` | github.com/pedrofale/iscc (public) | this repository: the package, its tests, `validation/`, the public docs |
| `iscc-markdown` | gits-15.sys.kth.se, `Lagergren-Lab-internal` (private) | designs, roadmap, session handoffs, experiment notes, related work, decisions |
| `iscc-overleaf` | gits-15.sys.kth.se, `Lagergren-Lab-internal` (private) | `paper.tex`, its bibliography and its figures |

**The paper is the truth.**
Methods and results are understood from `paper.tex`; this code serves it.
A change that alters a method or a result updates the paper in the same session, or a decision
record in `iscc-markdown/decisions/` says why not.

## What moved, and where to look now

The design documents (`DESIGN_*.md`), the backlog, the audit, the research questions and the
`handoffs/` prompts now live in `iscc-markdown`; the manuscript lives in `iscc-overleaf`.
Comments and docstrings in this repository still name them by their old filenames — `DESIGN_x.md`
is `iscc-markdown/methods/design-x.md`, and the mapping is spelled out in that repository's
`PROVENANCE.md`.
Their history is still here: `git log --follow DESIGN_x.md` on `dev` works.

`PARAMETERS.md` and `SCHEMA.md` deliberately stayed: they are reference documentation for this
code, symlinked into `docs/`, and they change with the code they describe.

## Figures

`validation/validate_*.py` writes into the paper repository through `validation/_paths.py`, which
resolves `$ISCC_PAPER_DIR`, then the workspace sibling `../iscc-overleaf`, then the legacy in-repo
`manuscript/` so a bare clone still runs.
If a figure lands in `manuscript/`, the paper repository is not next to this one.

## Setting the workspace up

Clone `iscc-markdown` and run `workspace/bootstrap.sh`; the full guide is its `workspace/SETUP.md`.
Work on `dev` and open pull requests against `dev`; `main` is written only by
`scripts/publish-main.sh`.

This file is internal — `scripts/publish-main.sh` strips it from the public `main` branch.
