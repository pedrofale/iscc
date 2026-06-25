# DESIGN: interactive web platform (long-term / aspirational)

Status: **vision capture, to be designed later** (noted 2026-06-25). This is a deliberate
placeholder — not yet scoped into milestones.

## Vision

A web application that lets users **tune each module's parameters through a UI**, which in practice
**generates the config files** the simulations consume, and that can **load and visualize results**
so users can make informed decisions about the *next* module — e.g. inspect the simulated ABM tumour
and use that view to choose the biopsy / sampling / molecular-assay / treatment steps. A
human-in-the-loop pipeline composer for the whole `tumor → sample → assay → treatment` chain.

## Key insight: the backend is (mostly) already the right shape

iscc is already **config-driven** (`isccsim`/`isccsample`/`isccdata` each take a YAML config) with a
**locked on-disk output schema** (`SCHEMA.md`) and **reproducible seeded** runs. So:
- "tune parameters in a UI" ≈ **a form that emits the existing YAML configs** — the UI is a thin
  layer over the current contract, not a rewrite.
- "load results to visualize" ≈ read the canonical `cell_data/` + traces and render them.

The main implication: **keep the modules config-driven, stateless, and their outputs serializable /
loadable** — every feature we add (DESIGN_features) should preserve that contract so the platform
stays a thin layer.

## The interactive feedback loop (the distinguishing feature)

The value isn't just form-filling — it's that *visualizing one module's output informs the next
module's parameters*:

```
ABM config → run → VISUALIZE tumour ──▶ user picks a biopsy region on the map
                                          └─▶ becomes the sample config → run → VISUALIZE sample
                                                └─▶ choose assay (protocol/breadth/batches) → VISUALIZE data
                                                      └─▶ choose treatment regimen → run → VISUALIZE response
```

E.g. selecting a biopsy region *directly on the tumour visualization* should produce the
corresponding `isccsample` config (region geometry → §A of DESIGN_features). This couples the UI to
the spatial/region features (F1) and to the visualization layer.

## What this needs from the rest of iscc (preserve as we build features)

- **Stable config schema** per module (already YAML; document each module's schema as it matures).
- **Loadable, serializable results** + a **web-friendly visualization API** (extend the existing
  `plot_muller` / `plot_grid` into something a frontend can render — e.g. JSON/array endpoints, not
  just matplotlib figures).
- **A run-time / job system** to execute simulations behind the UI (likely async jobs, not in-request).
- **Reproducibility** (seeds) so a shared config reproduces a shared result — already in place.

## Open questions (think about later)

- **Interactivity vs. cost.** A *realistic-scale* ABM (10⁶–10⁹ cells, see DESIGN_scalability §7) is
  almost certainly too slow for live interaction. Options: downsampled live previews, async job
  queue with progress, precomputed parameter sweeps, or a fast surrogate for the preview. The §7
  tau-leaping work directly gates how interactive this can be.
- **Region selection → config.** How does a drawn region on the tumour map translate to a sampling
  config (and similarly for picking treatment targets from an expression view)?
- **State / multi-user / persistence**, deployment, and where compute runs (browser? server? cloud?).
- **Relationship to the parameter-estimation layer** (DESIGN_inference): could the UI suggest
  realistic defaults learned via `estimate()` from a chosen reference dataset.

## Cross-refs
`SCHEMA.md` (the output contract the viz reads), `DESIGN_features.md` (the modules the UI exposes,
esp. F1 region-based sampling), `DESIGN_scalability.md` §7 (interactivity is gated by sim cost).
