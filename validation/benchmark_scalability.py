"""Benchmark the genotype-count engine's size/cost envelope (DESIGN_scalability.md §7).

Two engines are benchmarked:
  * the EXACT one-event-per-update engine (the §7 "required assessment" baseline), and
  * the TAU-LEAPING engine (generation-batched clonal update), added in this session.

For each we report cells/s and events/s (or generations/s), whether throughput degrades as
the tumour grows (#demes / #genotypes), and a projection of the wall-time to reach Noble parity
(10^6 cells) and diagnosis scale (10^9 cells).

Run:  python -u validation/benchmark_scalability.py            # both engines, default sizes
      python -u validation/benchmark_scalability.py --exact-steps 30000 --tau-target 1000000
"""
import argparse
import time

import numpy as np

from iscc.tumor.models import GenotypeTumor

# Canonical 10x1000 = 10k-gene genome (matches the §1 throughput table). carrying_capacity=1
# means crowd multiplier == capacity == 1, i.e. NO crowding ceiling -> unbounded growth, which
# is what we want when measuring the size/time envelope. A low death rate keeps the founder from
# stochastically dying out so a single run reaches large sizes.
GENOME = {"n_segments": 10, "segment_size": 1000}
SELECTION = {"prop_driver": 0.1, "prop_dispersal": 0.1,
             "prop_immune_resistance": 0.1, "prop_treatment_resistance": 0.1}
DEME = {"carrying_capacity": 1, "maximum_death_rate": 0.5}


def cancer_params(mutation_rate=0.01):
    # Low mutation_rate is the realistic clonal regime: most divisions grow an existing clone,
    # only a small fraction spawn a new genotype. (High mutation_rate makes #genotypes ~ #cells
    # in BOTH engines -- an infinite-sites property, not an engine cost -- which is the separate
    # §3 concern.) division >> death so growth is robustly positive.
    return {"division_rate": 0.3, "death_rate": 0.02, "max_birth_rate": 0.8,
            "mutation_rate": mutation_rate, "dispersal_rate": 0.1,
            "snv_prob": 0.5, "cnv_prob": 0.5, "n_events": 2, "amp_prob": 0.5}


def _make(grid_size=64, mutation_rate=0.01, update_mode="exact", **kw):
    return GenotypeTumor(
        seed=1, genome_params=GENOME, selection_params=SELECTION,
        cancer_cell_params=cancer_params(mutation_rate), deme_params=DEME,
        spatial_params={"grid_size": grid_size, "n_structures": 1, "structure_radius": 0},
        update_mode=update_mode, **kw)


def project(label, cells_per_s):
    for target, name in [(1e6, "10^6 (Noble)"), (1e9, "10^9 (diagnosis)")]:
        secs = target / cells_per_s
        print(f"    {label} -> {name:18s}: {secs:12.1f} s  = {secs/3600:8.2f} h  = {secs/86400:7.2f} d")


def bench_exact(total_steps, grid_size=64, mutation_rate=0.01, chunk=2000):
    """One event per update(): events/s == steps/wall; cells/s == final_size/wall.

    Run in chunks and time each so we can see whether per-event cost grows with #demes /
    #genotypes (bottleneck 2). The engine internally re-seeds per step, so chunking is exact.
    """
    print(f"\n=== EXACT engine: {total_steps} steps, grid {grid_size}x{grid_size}, "
          f"mut={mutation_rate} ===")
    t = _make(grid_size=grid_size, mutation_rate=mutation_rate, update_mode="exact")
    rng = t.rng
    done = 0
    win = []
    t0 = time.perf_counter()
    while done < total_steps:
        n = min(chunk, total_steps - done)
        c0 = time.perf_counter()
        for _ in range(n):
            t.update(rng)
            t.step += 1
        c1 = time.perf_counter()
        done += n
        size = t.get_cancer_size()
        eps = n / (c1 - c0)
        win.append((done, size, len(t.genotypes_counts), eps))
        print(f"  step {done:7d}  size {size:8d}  genotypes {len(t.genotypes_counts):6d}  "
              f"{eps:8.0f} events/s")
        if size == 0:
            print("  (extinct)"); break
    wall = time.perf_counter() - t0
    size = t.get_cancer_size()
    print(f"  TOTAL: {done} events, {size} cells, {wall:.2f}s  "
          f"-> {done/wall:.0f} events/s, {size/wall:.0f} cells/s")
    # degradation check: first vs last window events/s
    if len(win) >= 2:
        print(f"  throughput first window {win[0][3]:.0f} ev/s vs last {win[-1][3]:.0f} ev/s "
              f"(ratio {win[-1][3]/win[0][3]:.2f})")
    project("EXACT", size / wall if wall else 0)
    return t, win


def bench_tau(target_size, grid_size=128, mutation_rate=0.01, tau=1.0, max_gen=100000):
    """Tau-leaping: advance ALL clones once per generation. cells/s == final_size/wall;
    generations/s == n_gen/wall. Cost per generation scales with #occupied-demes x #clones,
    NOT #cells -- so a large clone advances in one Poisson draw."""
    print(f"\n=== TAU-LEAPING engine: target {target_size:.0f} cells, grid {grid_size}x{grid_size}, "
          f"mut={mutation_rate}, tau={tau} ===")
    t = _make(grid_size=grid_size, mutation_rate=mutation_rate, update_mode="tau",
              tau=tau, snapshot_every=1)
    rng = t.rng
    t0 = time.perf_counter()
    gen = 0
    last = 0.0
    while gen < max_gen:
        t._tau_generation(rng, tau)
        gen += 1
        size = t.get_cancer_size()
        now = time.perf_counter() - t0
        if now - last > 0.5 or size >= target_size:
            print(f"  gen {gen:6d}  size {size:10d}  genotypes {len(t.genotypes_counts):7d}  "
                  f"occ-demes {sum(1 for d in t.demes if d):6d}  {now:7.2f}s  "
                  f"({gen/now:.1f} gen/s, {size/now:.0f} cells/s)")
            last = now
        if size == 0:
            print("  (extinct)"); break
        if size >= target_size:
            break
    wall = time.perf_counter() - t0
    size = t.get_cancer_size()
    print(f"  TOTAL: {gen} generations, {size} cells, {wall:.2f}s  "
          f"-> {gen/wall:.1f} gen/s, {size/wall:.0f} cells/s")
    project("TAU", size / wall if wall else 0)
    return t, gen, size, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact-steps", type=int, default=20000)
    ap.add_argument("--exact-grid", type=int, default=64)
    ap.add_argument("--tau-target", type=float, default=1e6)
    ap.add_argument("--tau-grid", type=int, default=128)
    ap.add_argument("--mut", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--skip-tau", action="store_true")
    args = ap.parse_args()

    if not args.skip_exact:
        bench_exact(args.exact_steps, grid_size=args.exact_grid, mutation_rate=args.mut)
    if not args.skip_tau:
        bench_tau(args.tau_target, grid_size=args.tau_grid, mutation_rate=args.mut, tau=args.tau)


if __name__ == "__main__":
    main()
