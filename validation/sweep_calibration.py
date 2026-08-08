"""Calibration sweep: can the DUCT CONSTRAINT alone produce a truncal copy-number alteration?

THE HYPOTHESIS UNDER TEST. Cells cannot leave the duct until they breach the basement membrane, so a
clone with a proliferative advantage should take over the duct and everything that later invades
should descend from it -- giving a copy-number alteration shared by more than 90% of cancer cells
(a TRUNCAL CNA) out of confinement plus ordinary selection alone, with no founder CNA seeding and no
staged in-situ phase.

THE ANSWER: NO. Across 39 distinct configurations and 57 growth runs -- the 12-cell factorial the
sweep was commissioned for plus seven extension stages along every axis the diagnosis implicated --
``n_truncal_segments`` was **0 in every single run**, at both field scales and on all three seeds.
The best any alteration reached is ``max_cna_freq`` 0.634, in a configuration that never invades;
the best from a lesion whose DCIS->IDC arc survives is 0.506, and what pushes it there is growing
the lesion until the invasive mass dominates, not tightening the duct. This is not a near miss that
better tuning would close: section 4 shows the barrier is arithmetic, and the identity behind it is
exact to three decimals.

Truncal SNVs work throughout (``max_snv_freq`` 0.994-1.000 in every run) -- because they are SEEDED
into the founder before growth, never swept during it. That asymmetry is the whole finding.


1. REPRODUCING IT
-----------------
Every number below comes from this file. Each stage is a CLI subcommand; growth dominates the cost,
so configurations run in parallel worker processes::

    python validation/sweep_calibration.py core                                   # the 12-cell grid
    python validation/sweep_calibration.py diagnose                               # per-duct breakdown
    python validation/sweep_calibration.py grow   --traj 6000 12000 20000 30000
    python validation/sweep_calibration.py churn  --diagnose --target 30000
    python validation/sweep_calibration.py local  --target 30000 --max-gen 400
    python validation/sweep_calibration.py escape --diagnose --target 30000
    python validation/sweep_calibration.py narrow --traj 10000 20000 40000 --max-gen 400
    python validation/sweep_calibration.py gate   --traj 10000 40000 --max-gen 500
    python validation/sweep_calibration.py tradeoff --diagnose --target 30000
    python validation/sweep_calibration.py final  --scale mid --target 100000 --seeds 3 5 7

Scoring is ``sweep_score.score_tumour`` unchanged, so every row is judged by one rule. Seed 3
throughout unless stated. ``--diagnose`` swaps the one-line score for the per-compartment and
ceiling breakdown; ``--traj`` scores the SAME growing tumour at several sizes, which is the control
for the size confound (a lesion that stalls small looks clonal for reasons that have nothing to do
with the duct).

One harness note, because it changes what the numbers mean: growth is bounded (:func:`grow_bounded`).
Under ``crowding_mode="fixed"`` a deme equilibrates well below K, so the field saturates below the
growth target and the stock ``while size < target`` loop never returns. The budget turns that into a
measured result -- flagged ``stalled`` / ``gen-cap`` with the realised ``n_cancer`` in the table --
rather than a hang.


2. THE COMMISSIONED FACTORIAL (small scale, seed 3, target 12k)
---------------------------------------------------------------
A. ``trait_source`` dosage/mutation  x  B. ``crowding_mode`` own/fixed  x  C. ``division_rate``
0.7/0.4/0.25 against ``max_birth_rate`` 0.95::

    config                  n_cancer  clones  maxCNA  trunc  maxSNV    FGA  ploidy  %WGD  stroma%  flags
    dosage own div0.7          12139    3804   0.214      0   1.000  0.214    2.22  10.5     28.5  ploidy-low wgd-low
    dosage own div0.4          12076    3928   0.247      0   1.000  0.311    2.42  14.8     25.7  wgd-low
    dosage own div0.25         12093    3883   0.399      0   1.000  0.335    2.41  10.7     26.1  wgd-low
    dosage fixed div0.7         8364    2800   0.301      0   1.000  0.393    2.75  33.0      0.1  no-invasion stalled
    dosage fixed div0.4         7961    2680   0.285      0   1.000  0.374    2.91  33.3      0.1  no-invasion stalled
    dosage fixed div0.25        8477    2838   0.398      0   1.000  0.454    3.11  32.3      0.0  no-invasion fga/ploidy-high
    mutation own div0.7         9036    3173   0.199      0   0.999  0.236    2.30  15.5      2.0  ploidy-low wgd-low stalled
    mutation own div0.4        12035    3962   0.386      0   0.999  0.453    2.60  28.6     23.0  fga-high
    mutation own div0.25        9077    3098   0.617      0   0.999  0.403    2.27  18.4      3.2  fga-high ploidy-low stalled
    mutation fixed div0.7       8387    2720   0.257      0   0.999  0.349    2.43  21.2      0.1  no-invasion stalled
    mutation fixed div0.4       8333    2673   0.513      0   1.000  0.429    2.37  24.7      0.1  no-invasion fga-high stalled
    mutation fixed div0.25      8284    2652   0.634      0   1.000  0.468    2.13  16.0      0.2  no-invasion fga-high stalled

Row 1 reproduces the independently measured shipped baseline exactly (12139 cells, maxCNA 0.214), so
the harness agrees with the scorer's own control.

What the three axes actually did:

* **C (headroom) is the strongest of the three** and moves in the expected direction: 0.214 ->
  0.247 -> 0.399 (dosage/own), 0.257 -> 0.513 -> 0.634 (mutation/fixed). But it does NOT work by
  opening headroom. The per-clone division distribution shows the opposite: the share of cells
  pinned AT ``max_birth_rate`` 0.95 rises from 24.8% (div 0.7) to 31.5% (div 0.25) under dosage and
  reaches 88.3% in mutation/fixed/div0.25. Lowering the baseline makes the population MORE
  fitness-degenerate, not less, because reaching the target size then takes more generations and the
  extra generations accumulate more drivers. The clonality gain comes from the longer, more crowded
  growth, not from restored selection.
* **A (trait_source="mutation")** lifts maxCNA in 5 of 6 pairings, as predicted -- lifting the
  go-or-grow tax does let copy-number alterations be judged on driver content. It also makes breach
  a rare heritable event instead of a side effect of any CNA, which collapses invasion at this scale
  (stroma 2.0% at div 0.7 vs 28.5% under dosage).
* **B (crowding_mode="fixed")** raises maxCNA and destroys the biology: stroma <= 0.2% in all six
  fixed runs. The mechanism is mechanical -- duct WALL demes are seeded full of epithelial cells at
  K, and the fixed law charges crowding death against a reference of ``max_birth_rate``, so death
  reaches ~1.0 in a full deme and no cancer cell survives the crossing. The lesion is sealed in the
  lumen. ``crowding_ref`` would have to come down with it for "fixed" to be usable here at all.

Best cell that still invades: dosage/own/div0.25 at 0.399. Best overall: mutation/fixed/div0.25 at
0.634, disqualified by ``no-invasion``. Neither is within reach of 0.9.


3. WHERE THE SWEEP FAILS -- IT IS INSIDE ONE DUCT
--------------------------------------------------
``diagnose`` splits the lesion by compartment. If confinement worked and the problem were merely
that several ducts sweep independently, per-duct clonality would be high while the lesion-wide
number stayed low. It is not::

    dosage own div0.7      lesion maxCNA 0.214    duct 0: n=2976 clones=1176 maxCNA=0.206
                                                 duct 1: n=2901 clones=1171 maxCNA=0.166
                                                 duct 2: n=2806 clones=1103 maxCNA=0.163
    mutation fixed div0.25 lesion maxCNA 0.634    duct 0: n=2712 clones= 997 maxCNA=0.625
                                                 duct 1: n=2800 clones=1060 maxCNA=0.675
                                                 duct 2: n=2754 clones=1018 maxCNA=0.620

**Each duct carries ~1,100 distinct clones in ~2,950 cells -- a new clone every 2.6 cells.** A single
duct never sweeps, so the parallel-lineages story (and with it the ``n_glands`` /
``cross_gland_kappa`` axis) is not the blocker. The duct is in permanent clonal interference: with
``mutation_rate`` 0.3 and ``cnv_prob`` 0.35 a clone-defining event lands roughly every ninth
division, so new variants are created far faster than any one of them can expand across the
thousands of cells a 450 um duct holds. Fixation by selection needs ~ln(N)/s divisions; these runs
are tens of divisions deep, against an N of ~3,000 and an s that is mostly absent (see B above).

Duct occupancy sits at 1.2 K in every configuration, so the compartment is genuinely crowded -- the
confinement is real. It just does not select.


4. THE ARITHMETIC BARRIER (this is the actual answer)
-----------------------------------------------------
``tradeoff`` grows the shipped configuration to 30k with WGD off, varying only ``cnv_prob``, and
reads the fraction of the genome altered against the share of cells still carrying the founder's
state at a segment::

    cnv_prob   FGA     founder-state retention   best per-segment state freq
    0.35       0.299            0.700                      0.877
    0.25       0.234            0.765                      0.863
    0.15       0.219            0.781                      0.918
    0.08       0.157            0.843                      0.957
    0.03       0.108            0.892                      0.989

**FGA = 1 - retention, to three decimals, in all five rows.** That is not a coincidence, it is an
identity: every copy-number alteration in iscc is generated DURING growth at a constant per-division
rate, so a cell's altered fraction and the population's failure to agree on a segment are the same
quantity read from opposite ends. Requiring the realistic FGA window 0.2-0.4 therefore forces
retention below 0.8 -- and a truncal alteration needs a single state at 0.9.

Two further consequences fall out of the same table:

* The realism target ``%WGD`` 20-50 is independently fatal. A subclonal whole-genome doubling splits
  the population at EVERY segment: with WGD on, the best per-segment state frequency is 0.646; with
  ``wgd_rate=0`` and nothing else changed it is 0.877. A tumour in which 20-50% of cells are doubled
  is by construction a tumour in which no segment has a 90% consensus state.
* Lowering churn to buy retention costs exactly the FGA that made the tumour realistic. cnv_prob
  0.03 gives 0.892 retention and an FGA of 0.108 -- a nearly flat, diploid genome.

The measured lineage depth explains why the identity bites so hard. Inverting the retention against
the per-segment event rate ``mutation_rate * cnv_prob / n_segments`` gives ~40 divisions at
cnv_prob 0.35 and ~150 at 0.03 -- far more than the ~15 doublings the final cell count implies,
because crowding death makes every deme churn through many birth-death rounds. Deep lineages erase
shared states.


5. THE ONE MECHANISM THAT DOES RAISE CLONALITY, AND WHY IT STOPS AT 0.5
------------------------------------------------------------------------
Clonality is not flat -- it rises as the INVASIVE mass outgrows the residual in-situ disease, which
is the duct constraint acting as an escape bottleneck rather than as a selection chamber
(``grow``, shipped configuration, one growing tumour scored four times)::

    size     maxCNA   stroma%    FGA
     6069    0.044       0.3   0.035
    12098    0.250      30.2   0.182
    20904    0.434      58.3   0.231
    30560    0.506      71.2   0.269

0.506 at 30k is the highest number in this study from a tumour that still invades, and it comes
from the SHIPPED configuration with nothing changed but how long it grew. But the escape is not
narrow enough to be a real bottleneck, and every attempt to narrow it failed:

* ``prop_breach`` 0.005 (``grow``): maxCNA 0.357 at 30k -- WORSE, because under dosage traits breach
  is switched on by copy-number change rather than by breach genes, so shrinking the gene set does
  not make breaching rare.
* ``narrow`` -- one duct, breach as a mutation, no WGD, 92% of the lesion invaded, i.e. the
  bottleneck story at maximum strength: maxCNA 0.386-0.400, flat across sizes 10k/20k/40k.
* Cutting clone generation 6x (``mutation_rate`` 0.05, 802 clones at 10k instead of ~3,000): 0.362.

The invasive mass is polyclonal in every one of them. Once any breach-competent lineage reaches the
permissive stroma, the duct keeps supplying more, and each founds its own colony.

``gate`` then searched the breach rate itself for a window narrow enough to admit ONE lineage and
still admit invasion -- one duct, breach as a mutation, scored at 10k and again at 40k::

    prop_breach   maxCNA @10k   maxCNA @40k   stroma% @40k    FGA @40k   flags @40k
    0.0200           0.391         0.386          92.1         0.389     --
    0.0125           0.334         0.387          92.0         0.387     --
    0.0100           0.414         0.422          91.7         0.374     --
    0.0075           0.272         0.321          92.2         0.364     --

There is no window. The response is non-monotonic and confined to 0.27-0.42; a factor of 2.7 in the
breach rate changes nothing structural, because breach is a per-hop probability that any competent
clone can use rather than a licence one lineage holds. (0.01 at 40k is, incidentally, the most
realistic tumour in the study -- FGA 0.374, ploidy 2.80, 39.4% WGD, 91.7% invaded, no flags at all
-- and its best alteration still sits at 0.422.)

Scaling UP makes it worse, not better -- the mid-scale confirmation at 100k cells (section 8) drops
back to 0.316, because a bigger tumour is a deeper one.


6. WHAT WOULD ACTUALLY BE REQUIRED
-----------------------------------
The engine already produces a truncal SNV layer, and it does it by SEEDING: ``founder_mutations``
writes the transforming hits and the clock-like burden into the founder before growth, which is why
``max_snv_freq`` is ~1.0 in every run here. There is no copy-number equivalent -- founder seeding
covers point mutations only -- and that gap, not the duct, is what this sweep measured.

So a truncal CNA needs one of:

* a founder karyotype (the CNA analogue of the existing founder SNV seeding) -- the same modelling
  choice already made for point mutations, and explicitly out of scope for this test; or
* punctuated copy-number instability: a burst of alterations at or near transformation followed by a
  much lower ongoing rate, so that the altered fraction of the genome is inherited rather than
  accumulated. This also removes the FGA/retention identity, because burst alterations raise FGA
  without lowering retention; or
* a severe LATE bottleneck that resets the population's common ancestor. The breach gate is the only
  candidate the engine has, and section 5 shows it is a leaky one.

None of these is calibration.


7. THE OTHER NAMED AXES, TESTED AND NEGATIVE
---------------------------------------------
``local``, shipped base, all grown to the same 30k so size cannot flatter a stalled run::

    config                          n_cancer  clones  maxCNA    FGA  ploidy  %WGD  stroma%  flags
    shipped                            30653    8974   0.302  0.284    2.40  22.5     70.2  --
    dispersal 0.1                      30136   19729   0.248  0.628    3.48  63.9     68.5  fga/ploidy/wgd-high
    dispersal 0.01                      9235    7022   0.237  0.634    3.34  63.8      1.3  ... stalled
    closed ducts (cross_gland 0)       30112    9886   0.311  0.394    2.82  39.5     79.2  --
    dispersal 0.01 + closed ducts       2778    2140   0.265  0.577    3.06  50.2      0.2  no-invasion stalled

Both of these were expected to help and neither does:

* **dispersal_rate is not the lever it looked like.** Lowering it 0.9 -> 0.1 -> 0.01 LOWERS maxCNA
  (0.302 -> 0.248 -> 0.237) at matched size. An earlier reading in the other direction was a size
  artefact -- low dispersal stalls the lesion (9,235 cells at 0.01, 2,778 with ducts also sealed),
  and a small lesion is trivially more clonal. Holding size fixed reverses the sign. It also wrecks
  the genome: keeping daughters in their birth deme raises turnover, and FGA goes to 0.63 with
  ploidy 3.5 and 64% WGD.
* **Sealing the ducts does nothing.** ``cross_gland_kappa`` 0 makes each duct a genuinely closed
  compartment that must be won before the next is seeded -- the purest form of the confinement
  hypothesis. maxCNA 0.311 against the shipped 0.302. As section 3 already showed, the compartment
  being closed does not matter when the compartment itself never sweeps.


8. CONFIRMATION AT MID SCALE, THREE SEEDS
------------------------------------------
The two best-scoring configurations that keep the biology, grown to 100k cancer cells on the grid-96
five-duct field (``final``)::

    config                         n_cancer  clones  maxCNA  trunc  maxSNV    FGA  ploidy  %WGD  stroma%  flags
    shipped [s3]                     101576   31128   0.316      0   1.000  0.369    2.60  30.2     84.9  --
    shipped [s5]                     100416   31597   0.341      0   0.999  0.349    2.89  41.2     84.9  --
    shipped [s7]                     101232   32133   0.310      0   1.000  0.346    2.81  36.2     84.7  --
    1 duct + mutation breach [s3]    101162   32590   0.344      0   0.999  0.467    2.48  26.5     96.8  fga-high
    1 duct + mutation breach [s5]    100028   32196   0.280      0   0.997  0.463    2.59  30.2     96.7  fga-high
    1 duct + mutation breach [s7]    100737   32661   0.326      0   0.998  0.465    2.88  44.8     96.7  fga-high

Not a fluke in either direction: maxCNA is 0.310-0.341 across three seeds for the shipped
configuration and 0.280-0.344 for the calibrated one, and ``n_truncal_segments`` is 0 in all six.
The shipped tumour at this size is fully realistic -- every realism window satisfied, no flags, 85%
invaded -- and still has no clonal copy-number profile.

Note the direction: the SAME shipped configuration scored 0.506 at 30k (section 5) and 0.316 at
100k. Growing a bigger tumour lowers clonality, because a bigger tumour is a deeper one. Any
calibration judged only at small scale will read as more promising than it is.


9. RECOMMENDATION
-----------------
**Change nothing in ``notebooks/example_config.yaml`` for the sake of truncal CNAs.** No setting of
the parameters in that file produces one, and the two candidate changes both cost more than they
return:

* ``deme_params.crowding_mode: fixed`` -- do NOT adopt, and do not change the engine default. It is
  the right law in principle (it stops the fitness term cancelling at saturation) but in this ductal
  field it seals the lesion inside the duct: stroma <= 0.2% in all six of its runs, because the
  epithelium-filled wall demes sit at K and the law charges crowding death against a reference of
  ``max_birth_rate``, giving death ~1.0 there. If it is ever wanted here, ``crowding_ref`` has to
  come down with it -- roughly to the founder division rate, so that a full deme is a fitness
  threshold rather than a wall. Untested here, and it is a change to the invasion biology, not a
  calibration.
* ``selection_params.trait_source: mutation`` -- reasonable as an OPT-IN, wrong as a default. It
  lifts maxCNA in 5 of 6 factorial pairings and removes a real artefact (a copy-number change
  switching on an invasion trait), but it also makes breach rare enough to collapse invasion at
  small scale (stroma 2.0% vs 28.5%) and pushes FGA above the realistic window (0.45-0.47 in every
  mid-scale run). Leave the engine default at ``dosage``.

The one defensible change is not a parameter at all: **the shipped tumour should be grown until the
invasive mass dominates.** At its current tutorial size the lesion is still mostly in-situ and reads
``ploidy-low`` / ``wgd-low``; grown to ~85% invaded it satisfies every realism window with no flags
(section 8) and its best alteration frequency rises from 0.214 to ~0.32. That is a growth-target
point for the tutorials and benchmarks, not an edit to the YAML.

**What would actually deliver a truncal CNA, with the numbers it needs.** Copy-number seeding of the
founder -- the exact analogue of the ``founder_mutations`` block that already gives this tumour its
truncal SNV layer, which is why ``max_snv_freq`` is ~1.0 everywhere while ``max_cna_freq`` never
passes 0.64 in any configuration, or 0.51 in any that still invades. Seeding alone is not enough, though, and the tradeoff table says by how much: at the
shipped ``cnv_prob`` 0.35 a founder alteration would be eroded to 0.700 by ongoing churn, still
short of 0.9. Retention reaches 0.9 only at ``cnv_prob`` <= ~0.03, where churn contributes an FGA of
just 0.108. So a tumour with a realistic FGA of ~0.3 AND a truncal alteration needs roughly 0.2 of
its genome seeded as a founder karyotype and ``cnv_prob`` around 0.03 for the subclonal remainder --
i.e. the altered fraction becomes mostly inherited rather than mostly accumulated. Whole-genome
doubling has to move the same way: while 20-50% of cells are doubled and the rest are not, the best
per-segment consensus is 0.646 whatever else is done, so a WGD in this tumour should be either
clonal or rare, not half the population.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep_score as SS  # noqa: E402


# --------------------------------------------------------------------------- the factorial axes
#: A. Where a niche trait's value comes from. "dosage" (engine default) lets a pure copy-number change
#: switch a trait on, so the go-or-grow cost taxes nearly every CNA; "mutation" makes only mutated
#: copies count, so a CNA is judged on its driver content alone.
TRAIT_SOURCE = ("dosage", "mutation")

#: B. Crowding law. "own" (engine default) scales the cancer crowding term by the clone's OWN division
#: rate, so the fitness term cancels and a cancer-only deme caps independently of fitness. "fixed"
#: measures crowding death against a fixed reference, so survival under crowding is fitness-dependent.
CROWDING_MODE = ("own", "fixed")

#: C. Fitness headroom: the founder division rate against max_birth_rate 0.95. At 0.7 a clone can only
#: ever be 1.36x the founder before saturating the cap.
DIVISION_RATE = (0.7, 0.4, 0.25)


def spec(trait_source="dosage", crowding_mode="own", division_rate=0.7, label=None, **extra):
    """One sweep cell: ``(label, {parameter block: overrides})``, ready for :func:`grow_bounded`.

    ``extra`` keys are routed to the right parameter block, so an extension axis reads as
    ``spec(cnv_prob=0.1)`` rather than as a nested-dict literal.
    """
    selection = {"trait_source": trait_source}
    deme = {"crowding_mode": crowding_mode}
    cancer = {"division_rate": division_rate}
    spatial = {}
    for key, value in extra.items():
        if key in ("cnv_prob", "wgd_rate", "dispersal_rate", "mutation_rate", "max_birth_rate",
                   "death_rate"):
            cancer[key] = value
        elif key in ("prop_breach", "breach_effects", "breach_cost", "prop_stromal_survival",
                     "stromal_survival_cost", "prop_driver", "driver_effects"):
            selection[key] = value
        elif key in ("n_glands", "cross_gland_kappa", "cross_gland_lambda", "grid_size",
                     "gland_radius", "min_gland_sep", "stromal_hazard", "K_duct", "K_stroma"):
            spatial[key] = value
        elif key in ("crowding_ref", "crowding_margin", "carrying_capacity", "initial_cancer_cells"):
            deme[key] = value
        else:
            raise KeyError(f"unrouted sweep parameter {key!r}")
    if label is None:
        bits = [trait_source, crowding_mode, f"div{division_rate:g}"]
        bits += [f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                 for k, v in extra.items()]
        label = " ".join(bits)
    return label, dict(selection=selection, deme=deme, cancer=cancer, spatial=spatial)


#: Growth budget, in tau-generations. Needed because a configuration can SATURATE below the target:
#: under ``crowding_mode="fixed"`` a deme equilibrates at ``(div - death) / ((ref - death) * steep)``
#: of K rather than at K, so the same field holds far fewer cells and
#: ``realistic_regime.grow_realistic``'s ``while size < target`` loop never terminates. The budget
#: turns that into a measured RESULT (flagged ``stalled`` / ``gen-cap``, with the realised
#: ``n_cancer`` in the table) instead of a hang.
MAX_GEN = 200
#: Stall test: stop when the last ``STALL_WINDOW`` generations grew the lesion by less than this.
#: The window has to be LONG. A breach-gated lesion spends a real DCIS plateau confined in the duct
#: before any cell crosses the basement membrane, and a short window mistakes that plateau for death
#: -- which truncates precisely the confined phase this sweep is about. Measured: a 8-generation /
#: 2% window cut the shipped configuration off at 8.8k cells with 0% of its cancer in the stroma,
#: where the full run reaches 12k with 28% invaded.
STALL_GROWTH, STALL_WINDOW, STALL_MIN_GEN = 0.01, 40, 60


def grow_bounded(seed=3, scale="small", target=None, max_gen=MAX_GEN, verbose=False, **overrides):
    """``realistic_regime.grow_realistic`` with a growth budget: grow to ``target`` cancer cells, or
    until the lesion stops growing, or until ``max_gen`` tau-generations. Returns ``(tumour, note)``
    where ``note`` is ``""``, ``"stalled"`` or ``"gen-cap"``."""
    from iscc.tumor.models import GenotypeTumor
    if target is None:
        target = SS.DEFAULT_TARGET[scale]
    cfg = SS.RR.config_for(scale, **overrides)
    t = GenotypeTumor(seed=seed, update_mode="tau", tau=SS.RR.TAU, coarsen_passengers=SS.RR.COARSEN,
                      max_cells=1, founder_mutations=SS.RR.FOUNDER_MUTATIONS,
                      germline_params=SS.RR.GERMLINE, **cfg)
    gen, note, history = 0, "", []
    while t.get_cancer_size() < target and gen < max_gen:
        ratio = t.get_cancer_size() / target
        n = 8 if ratio < 0.15 else 3 if ratio < 0.6 else 2 if ratio < 0.9 else 1
        t.grow(n_steps=n, seed=seed)
        gen += n
        history.append((gen, t.get_cancer_size()))
        if verbose:
            print(f"    gen {gen}: {t.get_cancer_size()} cancer", flush=True)
        old = [s for g, s in history if g <= gen - STALL_WINDOW]
        if gen >= STALL_MIN_GEN and old and t.get_cancer_size() < old[-1] * (1.0 + STALL_GROWTH):
            note = "stalled"
            break
    else:
        if t.get_cancer_size() < target:
            note = "gen-cap"
    t.cell_data = None
    t.n_generations_grown = gen
    return t, note


def _one(job):
    """Worker entry point: grow + score one configuration. Returns ``(label, score, seconds)``."""
    label, kwargs, seed, scale, target, max_gen = job
    t0 = time.time()
    try:
        t, note = grow_bounded(seed=seed, scale=scale, target=target, max_gen=max_gen, **kwargs)
        score = SS.score_tumour(t)
        score["n_generations"] = int(t.n_generations_grown)
        if note:
            score["flags"] = (score["flags"] + " " + note).strip()
    except Exception as exc:                       # a configuration that cannot grow is a RESULT
        score = {"flags": f"ERROR {type(exc).__name__}: {exc}"[:70]}
    return label, score, time.time() - t0


def run_grid(specs, seed=3, scale="small", target=None, workers=5, max_gen=MAX_GEN, verbose=True):
    """Grow and score every spec, in parallel. Returns ``[(label, score), ...]`` in spec order."""
    jobs = [(label, kwargs, seed, scale, target, max_gen) for label, kwargs in specs]
    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for label, score, secs in pool.map(_one, jobs):
            done += 1
            if verbose:
                print(f"  [{done}/{len(jobs)}] {label}  ({secs:.0f}s)  "
                      f"n={score.get('n_cancer', 0)} "
                      f"maxCNA={score.get('max_cna_freq', float('nan')):.3f} "
                      f"trunc={score.get('n_truncal_segments', 0)} "
                      f"flags='{score.get('flags', '')}'", flush=True)
            rows.append((label, score))
    return rows


# --------------------------------------------------------------------------------------- stages
def stage_core():
    """A. trait_source x B. crowding_mode x C. division_rate -- the 12-cell factorial."""
    return [spec(trait_source=ts, crowding_mode=cm, division_rate=dr)
            for ts in TRAIT_SOURCE for cm in CROWDING_MODE for dr in DIVISION_RATE]


def stage_local():
    """Extension 2: how LOCAL the competition is, on the shipped base.

    ``dispersal_rate`` sets how fast a daughter leaves the deme it was born in -- lower values keep a
    clone competing against its own relatives instead of scattering across the field, which is the
    condition a local sweep needs. ``cross_gland_kappa`` 0 seals each duct off entirely, so a duct is
    a genuinely closed compartment that must be won before the next one is seeded."""
    return [spec(label="shipped"),
            spec(dispersal_rate=0.1, label="dispersal 0.1"),
            spec(dispersal_rate=0.01, label="dispersal 0.01"),
            spec(cross_gland_kappa=0.0, label="closed ducts"),
            spec(dispersal_rate=0.01, cross_gland_kappa=0.0, label="dispersal 0.01 + closed ducts")]


def stage_grow():
    """Extension 4: let the INVASIVE mass outgrow the residual DCIS.

    The per-duct diagnosis says a duct never sweeps, so the only bottleneck the duct constraint
    actually imposes is the ESCAPE: breach is rare, so the invasive mass descends from few cells. If
    that is the mechanism, clonality should rise as the lesion grows past the ducts -- and rise
    further when the escape is rarer (``prop_breach``), when there is only one duct to leave behind
    (``n_glands``), or when less copy-number churn overwrites what the escapee carried
    (``cnv_prob``)."""
    return [spec(label="shipped"),
            spec(n_glands=1, label="1 duct"),
            spec(prop_breach=0.005, label="rare breach"),
            spec(cnv_prob=0.10, label="low churn"),
            spec(prop_breach=0.005, cnv_prob=0.10, n_glands=1, label="all three")]


def stage_escape():
    """Extension 5: is the ceiling set by SUBCLONAL WGD, and is the ESCAPE a real bottleneck?

    Two questions in one grid. (i) ``wgd_rate=0`` removes the whole-genome-doubling channel: if the
    ceiling jumps, the thing splitting the population was WGD, not segment-level churn. (ii)
    ``trait_source="mutation"`` makes breach an actual mutation rather than a side effect of any
    copy-number change, so far fewer cells can cross the basement membrane -- which is what would
    turn the escape into the narrow bottleneck the duct hypothesis needs."""
    return [spec(label="shipped"),
            spec(wgd_rate=0.0, label="no WGD"),
            spec(trait_source="mutation", label="mutation breach"),
            spec(trait_source="mutation", wgd_rate=0.0, label="mutation + no WGD"),
            spec(trait_source="mutation", wgd_rate=0.0, cnv_prob=0.15,
                 label="mutation + no WGD + churn 0.15"),
            spec(trait_source="mutation", wgd_rate=0.0, prop_breach=0.005,
                 label="mutation + no WGD + rare breach")]


def stage_narrow():
    """Extension 6: everything the diagnosis implicates, together.

    ONE duct (a small DCIS reservoir the invasive mass can outgrow), breach as a rare MUTATION (so
    the escape is a narrow bottleneck rather than a quarter of the lesion crossing at once), no
    subclonal WGD (so the population is not split down the middle), and -- in the last cell -- a
    much lower clone-generation rate, since a duct that carries 1,100 clones in 3,000 cells is in
    permanent clonal interference and nothing can fix in it."""
    base = dict(trait_source="mutation", n_glands=1)
    return [spec(**base, label="1 duct + mutation breach"),
            spec(**base, wgd_rate=0.0, label="+ no WGD"),
            spec(**base, wgd_rate=0.0, cnv_prob=0.15, label="+ no WGD + churn 0.15"),
            spec(n_glands=1, wgd_rate=0.0, label="1 duct + dosage breach + no WGD"),
            spec(**base, wgd_rate=0.0, mutation_rate=0.05, label="+ no WGD + few clones")]


def stage_gate():
    """Extension 7: the user's hypothesis at its strongest -- is there a breach rate that makes the
    escape a ONE-LINEAGE bottleneck?

    With ``trait_source="mutation"`` breach is a heritable rare event (an SNV in a breach gene), so
    ``prop_breach`` sets how many independent lineages can ever cross the basement membrane. Too
    high and the invasion is polyclonal; too low and it never happens. If the duct constraint is
    going to produce a truncal alteration anywhere, it is in the window between."""
    base = dict(trait_source="mutation", n_glands=1)
    return [spec(**base, prop_breach=p, label=f"1 duct, mutation breach, prop_breach {p:g}")
            for p in (0.02, 0.0125, 0.01, 0.0075)]


def stage_tradeoff():
    """The trade-off that decides the whole question, with WGD switched off so only segment-level
    churn is in play. Run with ``--diagnose``: read ``FGA`` against ``founder retention``.

    ``frac_genome_altered`` is built entirely from copy-number events generated DURING growth at a
    constant per-division rate. ``founder retention`` is the share of cells that still carry the
    founder's state at a segment, i.e. the ceiling on how clonal any single alteration can be. The
    two are driven by the same rate in opposite directions, so this stage measures how much
    truncality a realistic ``FGA`` costs."""
    return [spec(wgd_rate=0.0, cnv_prob=p, label=f"no WGD, cnv_prob {p:g}")
            for p in (0.35, 0.25, 0.15, 0.08, 0.03)]


def stage_final():
    """The two configurations that scored best at small scale WITHOUT breaking realism, for the
    mid-scale / multi-seed confirmation. Run as
    ``sweep_calibration.py final --scale mid --target 100000 --seeds 3 5 7``."""
    return [spec(label="shipped"),
            spec(trait_source="mutation", n_glands=1, label="1 duct + mutation breach")]


def stage_diagnose():
    """The per-compartment breakdown of the interesting core cells (run through :func:`diagnose`)."""
    return [spec(),
            spec(division_rate=0.25),
            spec(trait_source="mutation", division_rate=0.4),
            spec(trait_source="mutation", crowding_mode="fixed", division_rate=0.25)]


def stage_churn():
    """Extension 1 proper: the CEILING as a function of copy-number churn.

    Run this with ``--diagnose`` -- the interesting number is not ``max_cna_freq`` but
    ``max_state_freq``, the largest fraction of cancer cells sharing ANY state at ANY segment. That
    is an upper bound on truncality that selection cannot touch: if no state is above 0.9, no
    configuration can produce a truncal alteration, whatever the duct does."""
    return [spec(cnv_prob=p, label=f"cnv_prob {p:g}") for p in (0.35, 0.15, 0.05, 0.02)]


STAGES = {"core": stage_core, "local": stage_local,
          "grow": stage_grow, "churn": stage_churn, "escape": stage_escape, "narrow": stage_narrow,
          "gate": stage_gate, "tradeoff": stage_tradeoff, "final": stage_final,
          "diagnose": stage_diagnose}


# ----------------------------------------------------------------------------------- diagnosis
def diagnose(label="?", seed=3, scale="small", target=None, max_gen=MAX_GEN, **kwargs):
    """Grow one configuration and report WHERE a sweep failed, not just that it did.

    ``score_tumour`` measures clonality over the whole lesion, which cannot distinguish "no clone
    ever won anywhere" from "each duct was won by a DIFFERENT clone". This adds the per-compartment
    breakdown that separates them:

    * ``per_gland`` -- for each colonised gland, its cancer count, its clone count and the largest
      fraction of ITS cells sharing one non-diploid segment state. If the duct constraint works at
      all, these are high even when the lesion-wide number is low.
    * ``duct_occupancy`` / ``stroma_occupancy`` -- mean cells per deme over K. A compartment sitting
      far below K is not crowded, and without crowding there is no competition to select on.
    * ``div_mean`` / ``div_p90`` / ``frac_at_cap`` -- the realised division-rate distribution. A
      population piled up at ``max_birth_rate`` is fitness-degenerate: no clone can out-divide it.
    """
    import numpy as np
    t, note = grow_bounded(seed=seed, scale=scale, target=target, max_gen=max_gen, **kwargs)
    score = SS.score_tumour(t)
    if note:
        score["flags"] = (score["flags"] + " " + note).strip()

    n_seg = int(t.n_segments)

    # THE CEILING. For a segment to be truncal SOME state must sit at >=0.9 of cancer cells, so the
    # largest per-segment state frequency is a hard upper bound on truncality that owes nothing to
    # selection. The founder is all-diploid, so ``founder_retention`` -- the mean fraction of cells
    # still at copy number 2 -- measures how much copy-number CHURN has erased the founder's state.
    # A segment changes at ``mutation_rate * cnv_prob / n_segments`` per division, so after D
    # divisions the retention is about exp(-that * D): the ceiling falls with churn and with depth,
    # and once it is under 0.9 NO configuration can produce a truncal CNA however hard it sweeps.
    counts = {}
    n_cancer_tot = 0
    for gid, cnt in t.genotypes_counts.items():
        if cnt > 0 and t._is_cancer(gid):
            n_cancer_tot += int(cnt)
            cns = t.genotypes[gid].genome_summary["seg_cns"]
            for s in range(n_seg):
                counts.setdefault(s, {})
                cn = int(cns[s])
                counts[s][cn] = counts[s].get(cn, 0) + int(cnt)
    max_state = [max(v.values()) / n_cancer_tot for v in counts.values()]
    diploid = [v.get(2, 0) / n_cancer_tot for v in counts.values()]

    per_gland, gland_cells = {}, {}
    for di, deme in enumerate(t.demes):
        g = int(t.gland_id[di]) if t.gland_id is not None else -1
        for gid, cnt in deme.items():
            if cnt > 0 and t._is_cancer(gid):
                gland_cells.setdefault(g, {}).setdefault(gid, 0)
                gland_cells[g][gid] += int(cnt)
    for g, clones in sorted(gland_cells.items()):
        n = sum(clones.values())
        best = 0.0
        for s in range(n_seg):
            states = {}
            for gid, cnt in clones.items():
                cn = int(t.genotypes[gid].genome_summary["seg_cns"][s])
                if cn != 2:
                    states[cn] = states.get(cn, 0) + cnt
            if states:
                best = max(best, max(states.values()) / n)
        per_gland[g] = dict(n_cancer=n, n_clones=len(clones), max_cna_freq=float(best))

    duct_occ, duct_n, stroma_occ, stroma_n = 0.0, 0, 0.0, 0
    for di, deme in enumerate(t.demes):
        tot = sum(deme.values())
        K = (t.carrying_capacity if t._deme_capacity is None else float(t._deme_capacity[di]))
        has_cancer = any(t._is_cancer(g) for g in deme)
        if not has_cancer:
            continue
        if t.gland_id is not None and t.gland_id[di] >= 0:
            duct_occ += tot / K; duct_n += 1
        else:
            stroma_occ += tot / K; stroma_n += 1

    divs, wts = [], []
    for gid, cnt in t.genotypes_counts.items():
        if cnt > 0 and t._is_cancer(gid):
            divs.append(t.genotypes[gid].evolutionary_parameters["division_rate"])
            wts.append(int(cnt))
    divs, wts = np.asarray(divs, float), np.asarray(wts, float)
    cap = float(getattr(t.genotypes[t.founder_id], "max_birth_rate", 0.95))
    order = np.argsort(divs)
    cum = np.cumsum(wts[order]) / wts.sum()

    return dict(
        label=label, score=score, per_gland=per_gland,
        max_state_freq=float(max(max_state)), mean_state_freq=float(np.mean(max_state)),
        founder_retention=float(np.mean(diploid)),
        duct_occupancy=float(duct_occ / duct_n) if duct_n else float("nan"),
        stroma_occupancy=float(stroma_occ / stroma_n) if stroma_n else float("nan"),
        div_mean=float((divs * wts).sum() / wts.sum()),
        div_p90=float(divs[order][int(np.searchsorted(cum, 0.9))]),
        frac_at_cap=float(wts[divs >= cap - 1e-9].sum() / wts.sum()),
        max_birth_rate=cap,
    )


def trajectory(label="?", seed=3, scale="small", checkpoints=(1000, 2000, 4000, 8000, 12000),
               max_gen=MAX_GEN, **overrides):
    """Score ONE growing tumour repeatedly, at each size in ``checkpoints``.

    This is the control for the size confound. Clonality falls as a lesion grows -- more divisions,
    more independent copy-number events -- so a configuration that stalls at 8k cells looks more
    clonal than one that reaches 12k for reasons that have nothing to do with the duct. Comparing
    two configurations along their own size trajectories removes that: the question becomes whether
    clonality is higher AT THE SAME SIZE, and whether it survives the invasive expansion.
    """
    from iscc.tumor.models import GenotypeTumor
    cfg = SS.RR.config_for(scale, **overrides)
    t = GenotypeTumor(seed=seed, update_mode="tau", tau=SS.RR.TAU, coarsen_passengers=SS.RR.COARSEN,
                      max_cells=1, founder_mutations=SS.RR.FOUNDER_MUTATIONS,
                      germline_params=SS.RR.GERMLINE, **cfg)
    out, gen = [], 0
    for target in checkpoints:
        while t.get_cancer_size() < target and gen < max_gen:
            t.grow(n_steps=2, seed=seed)
            gen += 2
        s = SS.score_tumour(t)
        s["n_generations"] = gen
        out.append((f"{label} @{t.get_cancer_size()}", s))
        if gen >= max_gen:
            break
    t.cell_data = None
    return out


def _traj_job(job):
    label, kwargs, seed, scale, checkpoints, max_gen = job
    return trajectory(label=label, seed=seed, scale=scale, checkpoints=checkpoints,
                      max_gen=max_gen, **kwargs)


def _diagnose_job(job):
    label, kwargs, seed, scale, target, max_gen = job
    return diagnose(label=label, seed=seed, scale=scale, target=target, max_gen=max_gen, **kwargs)


def print_diagnosis(d):
    print(f"--- {d['label']}")
    print(f"    lesion: n={d['score']['n_cancer']} maxCNA={d['score']['max_cna_freq']:.3f}"
          f" trunc={d['score']['n_truncal_segments']}/{d['score']['n_truncal_segments_vs_clonal']}"
          f" clones={d['score']['n_clones']} FGA={d['score']['frac_genome_altered']:.3f}"
          f" ploidy={d['score']['mean_ploidy']:.2f} %WGD={d['score']['pct_wgd']:.1f}"
          f" stroma%={d['score']['stroma_pct']:.1f} flags='{d['score']['flags']}'")
    print(f"    CEILING: best per-segment state freq {d['max_state_freq']:.3f}"
          f" (mean {d['mean_state_freq']:.3f}); founder diploid state retained in"
          f" {100 * d['founder_retention']:.1f}% of cells")
    print(f"    occupancy: duct {d['duct_occupancy']:.2f} K, stroma {d['stroma_occupancy']:.2f} K")
    print(f"    division: mean {d['div_mean']:.3f}, p90 {d['div_p90']:.3f},"
          f" at cap({d['max_birth_rate']:.2f}) {100 * d['frac_at_cap']:.1f}%")
    for g, pg in sorted(d["per_gland"].items()):
        name = "stroma" if g < 0 else f"duct {g}"
        print(f"    {name:>8}: n={pg['n_cancer']:>7d} clones={pg['n_clones']:>6d}"
              f" maxCNA={pg['max_cna_freq']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=sorted(STAGES) + ["all"])
    ap.add_argument("--scale", default="small")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="repeat the whole stage on each seed (fluke check)")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-gen", type=int, default=MAX_GEN)
    ap.add_argument("--diagnose", action="store_true",
                    help="per-compartment + ceiling breakdown instead of the one-line score")
    ap.add_argument("--traj", type=int, nargs="*", default=None,
                    help="score each configuration at these sizes as it grows, not just at the end")
    ap.add_argument("--json", default=None, help="also write the raw scores here")
    args = ap.parse_args(argv)

    if args.diagnose or args.stage == "diagnose":
        specs = STAGES[args.stage]()
        jobs = [(label, kwargs, args.seed, args.scale, args.target, args.max_gen)
                for label, kwargs in specs]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for d in pool.map(_diagnose_job, jobs):
                print_diagnosis(d)
        return None

    names = sorted(STAGES) if args.stage == "all" else [args.stage]
    seeds = args.seeds or [args.seed]
    rows = []
    for name in names:
        specs = STAGES[name]()
        for seed in seeds:
            tag = f"{name} @ {args.scale} seed {seed}"
            print(f"== {tag}: {len(specs)} configurations ==", flush=True)
            t0 = time.time()
            if args.traj is not None:
                jobs = [(label, kwargs, seed, args.scale, tuple(args.traj), args.max_gen)
                        for label, kwargs in specs]
                got = []
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    for part in pool.map(_traj_job, jobs):
                        for lab, s in part:
                            print(f"  {lab}: maxCNA={s['max_cna_freq']:.3f} "
                                  f"trunc={s['n_truncal_segments']} stroma%={s['stroma_pct']:.1f} "
                                  f"FGA={s['frac_genome_altered']:.3f}", flush=True)
                        got += part
            else:
                got = run_grid(specs, seed=seed, scale=args.scale, target=args.target,
                               workers=args.workers, max_gen=args.max_gen)
            if len(seeds) > 1:
                got = [(f"{lab} [s{seed}]", s) for lab, s in got]
            rows += got
            print(f"   ({time.time() - t0:.0f}s wall)\n", flush=True)
    print(SS.summarise(rows))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{"label": lab, **s} for lab, s in rows], f, indent=1)
    return rows


if __name__ == "__main__":
    main()
