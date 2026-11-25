"""Fig 5's batch levels, from the fitted technical parameters (``batch_fits.py``'s fits.json).

The levels of panels b-c and the batch model were registered before any run. Given the fits, the cohort's patients and a pooling
design, :func:`levels` returns, for each level, the scRNA overrides of every batch
(``run_scrna_batches(batch_kwargs=...)`` in the balanced design, one ``scRNA(**kwargs)`` per patient
in the confounded design) and every patient's dissociation settings (recovery, stored with the
cells; the stress strength, applied with ``iscc.sample.dissociation.apply_stress`` to the shared
stress-gene set when the counts are made). ``fig5_atlas.py integrate`` consumes it.

The rules, as registered:

* **Draws.** Each simulated batch takes the WHOLE fitted vector of one real tumour (Wu et al. 2021),
  drawn with replacement, so the parameters keep their real correlations; the tumour drawn is
  recorded per batch. Each patient's dissociation stress likewise comes from one drawn tumour. The
  patients' draws are the same in both designs (a patient is dissociated the same way however it
  is sequenced). In the confounded design a patient's batch takes the patient's own tumour, so its
  dissociation and its sequencing come from one real tumour, as in the atlas; in the balanced
  design every batch draws its own tumour.
* **Ambient.** A tumour whose ambient fraction was not identifiable has none in its vector; a batch
  that draws it takes a value drawn for that batch from the registered range in fits.json
  (``ambient_range``, :func:`batch_fits.draw_ambient`), recorded per batch. Without that range the
  levels refuse to be made (``batch_fits.require_ambient_range``).
* **fitted:** every batch its own drawn vector (``mu_lib``, ``sigma_lib``, ``dispersion``,
  ``ambient_frac``, ``dropout_mid``/``dropout_shape``, ``doublet_rate``), every patient its own
  stress; ``sigma_batch`` the 10x preset (0.29).
* **shared:** every batch the median fitted vector, every patient the median stress; batches then
  differ only by their realisation (the old 10x level at the atlas's median).
* **attribution arms** (``depth``, ``dispersion``, ``ambient``, ``dropout``, ``doublets``,
  ``stress``): the shared level with one source varying per batch (per patient for ``stress``) as at
  the fitted level, i.e. from the same draws.
* **reference:** the same cells, with their recovery, without the stress signature, in one batch
  without a batch effect (``sigma_batch`` 0, ``depth_batch_sigma`` 0) at the median vector.
* ``sigma_batch`` is the 10x preset (0.29, fitted on two 10x runs of one blood sample; 0.10 before
  2026-11-25) at every level but the reference. ``depth_batch_sigma``
  is 0 wherever ``mu_lib`` is drawn per batch (fitted, depth), so depth is not counted twice, and
  the 10x preset (0.05) where every batch shares the median ``mu_lib``.
* Recovery is the patient's own draw (fits.json ``recovery``, keyed by the cohort index) at every
  level, the reference included.

Interpretations fixed here (not spelled out by the registration): the median vector is the median
of each parameter over the fitted tumours, ``dropout_shape``'s over the tumours with dropout and
``ambient_frac``'s over the tumours whose fraction was fitted (if none was, the middle of the
registered range); a batch whose tumour has no dropout gets ``dropout_mid`` 0 and the median shape,
which is then unused.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(REPO, "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from iscc.data.rna import PROTOCOL_PRESETS  # noqa: E402

SIGMA_BATCH = PROTOCOL_PRESETS["10x"]["sigma_batch"]    # fitted on two 10x runs of one sample (0.29)
TECH = ("mu_lib", "sigma_lib", "dispersion", "ambient_frac", "dropout_mid", "dropout_shape", "doublet_rate")
SOURCES = {"depth": ("mu_lib", "sigma_lib"), "dispersion": ("dispersion",), "ambient": ("ambient_frac",),
           "dropout": ("dropout_mid", "dropout_shape"), "doublets": ("doublet_rate",)}
ARMS = tuple(SOURCES) + ("stress",)
LEVELS = ("fitted", "shared") + ARMS + ("reference",)
DESIGNS = ("confounded", "balanced")
DRAW_SEED = 20261117


def preset_depth_batch_sigma():
    return float(PROTOCOL_PRESETS["10x"]["depth_batch_sigma"])


def load(path):
    with open(path) as f:
        return json.load(f)


def tumour_vectors(fits):
    """The fitted tumours' vectors (``fits["wu"]["vectors"]``); a tumour whose ambient fraction was not
    fitted keeps ``ambient_frac`` None (drawn per batch from the registered range)."""
    return {t: dict(v) for t, v in fits["wu"]["vectors"].items()}


def median_vector(vectors, ambient_range=None):
    ts = sorted(vectors)
    out = {}
    for p in TECH + ("stress_strength",):
        if p == "dropout_shape":
            vals = [vectors[t][p] for t in ts if vectors[t].get("dropout_mid", 0) > 0 and vectors[t].get(p) is not None]
            out[p] = float(np.median(vals)) if vals else 1.0
        elif p == "ambient_frac":
            vals = [vectors[t][p] for t in ts if vectors[t].get(p) is not None]
            if vals:
                out[p] = float(np.median(vals))
            else:
                out[p] = 0.5 * (ambient_range["rho_low"] + ambient_range["rho_high"])   # the soup share
        else:
            out[p] = float(np.median([vectors[t][p] for t in ts]))
    return out


def draws(vectors, patients, n_balanced_batches=None, seed=DRAW_SEED):
    """Tumour draws with replacement: one per patient (dissociation stress, and the confounded
    design's batch) and one per balanced batch."""
    ts = sorted(vectors)
    rng = np.random.default_rng(seed)
    patients = [str(p) for p in patients]
    nb = len(patients) if n_balanced_batches is None else int(n_balanced_batches)
    pt = rng.choice(ts, size=len(patients), replace=True)
    bt = rng.choice(ts, size=nb, replace=True)
    return dict(seed=int(seed), tumours=ts, patient_tumour={p: str(t) for p, t in zip(patients, pt)},
                balanced_batch_tumour=[str(t) for t in bt])


def _ambient_draws(tumours, vectors, ambient_range, key):
    """Per batch: the tumour's fitted ambient fraction, or one drawn for that batch from the range
    (a generator keyed by the draw seed and the design, so every level sees the same draws)."""
    import batch_fits as BF
    rng = np.random.default_rng([DRAW_SEED, 1, key])
    out = []
    for t in tumours:
        if vectors[t].get("ambient_frac") is not None:
            out.append((float(vectors[t]["ambient_frac"]), "fitted"))
        else:
            a, rho = BF.draw_ambient(ambient_range, rng)
            out.append((a, f"drawn from the registered range (rho {rho:.4f})"))
    return out


def _batch_kwargs(vec, med, vary, depth_sigma_preset, ambient=None):
    """One batch's scRNA overrides: the median vector with the parameters in ``vary`` from ``vec``."""
    kw = {p: med[p] for p in TECH}
    for p in vary:
        kw[p] = ambient if (p == "ambient_frac" and ambient is not None) else vec[p]
    if kw["dropout_mid"] <= 0:
        kw["dropout_mid"], kw["dropout_shape"] = 0.0, med["dropout_shape"]
    elif kw.get("dropout_shape") is None:
        kw["dropout_shape"] = med["dropout_shape"]
    kw["sigma_batch"] = SIGMA_BATCH
    kw["depth_batch_sigma"] = 0.0 if "mu_lib" in vary else depth_sigma_preset
    return {k: float(v) for k, v in kw.items()}


def level_settings(fits, patients, design, level, n_balanced_batches=None, seed=DRAW_SEED):
    """The scRNA overrides per batch and the dissociation settings per patient of one level."""
    import batch_fits as BF
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    ambient_range = BF.require_ambient_range(fits)
    patients = [str(p) for p in patients]
    vectors = tumour_vectors(fits)
    med = median_vector(vectors, ambient_range)
    dr = draws(vectors, patients, n_balanced_batches, seed)
    rec = fits["recovery"]["per_patient"]
    missing = [p for p in patients if p not in rec]
    if missing:
        raise ValueError(f"no recovery draw for patient(s) {missing[:5]} (drawn for {len(rec)} patients)")
    dsig = preset_depth_batch_sigma()

    if level == "fitted":
        vary, stress_own = TECH, True
    elif level in SOURCES:
        vary, stress_own = SOURCES[level], False
    elif level == "stress":
        vary, stress_own = (), True
    else:                                   # shared, reference
        vary, stress_own = (), False

    pats = {}
    for p in patients:
        t = dr["patient_tumour"][p]
        stress = 0.0 if level == "reference" else (vectors[t]["stress_strength"] if stress_own
                                                   else med["stress_strength"])
        pats[p] = dict(recovery_probs=dict(rec[p]), stress_strength=float(stress),
                       stress_tumour=(t if stress_own and level != "reference" else None),
                       drawn_tumour=t)

    ambient_src = None
    if level == "reference":
        kw = _batch_kwargs(med, med, (), dsig)
        kw.update(sigma_batch=0.0, depth_batch_sigma=0.0)
        batches, labels, tumours = [kw], ["reference"], [None]
    else:
        if design == "confounded":
            tumours = [dr["patient_tumour"][p] for p in patients]
            labels = [f"batch{p}" for p in patients]
        else:
            tumours = list(dr["balanced_batch_tumour"])
            labels = [f"batch{i}" for i in range(len(tumours))]
        amb = _ambient_draws(tumours, vectors, ambient_range, DESIGNS.index(design))
        batches = [_batch_kwargs(vectors[t], med, vary, dsig, ambient=a) for t, (a, _) in zip(tumours, amb)]
        if "ambient_frac" in vary:
            ambient_src = [s for _, s in amb]
    return dict(level=level, design=design, batches=batches, batch_labels=labels,
                batch_tumours=[(t if vary else None) for t in tumours] if level != "reference" else [None],
                batch_tumours_drawn=tumours, batch_ambient_source=ambient_src, varies=list(vary),
                patients=pats, median_vector=med, draws=dr, ambient_range=ambient_range,
                sigma_batch=(0.0 if level == "reference" else SIGMA_BATCH))


def levels(fits, patients, design, n_balanced_batches=None, seed=DRAW_SEED):
    """Every level of one design: ``{level: level_settings(...)}``."""
    return {lv: level_settings(fits, patients, design, lv, n_balanced_batches, seed) for lv in LEVELS}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Print one design's levels as JSON")
    p.add_argument("fits")
    p.add_argument("--design", choices=DESIGNS, required=True)
    p.add_argument("--n-patients", type=int, default=50)
    p.add_argument("--n-balanced-batches", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    res = levels(load(a.fits), list(range(a.n_patients)), a.design, a.n_balanced_batches)
    txt = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()
