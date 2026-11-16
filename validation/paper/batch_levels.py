"""Fig 5's batch levels, from the fitted technical parameters (``batch_fits.py``'s fits.json).

Registered in exp-paper1-fig5-cohort, "Amendment, 2026-11-16 night", section "The levels of panels
b-c"; the model is decision-scrna-batches-from-technical-sources. Given the fits, the cohort's
patients and a pooling design, :func:`levels` returns, for each level, the scRNA overrides of every
batch (``run_scrna_batches(batch_kwargs=...)`` in the balanced design, one ``scRNA(**kwargs)`` per
patient in the confounded design) and every patient's dissociation settings
(``Resection.dissociate(recovery_probs=..., stress_strength=...)``, or the stress applied to the
recovered cells with ``iscc.sample.dissociation.apply_stress``).

The rules, as registered:

* **Draws.** Each simulated batch takes the WHOLE fitted vector of one real tumour (Wu et al. 2021),
  drawn with replacement, so the parameters keep their real correlations; the tumour drawn is
  recorded per batch. Each patient's dissociation stress likewise comes from one drawn tumour. The
  patients' draws are the same in both designs (a patient is dissociated the same way however it
  is sequenced). In the confounded design a patient's batch takes the patient's own tumour, so its
  dissociation and its sequencing come from one real tumour, as in the atlas; in the balanced
  design every batch draws its own tumour.
* **fitted:** every batch its own drawn vector (``mu_lib``, ``sigma_lib``, ``dispersion``,
  ``ambient_frac``, ``dropout_mid``/``dropout_shape``, ``doublet_rate``), every patient its own
  stress; ``sigma_batch`` 0.10.
* **shared:** every batch the median fitted vector, every patient the median stress; batches then
  differ only by their realisation (the old 10x level at the atlas's median).
* **attribution arms** (``depth``, ``dispersion``, ``ambient``, ``dropout``, ``doublets``,
  ``stress``): the shared level with one source varying per batch (per patient for ``stress``) as at
  the fitted level, i.e. from the same draws.
* **reference:** the same cells, with their recovery, without the stress signature, in one batch
  without a batch effect (``sigma_batch`` 0, ``depth_batch_sigma`` 0) at the median vector.
* ``sigma_batch`` is 0.10 (the 10x preset) at every level but the reference. ``depth_batch_sigma``
  is 0 wherever ``mu_lib`` is drawn per batch (fitted, depth), so depth is not counted twice, and
  the 10x preset (0.05) where every batch shares the median ``mu_lib``.
* Recovery is the patient's own draw (fits.json ``recovery``) at every level, the reference included.

Interpretations fixed here (not spelled out by the registration): the median vector is the median
of each parameter over the fitted tumours, ``dropout_shape``'s over the tumours with dropout (a
tumour with ``dropout_mid`` 0 has no fitted shape); a batch whose tumour has no dropout gets
``dropout_mid`` 0 and the median shape, which is then unused.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

SIGMA_BATCH = 0.10
TECH = ("mu_lib", "sigma_lib", "dispersion", "ambient_frac", "dropout_mid", "dropout_shape", "doublet_rate")
SOURCES = {"depth": ("mu_lib", "sigma_lib"), "dispersion": ("dispersion",), "ambient": ("ambient_frac",),
           "dropout": ("dropout_mid", "dropout_shape"), "doublets": ("doublet_rate",)}
ARMS = tuple(SOURCES) + ("stress",)
LEVELS = ("fitted", "shared") + ARMS + ("reference",)
DESIGNS = ("confounded", "balanced")
DRAW_SEED = 20261117


def preset_depth_batch_sigma():
    from iscc.data.rna import PROTOCOL_PRESETS
    return float(PROTOCOL_PRESETS["10x"]["depth_batch_sigma"])


def load(path):
    with open(path) as f:
        return json.load(f)


def tumour_vectors(fits, ambient_range=None, rng=None):
    """The fitted tumours' vectors (``fits["wu"]["vectors"]``). Without a fitted ambient fraction
    (identifiable in fewer than half the tumours) the registration draws it from a stated range,
    which must then be passed as ``ambient_range`` (low, high); each tumour vector then gets one
    uniform draw, recorded in the returned vectors."""
    vec = {t: dict(v) for t, v in fits["wu"]["vectors"].items()}
    missing = [t for t, v in vec.items() if v.get("ambient_frac") is None]
    if missing:
        if ambient_range is None:
            raise ValueError(f"the ambient fraction was not fitted ({len(missing)} tumours); the registration "
                             "draws it from a stated range: pass ambient_range=(low, high)")
        lo, hi = (float(x) for x in ambient_range)
        rng = rng if rng is not None else np.random.default_rng(DRAW_SEED + 1)
        for t in sorted(missing):
            vec[t]["ambient_frac"] = float(rng.uniform(lo, hi))
            vec[t]["ambient_drawn_from"] = [lo, hi]
    return vec


def median_vector(vectors):
    ts = sorted(vectors)
    out = {}
    for p in TECH + ("stress_strength",):
        if p == "dropout_shape":
            vals = [vectors[t][p] for t in ts if vectors[t].get("dropout_mid", 0) > 0 and vectors[t].get(p) is not None]
            out[p] = float(np.median(vals)) if vals else 1.0
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


def _batch_kwargs(vec, med, vary, depth_sigma_preset):
    """One batch's scRNA overrides: the median vector with the parameters in ``vary`` from ``vec``."""
    kw = {p: med[p] for p in TECH}
    for p in vary:
        kw[p] = vec[p]
    if kw["dropout_mid"] <= 0:
        kw["dropout_mid"], kw["dropout_shape"] = 0.0, med["dropout_shape"]
    elif kw.get("dropout_shape") is None:
        kw["dropout_shape"] = med["dropout_shape"]
    kw["sigma_batch"] = SIGMA_BATCH
    kw["depth_batch_sigma"] = 0.0 if "mu_lib" in vary else depth_sigma_preset
    return {k: float(v) for k, v in kw.items()}


def level_settings(fits, patients, design, level, n_balanced_batches=None, seed=DRAW_SEED,
                   ambient_range=None):
    """The scRNA overrides per batch and the dissociation settings per patient of one level."""
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    patients = [str(p) for p in patients]
    vectors = tumour_vectors(fits, ambient_range)
    med = median_vector(vectors)
    dr = draws(vectors, patients, n_balanced_batches, seed)
    rec = fits["recovery"]["per_patient"]
    if len(patients) > len(rec):
        raise ValueError(f"{len(patients)} patients but recovery was drawn for {len(rec)}")
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
    for i, p in enumerate(patients):
        t = dr["patient_tumour"][p]
        stress = 0.0 if level == "reference" else (vectors[t]["stress_strength"] if stress_own
                                                   else med["stress_strength"])
        pats[p] = dict(recovery_probs=dict(rec[str(i)]), stress_strength=float(stress),
                       stress_tumour=(t if stress_own and level != "reference" else None),
                       drawn_tumour=t)

    if level == "reference":
        kw = _batch_kwargs(med, med, (), dsig)
        kw.update(sigma_batch=0.0, depth_batch_sigma=0.0)
        batches, labels, tumours = [kw], ["reference"], [None]
    elif design == "confounded":
        tumours = [dr["patient_tumour"][p] for p in patients]
        labels = [f"batch{p}" for p in patients]
        batches = [_batch_kwargs(vectors[t], med, vary, dsig) for t in tumours]
    else:
        tumours = list(dr["balanced_batch_tumour"])
        labels = [f"batch{i}" for i in range(len(tumours))]
        batches = [_batch_kwargs(vectors[t], med, vary, dsig) for t in tumours]
    return dict(level=level, design=design, batches=batches, batch_labels=labels,
                batch_tumours=[(t if vary else None) for t in tumours] if level != "reference" else [None],
                batch_tumours_drawn=tumours, varies=list(vary), patients=pats, median_vector=med,
                draws=dr, sigma_batch=(0.0 if level == "reference" else SIGMA_BATCH))


def levels(fits, patients, design, n_balanced_batches=None, seed=DRAW_SEED, ambient_range=None):
    """Every level of one design: ``{level: level_settings(...)}``."""
    return {lv: level_settings(fits, patients, design, lv, n_balanced_batches, seed, ambient_range)
            for lv in LEVELS}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Print one design's levels as JSON")
    p.add_argument("fits")
    p.add_argument("--design", choices=DESIGNS, required=True)
    p.add_argument("--n-patients", type=int, default=50)
    p.add_argument("--n-balanced-batches", type=int, default=None)
    p.add_argument("--ambient-range", default=None, help="low,high (only if ambient was not fitted)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    rng = tuple(float(x) for x in a.ambient_range.split(",")) if a.ambient_range else None
    res = levels(load(a.fits), list(range(a.n_patients)), a.design, a.n_balanced_batches, ambient_range=rng)
    txt = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()
