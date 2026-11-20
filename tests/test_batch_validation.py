"""The machinery of Fig 5's batch fits and validation panel, on synthetic data only.

- the 10x multiplet table and its interpolation; the van den Brink gene list is the paper's 140;
- the SoupX-principle ambient estimate recovers the ambient fraction iscc itself planted;
- the contrast statistics see what is planted: a per-gene factor shared by every type gives a large
  S0 and fold changes correlated between types (S3); independent per-type factors do not;
- the registered scoring compares medians over seeds to the real value; the tuning interpolation
  and the runner's binary format.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "validation", "paper"))
import batch_fits as BF  # noqa: E402
import batch_validation as BV  # noqa: E402


def test_multiplet_rate_follows_10x_table():
    assert BF.multiplet_rate(1000) == pytest.approx(0.008)
    assert BF.multiplet_rate(5000) == pytest.approx(0.039)
    assert BF.multiplet_rate(2500) == pytest.approx(0.0195)
    assert BF.multiplet_rate(250) == pytest.approx(0.002)
    assert BF.multiplet_rate(20000) == pytest.approx(0.152)


def test_vdb_list_is_the_papers_140_genes():
    assert len(BF.VDB_MOUSE_GENES) == 140
    assert {"Fos", "Jun", "Hspa1a", "Egr1", "Kcne4"} <= set(BF.VDB_MOUSE_GENES)


def _three_type_batch(ambient, seed=0, n=1500, g=400):
    """iscc counts of three types: 'T', and 'X' and 'Y' with 40 genes each that only they express.
    Every profile sums to 1, so iscc's soup (the pooled expression) is the pooled counts' share."""
    from iscc.data import scRNA
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0, 1, g)
    prof = {t: base.copy() for t in ("T", "X", "Y")}
    for t, lo in (("X", 0), ("Y", 40)):
        for u in prof:
            prof[u][lo:lo + 40] = base[lo:lo + 40] * 30 if u == t else 0.0
    prof = {t: v / v.sum() for t, v in prof.items()}
    ct = rng.choice(["T", "X", "Y"], size=n, p=[0.5, 0.3, 0.2])
    names = [f"c{i}" for i in range(n)]
    exp = pd.DataFrame(np.vstack([prof[c] for c in ct]), index=names, columns=[f"g{i}" for i in range(g)])
    a = scRNA(protocol="10x", n_cells=n, seed=seed + 1, ambient_frac=ambient, doublet_rate=0.0,
              dropout_mid=0.0, mu_lib=3000.0)
    a.run({"cell_exp": exp}, cell_subset=names)
    return sp.csr_matrix(a.observed_counts.to_numpy()), ct


@pytest.mark.parametrize("ambient", [0.02, 0.08])
def test_ambient_estimate_recovers_iscc_ambient(ambient):
    X, ct = _three_type_batch(ambient)
    sets = {"X": np.arange(40), "Y": np.arange(40, 80)}
    res = BF.ambient_estimate(X, ct, ct == "T", sets)
    assert res["n_sets"] == 2 and res["identifiable"]
    assert res["ambient_frac"] == pytest.approx(ambient, rel=0.25)


def test_ambient_estimate_needs_agreeing_sets():
    X, ct = _three_type_batch(0.05)
    one = BF.ambient_estimate(X, ct, ct == "T", {"X": np.arange(40)})
    assert not one["identifiable"] and one["n_sets"] == 1          # one set cannot be checked
    # a set the T cells express themselves is not soup: the estimates disagree
    bad = BF.ambient_estimate(X, ct, ct == "T", {"X": np.arange(40), "Y": np.arange(80, 120)})
    assert not bad["identifiable"]


def _contrast(shared, specific, seed=0, n=1200, g=600, types=("a", "b", "c")):
    from iscc.data import scRNA
    rng = np.random.default_rng(seed)
    prof = {t: rng.lognormal(0, 1.2, g) for t in types}
    fac_shared = rng.lognormal(0, shared, g)
    fac_type = {t: rng.lognormal(0, specific, g) for t in types}
    parts, batch, ctype = [], [], []
    for r in ("A", "B"):
        ct = rng.choice(list(types), size=n)
        rows = [prof[t] * ((fac_shared * fac_type[t]) if r == "B" else 1.0) for t in ct]
        names = [f"{r}{i}" for i in range(n)]
        exp = pd.DataFrame(np.vstack(rows), index=names, columns=[f"g{i}" for i in range(g)])
        a = scRNA(protocol="10x", n_cells=n, seed=3 + len(parts), sigma_batch=0.0, depth_batch_sigma=0.0,
                  dropout_mid=0.0, doublet_rate=0.0, ambient_frac=0.0, mu_lib=4000.0)
        a.run({"cell_exp": exp}, cell_subset=names)
        parts.append(sp.csr_matrix(a.observed_counts.to_numpy()))
        batch += [r] * n
        ctype += list(ct)
    return sp.vstack(parts).tocsr(), np.array(batch), np.array(ctype)


def test_contrast_stats_see_a_shared_factor():
    X, b, c = _contrast(shared=0.5, specific=0.0)
    st = BV.contrast_stats(X, b, c, ["a", "b", "c"])
    s = st["summary"]
    assert s["S0"] > 0.5                     # sd of log2 of a LogNormal(0, 0.5) factor is ~0.72
    assert s["S3_median_corr"] > 0.8         # the same factor in every type
    assert len(st["S3_pairs"]) == 3
    assert all(len(r["curve"]["sd_log2fc"]) == 10 for r in st["per_type"].values())


def test_contrast_stats_see_type_specific_factors():
    X, b, c = _contrast(shared=0.0, specific=0.5)
    s = BV.contrast_stats(X, b, c, ["a", "b", "c"])["summary"]
    assert s["S0"] > 0.5
    assert s["S3_median_corr"] < 0.3


def test_score_compares_seed_medians_with_the_real_value():
    def v(**k):
        base = dict(S0=0.2, S1_spearman=0.0, S1_spread_ratio=1.0, S2_slope_low=0.0, S2_slope_mid=0.0,
                    S2_slope_high=0.0, S3_median_corr=0.5)
        base.update(k)
        return {"summary": base}
    versions = {"real": v(S1_spearman=0.3, S3_median_corr=0.4)}
    for s in BV.SEEDS:
        versions[f"iscc_s{s}"] = v(S1_spearman=0.25 + 0.01 * s, S3_median_corr=0.9)
        versions[f"splatter_s{s}"] = v(S1_spearman=0.0, S3_median_corr=0.95 - 0.5 * (s == 1))
    sc = BV.score(versions, size_differs=False)["against_splatter"]
    assert sc["S1"]["summaries"]["S1_spearman"]["iscc_closer"]
    assert sc["S1"]["summaries"]["S1_spearman"]["iscc_median"] == pytest.approx(0.28)
    assert sc["S3"]["summaries"]["S3_median_corr"]["iscc_closer"]          # 0.9 vs Splatter's median 0.95
    assert not sc["S1"]["reproduced"]        # spread ratio: tie (both 1.0) is not closer
    assert sc["S1"]["n_closer"] == 1 and sc["S1"]["n_summaries"] == 2


def test_interp_and_binary_round_trip(tmp_path):
    x, br = BV._interp([0, 1, 2, 3], [0.1, 0.2, 0.4, 0.8], 0.3)
    assert x == pytest.approx(1.5) and br == (1.0, 2.0)
    assert BV._interp([0, 1], [0.1, 0.2], 0.5) == (None, None)
    rng = np.random.default_rng(0)
    X = sp.random(30, 50, density=0.2, random_state=1, format="csr")
    X.data = rng.integers(1, 9, X.nnz).astype(float)
    BV.write_bin(X, str(tmp_path / "m.bin"))
    Y = BV.read_bin(str(tmp_path / "m.bin"))
    assert (Y != X).nnz == 0 and Y.shape == X.shape


def test_score_uses_the_size_matched_splatter_when_sizes_differ():
    def v(x):
        return {"summary": dict(S0=0.2, S1_spearman=x, S1_spread_ratio=1.0, S2_slope_low=0.0, S2_slope_mid=0.0,
                                S2_slope_high=0.0, S3_median_corr=0.5)}
    versions = {"real": v(0.3), **{f"iscc_s{s}": v(0.25) for s in BV.SEEDS},
                **{f"splatter_s{s}": v(0.31) for s in BV.SEEDS}}
    assert BV.score(versions, size_differs=False)["primary"] == "against_splatter"
    with pytest.raises(RuntimeError, match="matched to iscc"):
        BV.score(versions, size_differs=True)
    versions.update({f"splatteriscc_s{s}": v(0.0) for s in BV.SEEDS})
    sc = BV.score(versions, size_differs=True)
    assert sc["primary"] == "against_splatteriscc"
    assert sc["against_splatteriscc"]["S1"]["summaries"]["S1_spearman"]["iscc_closer"]
    assert not sc["against_splatter"]["S1"]["summaries"]["S1_spearman"]["iscc_closer"]


def test_tuning_that_cannot_reach_the_target_stops():
    class Fake:
        def simulate(self, runs, tag):
            return {r["name"]: (None, None, None) for r in runs}
    with pytest.raises(RuntimeError, match="cannot be tuned"):
        BV.tune(Fake(), "batch_facScale", (0.0, 0.1, 0.2), 5.0, lambda *a: 0.1, {})


def test_gene_filter_is_the_pooled_mean_and_a_zero_batch_mean_is_kept():
    # a gene expressed only in batch B passes the pooled filter and gets a finite fold change
    rng = np.random.default_rng(0)
    n, g = 200, 50
    XA = rng.poisson(2.0, (n, g)).astype(float)
    XB = rng.poisson(2.0, (n, g)).astype(float)
    XA[:, 0] = 0
    XB[:, 0] = rng.poisson(0.3, n)
    X = sp.csr_matrix(np.vstack([XA, XB]))
    st = BV.contrast_stats(X, np.array(["A"] * n + ["B"] * n), np.array(["t"] * (2 * n)), ["t"])
    assert st["per_type"]["t"]["n_genes"] == g
    assert st["per_type"]["t"]["n_genes_zero_in_a_batch"] == 1
    assert np.isfinite(st["summary"]["S0"])


def test_combine_by_type_follows_the_clarification():
    def f(n, mu, mid, shape=2.0):
        return dict(n_cells=n, mu_lib=mu, sigma_lib=0.4, dispersion=0.2, dropout_mid=mid, dropout_shape=shape)
    by_type = {"a": f(600, 1000.0, 0.5, 1.0), "b": f(300, 4000.0, 0.0), "c": f(100, 2000.0, 1.5, 3.0)}
    amb = {"a": dict(identifiable=True, ambient_frac=0.02), "b": dict(identifiable=False, ambient_frac=None),
           "c": dict(identifiable=True, ambient_frac=0.06)}
    c = BF.combine_by_type(by_type, amb, 1000)
    assert c["mu_lib"] == pytest.approx((600 * 1000 + 300 * 4000 + 100 * 2000) / 1000)
    # types with a midpoint above 0 hold 700 of 1000 cells: their weighted mean, the 0 never averaged in
    assert c["dropout_mid"] == pytest.approx((600 * 0.5 + 100 * 1.5) / 700)
    assert c["dropout_shape"] == pytest.approx((600 * 1.0 + 100 * 3.0) / 700)
    assert c["ambient_frac"] == pytest.approx((600 * 0.02 + 100 * 0.06) / 700)
    # dropout types holding half or less: no dropout; ambient types holding half or less: the range
    by_type["a"]["dropout_mid"] = 0.0
    amb["a"] = dict(identifiable=False, ambient_frac=None)
    c = BF.combine_by_type(by_type, amb, 1000)
    assert c["dropout_mid"] == 0.0 and c["ambient_frac"] is None


def test_label_transfer_fills_the_unlabelled_cells():
    from iscc.data import scRNA
    rng = np.random.default_rng(1)
    g, n = 300, 600
    prof = {t: rng.lognormal(0, 1.5, g) for t in ("x", "y", "z")}
    ct = rng.choice(list(prof), size=n)
    names = [f"c{i}" for i in range(n)]
    exp = pd.DataFrame(np.vstack([prof[t] for t in ct]), index=names, columns=[f"g{i}" for i in range(g)])
    a = scRNA(protocol="10x", n_cells=n, seed=2, dropout_mid=0.0, doublet_rate=0.0)
    a.run({"cell_exp": exp}, cell_subset=names)
    X = sp.csr_matrix(a.observed_counts.to_numpy())
    author = pd.Series(ct, dtype=object)
    author[rng.random(n) < 0.4] = np.nan
    author[:5] = "Unassigned"
    labels, rec = BF.transfer_labels(X, author, n_pcs=10)
    assert rec["n_transferred"] == int(author.isna().sum() + 5)
    assert rec["loo_accuracy"] > 0.95
    assert (labels == ct).mean() > 0.95
    assert set(labels) == {"x", "y", "z"}


def test_statistics_refuse_without_the_registered_range():
    fits = {"wu": {"ambient": {"needs_range": ["CID1"]}}, "ding": {"needs_range": []}, "ambient_range": None}
    with pytest.raises(RuntimeError, match="registered range"):
        BF.require_ambient_range(fits)
    assert BF.require_ambient_range({"wu": {"ambient": {"needs_range": []}}, "ding": {"needs_range": []}}) is None
    rng = np.random.default_rng(0)
    a, rho = BF.draw_ambient(dict(rho_low=0.03, rho_high=0.11), rng)
    assert 0.03 <= rho <= 0.11 and a == pytest.approx(rho / (1 - rho))


def test_embedding_labels_load_without_pickle(tmp_path):
    # the LISI step loads the embeddings with allow_pickle=False; labels from a pandas column are an
    # object array, which needs pickle (the smoke of 2026-11-20 failed there)
    obs = pd.DataFrame({"run": ["A", "B", "A"], "cell_type": ["T", "B cell", "T"]})
    assert obs["run"].to_numpy().dtype == object
    path = str(tmp_path / "emb.npz")
    BV.save_embedding(path, np.zeros((3, 2)), obs["run"].to_numpy(), obs["cell_type"].to_numpy())
    z = np.load(path, allow_pickle=False)
    assert list(z["batch"]) == ["A", "B", "A"] and list(z["ctype"]) == ["T", "B cell", "T"]
    assert z["pca"].shape == (3, 2)


def _summary(s0, rho):
    return {"summary": dict(S0=s0, S1_spearman=rho, S1_spread_ratio=1.0, S2_slope_low=0.0, S2_slope_mid=0.0,
                            S2_slope_high=0.0, S3_median_corr=0.5), "S4": {}}


def _write_parts(parts, differs=False, settings=None):
    meta = dict(settings=settings or json.loads(json.dumps(BV._settings())), fits={"sha256": "f"},
                synthetic=True, wall_s=1.0, commit="c", host="h", job=None, stage="s")
    BV._write_part(parts, "tune", dict(meta=meta, data={"n_cells": {"A": 1, "B": 1}}, splatter_image="img",
                                       versions={"real": _summary(0.4, 0.3)},
                                       splatter=dict(params="p.rds", runner_log=[{"args": ["estimate"]}],
                                                     de_facLoc=1.0, batch_facScale=0.1)))
    for s in BV.SEEDS:
        BV._write_part(parts, f"iscc_s{s}", dict(meta=meta, versions={f"iscc_s{s}": _summary(0.41, 0.28)}))
    BV._write_part(parts, "splatter", dict(meta=meta, runner_log=[{"args": ["simulate"]}],
                                           versions={f"splatter_s{s}": _summary(0.4, 0.0) for s in BV.SEEDS}))
    size = dict(meta=meta, S0=dict(real=0.4, iscc_median=0.41, ratio=1.025, differs_by_more_than_factor=differs),
                versions={}, splatter={}, runner_log=[])
    if differs:
        size["versions"] = {f"splatteriscc_s{s}": _summary(0.41, 0.0) for s in BV.SEEDS}
        size["splatter"] = dict(batch_facScale_iscc_size=0.12)
    BV._write_part(parts, "size", size)


def test_collect_assembles_the_stages(tmp_path):
    import json as _json
    parts = str(tmp_path / "parts")
    _write_parts(parts)
    res = BV.collect(parts, str(tmp_path / "validation.json"))
    assert set(res["versions_stats"]) == {"real", *(f"{p}_s{s}" for p in ("iscc", "splatter") for s in BV.SEEDS)}
    assert res["score"]["primary"] == "against_splatter"
    assert res["score"]["against_splatter"]["S1"]["summaries"]["S1_spearman"]["iscc_closer"]
    assert res["S0"]["splatter_median"] == pytest.approx(0.4)
    assert [d["args"][0] for d in res["splatter"]["runner_log"]] == ["estimate", "simulate"]
    assert "params" not in res["splatter"]
    with open(tmp_path / "validation.json") as f:
        assert _json.load(f)["stages"]["tune"]["commit"] == "c"
    with pytest.raises(SystemExit):                    # never overwritten
        BV.collect(parts, str(tmp_path / "validation.json"))


def test_collect_scores_against_the_size_matched_splatter_and_refuses_mixed_settings(tmp_path):
    parts = str(tmp_path / "parts")
    _write_parts(parts, differs=True)
    res = BV.collect(parts, str(tmp_path / "v.json"))
    assert res["score"]["primary"] == "against_splatteriscc"
    assert res["S0"]["splatter_iscc_size_median"] == pytest.approx(0.41)
    other = json.loads(json.dumps(BV._settings()))
    other["min_mean"] = 0.1
    BV._write_part(parts, "iscc_s3", dict(meta=dict(settings=other, fits={"sha256": "f"}, synthetic=True, wall_s=1.0),
                                          versions={"iscc_s3": _summary(0.41, 0.28)}))
    with pytest.raises(RuntimeError, match="other settings"):
        BV.collect(parts, str(tmp_path / "v2.json"))
