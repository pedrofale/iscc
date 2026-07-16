"""Run the REAL cell2location on an iscc reference + Visium section, write per-spot proportions.

Lives on the far side of the dedicated ``iscc-cell2location`` conda env so the core ``iscc`` env never
carries scvi-tools / pyro / torch. The deconvolution validation writes two AnnData files — a single-cell
reference (``obs['cell_type']``) and a spatial section — and this script:

  1. estimates per-gene, per-cell-type signatures with cell2location's negative-binomial
     ``RegressionModel`` (the reference step);
  2. deconvolves the spatial spots with the ``Cell2location`` model, given the signatures and the
     expected number of cells per spot;
  3. writes the posterior mean cell-type abundances per spot, normalised to proportions, to CSV.

Epoch counts are kept modest (the tutorials use tens of thousands) because the iscc benchmark is small
(hundreds of genes / spots) and RELATIVE — every scenario is deconvolved with identical settings, so the
comparison across references is fair even if the absolute abundances are not fully converged.

Usage:  python cell2location_runner.py <ref.h5ad> <spatial.h5ad> <out.csv>
                                       [epochs_ref] [epochs_sp] [n_cells_per_spot] [seed]
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


def main():
    ref_path, sp_path, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    epochs_ref = int(sys.argv[4]) if len(sys.argv) > 4 else 250
    epochs_sp = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    n_cells_per_spot = float(sys.argv[6]) if len(sys.argv) > 6 else 8.0
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0

    import scvi
    import cell2location
    from cell2location.models import RegressionModel, Cell2location
    from cell2location.utils.filtering import filter_genes

    scvi.settings.seed = seed

    ref = ad.read_h5ad(ref_path)
    sp = ad.read_h5ad(sp_path)
    ref.X = np.rint(np.asarray(ref.X)).astype(np.float32)     # counts
    sp.X = np.rint(np.asarray(sp.X)).astype(np.float32)

    # --- 1. reference signatures (per-gene NB regression on cell-type) ---
    RegressionModel.setup_anndata(adata=ref, labels_key="cell_type")
    reg = RegressionModel(ref)
    reg.train(max_epochs=epochs_ref, batch_size=min(1024, ref.n_obs), train_size=1.0)
    ref = reg.export_posterior(
        ref, use_quantiles=True, add_to_varm=["q05"],
        sample_kwargs={"batch_size": min(1024, ref.n_obs)})
    # per-gene per-cell-type expression signature (genes x cell types)
    if "means_per_cluster_mu_fg" in ref.varm:
        inf_aver = ref.varm["means_per_cluster_mu_fg"].copy()
    else:
        keys = [k for k in ref.varm.keys() if "per_cluster_mu_fg" in k]
        inf_aver = ref.varm[keys[0]].copy()
    inf_aver.columns = [c.split("means_per_cluster_mu_fg_")[-1].split("q05_per_cluster_mu_fg_")[-1]
                        for c in inf_aver.columns]

    # align genes between the signature and the spatial data
    genes = [g for g in sp.var_names if g in inf_aver.index]
    sp = sp[:, genes].copy()
    inf_aver = inf_aver.loc[genes]

    # --- 2. spatial deconvolution ---
    Cell2location.setup_anndata(adata=sp)
    mod = Cell2location(sp, cell_state_df=inf_aver,
                        N_cells_per_location=float(max(n_cells_per_spot, 1.0)),
                        detection_alpha=20.0)
    mod.train(max_epochs=epochs_sp, batch_size=None, train_size=1.0)
    sp = mod.export_posterior(sp, use_quantiles=True, add_to_obsm=["q05"],
                              sample_kwargs={"batch_size": sp.n_obs})

    # posterior cell-type abundance per spot -> proportions
    abund = sp.obsm.get("q05_cell_abundance_w_sf")
    if abund is None:
        key = [k for k in sp.obsm.keys() if "cell_abundance" in k][0]
        abund = sp.obsm[key]
    # the abundance array columns follow the order of the signature (cell_state_df) columns
    abund = pd.DataFrame(np.asarray(abund), index=sp.obs_names, columns=list(inf_aver.columns))
    props = abund.div(abund.sum(1).replace(0, np.nan), axis=0).fillna(1.0 / abund.shape[1])
    props.to_csv(out_csv)
    print("cell2location done: %d spots x %d cell types -> %s" % (props.shape[0], props.shape[1], out_csv))


if __name__ == "__main__":
    main()
