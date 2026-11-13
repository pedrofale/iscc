"""Cancer-Finder on a human-gene dataset written by iscc.data.human.write_visium or write_scrna.

Run with the Cancer-Finder environment (``bin/python-cancerfinder``)::

    python cancerfinder.py <dataset_dir> <out_dir> <checkpoint.pkl> [--threshold 0.5] [--code /opt/cancerfinder]

Cancer-Finder reads genes x cells (the transpose of scanpy's convention) by gene symbol and keeps its
checkpoint's gene list, filling absent genes with zero. Its own ``infer.py`` overwrites each
probability with the thresholded call, so this runs the same model and loader and keeps both.
Writes the common output of this directory (see calls.py): calls.tsv (malignant and p_malignant; no
clones), run.json, and the raw probabilities in native/probabilities.csv.
"""
import argparse
import functools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("checkpoint")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--code", default="/opt/cancerfinder")
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, a.code)
    sys.path.insert(0, HERE)
    import anndata
    import numpy as np
    import pandas as pd
    import scanpy
    import torch
    torch.load = functools.partial(torch.load, map_location="cpu")   # checkpoints saved on a GPU
    from models import model
    from utils import args_utils, opt_utils
    import calls as CALLS

    ad = scanpy.read_10x_h5(os.path.join(a.dataset, "outs", "filtered_feature_bc_matrix.h5"))
    ad.var_names_make_unique()
    X = ad.X.T.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X).T
    gx = anndata.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=list(ad.var_names)),
                         var=pd.DataFrame(index=list(ad.obs_names)))
    args = args_utils.create_args_from_infering(matrix=gx, ckp=a.checkpoint, threshold=a.threshold)
    alg = model.VREx(args)
    alg.load_state_dict(torch.load(a.checkpoint)["model_dict"])
    alg.eval()
    rows = []
    with torch.no_grad():
        for data, loader in opt_utils.InferLoaders(args):
            for batch in loader:
                p = torch.softmax(alg.predict(batch[0]), axis=1)[:, 1].numpy()
            rows.append(pd.DataFrame({"barcode": list(data.index), "p_malignant": p}))
    prob = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.join(a.out, "native"), exist_ok=True)
    prob.to_csv(os.path.join(a.out, "native", "probabilities.csv"), index=False)
    overlap = int(len(set(args.HVG_list) & set(ad.var_names)))
    CALLS.write(a.out, prob.assign(malignant=(prob["p_malignant"] > a.threshold).astype(int), clone=np.nan),
                tool="cancerfinder", code=a.code, checkpoint=a.checkpoint, threshold=a.threshold,
                n_barcodes=int(ad.n_obs), n_model_genes=len(args.HVG_list), n_model_genes_present=overlap,
                wall_s=round(time.time() - t0, 1))
    print(f"cancerfinder done: {int((prob['p_malignant'] > a.threshold).sum())} of {len(prob)} called "
          f"malignant; {overlap} of {len(args.HVG_list)} model genes present", flush=True)


if __name__ == "__main__":
    main()
