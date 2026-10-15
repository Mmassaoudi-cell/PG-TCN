"""Bus-level localisation of FDIAs: PG-TCN bus head vs. two baselines on the same test episodes.

* drift-score baseline: per-bus mean |multi-horizon drift| over the last 30 scans, threshold tuned on validation (F1);
* tree localiser: LightGBM on per-bus token rows (token at the last scan + 32-scan mean), trained on training FDIAs.
Metrics on all FDIA test episodes at the final scan: bus-level precision / recall / F1 and top-1 accuracy.

python loc_rev.py ieee14 [stage]
Output: results/rev_final/localisation_<system>.csv
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import torch

from features import region_of, DATA
from train_nn import load_split, ROOT

FEAT = os.path.join(DATA, "feat")
HOR = (60, 120)
OUT = os.path.join(ROOT, "results", "lib_final")
os.makedirs(OUT, exist_ok=True)


def tokens(fz, X, bs=50):
    out = []
    for s in range(0, len(X), bs):
        x = X[s:s + bs].cuda().float()
        tok = fz.node_tokens(x, HOR)                                  # (b,T,nR,F)
        last = tok[:, -1]
        mean = tok[:, -32:].mean(1)
        out.append(torch.cat([last, mean], -1).cpu().numpy())
    return np.concatenate(out)                                        # (E,nR,2F)


def drift_score(fz, X, bs=50):
    out = []
    for s in range(0, len(X), bs):
        x = X[s:s + bs].cuda().float()
        tok = fz.node_tokens(x, HOR)
        out.append(tok[:, -30:, :, 30:40:2].abs().mean((1, 3)).cpu().numpy())   # |mean drift|, W=120, all types
    return np.concatenate(out)


def prf(score, true, thr):
    pred = score > thr
    tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum()
    top1 = true[np.arange(len(true)), score.argmax(1)].mean()
    return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1), top1=top1)


def main(system, stage="lib"):
    fz = pickle.load(open(os.path.join(FEAT, f"{system}_featurizer.pkl"), "rb"))
    fz.r = region_of(system)
    data = {}
    for sp in ["train", "val", "test"]:
        X, m = load_split(system, "hard", sp)
        att = np.isin(m["cls"], [3, 4])
        data[sp] = (X[att], m["att_bus"][att].astype(bool))
    rows = []
    # drift-score baseline
    sv = drift_score(fz, data["val"][0])
    thr = max(np.quantile(sv, np.linspace(0.5, 0.99, 50)), key=lambda t: prf(sv, data["val"][1], t)["f1"])
    st = drift_score(fz, data["test"][0])
    rows.append(dict(method="Drift score", seed=0, **prf(st, data["test"][1], thr)))
    # tree localiser
    import lightgbm as lgb
    Ttr, Tva, Tte = (tokens(fz, data[s][0]) for s in ["train", "val", "test"])
    nR = Ttr.shape[1]
    for seed in range(3):
        clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.5, random_state=seed, verbose=-1, n_jobs=12)
        clf.fit(Ttr.reshape(-1, Ttr.shape[-1]), data["train"][1].ravel())
        pv = clf.predict_proba(Tva.reshape(-1, Tva.shape[-1]))[:, 1].reshape(-1, nR)
        thr = max(np.linspace(0.05, 0.95, 37), key=lambda t: prf(pv, data["val"][1], t)["f1"])
        pt = clf.predict_proba(Tte.reshape(-1, Tte.shape[-1]))[:, 1].reshape(-1, nR)
        rows.append(dict(method="Tree localiser", seed=seed, **prf(pt, data["test"][1], thr)))
    # PG-TCN bus head; threshold tuned on validation F1 exactly as for the baselines
    import json
    from train_nn import Runner
    for f in sorted(glob.glob(os.path.join(ROOT, "results", stage, system, "bus_graph_tcn__final__seed*_testnodes.npz"))):
        seed = int(f.split("seed")[-1].split("_")[0])
        base = f.replace("_testnodes.npz", "")
        res = json.load(open(base + ".json"))
        r = Runner(system, "bus_graph_tcn", res["hp"], seed)
        r.net.load_state_dict(torch.load(base + ".pt", map_location="cuda"))
        pv = r.predict_nodes(data["val"][0])[:, -1]
        thr = max(np.linspace(0.05, 0.95, 37), key=lambda t: prf(pv, data["val"][1], t)["f1"])
        d = np.load(f)
        att = np.isin(d["cls"], [3, 4])
        rows.append(dict(method="PG-TCN", seed=seed, **prf(d["pn_last"][att].astype(np.float32), d["att_bus"][att].astype(bool), thr)))
    df = pd.DataFrame(rows).assign(system=system)
    df.to_csv(os.path.join(OUT, f"localisation_{system}.csv"), index=False)
    print(df.groupby("method")[["precision", "recall", "f1", "top1"]].mean().round(3))


if __name__ == "__main__":
    main(*sys.argv[1:])
