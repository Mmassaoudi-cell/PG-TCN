"""Test-set statistics from saved per-step probabilities.

1. Episode-bootstrap 95% CIs of the seed-averaged step macro-F1 per (system, model) and paired bootstrap
   differences PG-TCN - benchmark (episodes resampled jointly, so seeds and models stay paired).
2. Matched-FAR operating characteristics: the CUSUM threshold is swept on the TEST set and the detection rate
   and mean delay are read at test episode FAR = 1%, 2% and 5% (no validation calibration involved).

python rev_stats.py [stage]
Outputs: results/rev_final/bootstrap_ci.csv, bootstrap_diff.csv, amoc.csv
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

import metrics as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results", "lib_final")
os.makedirs(OUT, exist_ok=True)
P = "bus_graph_tcn"
B = 1000
HS = np.concatenate([np.linspace(0.2, 20, 60), np.linspace(21, 400, 80)])


def episode_cms(prob, cls, tau):
    E, T, K = prob.shape
    y = M.step_labels(cls, tau, T)[:, M.T_EVAL:]
    yh = prob[:, M.T_EVAL:].argmax(-1)
    cms = np.zeros((E, K, K), np.int32)
    for k in range(K):
        for j in range(K):
            cms[:, k, j] = ((y == k) & (yh == j)).sum(1)
    return cms


def macro_f1(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    fp = cm.sum(-2) - tp
    fn = cm.sum(-1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean(-1)


def amoc(prob, cls, tau, targets=(0.01, 0.02, 0.05)):
    llr = M.attack_llr(prob)
    T = prob.shape[1]
    curve = []
    for h in HS:
        e = M.episode_metrics(M.cusum_alarms(llr, h), cls, tau, T)
        curve.append((e["far"], e["det_rate"], e["delay"]))
    c = np.array(curve, float)
    out = {}
    for f in targets:
        ok = c[:, 0] <= f
        i = np.where(ok)[0][0] if ok.any() else len(c) - 1   # smallest threshold meeting the FAR target
        out[f"det@{f:g}"], out[f"delay@{f:g}"] = c[i, 1], c[i, 2]
    return out


def main(stage="rev"):
    files = sorted(glob.glob(os.path.join(ROOT, "results", stage, "*", "*__seed*_testprob.npz")))
    groups = {}
    for f in files:
        system = os.path.basename(os.path.dirname(f))
        model, tag, seed = os.path.basename(f).replace("_testprob.npz", "").split("__")
        if model == P and tag != "final":
            continue
        if model == "gcn_gru" and tag == "matched":
            model = "gcn_gru_matched"
        elif model != P and tag != "default":
            continue
        groups.setdefault((system, model), []).append(f)
    rng = np.random.default_rng(0)
    ci, diff, am = [], [], []
    for system in sorted({s for s, _ in groups}):
        cms = {}
        for (s, model), fl in groups.items():
            if s != system:
                continue
            cs = []
            for f in fl:
                d = np.load(f)
                p = d["prob"].astype(np.float32)
                cs.append(episode_cms(p, d["cls"], d["tau"]))
                am.append(dict(system=system, model=model, seed=f.split("seed")[-1].split("_")[0], **amoc(p, d["cls"], d["tau"])))
            cms[model] = np.stack(cs)                                  # (S,E,K,K)
        E = next(iter(cms.values())).shape[1]
        idx = rng.integers(0, E, size=(B, E))
        boot = {}
        for model, c in cms.items():
            point = macro_f1(c.sum(1)).mean()
            bs = np.array([macro_f1(c[:, ii].sum(1)).mean() for ii in idx])
            boot[model] = bs
            ci.append(dict(system=system, model=model, n_seeds=c.shape[0], macro_f1=point,
                           lo=np.quantile(bs, 0.025), hi=np.quantile(bs, 0.975)))
        if P in boot:
            for model, bs in boot.items():
                if model == P:
                    continue
                dlt = boot[P] - bs
                diff.append(dict(system=system, baseline=model, diff=dlt.mean(), lo=np.quantile(dlt, 0.025),
                                 hi=np.quantile(dlt, 0.975), p_boot=float(min(1.0, 2 * min((dlt <= 0).mean(), (dlt >= 0).mean())))))
        print("bootstrap done", system, flush=True)
    pd.DataFrame(ci).to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)
    pd.DataFrame(diff).to_csv(os.path.join(OUT, "bootstrap_diff.csv"), index=False)
    pd.DataFrame(am).to_csv(os.path.join(OUT, "amoc.csv"), index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
