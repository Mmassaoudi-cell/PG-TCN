"""Recompute the validation-calibrated decision rule (raw, calibrated-LLR and two-parameter CUSUM) for every trained
model of a stage with the final metrics code, using only validation episodes and validation-period streams, then
re-score the saved test probabilities. Updates the result JSON in place (idempotent; marks 'decision_v' = 2).

python redecide.py [stage] [system]
"""
import os
import sys
import glob
import json
import numpy as np

import metrics as M
from train_nn import load_split, decision_rule, _summ, ROOT
from eval_rev import Predictor

stage = sys.argv[1] if len(sys.argv) > 1 else "lib"
systems = sys.argv[2].split(",") if len(sys.argv) > 2 else ["ieee14", "ieee30", "ieee57", "ieee118"]
for system in systems:
    Xva, mva = load_split(system, "hard", "val")
    Xs, _ = load_split(system, "ood_streamcal", "test")
    for f in sorted(glob.glob(os.path.join(ROOT, "results", stage, system, "*__seed*.json"))):
        res = json.load(open(f))
        base = f[:-5]
        ck = base + ".pt" if os.path.exists(base + ".pt") else base + ".pkl"
        if res.get("decision_v") == 2 or not os.path.exists(ck) or not os.path.exists(base + "_testprob.npz"):
            continue
        pred = Predictor(system, res, ck)
        vp = base + "_valprob.npz"
        pv = np.load(vp)["prob"].astype(np.float32) if os.path.exists(vp) else pred(Xva)
        ps = pred(Xs, long=True)
        D = decision_rule(pv, mva, ps)
        res.update(D)
        h, cal, h_cal = D["cusum_h"], tuple(D["llr_cal"]), D["cusum_h_cal"]
        loc = {k: v for k, v in res.get("test", {}).items() if k.startswith("loc_")}
        res["val"] = {**_summ(pv, mva, h, cal, h_cal, D)}
        d = np.load(base + "_testprob.npz")
        mt = {k: d[k] for k in d.files if k != "prob"}
        res["test"] = {**_summ(d["prob"].astype(np.float32), mt, h, cal, h_cal, D), **loc}
        res["val"].pop("cm", None)
        res["decision_v"] = 2
        json.dump(res, open(f, "w"), default=float)
        print("redecided", system, os.path.basename(base), "k", D["cusum_k2"], "h", round(D["cusum_h2"], 1),
              "test det", round(res["test"]["epk_det_rate"], 4), "far", round(res["test"]["epk_far"], 4), flush=True)
