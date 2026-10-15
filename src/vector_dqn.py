"""Candidate C5: vector-state DQN (source-method extension) on causal window summaries.
Binary stop/continue policy -> episode detection metrics + latched binary step metrics (no 5-class output).
python vector_dqn.py --system ieee14 --seed 0 --stage screen
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

import metrics as M
from train_nn import load_split, ROOT
from tabular import summaries
from classical import binary_step_summary

sys.path.insert(0, os.path.join(ROOT, "SOURCE_METHOD_REPRODUCTION"))
from dqn_detector import DQNDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="screen")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", a.stage, a.system)
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"vector_dqn__default__seed{a.seed}.json")
    if os.path.exists(fn):
        return
    Xtr, mtr = load_split(a.system, "hard", "train")
    Xva, mva = load_split(a.system, "hard", "val")
    Str, Sva = summaries(Xtr).float().numpy(), summaries(Xva).float().numpy()
    mu, sd = Str[:, M.T_EVAL:].reshape(-1, Str.shape[-1]).mean(0), Str[:, M.T_EVAL:].reshape(-1, Str.shape[-1]).std(0) + 1e-6
    Str, Sva = (Str - mu) / sd, (Sva - mu) / sd
    Str[:, :M.T_EVAL] = 0; Sva[:, :M.T_EVAL] = 0
    is_att = np.isin(mtr["cls"], M.ATT)
    best = None
    for cb in [0.02, 0.1]:
        det = DQNDetector(c_beta=cb, seed=a.seed, hidden=128).set_vector_input(Str.shape[-1])
        det.fit(Str, mtr["tau"], is_att)
        al = det.alarm_times(Sva)
        em = M.episode_metrics(al, mva["cls"], mva["tau"], Sva.shape[1])
        d = dict(eta=Sva[..., 0], cls=mva["cls"], tau=mva["tau"])
        bs = binary_step_summary(al, d)
        key = (em["far"] <= 0.05, bs["bin_f1"])
        if best is None or key > best[0]:
            best = (key, cb, em, bs, det.train_time)
    _, cb, em, bs, tt = best
    res = dict(system=a.system, model="vector_dqn", seed=a.seed, c_beta=cb, train_time_s=tt,
               val={**{f"ep_{k}": float(v) for k, v in em.items()}, **{k: float(v) for k, v in bs.items()},
                    "macro_f1": float("nan")})
    json.dump(res, open(fn, "w"))
    print("DONE vector_dqn", res["val"])


if __name__ == "__main__":
    main()
