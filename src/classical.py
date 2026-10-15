"""Binary (attack / no-attack) sequential detectors on the new scenario:
  eta_shewhart : threshold on the SP metric eta
  eta_cusum    : Page CUSUM on standardized log eta
  chi2_cusum   : Page CUSUM on the normalized innovation chi^2 statistic sum(nun^2) (classical DSE BDD)
  source_dqn   : SOURCE_METHOD_REPRODUCTION (DQN on eta) retrained on the new scenario (c_beta chosen on VAL)
Thresholds are calibrated on VALIDATION to episode FAR <= 1% (same rule as learned models).
python classical.py --system ieee14 [--seed k]
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

import metrics as M
from train_nn import ROOT
from features import load_raw

sys.path.insert(0, os.path.join(ROOT, "SOURCE_METHOD_REPRODUCTION"))
from dqn_detector import DQNDetector  # noqa: E402


def stat_series(d, kind, ref):
    if kind == "eta_shewhart" or kind == "eta_cusum":
        z = (np.log(d["eta"]) - ref["mu"]) / ref["sd"]
    else:
        g = (d["nun"].astype(np.float32) ** 2).sum(-1)
        z = (g - ref["mu"]) / ref["sd"]
    return z


def alarms_from(z, kind, h, k=0.5):
    E, T = z.shape
    if kind == "eta_shewhart":
        s = z >= h
        s[:, :M.T_EVAL] = False
        return np.where(s.any(1), s.argmax(1), T)
    return M.cusum_alarms(z - k, h)


def binary_step_summary(alarm, d):
    """Step-level binary metrics when an alarm latches 'attack' from the alarm time onwards."""
    E, T = d["eta"].shape
    y = M.step_labels(d["cls"], d["tau"], T)[:, M.T_EVAL:]
    yb = np.isin(y, M.ATT)
    pred = (np.arange(T)[None] >= alarm[:, None])[:, M.T_EVAL:]
    tp = (yb & pred).sum(); fp = (~yb & pred).sum(); fn = (yb & ~pred).sum(); tn = (~yb & ~pred).sum()
    return dict(bin_f1=2 * tp / max(2 * tp + fp + fn, 1), step_fpr=fp / max(fp + tn, 1), step_fnr=fn / max(fn + tp, 1))


def calib(zv, kind, dv):
    grid = np.concatenate([np.linspace(0.5, 20, 60), np.linspace(21, 400, 120)])
    for h in grid:
        m = M.episode_metrics(alarms_from(zv, kind, h), dv["cls"], dv["tau"], zv.shape[1])
        if m["far"] <= M.FAR_TARGET:
            return float(h)
    return float(grid[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tests", default="")
    ap.add_argument("--val_only", action="store_true")
    ap.add_argument("--stage", default="bench")
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", "diag" if a.val_only else a.stage, a.system)
    os.makedirs(out_dir, exist_ok=True)
    tr = load_raw(a.system, "hard", "train")
    va = load_raw(a.system, "hard", "val")
    tests = {"val": va} if a.val_only else {"test": load_raw(a.system, "hard", "test")}
    for t in [x for x in a.tests.split(",") if x]:
        tests[t] = load_raw(a.system, t, "test")
    norm = tr["cls"] == 0
    for kind in ["eta_shewhart", "eta_cusum", "chi2_cusum"]:
        fn = os.path.join(out_dir, f"{kind}__default__seed{a.seed}.json")
        if a.seed > 0 or os.path.exists(fn):
            continue  # deterministic detectors: one run
        if kind.startswith("eta"):
            le = np.log(tr["eta"][norm][:, M.T_EVAL:])
            ref = dict(mu=float(le.mean()), sd=float(le.std()))
        else:
            g = (tr["nun"][norm][:, M.T_EVAL:].astype(np.float32) ** 2).sum(-1)
            ref = dict(mu=float(g.mean()), sd=float(g.std()))
        h = calib(stat_series(va, kind, ref), kind, va)
        res = dict(system=a.system, model=kind, seed=0, cusum_h=h, ref=ref)
        for name, d in tests.items():
            t0 = time.time()
            al = alarms_from(stat_series(d, kind, ref), kind, h)
            lat = (time.time() - t0) / d["eta"].size * 1e3
            em = M.episode_metrics(al, d["cls"], d["tau"], d["eta"].shape[1])
            res[name] = {**{f"ep_{k}": float(v) for k, v in em.items()}, **{k: float(v) for k, v in binary_step_summary(al, d).items()}}
        res["latency_ms"] = lat
        json.dump(res, open(fn, "w"))
        print("DONE", kind, res[list(tests)[0]])
    # ------------- SOURCE_METHOD_REPRODUCTION retrained on the new scenario
    fn = os.path.join(out_dir, f"source_dqn__default__seed{a.seed}.json")
    if os.path.exists(fn):
        return
    is_att = lambda d: np.isin(d["cls"], M.ATT)
    sel_file = os.path.join(ROOT, "results", "tuning", a.system, "source_dqn.json" if a.stage == "bench" else f"source_dqn_{a.stage}.json")
    os.makedirs(os.path.dirname(sel_file), exist_ok=True)
    if os.path.exists(sel_file):
        cb = json.load(open(sel_file))["c_beta"]
    else:
        best = None
        rows = []
        for cb_ in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            det = DQNDetector(c_beta=cb_, seed=0).fit(tr["eta"], tr["tau"], is_att(tr))
            al = det.alarm_times(va["eta"])
            m = M.episode_metrics(al, va["cls"], va["tau"], va["eta"].shape[1])
            rows.append(dict(c_beta=cb_, **{k: float(v) for k, v in m.items()}))
            # SP rule adapted: FAR < 5%, then maximise detection, then minimise delay
            key = (m["far"] < 0.05, m["det_rate"], -np.nan_to_num(m["delay"], nan=1e9))
            if best is None or key > best[0]:
                best = (key, cb_)
        cb = best[1]
        json.dump(dict(c_beta=cb, rows=rows), open(sel_file, "w"), indent=1)
    det = DQNDetector(c_beta=cb, seed=a.seed).fit(tr["eta"], tr["tau"], is_att(tr))
    res = dict(system=a.system, model="source_dqn", seed=a.seed, c_beta=cb, train_time_s=det.train_time)
    for name, d in tests.items():
        t0 = time.time()
        al = det.alarm_times(d["eta"])
        lat = (time.time() - t0) / d["eta"].size * 1e3
        em = M.episode_metrics(al, d["cls"], d["tau"], d["eta"].shape[1])
        res[name] = {**{f"ep_{k}": float(v) for k, v in em.items()}, **{k: float(v) for k, v in binary_step_summary(al, d).items()}}
    res["latency_ms"] = lat
    res["params"] = sum(p.numel() for p in det.q.parameters())
    json.dump(res, open(fn, "w"))
    print("DONE source_dqn", res[list(tests)[0]])


if __name__ == "__main__":
    main()
