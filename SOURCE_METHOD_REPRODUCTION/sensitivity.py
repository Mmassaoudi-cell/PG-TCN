"""Sensitivity of the reproduced source detector to the implementation choices not fixed by the source paper.

For every configuration the stop-cost c_beta is selected on VALIDATION (same rule as run_reproduction.py),
and the timely-detection rate is reported on the TEST set of the IEEE-14 source setting.
Output: results/reproduction/sensitivity.csv (+ per-unit JSON cache).
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dqn_detector import DQNDetector  # noqa: E402
from run_reproduction import load, evaluate, is_att, OUT  # noqa: E402

UNITS = os.path.join(OUT, "units_sens")
os.makedirs(UNITS, exist_ok=True)
CONFIGS = {
    "reference": {},
    "lr_1e-4": dict(lr=1e-4),
    "lr_3e-3": dict(lr=3e-3),
    "hidden_128": dict(hidden=128),
    "gamma_0.9": dict(gamma=0.9),
    "eps_0.05": dict(eps=0.05),
    "eps_0.2": dict(eps=0.2),
    "standard_replay": dict(counterfactual=False),
    "sgd_lr_0.1": dict(optim="sgd", lr=0.1),
}
C_BETAS = [0.02, 0.05, 0.1]
SEEDS = [0, 1, 2]


def run(system="ieee14"):
    tr, va, te = load(f"{system}_src_train"), load(f"{system}_src_val"), load(f"{system}_src_test")
    rows = []
    for name, kw in CONFIGS.items():
        for seed in SEEDS:
            cand = []
            for cb in C_BETAS:
                p = os.path.join(UNITS, f"{system}_{name}_cb{cb}_s{seed}.json")
                if os.path.exists(p):
                    r = json.load(open(p))
                else:
                    det = DQNDetector(c_beta=cb, seed=seed, **kw).fit(tr["eta"], tr["tau"], is_att(tr))
                    r = dict(val=evaluate(det, va), test=evaluate(det, te), train_time=det.train_time)
                    json.dump(r, open(p, "w"))
                cand.append((cb, r))
            # source-paper rule on validation: FAR < 5 %, then minimum delay
            feas = [c for c in cand if c[1]["val"]["far"] < 0.05] or cand
            cb, r = min(feas, key=lambda c: (np.nan_to_num(c[1]["val"]["delay_s"], nan=1e9), c[1]["val"]["far"]))
            rows.append(dict(system=system, config=name, seed=seed, c_beta=cb, val_timely=r["val"]["timely_rate"],
                             test_timely=r["test"]["timely_rate"], test_far=r["test"]["far"],
                             test_delay=r["test"]["delay_s"]))
            print(name, seed, cb, round(r["test"]["timely_rate"], 4), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    print(df.groupby("config")[["test_timely", "test_far"]].agg(["mean", "std", "max"]).round(4))


if __name__ == "__main__":
    run(*(sys.argv[1:] or ["ieee14"]))
