"""Cross-grid transfer of PG-TCN (the parameter count does not depend on the grid).

For a source->target pair and seed: (a) zero-shot: source weights, target adjacency and target featurizer, CUSUM
threshold calibrated on target validation; (b) fine-tune: source weights trained further on 10% of the target
training episodes; (c) scratch: the same 10% subset from random initialisation.

python transfer_rev.py ieee14 ieee30 0
Output: results/rev_transfer/<src>_to_<tgt>__seed<k>.json
"""
import os
import sys
import json
import numpy as np
import torch

import metrics as M
from train_nn import Runner, load_split, evaluate_full, ROOT

src, tgt, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = os.path.join(ROOT, "results", "rev_transfer")
os.makedirs(OUT, exist_ok=True)
fn = os.path.join(OUT, f"{src}_to_{tgt}__seed{seed}.json")
if os.path.exists(fn):
    sys.exit(0)
base = os.path.join(ROOT, "results", "lib", src, f"bus_graph_tcn__final__seed{seed}")
res_src = json.load(open(base + ".json"))
hp = res_src["hp"]
state = {k: v for k, v in torch.load(base + ".pt", map_location="cuda").items() if k != "A"}
Xva, mva = load_split(tgt, "hard", "val")
tests = {"test": load_split(tgt, "hard", "test")}
out = {}

r = Runner(tgt, "bus_graph_tcn", hp, seed)
r.net.load_state_dict(state, strict=False)
out["zero_shot"] = {k: v for k, v in evaluate_full(r, Xva, mva, tests, system=tgt)["test"].items() if k != "cm"}

Xtr, mtr = load_split(tgt, "hard", "train", 0.1, seed)
for name, init in [("fine_tune", True), ("scratch", False)]:
    r = Runner(tgt, "bus_graph_tcn", {**hp, "epochs": 30}, seed)
    if init:
        r.net.load_state_dict(state, strict=False)
    r.fit(Xtr, mtr, Xva, mva)
    out[name] = {k: v for k, v in evaluate_full(r, Xva, mva, tests, system=tgt)["test"].items() if k != "cm"}
    out[name]["train_time_s"] = r.train_time
json.dump(dict(src=src, tgt=tgt, seed=seed, **out), open(fn, "w"))
print(src, tgt, seed, {k: round(v["macro_f1"], 4) for k, v in out.items()}, flush=True)
