"""Controlled efficiency benchmark (same conditions for every model; run when GPU is otherwise idle).
Outputs results/final/efficiency_controlled.csv
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import yaml

from models import MODELS, GRAPH_MODELS, n_params
from features import region_of
from train_nn import FEAT, ROOT

DEV = torch.device("cuda")
cfg = yaml.safe_load(open(os.path.join(ROOT, "FINAL_MODEL_CONFIG.yaml")))
P = cfg["model"]["name"]


def hp_of(m):
    if m == P:
        return cfg["model"]["hyperparameters"]
    mm = "gcn_gru" if m.startswith("gcn_gru") else m
    f = os.path.join(ROOT, "results", "tuning", "ieee14", f"nn_{mm}{'_tok' if mm == 'gcn_gru' else ''}.json")
    hp = json.load(open(f))["best"] if os.path.exists(f) else {}
    if mm == "gcn_gru":
        hp["horizons"] = cfg["model"]["hyperparameters"]["horizons"]
    if m == "gcn_gru_matched":
        hp["hid"] = 112
    return hp


def build(system, m):
    fz = pickle.load(open(os.path.join(FEAT, f"{system}_featurizer.pkl"), "rb"))
    fz.r = region_of(system)
    hp = hp_of(m)
    kw = {k: v for k, v in hp.items() if k in ("hid", "drop")}
    cin = fz.r.M * 2 + len(fz.r.mPw) + 1
    hor = tuple(hp.get("horizons", ()))
    gm = "gcn_gru" if m.startswith("gcn_gru") else m
    net = GRAPH_MODELS[gm](24 + 10 * len(hor), fz.r.A, **kw) if gm in GRAPH_MODELS else MODELS[m](cin, **kw)
    net.hor = hor
    return net.eval(), fz, cin


def fwd(net, fz, x, graph):
    nodes = fz.node_tokens(x, net.hor) if graph else None
    return net(x, nodes=nodes)


@torch.no_grad()
def bench(system, m, dev, reps=20):
    net, fz, cin = build(system, m)
    net = net.to(dev)
    graph = m == P or m.startswith("gcn_gru")
    x1 = torch.randn(1, 160, cin, device=dev)           # streaming decision: last 160 scans (>= drift window + field)
    for _ in range(5):
        fwd(net, fz, x1, graph)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fwd(net, fz, x1, graph)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / reps * 1e3
    out = dict(system=system, model=m, device=dev.type, params=n_params(net), latency_ms=lat)
    if dev.type == "cuda":
        xb = torch.randn(64, 200, cin, device=dev)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            fwd(net, fz, xb, graph)
        torch.cuda.synchronize()
        out["throughput_steps_s"] = 3 * 64 * 200 / (time.perf_counter() - t0)
        out["peak_infer_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        with FlopCounterMode(display=False) as fc:
            fwd(net, fz, x1, graph)
        out["gflops_per_decision"] = fc.get_total_flops() / 1e9
    out["model_size_mb"] = sum(p.numel() * p.element_size() for p in net.parameters()) / 2 ** 20
    return out


@torch.no_grad()
def ckf_cost(system, reps=30):
    """Shared front end: one coupled-CKF step (source paper) for a single episode on GPU/CPU."""
    import sim
    r = region_of(system)
    ckf = sim.CoupledCKF(r, np.eye(2 * len(r.mPw)) * 1e-5)
    x0 = torch.tensor(np.concatenate([np.zeros(len(r.th_idx)), np.ones(r.nR)]), dtype=torch.float64, device=DEV)[None]
    z = r.h(x0)[:, None].repeat(1, reps, 1)
    ckf.run(z[:, :3], x0)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ckf.run(z, x0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    rows = []
    models = [P, "flat_tcn", "gcn_gru", "gcn_gru_matched", "stdcnn", "transformer", "gru", "lstm", "cnn1d", "mlp", "mamba"]
    torch.set_num_threads(4)
    for s in cfg["data"]["systems"]:
        for m in models:
            for dev in [DEV, torch.device("cpu")]:
                rows.append(bench(s, m, dev))
                print(rows[-1], flush=True)
                import pandas as pd
                pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "lib_final", "efficiency_controlled.csv"), index=False)
        rows.append(dict(system=s, model="ckf_front_end", device="cuda", latency_ms=ckf_cost(s)))
        # source DQN policy network (1->64->64->2)
        q = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
        x = torch.randn(1, 1)
        t0 = time.perf_counter()
        for _ in range(1000):
            q(x)
        rows.append(dict(system=s, model="source_dqn", device="cpu", params=sum(p.numel() for p in q.parameters()),
                         latency_ms=(time.perf_counter() - t0)))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "lib_final", "efficiency_controlled.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results", "lib_final"), exist_ok=True)
    main()
