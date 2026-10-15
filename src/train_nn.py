"""Train / evaluate causal per-step neural models on cached features.

python train_nn.py --system ieee14 --model bus_graph_tcn --seed 0 --stage screen
Results: results/<stage>/<system>/<model>__<tag>__seed<k>.json (+ test probabilities .npz)
"""
import os
import sys
import json
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import region_of, DATA
import metrics as M
from models import MODELS, GRAPH_MODELS, n_params

DEV = torch.device("cuda")
FEAT = os.path.join(DATA, "feat")
ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_split(system, name, split, frac=1.0, seed=0):
    X = np.load(os.path.join(FEAT, f"{system}_{name}_{split}.npy"), mmap_mode="r")
    meta = dict(np.load(os.path.join(FEAT, f"{system}_{name}_{split}_meta.npz")))
    idx = np.arange(len(X))
    if frac < 1.0:  # stratified subsample of TRAIN episodes (low-data study)
        rng = np.random.default_rng(1000 + seed)
        idx = np.concatenate([rng.choice(np.where(meta["cls"] == c)[0], max(2, int(round(frac * (meta["cls"] == c).sum()))),
                                         replace=False) for c in range(M.K)])
        idx.sort()
    X = torch.tensor(np.asarray(X[idx]), dtype=torch.float16)
    meta = {k: v[idx] for k, v in meta.items()}
    return X, meta


class Runner:
    def __init__(self, system, model, hp, seed=0, feature_mask=None):
        self.system, self.model_name, self.hp, self.seed = system, model, hp, seed
        torch.manual_seed(seed); np.random.seed(seed)
        fz = pickle.load(open(os.path.join(FEAT, f"{system}_featurizer.pkl"), "rb"))
        fz.r = region_of(system)
        self.fz = fz
        self.graph = model in GRAPH_MODELS
        self.feature_mask = feature_mask  # optional channel ablation (callable on X)
        cin = fz.r.M * 2 + len(fz.r.mPw) + 1
        kw = {k: v for k, v in hp.items() if k in ("hid", "drop", "layers", "W", "dil", "use_graph")}
        self.horizons = tuple(hp.get("horizons", ()))
        if self.graph:
            self.net = GRAPH_MODELS[model](24 + 10 * len(self.horizons), fz.r.A, **kw).to(DEV)
        else:
            self.net = MODELS[model](cin, **kw).to(DEV)

    def prep(self, xb):
        x = xb.to(DEV, non_blocking=True).float()
        if self.feature_mask is not None:
            x = self.feature_mask(x)
        nodes = self.fz.node_tokens(x, self.horizons) if self.graph else None
        return x, nodes

    def forward(self, xb, return_nodes=False):
        x, nodes = self.prep(xb)
        if self.graph and return_nodes:
            return self.net(x, nodes=nodes, return_nodes=True)
        return self.net(x, nodes=nodes)

    @torch.no_grad()
    def predict(self, X, bs=128):
        self.net.eval()
        out = []
        for s in range(0, len(X), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.hp.get("amp", True)):
                lg = self.forward(X[s:s + bs])
            out.append(torch.softmax(lg.float(), -1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def predict_nodes(self, X, bs=128):
        self.net.eval()
        out = []
        for s in range(0, len(X), bs):
            _, nl = self.forward(X[s:s + bs], return_nodes=True)
            out.append(torch.sigmoid(nl.float()).cpu().numpy())
        return np.concatenate(out)

    def fit(self, Xtr, mtr, Xva, mva, log=None, trial=None):
        hp = self.hp
        E, T, _ = Xtr.shape
        ytr = torch.tensor(M.step_labels(mtr["cls"], mtr["tau"], T), dtype=torch.long)
        cnt = np.bincount(ytr[:, M.T_EVAL:].ravel().numpy(), minlength=M.K).astype(float)
        w = torch.tensor((cnt.sum() / (M.K * np.maximum(cnt, 1))) ** hp.get("cw_pow", 1.0), dtype=torch.float32, device=DEV)
        att_bus = torch.tensor(mtr["att_bus"], dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp.get("wd", 1e-4))
        epochs = hp.get("epochs", 30)
        steps = epochs * int(np.ceil(E / hp["bs"]))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps, pct_start=0.1)
        best, best_state, bad = -1, None, 0
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        hist = []
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(E)
            for s in range(0, E, hp["bs"]):
                b = perm[s:s + hp["bs"]]
                yb = ytr[b].to(DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=hp.get("amp", True)):
                    if self.graph and hp.get("loc_w", 0) > 0:
                        lg, nl = self.forward(Xtr[b], return_nodes=True)
                    else:
                        lg = self.forward(Xtr[b]); nl = None
                lg = lg.float()[:, M.T_EVAL - 12:]
                loss = F.cross_entropy(lg.reshape(-1, M.K), yb[:, M.T_EVAL - 12:].reshape(-1), weight=w,
                                       label_smoothing=hp.get("ls", 0.0))
                if nl is not None:
                    # localisation: attacked buses active after onset in FDIA episodes; zero elsewhere
                    act = (yb >= 3).float()[..., None]                           # (B,T,1)
                    tgt = act * att_bus[b].to(DEV)[:, None, :]
                    loss = loss + hp["loc_w"] * F.binary_cross_entropy_with_logits(nl.float()[:, M.T_EVAL:], tgt[:, M.T_EVAL:])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
            pv = self.predict(Xva)
            f1 = M.step_metrics(pv, mva["cls"], mva["tau"])["macro_f1"]
            hist.append(dict(epoch=ep, loss=float(loss), val_macro_f1=float(f1)))
            if log:
                print(f"    ep {ep} loss {float(loss):.4f} val macroF1 {f1:.4f}", flush=True)
            if trial is not None:
                import optuna
                trial.report(f1, ep)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if f1 > best + 1e-4:
                best, bad = f1, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= hp.get("patience", 6):
                    break
        self.net.load_state_dict(best_state)
        self.train_time = time.time() - t0
        self.peak_mem_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        self.hist = hist
        self.best_val = best
        return self

    @torch.no_grad()
    def latency(self, X, reps=20):
        """Streaming latency: one episode prefix of 64 steps (covers receptive field), batch 1, per decision."""
        self.net.eval()
        x = X[:1, -64:]
        for _ in range(3):
            self.forward(x)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(reps):
            self.forward(x)
        torch.cuda.synchronize()
        lat = (time.time() - t0) / reps
        xb = X[:128]
        torch.cuda.synchronize(); t0 = time.time()
        self.forward(xb)
        torch.cuda.synchronize()
        thr = xb.shape[0] * xb.shape[1] / (time.time() - t0)
        return lat * 1e3, thr


DEFAULT_HP = dict(lr=2e-3, bs=64, epochs=30, patience=6, hid=64, drop=0.1, wd=1e-4, amp=True)
MODEL_HP = {"bus_graph_tcn": dict(hid=48, loc_w=0.2), "mlp": dict(hid=256), "stdcnn": dict(hid=32),
            "mamba": dict(amp=False), "transformer": dict(lr=1e-3)}


def stream_cal_probs(r, system):
    """Predictions on the VALIDATION-period hour-long streams used to calibrate the long-run alarm rate."""
    if not os.path.exists(os.path.join(FEAT, f"{system}_ood_streamcal_test.npy")):
        return None
    Xs, _ = load_split(system, "ood_streamcal", "test")
    return r.predict(Xs, bs=2) if hasattr(r, "net") else r(Xs, long=True)


def decision_rule(pv, mva, ps):
    """Validation calibration of all decision statistics: raw log-odds CUSUM (h), calibrated-LLR CUSUM (h_cal) and
    the two-parameter calibrated-LLR CUSUM (k, h2) that also bounds validation stream alarms per day."""
    h = M.calibrate_h(M.attack_llr(pv), mva["cls"], mva["tau"])
    cal = M.fit_llr_calibration(pv, mva["cls"], mva["tau"])
    lv = M.calibrated_llr(pv, cal)
    h_cal = M.calibrate_h(lv, mva["cls"], mva["tau"])
    k2, h2 = (0.0, h_cal) if ps is None else M.calibrate_hk(lv, mva["cls"], mva["tau"], M.calibrated_llr(ps, cal))
    return dict(cusum_h=h, llr_cal=list(cal), cusum_h_cal=h_cal, cusum_k2=k2, cusum_h2=h2)


def evaluate_full(r, Xva, mva, tests, save_probs=None, save_all=False, system=None):
    pv = r.predict(Xva)
    ps = stream_cal_probs(r, system) if system else None
    D = decision_rule(pv, mva, ps)
    h, cal, h_cal = D["cusum_h"], tuple(D["llr_cal"]), D["cusum_h_cal"]
    res = dict(val=_summ(pv, mva, h, cal, h_cal, D))
    if save_probs and save_all:
        np.savez_compressed(save_probs.replace("_testprob.npz", "_valprob.npz"), prob=pv.astype(np.float16),
                            cls=mva["cls"], tau=mva["tau"], kappa=mva["kappa"], ramp_L=mva["ramp_L"])
    for name, (X, m) in tests.items():
        p = r.predict(X)
        res[name] = _summ(p, m, h, cal, h_cal, D)
        if save_probs and (name == "test" or save_all):
            fn = save_probs if name == "test" else save_probs.replace("_testprob.npz", f"_{name}prob.npz")
            np.savez_compressed(fn, prob=p.astype(np.float16), **{k: m[k] for k in ("cls", "tau", "kappa", "ramp_L") if k in m})
    res.update(D)
    return res


def _summ(p, m, h, cal=None, h_cal=None, D=None):
    sm = M.step_metrics(p, m["cls"], m["tau"])
    cm = sm.pop("cm").tolist()
    al = M.cusum_alarms(M.attack_llr(p), h)
    em = M.episode_metrics(al, m["cls"], m["tau"], p.shape[1])
    out = {**{k: float(v) for k, v in sm.items()}, **{f"ep_{k}": float(v) for k, v in em.items()},
           "ep_class_macro_f1": float(M.episode_class_f1(p, m["cls"])), "cm": cm}
    if cal is not None:
        al = M.cusum_alarms(M.calibrated_llr(p, cal), h_cal)
        out.update({f"epcal_{k}": float(v) for k, v in M.episode_metrics(al, m["cls"], m["tau"], p.shape[1]).items()})
    if D is not None:   # primary decision rule
        al = M.cusum_alarms(M.calibrated_llr(p, tuple(D["llr_cal"])), D["cusum_h2"], k=D["cusum_k2"])
        out.update({f"epk_{k}": float(v) for k, v in M.episode_metrics(al, m["cls"], m["tau"], p.shape[1]).items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--model", default="flat_tcn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="screen")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--hp", default="{}")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--tests", default="")           # comma list of extra test sets (names), e.g. ood_mimic
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--ablate", default="")          # no_vel | no_wres | no_eta | no_innov (channel zeroing)
    ap.add_argument("--save_all", action="store_true")  # save checkpoint + val/OOD probabilities
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", a.stage, a.system)
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{a.model}__{a.tag}__seed{a.seed}.json")
    if os.path.exists(fn):
        print("exists", fn); return
    hp = {**DEFAULT_HP, **MODEL_HP.get(a.model, {}), **json.loads(a.hp)}
    Xtr, mtr = load_split(a.system, "hard", "train", a.frac, a.seed)
    Xva, mva = load_split(a.system, "hard", "val")
    r = Runner(a.system, a.model, hp, a.seed, feature_mask=make_mask(a.system, a.ablate))
    r.fit(Xtr, mtr, Xva, mva, log=True)
    tests = {}
    if not a.no_test:
        tests["test"] = load_split(a.system, "hard", "test")
    for t in [x for x in a.tests.split(",") if x]:
        tests[t] = load_split(a.system, t, "test")
    probs = None if (a.no_test and not a.save_all) else fn.replace(".json", "_testprob.npz")
    res = evaluate_full(r, Xva, mva, tests, save_probs=probs, save_all=a.save_all, system=a.system)
    if a.save_all:
        torch.save(r.net.state_dict(), fn.replace(".json", ".pt"))
    lat, thr = r.latency(Xva)
    res.update(ablate=a.ablate, system=a.system, model=a.model, seed=a.seed, tag=a.tag, hp=hp, frac=a.frac, params=n_params(r.net),
               train_time_s=r.train_time, peak_mem_mb=r.peak_mem_mb, latency_ms=lat, throughput_steps_s=thr,
               best_val_macro_f1=r.best_val, hist=r.hist, epochs_run=len(r.hist))
    if a.model == "bus_graph_tcn" and not a.no_test:
        X, m = tests["test"]
        pn = r.predict_nodes(X)
        res["test"].update(_loc_metrics(pn, m))
        if a.save_all:
            att = np.isin(m["cls"], [3, 4])
            np.savez_compressed(fn.replace(".json", "_testnodes.npz"), pn_last=pn[:, -1].astype(np.float16),
                                pn_mean=pn[:, -30:].mean(1).astype(np.float16), att_bus=m["att_bus"], cls=m["cls"])
    json.dump(res, open(fn, "w"))
    print(f"DONE {a.system} {a.model} seed{a.seed}: val F1 {res['val']['macro_f1']:.4f}"
          + ("" if a.no_test else f" test F1 {res['test']['macro_f1']:.4f}") + f" params {res['params']} time {r.train_time:.0f}s", flush=True)


def make_mask(system, ablate):
    if not ablate:
        return None
    r = region_of(system)
    Mm, nw = r.M, len(r.mPw)
    sl = {"no_innov": slice(0, Mm), "no_vel": slice(Mm, 2 * Mm), "no_wres": slice(2 * Mm, 2 * Mm + nw),
          "no_eta": slice(2 * Mm + nw, 2 * Mm + nw + 1)}
    cut = [sl[k] for k in ablate.split("+")]

    def f(x):
        x = x.clone()
        for c in cut:
            x[..., c] = 0.0
        return x
    return f


def _loc_metrics(pn, m, thr=0.5):
    """Bus-level localisation F1 at the last step of FDIA episodes."""
    att = np.isin(m["cls"], [3, 4])
    if not att.any():
        return {}
    pred = pn[att, -1] > thr
    true = m["att_bus"][att].astype(bool)
    tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum()
    return dict(loc_f1=float(2 * tp / max(2 * tp + fp + fn, 1)), loc_precision=float(tp / max(tp + fp, 1)),
                loc_recall=float(tp / max(tp + fn, 1)))


if __name__ == "__main__":
    main()
