"""Tabular benchmarks on causal per-step window summaries (mean, mean|.|, last over W=32 steps).

python tabular.py --system ieee14 --model lgbm --seed 0 [--tune 12]
Hyper-parameters are tuned with Optuna TPE on VALIDATION macro-F1 only (seed 0), then frozen for all seeds.
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import optuna

import metrics as M
from train_nn import load_split, _summ, decision_rule, ROOT, FEAT as DATA_FEAT
from models import window_summary

optuna.logging.set_verbosity(optuna.logging.WARNING)
DEV = torch.device("cuda")
W = 32
N_JOBS = 12


TOK = dict(fz=None, horizons=(60, 120))


def summaries(X, bs=100):
    out = []
    for s in range(0, len(X), bs):
        x = X[s:s + bs].to(DEV).float()
        if TOK["fz"] is not None:   # same multi-horizon bus tokens as the proposed model, flattened over buses
            tok = TOK["fz"].node_tokens(x, TOK["horizons"])            # (b,T,nR,F)
            tok = tok.reshape(tok.shape[0], tok.shape[1], -1)
            c = torch.cumsum(torch.nn.functional.pad(tok, (0, 0, W, 0)), 1)
            out.append(torch.cat([tok, (c[:, W:] - c[:, :-W]) / W], -1).half().cpu())
        else:
            out.append(window_summary(x, W).half().cpu())
    return torch.cat(out)                                              # (E,T,D) fp16


def train_rows(S, meta, per_ep, rng):
    E, T, D = S.shape
    y = M.step_labels(meta["cls"], meta["tau"], T)
    rows = []
    for i in range(E):
        a = rng.integers(M.T_EVAL, T, size=per_ep // 2)
        b = rng.integers(max(M.T_EVAL, meta["tau"][i]), T, size=per_ep - per_ep // 2)
        t = np.concatenate([a, b])
        rows.append(np.stack([np.full(len(t), i), t], 1))
    r = np.concatenate(rows)
    return S[r[:, 0], r[:, 1]].float().numpy(), y[r[:, 0], r[:, 1]]


def balanced_w(y):
    c = np.bincount(y, minlength=M.K).astype(float)
    return (len(y) / (M.K * np.maximum(c, 1)))[y]


class Std:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


def make(model, hp, seed):
    if model == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=hp["C"], max_iter=300, class_weight="balanced")
    if model == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.kernel_approximation import Nystroem
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        return make_pipeline(Nystroem(gamma=hp["gamma"], n_components=1000, random_state=seed),
                             CalibratedClassifierCV(LinearSVC(C=hp["C"], class_weight="balanced", max_iter=3000), cv=3))
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=hp["n"], max_depth=hp["depth"], max_features=hp["mf"],
                                      min_samples_leaf=hp["leaf"], class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=seed)
    if model == "et":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=hp["n"], max_depth=hp["depth"], max_features=hp["mf"],
                                    min_samples_leaf=hp["leaf"], class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=seed)
    if model == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=hp["n"], max_depth=hp["depth"], learning_rate=hp["lr"], subsample=hp["sub"],
                                 colsample_bytree=hp["col"], tree_method="hist", device="cpu", n_jobs=N_JOBS, random_state=seed,
                                 reg_lambda=hp["l2"], eval_metric="mlogloss")
    if model == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=hp["n"], num_leaves=hp["leaves"], learning_rate=hp["lr"], subsample=hp["sub"],
                                  subsample_freq=1, colsample_bytree=hp["col"], reg_lambda=hp["l2"], n_jobs=N_JOBS,
                                  random_state=seed, verbose=-1)
    if model == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=hp.get("n_cat", hp["n"]), depth=hp["depth"], learning_rate=hp["lr"], l2_leaf_reg=hp["l2"],
                                  task_type="CPU", thread_count=N_JOBS, random_seed=seed, verbose=0, loss_function="MultiClass")
    if model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=hp["n"], learning_rate=hp["lr"], max_leaf_nodes=hp["leaves"],
                                              l2_regularization=hp["l2"], class_weight="balanced", random_state=seed)
    raise ValueError(model)


def space(model, t):
    if model == "lr":
        return dict(C=t.suggest_float("C", 1e-3, 10, log=True))
    if model == "svm":
        return dict(C=t.suggest_float("C", 1e-3, 10, log=True), gamma=t.suggest_float("gamma", 1e-5, 1e-2, log=True))
    if model in ("rf", "et"):
        return dict(n=t.suggest_int("n", 100, 500, step=100), depth=t.suggest_categorical("depth", [None, 12, 20, 30]),
                    mf=t.suggest_categorical("mf", ["sqrt", "log2", 0.1]), leaf=t.suggest_int("leaf", 1, 10))
    if model in ("xgb", "lgbm", "cat", "hgb"):
        hp = dict(n=t.suggest_int("n", 150, 800, step=50), lr=t.suggest_float("lr", 0.02, 0.2, log=True),
                  l2=t.suggest_float("l2", 1e-3, 10, log=True))
        if model == "xgb":
            hp["depth"] = t.suggest_int("depth", 4, 10)
        if model == "cat":
            hp["depth"] = t.suggest_int("depth", 4, 8)
            hp["n"] = t.suggest_int("n_cat", 150, 600, step=50)
        if model in ("lgbm", "hgb"):
            hp["leaves"] = t.suggest_int("leaves", 15, 127)
        if model in ("xgb", "lgbm"):
            hp["sub"] = t.suggest_float("sub", 0.6, 1.0)
            hp["col"] = t.suggest_float("col", 0.1, 0.8)
        return hp
    return {}


def fit_model(model, hp, Xtr, ytr, seed):
    if model == "hgb":  # avoid OpenMP oversubscription on the shared CPU
        from threadpoolctl import threadpool_limits
        threadpool_limits(4)
    clf = make(model, hp, seed)
    sw = balanced_w(ytr)
    if model in ("lr", "svm", "rf", "et", "hgb"):
        if model == "svm":
            clf.fit(Xtr, ytr)
        elif model in ("rf", "et", "hgb", "lr"):
            clf.fit(Xtr, ytr) if model != "hgb" else clf.fit(Xtr, ytr)
    else:
        clf.fit(Xtr, ytr, sample_weight=sw)
    return clf


class KNN:
    """GPU brute-force k-NN with distance-free class-balanced voting."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = torch.tensor(X, device=DEV, dtype=torch.float16)
        self.y = torch.tensor(y, device=DEV)
        c = torch.bincount(self.y, minlength=M.K).float()
        self.w = (c.sum() / (M.K * c.clamp(min=1)))
        return self

    def predict_proba(self, X):
        out = []
        for s in range(0, len(X), 4096):
            q = torch.tensor(X[s:s + 4096], device=DEV, dtype=torch.float16)
            d = torch.cdist(q.float(), self.X.float())
            nn_ = d.topk(self.k, largest=False).indices
            lab = self.y[nn_]
            p = torch.zeros(len(q), M.K, device=DEV)
            p.scatter_add_(1, lab, self.w[lab])
            out.append((p / p.sum(1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)


def predict_episodes(clf, S, std, bs=100):
    E, T, D = S.shape
    P = np.zeros((E, T, M.K), np.float32)
    P[:, :, 0] = 1.0
    for s in range(0, E, bs):
        x = std(S[s:s + bs, M.T_EVAL:].reshape(-1, D).float().numpy())
        p = clf.predict_proba(x)
        P[s:s + bs, M.T_EVAL:] = p.reshape(-1, T - M.T_EVAL, M.K)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="bench")
    ap.add_argument("--tune", type=int, default=12)
    ap.add_argument("--per_ep", type=int, default=12)
    ap.add_argument("--tests", default="")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--feat", default="flat")          # flat | tok (multi-horizon bus tokens)
    ap.add_argument("--save_all", action="store_true")
    a = ap.parse_args()
    mname = a.model + ("_tok" if a.feat == "tok" else "")
    if a.feat == "tok":
        import pickle as _pk
        from features import region_of, DATA as _D
        fz = _pk.load(open(os.path.join(_D, "feat", f"{a.system}_featurizer.pkl"), "rb"))
        fz.r = region_of(a.system)
        TOK["fz"] = fz
    out_dir = os.path.join(ROOT, "results", a.stage, a.system)
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{mname}__{a.tag}__seed{a.seed}.json")
    if os.path.exists(fn):
        print("exists", fn); return
    rng = np.random.default_rng(a.seed)
    Xtr, mtr = load_split(a.system, "hard", "train", a.frac, a.seed)
    Xva, mva = load_split(a.system, "hard", "val")
    Str, Sva = summaries(Xtr), summaries(Xva)
    Rtr, ytr = train_rows(Str, mtr, a.per_ep, rng)
    std = Std().fit(Rtr)
    Rtr = std(Rtr)
    cap = {"svm": 15000, "knn": 30000}.get(a.model)
    if cap and len(Rtr) > cap:
        # class-stratified cap for kernel/instance methods
        sel = rng.choice(len(Rtr), cap, replace=False)
        Rtr, ytr = Rtr[sel], ytr[sel]
    # --------- hyper-parameters: tuned once (seed 0) on VALIDATION rows, cached
    # hyper-parameters are tuned on IEEE-14 validation and kept fixed for all systems (same protocol as neural models)
    hp_file = os.path.join(ROOT, "results", "tuning", "ieee14", f"{mname}.json")
    os.makedirs(os.path.dirname(hp_file), exist_ok=True)
    Rva, yva = train_rows(Sva, mva, 6, np.random.default_rng(99))
    Rva = std(Rva)
    t_tune = 0.0
    if a.seed != 0:  # hyper-parameters are tuned once by the seed-0 job; other seeds wait for them
        waited = 0
        while not os.path.exists(hp_file) and waited < 6 * 3600:
            time.sleep(30)
            waited += 30
    if os.path.exists(hp_file):
        hp = json.load(open(hp_file))["best"]
    else:
        t0 = time.time()
        from sklearn.metrics import f1_score
        sub = np.random.default_rng(7).choice(len(Rtr), min(len(Rtr), 30000), replace=False)

        def obj(t):
            if a.model == "knn":
                k = t.suggest_int("k", 3, 51, step=2)
                clf = KNN(k).fit(Rtr[sub], ytr[sub])
            else:
                clf = fit_model(a.model, space(a.model, t), Rtr[sub], ytr[sub], 0)
            return f1_score(yva, clf.predict_proba(Rva).argmax(1), average="macro")
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        st.optimize(obj, n_trials=a.tune)
        hp = st.best_params
        t_tune = time.time() - t0
        json.dump(dict(best=hp, value=st.best_value, trials=[dict(p=tr.params, v=tr.value) for tr in st.trials],
                       tune_time_s=t_tune), open(hp_file, "w"), indent=1)
    if a.model in ("rf", "et") and "depth" in hp and hp["depth"] == "None":
        hp["depth"] = None
    t0 = time.time()
    clf = KNN(hp["k"]).fit(Rtr, ytr) if a.model == "knn" else fit_model(a.model, hp, Rtr, ytr, a.seed)
    train_time = time.time() - t0
    pv = predict_episodes(clf, Sva, std)
    ps = None
    if os.path.exists(os.path.join(DATA_FEAT, f"{a.system}_ood_streamcal_test.npy")):
        Xs, _ = load_split(a.system, "ood_streamcal", "test")
        ps = predict_episodes(clf, summaries(Xs, bs=4), std, bs=2)
    D = decision_rule(pv, mva, ps)
    h, cal, h_cal = D["cusum_h"], tuple(D["llr_cal"]), D["cusum_h_cal"]
    res = dict(val=_summ(pv, mva, h, cal, h_cal, D), **D)
    if a.save_all:
        import pickle as _pk
        _pk.dump(dict(clf=clf, std=std, feat=a.feat), open(fn.replace(".json", ".pkl"), "wb"))
        np.savez_compressed(fn.replace(".json", "_valprob.npz"), prob=pv.astype(np.float16), cls=mva["cls"], tau=mva["tau"],
                            kappa=mva["kappa"], ramp_L=mva["ramp_L"])
    tests = {} if a.no_test else {"test": load_split(a.system, "hard", "test")}
    for t in [x for x in a.tests.split(",") if x]:
        tests[t] = load_split(a.system, t, "test")
    for name, (X, m) in tests.items():
        S = summaries(X)
        t0 = time.time()
        p = predict_episodes(clf, S, std)
        inf = time.time() - t0
        res[name] = _summ(p, m, h, cal, h_cal, D)
        if name == "test":
            np.savez_compressed(fn.replace(".json", "_testprob.npz"), prob=p.astype(np.float16), cls=m["cls"], tau=m["tau"],
                                kappa=m["kappa"], ramp_L=m["ramp_L"])
            res["throughput_steps_s"] = (S.shape[0] * (S.shape[1] - M.T_EVAL)) / inf
    # single-decision latency (one feature row)
    x1 = std(Sva[:1, -1].float().numpy())
    t0 = time.time()
    for _ in range(20):
        clf.predict_proba(x1)
    lat = (time.time() - t0) / 20 * 1e3
    import pickle
    size = len(pickle.dumps(clf)) / 2 ** 20 if a.model != "knn" else Rtr.nbytes / 2 ** 20
    res.update(system=a.system, model=mname, seed=a.seed, hp=hp, train_time_s=train_time, tune_time_s=t_tune,
               latency_ms=lat, model_size_mb=size, n_train_rows=len(Rtr), n_features=Rtr.shape[1], frac=a.frac, tag=a.tag)
    json.dump(res, open(fn, "w"), default=float)
    print(f"DONE {a.system} {a.model} seed{a.seed}: val F1 {res['val']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
