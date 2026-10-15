"""Generate all datasets. The filter statistics S_hat are always fitted on TRAIN and re-used for
val/test/OOD sets (no leakage). Seeds are unique per (system, dataset, split).

Usage: python gen_data.py --system ieee14 --what src,hard,ood
"""
import os
import argparse
import hashlib
import time
import numpy as np
import sim

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
SIZES_SRC = dict(train=10000, val=2000, test=10000)
SIZES_HARD = dict(train=6000, val=1500, test=3000)
N_OOD = 1500
# compute-constrained sizes for the larger systems (shared GPU); documented in DATA_AUDIT.md
BIG = {"ieee57", "ieee118"}
SIZES_SRC_BIG = dict(train=5000, val=2000, test=3000)
SIZES_HARD_BIG = dict(train=4000, val=1000, test=2000)
N_OOD_BIG = 1000


def seed_of(*keys):
    return int(hashlib.md5("|".join(map(str, keys)).encode()).hexdigest()[:8], 16)


def save(path, r):
    keep = {k: v for k, v in r.items() if isinstance(v, np.ndarray) or np.isscalar(v)}
    np.savez(path, **keep)


def gen(system, name, split, n, S_hat=None, fsplit=None, **kw):
    path = os.path.join(OUT, f"{system}_{name}_{fsplit or split}.npz")
    if os.path.exists(path):
        try:
            d = dict(np.load(path))
            _ = d["conv"].sum()
            print("exists", path)
            return d
        except Exception:
            os.remove(path)
    tmp = path.replace(".npz", ".tmp.npz")
    t0 = time.time()
    r = sim.generate(system, split, n, seed=seed_of(system, name, split), S_hat=S_hat, verbose=False, **kw)
    r["t_total"] = time.time() - t0
    save(tmp, r)
    os.replace(tmp, path)
    print(f"saved {path} n={n} conv={r['conv'].mean():.4f} time={r['t_total']:.1f}s", flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--what", default="src,hard,ood")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    what = a.what.split(",")
    s = a.system
    if s in BIG:
        SIZES_SRC, SIZES_HARD, N_OOD = SIZES_SRC_BIG, SIZES_HARD_BIG, N_OOD_BIG
    if "src" in what:
        tr = gen(s, "src", "train", SIZES_SRC["train"], mode="src", save_nu=False)
        for sp in ["val", "test"]:
            gen(s, "src", sp, SIZES_SRC[sp], S_hat=tr["S_hat"], mode="src", save_nu=False)
        if s == "ieee14":  # SP Table II: without correlated wind generation (WOCWG)
            trw = gen(s, "srcwocwg", "train", SIZES_SRC["train"], mode="src", save_nu=False,
                      corr_wind=False, use_corr_filter=False)
            for sp in ["val", "test"]:
                gen(s, "srcwocwg", sp, SIZES_SRC[sp], S_hat=trw["S_hat"], mode="src", save_nu=False,
                    corr_wind=False, use_corr_filter=False)
            # cross conditions: WCWG-trained filter on WOCWG data and vice versa
            gen(s, "src_on_wocwg", "test", SIZES_SRC["test"], S_hat=tr["S_hat"], mode="src", save_nu=False,
                corr_wind=False)
            gen(s, "srcwocwg_on_wcwg", "test", SIZES_SRC["test"], S_hat=trw["S_hat"], mode="src", save_nu=False,
                use_corr_filter=False)
            # Diagnostic: single fixed operating point (hypothesis for the published 100%)
            trf = gen(s, "srcfixed", "train", SIZES_SRC["train"], mode="src", save_nu=False, fixed_anchor=100, load_scale=1.0)
            for sp in ["val", "test"]:
                gen(s, "srcfixed", sp, SIZES_SRC[sp], S_hat=trf["S_hat"], mode="src", save_nu=False, fixed_anchor=100,
                    load_scale=1.0)
            # SP Table III: noise x {0.5,1,1.5} and load x {0.7,1.0,1.3}
            for ns in [0.5, 1.0, 1.5]:
                for ls in [0.7, 1.0, 1.3]:
                    gen(s, f"src_n{ns}_l{ls}", "test", 2000, S_hat=tr["S_hat"], mode="src", save_nu=False,
                        noise_scale=ns, load_scale=ls)
    if "hard" in what:
        tr = gen(s, "hard", "train", SIZES_HARD["train"], mode="hard", library=True)
        for sp in ["val", "test"]:
            gen(s, "hard", sp, SIZES_HARD[sp], S_hat=tr["S_hat"], mode="hard", library=True)
    if "ood" in what:
        tr = dict(np.load(os.path.join(OUT, f"{s}_hard_train.npz")))
        S = tr["S_hat"]
        L = dict(mode="hard", library=True)
        gen(s, "ood_mimic", "test", N_OOD, S_hat=S, mimic=True, **L)
        gen(s, "ood_wocwg", "test", N_OOD, S_hat=S, corr_wind=False, **L)
        gen(s, "ood_noise1.5", "test", N_OOD, S_hat=S, noise_scale=1.5, **L)
        gen(s, "ood_load0.7", "test", N_OOD, S_hat=S, load_scale=0.7, **L)
        gen(s, "ood_load1.3", "test", N_OOD, S_hat=S, load_scale=1.3, **L)
    if "rev" in what:  # attack continuum, held-out benign events, unseen real drivers, hour-long streams
        S = dict(np.load(os.path.join(OUT, f"{s}_hard_train.npz")))["S_hat"]
        L = dict(mode="hard", library=True)
        for bl in [0.25, 0.5, 0.75]:
            gen(s, f"ood_mix{int(bl * 100)}", "test", N_OOD, S_hat=S, mimic_blend=bl, **L)
        for ev in ["multi_step", "ext_gen_trip", "ext_line_out"]:   # event types never used for training
            gen(s, f"ood_bn_{ev}", "test", N_OOD, S_hat=S, mode="hard", class_probs=[1, 0, 0, 0, 0], benign_ood=ev)
        gen(s, "ood_zones58", "test", N_OOD, S_hat=S, ext=((5, 6, 7, 8), "isone"), **L)
        gen(s, "ood_pjm", "test", N_OOD, S_hat=S, ext=((1, 2, 3, 4), "pjm"), **L)
        # hour-long streams of normal operation with one library benign event: validation (calibration) and test
        for sp, fs in [("val", "streamval"), ("test", "test")]:
            gen(s, "ood_stream" if sp == "test" else "ood_streamcal", sp, 60, S_hat=S, fsplit="test", T=3600,
                class_probs=[0.34, 0.33, 0.33, 0, 0], tau_range=(300, 3300), chunk=60, **L)
