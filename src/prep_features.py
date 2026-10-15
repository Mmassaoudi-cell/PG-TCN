"""Fit the Featurizer on TRAIN only and cache float16 flat features for every split / OOD set."""
import os
import sys
import glob
import pickle
import numpy as np
from features import Featurizer, load_raw, DATA

FEAT = os.path.join(DATA, "feat")
os.makedirs(FEAT, exist_ok=True)
META = ["cls", "tau", "kappa", "ramp_L", "att_bus", "scale", "eta", "ev_kind"]


def cache(system, fz, name, split):
    out = os.path.join(FEAT, f"{system}_{name}_{split}")
    if os.path.exists(out + ".npy"):
        return
    d = load_raw(system, name, split)
    d = fz.transform(d)
    np.save(out + ".tmp.npy", d["X_flat"].astype(np.float16))
    os.replace(out + ".tmp.npy", out + ".npy")
    np.savez(out + "_meta.npz", **{k: d[k] for k in META if k in d})
    print("cached", out, d["X_flat"].shape, flush=True)


def run(system):
    pk = os.path.join(FEAT, f"{system}_featurizer.pkl")
    if os.path.exists(pk):
        fz = pickle.load(open(pk, "rb"))
    else:
        tr = load_raw(system, "hard", "train")
        fz = Featurizer(system).fit(tr)
        tr = fz.transform(tr)
        out = os.path.join(FEAT, f"{system}_hard_train")
        np.save(out + ".npy", tr["X_flat"].astype(np.float16))
        np.savez(out + "_meta.npz", **{k: tr[k] for k in META if k in tr})
        fz.r = None
        pickle.dump(fz, open(pk, "wb"))
        print("fitted+cached", system, tr["X_flat"].shape, flush=True)
    from features import region_of
    fz.r = region_of(system)
    for f in sorted(glob.glob(os.path.join(DATA, f"{system}_*_test.npz")) + glob.glob(os.path.join(DATA, f"{system}_hard_val.npz"))):
        base = os.path.basename(f)[len(system) + 1:-4]
        name, split = base.rsplit("_", 1)
        if name.startswith("hard") or name.startswith("ood"):
            cache(system, fz, name, split)


if __name__ == "__main__":
    for s in sys.argv[1].split(","):
        run(s)
