"""Real correlated wind (GEFCom2014-W, zones 1-4) and load (ISO-NE) hourly profiles with a temporal split.

Train anchors: 2012-01-01 .. 2012-12-31
Val   anchors: 2013-01-01 .. 2013-05-31
Test  anchors: 2013-06-01 .. 2013-12-31
"""
import os
import numpy as np
import pandas as pd

# Folder with the raw public data (see README): GEFCom2014_Dataset/, isone_load.csv, PJM_Load_hourly.csv.
# Set the environment variable PGTCN_DATA to its location; defaults to <repo>/data_raw.
DATA = os.environ.get("PGTCN_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_raw"))
WDIR = os.path.join(DATA, "GEFCom2014_Dataset", "GEFCom2014 Data", "GEFCom2014-W_V2", "Wind")
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "profiles.parquet")
ZONES = [1, 2, 3, 4]
SPLITS = {"train": ("2012-01-01", "2012-12-31 23:00"),
          "val": ("2013-01-01", "2013-05-31 23:00"),
          "test": ("2013-06-01", "2013-12-31 22:00")}


def _read_wind():
    frames = []
    for z in ZONES:
        f = os.path.join(WDIR, "Task 15", "Task15_W_Zone1_10", "Task15_W_Zone1_10", f"Task15_W_Zone{z}.csv")
        d = pd.read_csv(f)[["TIMESTAMP", "TARGETVAR"]]
        s = pd.read_csv(os.path.join(WDIR, "Solution to Task 15", "solution15_W.csv"))
        s = s[s.ZONEID == z][["TIMESTAMP", "TARGETVAR"]]
        d = pd.concat([d, s])
        d["t"] = pd.to_datetime(d.TIMESTAMP, format="%Y%m%d %H:%M")
        frames.append(d.set_index("t")["TARGETVAR"].rename(f"w{z}"))
    w = pd.concat(frames, axis=1).sort_index()
    w = w[~w.index.duplicated()]
    return w.interpolate(limit=6).clip(0, 1)


def _read_load():
    d = pd.read_csv(os.path.join(DATA, "isone_load.csv"), sep=";")
    t = pd.to_datetime(d.date, format="%d/%m/%Y") + pd.to_timedelta(d.hour - 1, unit="h")
    s = pd.Series(d.load.values, index=t).sort_index()
    s = s[~s.index.duplicated()]
    return s.rename("load")


def get_profiles():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    w = _read_wind()
    l = _read_load()
    df = w.join(l, how="inner").dropna()
    df = df.loc["2012-01-01":"2013-12-31"]
    tr = df.loc[slice(*SPLITS["train"])]
    df["load"] = df["load"] / tr["load"].mean()  # normalise by TRAIN mean only
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_parquet(CACHE)
    return df


def split_frame(split):
    df = get_profiles()
    a, b = SPLITS[split]
    return df.loc[a:b]


def increment_corr(split="train"):
    """Correlation of hourly wind increments among farms (TRAIN only)."""
    d = split_frame(split)[[f"w{z}" for z in ZONES]].diff().dropna().values
    return np.corrcoef(d.T)


if __name__ == "__main__":
    df = get_profiles()
    print(df.shape, df.index.min(), df.index.max())
    for s in SPLITS:
        f = split_frame(s)
        print(s, len(f), "wind mean", f[[f"w{z}" for z in ZONES]].mean().round(3).tolist(),
              "load range", round(f.load.min(), 3), round(f.load.max(), 3))
    print("level corr (train)\n", np.round(split_frame("train")[[f"w{z}" for z in ZONES]].corr().values, 3))
    print("increment corr (train)\n", np.round(increment_corr(), 3))


# ----------------------------------------------------------------------------------- external (unseen) drivers
_EXT = {}


def _read_wind_zones(zones):
    frames = []
    s = pd.read_csv(os.path.join(WDIR, "Solution to Task 15", "solution15_W.csv"))
    for z in zones:
        f = os.path.join(WDIR, "Task 15", "Task15_W_Zone1_10", "Task15_W_Zone1_10", f"Task15_W_Zone{z}.csv")
        d = pd.read_csv(f)[["TIMESTAMP", "TARGETVAR"]]
        d = pd.concat([d, s[s.ZONEID == z][["TIMESTAMP", "TARGETVAR"]]])
        d["t"] = pd.to_datetime(d.TIMESTAMP, format="%Y%m%d %H:%M")
        frames.append(d.set_index("t")["TARGETVAR"])
    w = pd.concat(frames, axis=1).sort_index()
    w = w[~w.index.duplicated()]
    w.columns = [f"w{z}" for z in ZONES[:len(zones)]]       # stored under the default column names
    return w.interpolate(limit=6).clip(0, 1)


def _read_pjm():
    d = pd.read_csv(os.path.join(DATA, "PJM_Load_hourly.csv"))
    t = pd.to_datetime(d.Datetime) + pd.DateOffset(years=13)  # 1999-2000 calendar re-anchored to 2012-2013
    s = pd.Series(d.PJM_Load_MW.values, index=t).sort_index()
    return s[~s.index.duplicated()].rename("load")


def ext_frame(split, zones=(1, 2, 3, 4), load_src="isone"):
    """Profiles from drivers never used for training: other GEFCom wind zones and/or the PJM load."""
    key = (tuple(zones), load_src)
    if key not in _EXT:
        w = _read_wind_zones(zones)
        l = _read_pjm() if load_src == "pjm" else _read_load()
        df = w.join(l, how="inner").dropna().loc["2012-01-01":"2013-12-31"]
        df["load"] = df["load"] / df.loc[slice(*SPLITS["train"]), "load"].mean()
        _EXT[key] = df
    a, b = SPLITS[split]
    return _EXT[key].loc[a:b]
