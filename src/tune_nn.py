"""Stage-3 validation-only tuning of top neural candidates (Optuna TPE + median pruning).
python tune_nn.py --system ieee14 --model bus_graph_tcn --trials 12
"""
import os, json, time, argparse
import optuna
from train_nn import Runner, load_split, DEFAULT_HP, MODEL_HP, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="ieee14"); ap.add_argument("--model", default="bus_graph_tcn")
ap.add_argument("--trials", type=int, default=12); ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--horizons", default="")   # e.g. 60,120: tune on the multi-horizon bus tokens
a = ap.parse_args()
HOR = [int(x) for x in a.horizons.split(",") if x]
SUF = "_tok" if HOR else ""
out = os.path.join(ROOT, "results", "tuning", a.system, f"nn_{a.model}{SUF}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
db = f"sqlite:///{os.path.join(ROOT, 'results', 'tuning', a.system, f'optuna_{a.model}{SUF}.db')}".replace("\\", "/")
Xtr, mtr = load_split(a.system, "hard", "train")
Xva, mva = load_split(a.system, "hard", "val")


def obj(t):
    hp = {**DEFAULT_HP, **MODEL_HP.get(a.model, {})}
    hp.update(lr=t.suggest_float("lr", 5e-4, 4e-3, log=True), hid=t.suggest_categorical("hid", [32, 48, 64, 96]),
              drop=t.suggest_float("drop", 0.0, 0.3), cw_pow=t.suggest_categorical("cw_pow", [0.5, 1.0]),
              wd=t.suggest_float("wd", 1e-5, 1e-3, log=True), epochs=a.epochs, patience=5)
    if HOR:
        hp["horizons"] = HOR
    if a.model == "bus_graph_tcn":
        hp["loc_w"] = t.suggest_categorical("loc_w", [0.0, 0.2, 0.5])
    r = Runner(a.system, a.model, hp, seed=0)
    r.fit(Xtr, mtr, Xva, mva, trial=t)
    return r.best_val


st = optuna.create_study(study_name=a.model, storage=db, load_if_exists=True, direction="maximize",
                         sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
t0 = time.time()
from optuna.trial import TrialState
left = a.trials - len([x for x in st.trials if x.state in (TrialState.COMPLETE, TrialState.PRUNED)])
if left > 0:
    while left > 0:
        st.optimize(obj, n_trials=1, catch=())
        left = a.trials - len([x for x in st.trials if x.state in (TrialState.COMPLETE, TrialState.PRUNED)])
json.dump(dict(best=st.best_params, value=st.best_value, tune_time_s=time.time() - t0,
               trials=[dict(p=x.params, v=x.value, state=str(x.state)) for x in st.trials if x.state != TrialState.FAIL]), open(out, "w"), indent=1)
print("TUNED", a.system, a.model, st.best_value, st.best_params, flush=True)
