# PG-TCN: Posterior-Drift Graph Temporal Networks for FDIA Detection in Partially Observable Grids

Source code for **PG-TCN**, a detector that identifies benign events and filter-absorbed false data injection attacks
(FDIAs) on the reduced-order dynamic state estimator of a partially observable power grid with correlated wind
generation, together with the complete experimental pipeline (simulation, benchmarks, statistics).

Author: Mohamed Massaoudi.

## Method in brief

1. **Reduced-order estimator (front end).** A coupled cubature Kalman filter on the known region returns whitened
   innovations, their covariance and the posterior state (`src/sim.py`).
2. **Evidence channels** (`src/features.py`): standardised innovation, posterior measurement velocity
   `u_k = (h(x_k|k) - h(x_k-1|k-1)) / sigma_v`, which exposes the drift absorbed by the filter, its multi-horizon
   accumulation over 60 and 120 scans, a wind correlation-consistency residual and the normalised innovation metric.
3. **Measurement-to-bus typed tokens.** Mean and mean-absolute aggregates per bus and measurement type give 44 features
   per bus, independent of the grid size, so one network can be applied to any grid.
4. **Spatio-temporal backbone** (`BusGraphTCN` in `src/models.py`): four causal dilated temporal convolution blocks
   (dilations 1, 2, 4, 8) interleaved with graph propagation over the known topology, attention plus max pooling,
   a five-class head (normal, wind event, load event, FDIA step, FDIA ramp) and a bus-level localisation head.
5. **Calibrated-likelihood-ratio CUSUM** (`src/metrics.py`): the attack log-odds are Platt-calibrated on validation data,
   the prior log-odds are removed, and a CUSUM with offset and threshold selected on validation episodes and
   validation-period hour-long streams bounds the episode false-alarm rate (1%) and the alarms per day (1).

The frozen configuration is in `FINAL_MODEL_CONFIG.yaml`.

## Repository layout

```
src/                          method and experiment pipeline
  grid.py                     IEEE cases, batched Newton-Raphson AC power flow, reduced-order region model
  profiles.py                 real wind (GEFCom2014-W) and load (ISO-NE, PJM) profiles, temporal split
  sim.py                      episodes, benign-event library, AC-consistent FDIAs, coupled CKF
  gen_data.py, prep_features.py
  features.py                 evidence channels and bus tokens
  models.py                   PG-TCN and all deep / graph benchmarks
  train_nn.py, tune_nn.py     training, evaluation, Optuna tuning of neural models
  tabular.py                  tabular benchmarks (window statistics or bus tokens)
  classical.py                source DQN detector and innovation-norm sequential tests
  metrics.py, stats.py        metrics, calibrated-LLR CUSUM, paired statistics
  redecide.py                 validation-only calibration of the decision rule for saved models
  eval_rev.py                 held-out sets and hour-long streams
  loc_rev.py, transfer_rev.py localisation comparison, cross-grid transfer
  rev_stats.py, paired_delay.py, efficiency.py, analyze_lib.py   statistics and analysis
  launch.py, make_jobs_rev.py resumable job lists and launcher
SOURCE_METHOD_REPRODUCTION/   re-implementation of the innovation-norm DQN detector (Ran et al., IEEE TSG 2026)
FINAL_MODEL_CONFIG.yaml       frozen PG-TCN configuration
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt        # Python 3.13, PyTorch with CUDA recommended
```

Public data (not redistributed here), placed in one folder:

- GEFCom2014 data set (wind track, Task 15 and its solution file): `GEFCom2014_Dataset/GEFCom2014 Data/GEFCom2014-W_V2/Wind/...`
- ISO New England hourly system load: `isone_load.csv` (columns `date;hour;load`)
- PJM hourly load: `PJM_Load_hourly.csv` (columns `Datetime,PJM_Load_MW`)

Point the code to that folder:

```bash
export PGTCN_DATA=/path/to/data          # Windows: setx PGTCN_DATA "C:\path\to\data"
```

IEEE 14-, 30-, 57- and 118-bus case data come from PYPOWER and pandapower.

## Reproduction

All commands run from `src/`. Every step is resumable: finished units are skipped on re-run.

```bash
# 1. Data: source setting, benign-event library, held-out sets, hour-long streams
for s in ieee14 ieee30 ieee57 ieee118; do
  python gen_data.py --system $s --what src,hard,ood,rev
  python prep_features.py $s
done

# 2. Re-implementation and diagnosis of the innovation-norm DQN detector
cd ../SOURCE_METHOD_REPRODUCTION
python check_ckf_equivalence.py
for s in ieee14 ieee30 ieee57 ieee118; do python run_reproduction.py $s; done
python run_reproduction.py fixed
python sensitivity.py ieee14
cd ../src

# 3. Hyper-parameter tuning on IEEE-14 validation data (12 trials per neural model)
python tune_nn.py --system ieee14 --model bus_graph_tcn --trials 12
for m in flat_tcn gru lstm transformer mlp cnn1d stdcnn mamba; do python tune_nn.py --system ieee14 --model $m --trials 12; done
python tune_nn.py --system ieee14 --model gcn_gru --trials 12 --horizons 60,120

# 4. Training: PG-TCN (10 seeds), ablations, deep/graph, tabular and classical benchmarks
python make_jobs_rev.py 60,120          # writes ../jobs_rev_{final,ablation,deep,tab,classical}.txt
for g in final ablation deep tab classical; do python launch.py ../jobs_rev_$g.txt 2; done

# 5. Validation-only decision rule, held-out evaluation, localisation, transfer
python redecide.py lib
for s in ieee14 ieee30 ieee57 ieee118; do
  for k in 0 1 2 3 4 5 6 7 8 9; do python eval_rev.py --system $s --model bus_graph_tcn --tag final --seed $k; done
  python loc_rev.py $s
done
python transfer_rev.py ieee14 ieee30 0   # likewise for other source/target pairs and seeds

# 6. Statistics and analysis
python rev_stats.py lib
python paired_delay.py
python efficiency.py
python analyze_lib.py
```

Benchmarks are evaluated on the held-out sets with
`python eval_rev.py --system <s> --model <model> --tag <default|matched> --seed <k>`.

## License

See `LICENSE`.
