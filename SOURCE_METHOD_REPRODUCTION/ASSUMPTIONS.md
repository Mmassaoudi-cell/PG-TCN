# SOURCE_METHOD_REPRODUCTION — implementation assumptions and deviations

Code: `src/grid.py` (IEEE cases, batched AC power flow, reduced-order region model), `src/sim.py`
(profiles, events, attack, coupled CKF), `src/gen_data.py` (datasets), `SOURCE_METHOD_REPRODUCTION/dqn_detector.py`
(DQN detector), `SOURCE_METHOD_REPRODUCTION/run_reproduction.py` (Tables I–IV), `check_ckf_equivalence.py`.

## Faithfully reproduced (as specified in the source paper, SP)
* IEEE 14/30/57/118 test systems, SP attack regions, boundary buses and wind-farm buses; 100 MW wind farms.
* Reduced-order measurement model with boundary bus as local angle reference (SP Eqs. 13–15).
* Correlation-aware attack: AC-consistent a = h(x̂+Δx) − h(x̂) on internal-bus states, correlated wind-bus
  injection target, operating-limit feasibility (SP Eqs. 5–7, Steps 1–4).
* VAR(1) Yule–Walker wind model → Ŝ; coupled R^{rd,corr}; Cholesky whitening (SP Eqs. 8–20).
* Cubature Kalman filter with 2N points (SP Eqs. 21–31); detection metric η (SP Eq. 33).
* POMDP cost structure and DQN with target network + experience replay (SP Eqs. 34–38, Algorithms 1–2).
* ε = 0.1, T = 200, 10,000 training trials, 10,000 test trials; FAR/timely/delay/precision/recall/F-score.
* c_β chosen by the SP Pareto rule (FAR < 5%, delay < 5 s) on a *validation* set (the SP has none; we add it).

## Assumptions (information missing in the SP)
| ID | Item | Choice | Rationale |
|---|---|---|---|
| A1 | Wind series | Real GEFCom2014-W zones 1–4 (normalised, hourly, 2012–2013) set each episode's level; intra-episode VAR(1) fluctuation (a = 0.95, σ = 0.006 p.u.) with spatial correlation = TRAIN level correlation. | Real correlated wind; temporal split possible. |
| A2 | Load | IEEE base loads × ISO-NE hourly level mapped to [0.7,1.3] (TRAIN 1–99% quantiles) × AR(1) 0.2% noise; PV-generator dispatch scaled with load. | SP robustness grid uses 0.7–1.3×. |
| A3 | Sampling interval | 1 s | Table IV: delay 0.0012 s ⇔ 0.12% one-step delays. |
| A4 | Measurements | V at all region buses; P,Q injections at internal buses; P,Q flows at both ends of region lines (IEEE-14: 49). | Fig. 7 shows ≈54 measurements. |
| A5 | Noise | σ_w² = 2e-4 (process, random walk), σ_v² = 1e-4 (measurement). | Text order ambiguous. |
| A6 | Transition f(·) | identity (random walk) | Not stated. |
| A7 | Attack vector | Δθ_int ~ N(0,κ²), ΔV_int ~ N(0,(0.2κ)²), projected so wind-bus ΔP = d ~ N(0, (4κ)² C_w); κ ~ logU[0.01,0.1] (source scenario). | Calibrated so amplitude statistics match SP Figs. 7–8 (median 0.026 p.u., q99 0.16 p.u.). |
| A8 | Attacker state knowledge | x̂ = x + N(0, 0.003²) | ≈ local WLS accuracy. |
| A9 | Limits | P/Q injection within [0.5,1.5]×current ±0.1 p.u.; |flow| ≤ 1.5·max(|current|,0.2); V ∈ [0.8,1.1] and |Δθ_line| ≤ 15° never worsened; enforced by bisection on the attack scale. | Case-file-anchored limits made nearly all attacks infeasible at wind-heavy operating points. |
| A10 | Onset | τ ~ U{60,…,140} | Not stated. |
| A11 | DQN | MLP 1→64→64→2 (ReLU), Adam 1e-3, γ = 0.99, replay 200k, batch 1024, target sync 500 updates, input log(η/η_ref) with η_ref = TRAIN pre-attack geometric mean. | Not stated. |
| A12 | c_β grid | {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} | SP states c_β ∈ [0,1]. |
| A13 | Recall/F-score | Trial level. | Not defined in SP. |
| A14 | Boundary injections | Excluded from the defender's reduced measurement set (contain unknown external flows). | SP estimates them via Eq. 4 (path-based angle differences); unspecified details. |
| A15 | Wind reactive power | Farms absorb reactive power, PF ∈ [0.9,1.0]. | Injecting Q pushed 68–76% of IEEE-14/30 operating points above 1.1 p.u. |
| A16 | IEEE-57 boundary | Buses 45, 46 added to the stated boundary set (they have external ties). | Topology consistency. |
| A17 | IEEE-30 | True IEEE-30 (pandapower `case_ieee30`, 283.4 MW), not PYPOWER's Alsac–Stott `case30`. | Faithfulness. |
| A18 | Generator Q limits | Ignored in the power flow. | Standard for PF-driven data generation. |

## Deviations (needed to make the method run; each reported with and without)
* **D1 — learning rate.** Literal α = 0.1 as the SGD learning rate collapses to "never stop" (0% detection on
  IEEE-14 validation). Adam 1e-3 is used.
* **D2 — counterfactual replay.** With literal ε-greedy exploration (ε = 0.1), random *stop* actions terminate
  ≈95% of training episodes before the attack onset (0.95^60), so post-attack states are rarely replayed:
  timely rate 61–78% on IEEE-14 validation. Because observations are exogenous (η does not depend on the
  action), we store the outcomes of both actions at every step (same MDP, exhaustive exploration). This is the
  most favourable configuration for the SP detector and is used for all reported numbers.
* **D3 — Joseph-form covariance update.** SP Eq. 31 is evaluated in the algebraically identical Joseph form to
  keep P positive definite in float64 on long runs.
* **D4 — whitened noise covariance.** SP Eq. 28 adds R^{rd,corr} after whitening by Ψ⁻¹; since Ψ⁻¹RΨ⁻ᵀ = I, we
  add I (mathematically consistent version).
