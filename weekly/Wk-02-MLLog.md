# Week 02 ML Log — Preprocessing, RF Benchmark, Sequence & RL Scaffolding

## Objectives

1. **Synthetic dataset generation** — build a physically-coupled Aido Rover patrol dataset from a shared,
   stepable world core (not independent random channels), matching the platform's sensor/dynamics
   spec.
2. **Preprocessing pipeline** — replicate the CNC methodology (FFT → PCA → RF) on the new domain, on a
   split that prevents window/event leakage.
3. **Sequence tensor scaffolding** — build sliding-window `[N, window, channels]` tensors (raw + FFT view)
   for Phase B sequence models, on the same no-leakage split.
4. **MDP / offline-RL scaffolding** — formalise the patrol task as an MDP over the same world core with
   blockages enabled, implement a scripted baseline policy, and generate a transition dataset for Phase C.

## What Was Built

**World Core.** Stepable core (`RoverWorld.step`) shared by every downstream artefact: procedural map,
terrain-modulated torque/battery, LiDAR, optional blockage mechanic. Anomalies are slip/stuck mechanical
faults, statistically (not deterministically) coupled to terrain. Result: `shared_modules/rover_world.py`,
reused unchanged by the RL scaffold below.

**Dataset Generation.** 15,000 steps @10Hz, 9 channels, `MAP_SEED=6`, `HAZARD=0.05` auto-calibrated, blockages
off. Result: 15.1% anomaly (1,369 slip + 889 stuck), 718 injected missing values (0 after fill), 298 LiDAR
outliers, 0 torque outliers.

**Stratified Block Split.** 23 event-respecting blocks → `StratifiedGroupKFold` (7 folds) → val/test roles
chosen from pre-training per-fold structural stats rather than arbitrarily → purge 50 rows/block. Reason:
row-level (even stratified) splits leak given the 50-step FFT lookback and ~31-step fault durations —
full rationale in `reports/W02_Feature_Analysis_Report.md` §3. Result: train 9,734/val 2,215/test 1,901
(~16% anomaly each), 0/72 events split; canonical fold moved from 14th to 57th percentile of the 7-fold
score distribution.

**Preprocessing Pipeline.** GPS fed as deltas (not absolute position); FFT(50-step, 5 channels) + 9 raw + 6
cross-channel physical features (`inter_wheel_std`, `stall_ratio`) → 40-D matrix; PCA (95% target); RF
feature-selection as an interpretability side-analysis. Code shared via `shared_modules/features.py`.
Result: 19/40 PCA components (95.14% var); F1-criterion selects K=5 (0.8275 val F1) vs. accuracy's K=4
(0.7701) — deployed model still uses the full 19-component set (see RF Benchmark).

**RF Benchmark.** `StandardScaler → PCA(0.95) → RandomForestClassifier` as one `Pipeline` so every CV fold
refits scaler/PCA on its own rows only; grid search over `n_estimators × max_depth`, `StratifiedGroupKFold`
5-fold, `class_weight='balanced'`, `SEED=42`. Result: best `n_estimators=200, max_depth=10`; test F1 0.7350
default / 0.7359 val-tuned threshold, AUC 0.9668; 7-fold rotation F1 0.7544 ± 0.0596, AUC 0.9595 ± 0.0092;
latency 7.856 ms single-sample — **PASS** (≤100 ms gate). Full metrics table/confusion matrices in the
Results Summary below and `W02_RF_Benchmark.ipynb`.

**Sequence Tensor.** Windows {10, 20, 50} built on the same canonical split and purged rows, 11 channels
(9 raw + 2 physical, instantaneous only — sequence models aggregate over time themselves) + a windowed FFT
view. Result: saved to `data/rover_windows.npz`, same row set/anomaly rate as the classification split.

**MDP / Offline-RL Scaffolding.** MDP over the same world core (`blockages=True`), independent of the
classification split (episode-level, not block-level). 9-D state (no absolute GPS, same leakage reasoning as
above), 5 actions, reward table conditioned on `halted`/`main_blocked`/`rough_terrain` so context-dependent
correctness of an action is captured rather than a flat `(label, action)` table. Scripted priority-rule
policy + 5% ε-explore. 8 episodes (different blockage/fault seeds), chained to 48,000 rows. Full design
rationale and calibration: `reports/W02_RL_scaffold.md`; formal spec: `rl/mdp_schema.md`. Result: action
distribution continue 50.2% / raise-alert 18.7% / slow 16.1% / reroute 15.0% / return-to-base <0.1%; 96.4%
anomaly response rate; 1/8 episodes reached low battery.

## Key Decisions

- **Blocks, not rows, are the split unit** for every classification/sequence model this week and beyond —
  prevents FFT-window and fault-event leakage across train/val/test.
- **Val/test fold roles chosen structurally, not arbitrarily** — fixes a fold-selection blind spot in
  `StratifiedGroupKFold` (balances anomaly rate but not event count/duration).
- **GPS as delta, not absolute position** — avoids a fixed-map position shortcut; mirrors the MDP state design.
- **Cross-channel physical features (`inter_wheel_std`, `stall_ratio`) close a gap per-channel FFT can't
  express** — empirically the largest F1 gain of the week (ablation in `W02_Feature_Analysis_Report.md` §4).
- **Deployed RF uses the full 19-component PCA set**, not the minimal-subset finding — minimal-subset was an
  interpretability analysis, not a deployment choice.
- **Feature/window code centralized in `shared_modules/features.py`** — a prior duplication had already caused
  a silent divergence on 5/23 blocks.
- **CNC-vs-Rover:** CNC needed 2 features from 1 channel for 97.8% accuracy with no leakage structure to guard
  against; Rover needs 5 PCA components + purpose-built physical features on a harder, leakage-guarded,
  imbalanced split — reflects task difficulty, not a weaker pipeline.
- **MDP reward conditions on `halted`/`main_blocked`/`rough_terrain`** — a flat `(label, action)` table would
  let an agent collect "productive patrol" reward while stalled at a blockage.
- **World core reused as-is for RL** (`blockages=True`) — no dynamics divergence from the W2 classifier data.
- **8-episode / 48,000-row budget** — sized so full-block and low-battery termination paths are each hit a
  few times rather than never (dead-end blockage layouts occur in only ~4/10 random seeds).

## Results Summary

| Item                                               | Result                                                                                                                                                                                             |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classifier dataset                                 | 15,000 samples, 9 channels, 15.1% anomaly (1,369 slip + 889 stuck)                                                                                                                                 |
| Block split                                        | 23 blocks; train 9,734 (16.4%) / val 2,215 (16.0%) / test 1,901 (16.0%); 7.7% purged; 0/72 events split; fold roles structurally selected                                                          |
| Features                                           | 40-D: 9 raw + 25 FFT (5 channels × 5 spectral features, 50-step window) + 6 physical (inter-wheel std, stall ratio); GPS fed as deltas; shared across notebooks via`shared_modules/features.py` |
| PCA                                                | 40 → 19 components, 95.14% variance                                                                                                                                                               |
| RF-selected features (analysis only)               | 5 PCA components, F1-selection criterion (94.67% val acc. / 0.8275 val F1)                                                                                                                         |
| RF benchmark (test, deployed on all 19 components) | F1 (anomaly) 0.7350 default / 0.7359 tuned threshold, AUC-ROC 0.9668                                                                                                                               |
| RF 7-fold rotation                                 | F1 0.7544 ± 0.0596, AUC 0.9595 ± 0.0092 (canonical fold at 57th percentile)                                                                                                                      |
| RF latency                                         | 7.856 ms single-sample (full pipeline) — PASS (≤100 ms gate)                                                                                                                                     |
| Sequence tensor                                    | window ∈ {10,20,50}, 11 channels; train (9734,W,11) 16.4% / val (2215,W,11) 16.0% / test (1901,W,11) 16.0% + FFT view (N,25)                                                                      |
| MDP state / action space                           | 9-D state · 5 discrete actions                                                                                                                                                                    |
| RL transitions                                     | 48,000 rows, 8 episodes, 96.4% anomaly response rate, 1/8 episodes reach low battery                                                                                                               |

---

## Deliverables Completed

- `W02_Rover_World_Core.ipynb` + `W02_Rover_World_Core_Design.md` — shared world core, design & validation
- `W02_Preprocessing_Pipeline.ipynb` — dataset generation, cleaning, structural fold-role split, FFT/physical features, PCA, RF selection analysis
- `W02_Feature_Analysis_Report.md` — data quality / block split / PCA / RF / CNC-comparison report
- `W02_RF_Benchmark.ipynb` — Pipeline-based group-aware grid search, metrics, threshold tuning, 7-fold rotation, latency benchmark
- `W02_Sequence_and_RL_Scaffolding.ipynb` — 11-channel sliding-window tensors (3 window sizes) + FFT view + MDP transition rollout
- `shared_modules/features.py` — shared FFT/physical-feature/tabular-matrix functions (preprocessing, sequence, MLP notebooks)
- `synthetic_rover_data.csv`, `rover_stratified_block_split.csv`, `rover_windows.npz`, `rover_transitions.csv`
- `mdp_schema.md` — formal MDP specification
- `W02_RL_scaffold.md` — RL design rationale and calibration
