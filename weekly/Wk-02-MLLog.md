# Week 02 ML Log — Preprocessing, RF Benchmark, Sequence & RL Scaffolding

## Objectives

1. **Synthetic dataset generation** — build a physically-coupled Aido Rover patrol dataset from a shared,
   stepable world core (not independent random channels), matching the platform's sensor/dynamics
   spec.
2. **Preprocessing pipeline** — replicate the CNC methodology (FFT → PCA → RF) on the new domain.
3. **Sequence tensor scaffolding** — build a sliding-window `[N, window, channels]` tensor (raw + FFT view)
   for Phase B sequence models, with no look-back leakage across splits.
4. **MDP / offline-RL scaffolding** — formalise the patrol task as an MDP over the same world core with
   blockages enabled, implement a scripted baseline policy, and generate a transition dataset for Phase C.

## What Was Built

### World Core

A stepable core (`RoverWorld.step(action) → 10 channels + label + info`) drives every downstream
artefact: a 2D procedurally-generated map (boustrophedon main loop + rejoining branch detours),
terrain-modulated torque/battery, LiDAR ray-casting, and an optional blockage mechanic (off for the
classifier dataset, on for RL). Anomalies are a mechanical self-fault (slip: one wheel spikes while others
shed load; stuck: all four wheels rise, displacement collapses) with severity `s ~ Beta(2,5)`, statistically
— not deterministically — coupled to terrain, so torque alone cannot trivially separate fault from rough
ground.

### Dataset Generation

15,000 timesteps at 10 Hz (25 min), 9 sensor channels (gps_lat, gps_lon, lidar_distance, battery_soc,
torque_0–3, ambient_temp), driven by the world core on `MAP_SEED=6`, `HAZARD=0.05` (auto-calibrated to land
the anomaly rate in the 14–16% band), blockages off, `continue` policy throughout. Realised:  15.1% anomaly
(2,258 samples: 1,369 slip + 889 stuck), 0.5% random missingness injected (718 missing values before
cleaning, 0 after forward/backward fill). 298 LiDAR 3σ outliers (the 200 m no-return dropout spikes); 0 torque
outliers — terrain already spreads torque over a wide range, so a fault's added deviation is usually absorbed
inside that natural spread rather than crossing 3σ.

### Preprocessing pipeline

FFT (50 step window) on torque_0–3 + lidar_distance → 25 spectral features (dominant frequency,
centroid, bandwidth, total power, peak-to-mean per channel), plus the 9 raw channels = 34-D feature matrix.
PCA (95% variance target) retained 17 of 34 components, 96.04% variance; PC1 alone carries 34.04%, loading
on torque spectral centroid and peak-to-mean — the causal fault model routes both slip and stuck through one
shared driver (terrain-modulated torque), concentrating variance more than independently-randomised burst
parameters would. RF feature selection (100 trees) over the 17 PCA components found the top-4
(PC2, PC1, PC3, PC4) sufficient — 94.25% val accuracy vs. 96.21% full-set baseline (1.96 pp delta, within the
2 pp threshold).

### RF Benchmark

Grid search `n_estimators ∈ {50,100,200} × max_depth ∈ {None,5,10,20}`, 5-fold CV, F1 scoring,
`class_weight='balanced'`, `random_state=SEED=42`. Split: stratified 70/15/15 (train 10,465 / val 2,242 /
test 2,243, each 15.1% anomaly).

Best: `n_estimators=200, max_depth=None`, CV F1 = 0.7582, train time 8.31 s.

| Split | Precision (anomaly) | Recall (anomaly) | F1 (anomaly) | AUC-ROC |
| ----- | ------------------- | ---------------- | ------------ | ------- |
| Train | 0.9838              | 1.0000           | 0.9918       | 1.0000  |
| Val   | 0.7167              | 0.8757           | 0.7883       | 0.9707  |
| Test  | 0.7261              | 0.8525           | 0.7843       | 0.9742  |

Test confusion matrix: TN=1795, FP=109, FN=50, TP=289 (N=2,243).

Latency: single-sample 7.642 ms, 1,000-sample batch 19.75 ms (0.0197 ms/sample). Constraint: Aido Rover
general gate ≤100 ms @10 Hz → **PASS**.

Train F1 (0.99) far exceeding val/test F1 (0.78–0.79) reflects genuine task difficulty carried over from the
causal, terrain-confounded fault model — not a data-quality problem, and consistent with the Feature Analysis
Report's finding that this task needs more PCA components than the CNC precedent (4 vs. 2) to reach a
comparable accuracy band.

### Sequence Tensor

Sequential 70/15/15 split on the cleaned trace, `WINDOW=50` (matches the FFT window),
`STRIDE=1`: train (10,450, 50, 9) 13.7% anomaly, val (2,200, 50, 9) 20.6%, test (2,200, 50, 9) 17.0%. Anomaly
rate varies by split because splits are time-ordered, not stratified — expected for a sequential evaluation
setup. A parallel windowed FFT-feature view (`Xfft_{train,val,test}`, shape `(N, 25)`) was added,
reusing the same `fft_features` extractor as the preprocessing pipeline and the same split indices, so Phase
B can train on raw, FFT, or both without re-deriving splits. Saved to `data/rover_windows.npz`.

### MDP / Offline-RL Scaffolding

Formalised as an MDP over the same world core (`blockages=True`) — the same core the W05 Gymnasium env will wrap, so offline data and the online env share identical dynamics. Full design rationale, calibration evidence, and every threshold's justification: `reports/RL_scaffold.md`. Formal spec: `rl/mdp_schema.md`.

- **State (9-D):** `torque_mean/max/std`, `lidar_mean` (m), `soc_slope`, `battery_soc`, `route_progress`,
  `next_main_block_dist`, `branch_block_dist`. No absolute GPS (position leakage — fixed terrain means faults
  recur at fixed coordinates).
- **Actions (5):** continue, slow (real fault-mitigating effect via `speed_factor`), reroute, raise-alert,
  return-to-base (ends the episode).
- **Reward:** a `(label, action)` table conditioned on `halted`, `main_blocked`, and `rough_terrain` (window
  `torque_mean > 30 Nm`, isolating wet-grass/mud) — see Key Decisions below.
- **Scripted policy:** a priority-ordered rule table (`POLICY_RULES`) — abort on low battery or an unresolved
  full block; alert on ground-truth fault or a full block; reroute around a blocked main route; slow on
  high-slip terrain; continue otherwise — plus 5% ε-explore over the four non-terminal actions.
- **Generation:** multiple episodes (different blockage/fault seeds, same `MAP_SEED=6`), each capped at one
  full 960 m loop (9,600 steps), chained to 48,000 rows. Episode 0 starts at 100% battery; subsequent episodes
  draw `Uniform(40,100)%` (fleet-dispatch realism).

Execution (`ROUGH_TERRAIN_TORQUE=30` Nm): 48,000 rows across 8 episodes. Action distribution: continue
50.2%, raise-alert 18.7%, slow 16.1%, reroute 15.0%, return-to-base <0.1% (5 rows). Event-level onsets (a
truer read of how often each action is *chosen*, not how long it's sustained): continue 1,303, raise-alert
1,034, slow 929, reroute 836, return-to-base 5. Anomaly response rate (alert or abort on a true fault): 8,240/8,549 = 96.4%. 1 of 8
episodes genuinely reached `battery_soc < 20%`.

## Key Decisions

**CNC-vs-Rover sensor difference.** The CNC wear-detection pipeline needed only 2 features (both from a
single Z-axis positional channel) to reach 97.81% test accuracy — a clean, single-axis wear signature. The
Rover anomaly signature needs 4 PCA components (94.25% val accuracy) spanning both torque spectral shape and LiDAR spectral shape, because a mechanical self-fault here is a distributed, multi-wheel dynamic event (slip: one wheel spikes, others shed load; stuck: all four rise together) confounded by terrain that also drives normal torque up — a single-channel positional signature has no analogue in this task. The RF benchmark's train-vs-test F1 gap (0.99 vs. 0.78) is the direct downstream consequence of that added difficulty.

**MDP reward-design choice: conditioning on `halted`/`main_blocked`/`rough_terrain`, not just `(label, action)`.** A flat `(label, action)` reward table cannot express that the *same* nominal action is correct in
one context and wrong in another once blockages and terrain risk are in play: `continue` while genuinely
patrolling deserves `+1.0`, but `continue` while parked at an unresolved blockage does not (`−0.5`); `reroute`
around a real obstruction deserves `0.0` (not penalised), but a gratuitous reroute on a clear route is `−0.3`;
`slow` on high-slip terrain (wet-grass/mud, `torque_mean > 30 Nm`) deserves `0.0`, but `slow` on asphalt for no
reason is `−0.1`. Without these three conditioning terms, an RL agent trained on this data could learn to sit
motionless at a blockage and still collect the "productive patrol" reward every step — the conditioning closes
that gap while keeping the reward a simple, auditable table.

**World core reused as-is for RL.** The core's `world.step` / `env.step` split meant the RL scaffolding work only had to use the existing
core with `blockages=True` — the same dataset-generation guarantee (byte-identical W2 output when blockages are off) holds regardless of what W05 does with the reward.

**Multi-episode generation.** Each episode's blockage layout is one random draw;
a genuinely dead-end "both branches blocked" configuration occurs in only ~4/10 independently-drawn seeds.
48,000 rows (~5 full-loop episodes) was sized so the dataset has a realistic chance of exercising the
full-block and low-battery termination paths at least a few times, which a single long rollout or an
under-sized budget does not reliably do.

## Results Summary

| Item                     | Result                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Classifier dataset       | 15,000 samples, 9 channels, 15.1% anomaly (1,369 slip + 889 stuck)                                          |
| FFT features             | 25 (5 channels × 5 spectral features, 50-step window)                                                      |
| PCA                      | 34 → 17 components, 96.04% variance                                                                        |
| RF-selected features     | 4 PCA components (94.25% val acc., within 2 pp of 96.21% full-set)                                          |
| RF benchmark (test)      | F1 (anomaly) 0.7843, AUC-ROC 0.9742, confusion TN=1795/FP=109/FN=50/TP=289                                  |
| RF latency               | 7.642 ms single-sample — PASS (≤100 ms gate)                                                             |
| Sequence tensor          | X_train (10450,50,9) 13.7% anomaly · X_val (2200,50,9) 20.6% · X_test (2200,50,9) 17.0% + FFT view (N,25) |
| MDP state / action space | 9-D state · 5 discrete actions                                                                             |
| RL transitions           | 48,000 rows, 8 episodes, 96.4% anomaly response rate, 1/8 episodes reach low battery                        |

---

## Deliverables Completed

- `W02_Rover_World_Core.ipynb` + `W02_Rover_World_Core_Design.md` — shared world core, design & validation
- `W02_Preprocessing_Pipeline.ipynb` — dataset generation, cleaning, FFT, PCA, RF selection
- `W02_Feature_Analysis_Report.md` —  data quality / PCA / RF / CNC-comparison report
- `W02_RF_Benchmark.ipynb` — grid search, metrics, latency benchmark
- `W02_Sequence_and_RL_Scaffolding.ipynb` — sliding-window tensor + FFT view + MDP transition rollout
- `synthetic_rover_data.csv`, `rover_windows.npz`, `rover_transitions.csv`
- `mdp_schema.md` — formal MDP specification
- `RL_scaffold.md` — RL design rationale and calibration
