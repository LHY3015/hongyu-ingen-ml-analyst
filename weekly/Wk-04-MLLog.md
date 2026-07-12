# Week 04 ML Log — Trajectory Prediction (Linear / MLP / LSTM / Transformer seq2seq)

## What was done

- Synthetic Aido Humanoid motion dataset generated (`data/synthetic_humanoid_motion.csv`): 5,000 independent 1.5 s episodes at a 10 Hz planner tick — quadratic base path + speed-preserving obstacle-avoidance steering + ZMP gait sway (1–3 cm) + cm-level measurement noise, all parameters grounded in the Humanoid product doc (1.1 m/s walk limit enforced; min obstacle clearance 0.14 m; avoidance active in 38.1% of episodes). i.i.d. 70/15/15 split by sequence (episodes are independent — no group split needed).
- Four regressors + CV-extrapolation floor trained under the W03 protocol (Adam 1e-3, batch 128, early stopping, patience 10), 5 seeds each for the neural models; targets = next 5 CoM waypoints relative to the last estimated pose.
- Per-horizon Euclidean error, avoidance-vs-clean split, attention-over-horizon extraction (5 learned horizon queries), latency vs the plan's 50 ms motion-planning gate (plan-defined; product docs give 10 ms MPC / 150 ms grasp — flagged).
- Mid-point review deck built (`W04_Mid_Review_Deck.pptx`): headline = latency-vs-F1 across all six rover models + trajectory-task slide.

## Results

| Model            | Test mean Euclid (cm)  | h5 (cm)        | Latency (ms) | ≤50 ms |
| ---------------- | ---------------------- | -------------- | ------------ | ------- |
| CV extrapolation | 2.77                   | 4.28           | 0.003        | PASS    |
| Linear           | 1.50                   | 2.56           | 0.087        | PASS    |
| MLP              | 1.63 ± 0.06           | 2.43           | 0.033        | PASS    |
| **LSTM**   | **1.44 ± 0.04** | **2.18** | 0.338        | PASS    |
| Transformer      | 1.60 ± 0.15           | 2.32           | 0.513        | PASS    |

- All learned models beat CV; the task is mostly linear — Linear wins waypoints 1–2 (optimal filtering of sway/noise), the LSTM/Transformer win waypoints 4–5 (anticipating steering). The nonlinear gain concentrates in avoidance episodes (LSTM h5 2.86 vs Linear 3.52 cm); on clean episodes Linear ≈ LSTM.
- Transformer accuracy does not justify its latency here: slower *and* less accurate than the LSTM, with 3× the seed variance (84.7K params vs 3,500 short sequences — attention under-determined at this scale). Transformer-vs-MLP/Linear gaps not significant.
- Param-matched control (Transformer-S, d_model=32, 21.9K params; notebook §5.3): shrinking **halves the seed std** (0.15→0.07 cm) and nudges the mean to 1.52 cm but still trails the LSTM — the instability was over-parameterization, not mis-tuning. Mirror image of the W03 control, where the *same* d=32 shrink *cost* 0.07 F1: capacity paid off on 50-step windows, wasted on 10-step sequences. (Length-vs-payoff is a consistent 2-task pattern, not a proven mechanism — task type and data size also differ.)
- MLP regularisation check (only model with a train/val gap): the gap is closable (wd 1e-4 → 0.47→0.13 ×1e-4; wd 1e-3 → zero) but every config that closes it worsens val and test error (1.63 → 1.79 / 1.87 / 3.16 cm) — benign capacity under early stopping, baseline kept.
- **Attention over horizon vs planner state use:** the attention centre-of-mass moves monotonically earlier in the history as the predicted waypoint moves further out (h1 ≈ step 6.7 of 0–9, h5 ≈ step 3.1) — near waypoints read the current state, far waypoints read the trend/manoeuvre context. The thesis planner weights information the same way: the committed near segment is pinned to the current state, while the far tail's shape is set by corridor context from older map information — and that uncertain far tail is exactly what FASTER's whole/safe dual-trajectory design hedges. Full quality/compute mapping (replan-cycle decomposition, P_max horizon truncation, δt=1.25×replan coupling): W04_Trajectory_Report §7.

## Deliverables Completed

- `W04_Trajectory_Prediction.ipynb` — data generation + 4 regressors + CV floor; per-horizon errors; attention-over-horizon; latency vs 50 ms
- `W04_Trajectory_Report.md` — horizon-error analysis, learning-curve fit diagnosis, avoidance split, Transformer-vs-LSTM tradeoff, thesis-connection mapping (§7)
- `data/synthetic_humanoid_motion.csv`
- `W04_Mid_Review_Deck.pptx`
