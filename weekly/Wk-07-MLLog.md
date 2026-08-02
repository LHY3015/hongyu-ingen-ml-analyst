# Week 07 ML Log — Cross-Family Explainability, Pareto and One Optimisation

## What was done

- **Week-6 review batch landed first** (four commits): the MARL wrapper gained termination parity
  with the flat env and was retrained — team returns moved from −2,522/−2,386 to +3,819/+3,840,
  the old negatives being a stuck-rover artifact; the 1M option arms were extended to five seeds
  and all four factorial effects now carry paired Wilcoxon significance (`p = 0.002–0.037`); the
  evaluation module gained a per-step trace API (`rollout(record=True)`, `load_policy_net`) that
  this week's saliency work runs on.
- **Deterministic refits with saved checkpoints** (`shared_modules/refit.py`, `saved_models/`).
  No Week 2–4 notebook had persisted a model, so all thirteen were refit from their documented
  configurations and verified against the published numbers before any attribution: twelve
  reproduce every digit (threshold, F1, AUC, confusion matrix, epoch counts), with round-trip
  load checks exact over the full test set.
- **SHAP across three tasks** (`W07_Explainability.ipynb`): TreeExplainer on the Rover RF in its
  PCA space and on Fari/Senpai, DeepExplainer on the Rover and Fari MLPs, with the Week-2 MDI
  selection as the comparison target and the Fari generative weights as ground truth.
- **RL value and saliency**: critic value and per-channel input gradients along a canonical
  episode for PPO (BC-init) against the BC prior it was tuned from, event-conditioned on the same
  masks the Week-6 tables count on.
- **Senpai learner-state task built** (2,000 × 5, three classes): fixed-difficulty IRT-3PL
  assessment scenario with an in-notebook data card separating documented anchors from invented
  distributions, a θ-oracle ceiling, RF + multi-class SHAP under the full reporting standard.
- **Cross-family Pareto + optimisation** (`W07_Pareto_and_Optimisation.ipynb`): one latency
  convention re-measured from checkpoints, three per-platform panels, the full 15-pair
  significance matrix deferred from Week 3, and the RF depth/width reduction.
- Reports: `W07_Explainability_Pareto_Report.md` (the findings report — every figure and the full
  argument live there; the notebooks produce code, data and images), `W07_Methodology_Report.md`,
  `W07_PIC20_ML_Analysis.md`.

## Results

### The refit that would not reproduce is a finding

The 1D-CNN's training is nondeterministic at the hardware level: cuDNN's convolution backward
accumulates gradients atomically, and eight runs of the unchanged code span test F1 0.683–0.738
(published 0.7317 ≈ 75th percentile) while AUC holds at 0.9585 ± 0.0010. The module pins the
deterministic kernel and records F1 0.6977 with the spread documented. Same moral as the Week-3
threshold-swing caveat: threshold-dependent F1 carries noise that AUC does not.

### SHAP agrees with Week 2, and the theme is cross-channel

The RF's SHAP top-5 components are the Week-2 MDI selection — same set, same order, Spearman 0.919
over all 19 components — so the selection survives a change of attribution method. Resolving the
three previously undocumented components (PC7, PC12, PC17) shows all three dominated by
`inter_wheel_std` terms; the MLP, explained in its native 40-D space, puts `inter_wheel_std` first
as well. Two model families, different spaces, same primary evidence: the engineered cross-channel
features. On Fari, where the generative weights are known, both models recover the true importance
ordering exactly (ρ = 1.0, exact-permutation `p = 0.017`) — the attribution pipeline passes the one
test with a ground truth.

### The most surprising finding: the misses are not localisation failures

Joining two Week-3 results that had never been joined — attention lands on fault spans (5.3×
selectivity) and stuck-type recall is only 39.5 % — the stuck-type test windows split into two
independent fault episodes: one 90 % detected, one missed entirely. In the fully missed episode,
every window's attention selectivity is above 1 (median 8.7×): the encoder finds the fault span
and the classification head discards it. The n = 2 episode count is printed with the figure and
kills any detected-vs-missed ranking, but the one-sided claim stands, and it moves the stuck-type
recall ceiling from the encoder to the decision head.

### RL fine-tuning is feature reallocation

BC keeps a standing dual sensitivity (`torque_max` and `next_main_block_dist` comparable in every
condition); PPO concentrates 42–46 % of its gradient on `torque_max` and nearly drops the blockage
distance except when one is in sensor range. The raise-alert logit is a `torque_max` detector
(43 % of its gradient — the world's fault signature is a torque surge, 21.5 → 44.1 Nm); the
reroute logit has no feature at all (top channel 16 %, uniform is 11 %). That is the
attribution-level form of the Week-6 behaviour trade (reroute 0.84 → 0.34, false alarms 55 → 241),
and the value trace shows the critic pricing a dead end at −57 while the actor continues into the
stuck timeout.

### Senpai, Pareto, reduction

Senpai RF reaches macro-F1 0.667 against a θ-oracle ceiling of 0.764 (~87 % of the extractable
information), 8.7 ms per sample, PASS against the 100 ms gate; `hint_requests` attributes twice as
strongly for the beginner class as for intermediate — struggle signals find who needs help. The
Pareto panels: Rover classification frontier {MLP, Transformer} with the full 15-pair matrix
showing Transformer > {RF, CNN, LSTM, GRU} but Transformer ≈ MLP (`p = 0.34`) — MLP stays the
deployment recommendation; Humanoid frontier {CV, Linear, MLP, LSTM} with the Transformer
dominated by LSTM; patrol frontier {REINFORCE+baseline, PPO} with DQN below the deployable floor
and the Week-6 layout caveat attached. The RF reduction lands at 50 trees × depth 6: F1
0.7359 → 0.7153 (−0.021, inside the ±0.048 fold spread), 9.07 → 2.52 ms, SHAP top-5 unchanged —
from 91 % of the ≤10 ms breach-detection tier to 25 %. Calibration against the STUM target: the
MLP's on-distribution ECE is 0.025 — under the deployed 0.031 benchmark with no calibration layer —
and the Transformer's 0.041 drops to 0.032 with validation-fitted temperature scaling; the open
calibration problem is cross-block transfer (the threshold swings), not marginal reliability.

## Deliverables Completed

- `shared_modules/refit.py` + `saved_models/` (13 verified checkpoints + manifest)
- `notebooks/W07_Explainability.ipynb` — refits, SHAP × 3 tasks, attention-on-misses, RL
  value/saliency, learning curves, Senpai task + data card
- `notebooks/W07_Pareto_and_Optimisation.ipynb` — latency convention, three panels, significance
  matrix, RF reduction; `data/pareto_points.csv`
- `data/senpai_learner_state.csv`
- `reports/W07_Explainability_Pareto_Report.md`, `reports/W07_Methodology_Report.md`,
  `reports/W07_PIC20_ML_Analysis.md`
- Week-6 review corrections across `rl/`, `shared_modules/`, the W06 notebooks, report and log
