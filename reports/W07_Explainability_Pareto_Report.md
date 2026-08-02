Hongyu LIU
InGen Dynamics - ML & NN Analyst Intern, August 2026

---

**Platform:** all platforms — explainability & deployment · full PIC 2.0
**Protocol:** every attribution runs on a refit model verified against its published metrics; canonical block split for Rover, 70/15/15 stratified for Fari/Senpai, ten-world Week-6 protocol for RL
**Deployment gates:** Aido Rover ≤100 ms (≤10 ms breach tier in §9) · Aido Humanoid ≤50 ms · Fari ≤35 ms · Senpai ≤100 ms

## 1. Overview

Week 7 asks *why* the Phase A–C models decide what they decide — SHAP for the classical model and
the MLP on three tasks, attention for the transformer, value and input-saliency for the RL policies
— then places every family on a latency–score Pareto and runs one optimisation experiment. The
classical family is RandomForest alone (one model, not the plan's five), and every "classical"
claim below is scoped to it. Five results carry the week.

| finding                                                        | evidence                                                                                     |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| One feature story spans every model family                     | cross-channel `inter_wheel_std` leads RF components, MLP 40-D SHAP, and (as `torque_max`) the RL alert head (§4–5, §8) |
| The stuck-type misses are not localisation failures            | in a fully missed fault episode, 100 % of windows have attention on the fault span, median 8.7× (§7) |
| RL fine-tuning is feature reallocation                         | PPO concentrates 42–46 % of gradient on `torque_max` and drops the blockage distance the BC prior watched (§8) |
| The frontier confirms the Week-3 defaults                      | classification frontier {MLP, Transformer}, Transformer ≈ MLP (`p = 0.34`); trajectory {…, LSTM}; patrol {REINFORCE+baseline, PPO} (§9) |
| A model that cannot be refit bit-identically is a finding      | the 1D-CNN's F1 spans 0.683–0.738 across identical runs from cuDNN nondeterminism alone (§2) |

## 2. Refits — the Precondition, and What Refused to Reproduce

No Week 2–4 notebook persisted a fitted model, so every explainability target was refit from its
documented configuration and verified against the ledger before any attribution was computed
(`shared_modules/refit.py`; checkpoints and a verification manifest under `saved_models/`). Twelve
of thirteen models reproduce every published digit — threshold, F1, AUC, confusion matrix, epochs
run, best epoch — and reload bit-identically over the full test set.

The exception is the 1D-CNN, and the cause is hardware rather than the port: cuDNN's convolution
backward accumulates gradients atomically, and eight runs of the unchanged training code span test
F1 **0.683–0.738** (the published 0.7317 sits near the 75th percentile) while AUC holds at
0.9585 ± 0.0010. The refit pins the deterministic kernel — which leaves the other seven torch
models bit-identical — and records the pinned draw (F1 0.6977) with the spread documented. The
lesson recurs throughout this report: threshold-dependent F1 carries run noise that AUC does not.

## 3. Rover SHAP — the Classical Model Agrees With Week 2

The deployed RF is a `Pipeline(scaler → PCA(19) → forest)`, so its Shapley values are computed
exactly in the 19-component space the forest splits on, over the full 1,901-row canonical test
fold (additivity residual 3×10⁻¹⁵ against the model's own probabilities).

![Figure 1](image/W07_Explainability/rover_rf_shap_pca.png)

*Figure 1 — Left: per-row SHAP over the 19 PCA components with feature-value colouring. Right:
mean-|SHAP| ranking; orange bars are the Week-2 MDI selection.*

**SHAP's top five components are the Week-2 selection — same set, same order** (PC1, PC2, PC12,
PC7, PC17), and the full 19-component rank correlation between MDI and mean-|SHAP| is Spearman
0.919 (`p = 3×10⁻⁸`). The Week-2 methodology caveat — MDI is biased toward high-variance features,
and PCA components mix sensors — could have made that selection a method artifact; on this model
it is not. Divergence appears only in tail components neither method would select.

Week 2 documented loadings for PC1–PC5 only, so the three high-index selected components are
resolved here from the fitted PCA:

| component | top loadings (signed)                                                                     |
| --------- | ------------------------------------------------------------------------------------------ |
| PC7       | `inter_wheel_std_roll_max` +.43 · `inter_wheel_std_roll_mean` +.42 · `inter_wheel_std` +.37 · `gps_dlat` +.24 |
| PC12      | `inter_wheel_std` +.48 · `gps_dlat` −.39 · torque `dom_freq` terms                          |
| PC17      | `inter_wheel_std` +.51 · `torque_3_dom_freq` −.45 · `torque_2_bandwidth` +.43               |

All three are dominated by the engineered cross-channel dispersion features. With PC1/PC2 carrying
the torque spectral centroids and `stall_ratio`, the five selected components tell one story: the
classical model's usable signal concentrates in the features Week-2 engineering added — the
attribution-level restatement of the +0.27 F1 those features bought.

## 4. Rover SHAP — the MLP, in Its Native Space

The MLP consumes the standardised 40-D matrix directly (bit-identical to the RF pipeline's input
space), so DeepExplainer attributes over named features with no back-projection; background is 200
seed-42 train rows, the full test fold is explained, additivity residual 2×10⁻⁶ on the anomaly
logit.

![Figure 2](image/W07_Explainability/rover_mlp_shap_40d.png)

*Figure 2 — MLP SHAP, top 10 of 40 features. High `inter_wheel_std` (red, right tail to +3.8
logits) is the single strongest anomaly driver.*

Top-3: **`inter_wheel_std`, `torque_1`, `torque_0`** — the cross-channel dispersion feature first,
then raw torques, with the `stall_ratio` family filling the next ranks. Both model families,
trained in different spaces with different inductive biases, select the same engineered feature as
their primary anomaly evidence. A formal rank comparison through a loading projection
(`Σ_k mean|SHAP_k|·|loading_kj|`) gives Spearman 0.18, not significant — but that projection is
magnitude-only and mass-spreading, so it bounds what the comparison can show rather than
contradicting the top-of-ranking agreement that is visible directly.

## 5. Fari — Attribution Scored Against a Known Truth

Fari's label is generated from an explicit linear score (weights `topic_coherence` 1.1,
`sentiment_score` 0.9, `latency_ms` −0.8, `follow_up_rate` 0.7, `response_length` 0.3 on z-scored
features), so here the true importance ordering exists and SHAP can be scored rather than compared.

![Figure 3](image/W07_Explainability/fari_shap_vs_true_weights.png)

*Figure 3 — Per-feature attribution share for the RF and MLP against the true |weight| share; each
series normalised to sum to 1.*

**Both models recover the true ordering exactly** (Spearman ρ = 1.0; exact-permutation
`p = 0.017` at n = 5) — the attribution pipeline passes the one test where a ground truth exists.
The share profiles differ in character: the RF exaggerates the strongest feature (0.35 vs true
0.29) and nearly zeroes the weakest (0.03 vs 0.08), while the MLP tracks the whole profile
including the tail — trees spend their splits where signal is strong, the small dense network keeps
weak features alive. It is the same contrast §3–4 show on Rover, in miniature.

## 6. Senpai — a Three-Class Task With a Measured Ceiling

Senpai had no dataset anywhere, and its product documents carry design parameters but no measured
data, so the 2,000 × 5 task is generated under a fixed-difficulty IRT-3PL assessment scenario: the
documents specify the 3PL response model and an adaptive 57–69 % success band — which would make
`correct_rate` level-invariant under tutoring, hence the fixed-difficulty probe — and the class
prior 21.4/64.3/14.3 % maps the plan's beginner/intermediate/advanced onto the documented
below/at/above-expected bands (a real class composition of 6/18/4). Response times anchor on the
documented 1.2 s pseudo-mastery threshold, hints on the σ > 0.65 struggle trigger, sessions on the
30-min cap; every invented distribution is recorded in the notebook's data card. Labels are the
generating latent ability's class, not feature thresholds, so irreducible error exists by
construction and is measured: a Bayes rule on the latent θ reaches accuracy 0.812 / macro-F1
0.764.

![Figure 4](image/W07_Explainability/senpai_class_distributions.png)

*Figure 4 — Class-conditional feature distributions. `correct_rate`, `hint_requests` and
`response_time_mean` separate the classes; `topic_switch_count` barely does (by design); the
`session_duration` spike at 30 min is the documented session cap.*

The classical model (RF, grid winner 200 trees × depth 10, stratified 70/15/15 split of
1,400/300/300) reaches **test macro-F1 0.667 against the 0.764 ceiling** — about 87 % of the
information the features hold — at 8.7 ms per sample (PASS vs the 100 ms conversational gate).

![Figure 5](image/W07_Explainability/senpai_confusion.png)

*Figure 5 — Test confusion matrix. The smallest class (advanced, 43 rows) is where the remaining
error concentrates: per-class F1 0.685 / 0.750 / 0.566.*

![Figure 6](image/W07_Explainability/senpai_shap_summary.png)

*Figure 6 — Per-class SHAP. High `correct_rate` pushes away from beginner and toward advanced;
hints do the reverse.*

Multi-class SHAP matches the generator's design and adds one asymmetry the design did not state:
`hint_requests` attributes nearly twice as strongly for the beginner class (0.123) as for
intermediate (0.071) — struggle signals identify who needs help more sharply than they separate the
upper levels, which is the asymmetry a differentiation product wants. `topic_switch_count`
attributes almost nothing anywhere, matching its weak class signal by construction; a fielded model
would have to earn its switches from engagement telemetry the synthetic set does not carry.

## 7. Attention on the Misses

Week 3 left two findings unjoined: the Transformer's attention lands on true fault spans (median
selectivity 5.3×, above 1 in 98 % of eligible anomalous windows, against 1.15× for the LSTM
hidden-state norm), and detection collapses on stuck-type faults (39.5 % caught vs 76.2 % for
slip). If attention still finds the fault span on the missed windows, the miss is a
decision-boundary failure, not a localisation failure — and the fix is a different one.

Fault type is recovered by Week 3's run-length proxy (slip lives < 40 steps, stuck up to 80, so
runs longer than 40 steps are unambiguously stuck-type); the refit Transformer is replayed at its
canonical 0.87 threshold and reproduces the ledger confusion matrix before any attention is read.

![Figure 7](image/W07_Explainability/attention_stuck_detected_vs_missed.png)

*Figure 7 — Attention selectivity (fault steps / normal steps, log scale) for stuck-type windows by
detection outcome, coloured by fault episode. The 99 eligible windows tile only two independent
episodes — one 90 % detected (blue), one missed entirely (orange) — so the two groups are very
nearly the two episodes.*

The sample structure kills any detected-vs-missed ranking (n = 2 episodes; the window-level medians
7.0 vs 8.7 compare episode identity, not populations). What stands is the one-sided claim, and it
is the week's most surprising result: **in the fully missed episode, every one of the 49 windows
has attention selectivity above 1 (median 8.7×) while the classification head calls all of them
normal.** The encoder finds the fault span and the head discards it. The stuck-type recall ceiling
is a property of the decision head under the training distribution, not of the encoder — which is
where a Week-8 fix should start.

## 8. RL Value and Saliency — What Fine-Tuning Changed

A policy's counterpart to SHAP is two objects: the critic's value of the state it stands in, and
the sensitivity of its action logits to each observation channel. Both are read on the Week-6
deployable leader (PPO 250k, BC-init, seed 0) against the BC prior it was tuned from; both
reproduce their published world-0 returns to the digit before attribution, and the analysis runs on
one canonical episode — it explains a policy, it does not rank policies.

![Figure 8](image/W07_Explainability/rl_value_trace.png)

*Figure 8 — Top: critic value V(s) along the episode; shaded spans are active anomaly labels,
vertical markers the main-route and branch blockage onsets. Bottom: the full action distribution
π(a|s) on the same axis.*

The trace shows the mismatch the Week-6 tables measured. The critic prices route state correctly —
V holds 50–85 on normal patrol, collapses at the main-route blockage and settles near −57 in the
dead end — but the actor barely answers: reroute mass appears late and never dominates, and after
the branch also blocks, the policy returns to `continue` at 74 % inside a state its own critic
scores as a dead end, until the stuck timeout ends the episode. Event-conditioned: mean V is 69.7
on normal patrol (modal action `continue` 74 %), 34.6 on anomaly steps (modal `raise-alert` 90 %),
−39.8 on blockage-approach and −56.8 in the dead end — both still modal `continue`. Alert mass
also fires on unshaded stretches: the 241 false alerts per 1,000 normal steps, visible on one
trace.

![Figure 9](image/W07_Explainability/rl_saliency_conditions_share.png)

*Figure 9 — Input-gradient saliency as a within-policy share, per event condition, PPO vs the BC
prior. Gradients are taken at each policy's own greedy action on the normalised observation.*

**Fine-tuning reallocated attention in one direction.** BC keeps a standing dual sensitivity —
`torque_max` and `next_main_block_dist` carry comparable gradient in every condition (27 %/27 % on
normal patrol) — while PPO concentrates on `torque_max` (46 % normal, 42 % anomaly) and leaves the
blockage distance under 9 % except when a blockage is already in sensor range. That is the
attribution-level form of the Week-6 behaviour trade (reroute 0.84 → 0.34, false alarms 55 → 241):
PPO's updates grew the alert pathway, which pays on every anomaly step, and let the navigation
pathway inherited from BC atrophy. The two policies choose the same greedy action on only 61.7 %
of trace steps — the fine-tuned policy is not a sharpened copy of its prior.

![Figure 10](image/W07_Explainability/rl_saliency_per_action.png)

*Figure 10 — One logit per trigger: the raise-alert logit on anomaly steps, the reroute logit on
blockage-approach steps.*

The per-logit view is the sharpest statement. The raise-alert logit is essentially a `torque_max`
detector (43 % of its gradient on one channel — mechanistically consistent, since the world's fault
signature is a torque surge: 21.5 → 44.1 Nm on anomaly steps, point-biserial r = 0.49). The
reroute logit has no feature at all: its largest channel carries 16 % of the gradient against a
uniform 11 %. A head with no sharp input dependence cannot fire reliably — the mechanistic form of
the Week-6 finding that rerouting is what the flat 10 Hz policy cannot learn and the node-level
semi-MDP can (0.34 vs 0.82). The observation-only IRL policy failed for the complementary reason,
so across all three lenses the alert pathway is the learnable part of this task and routing is the
hard part.

## 9. Learning Curves — Neither Model Has Seen Enough Data

The curve grows the training set by whole blocks (rows within a block are serially dependent), and
refits the full protocol at every size: standardisation, class weights, early stopping and the
val-tuned threshold.

![Figure 11](image/W07_Explainability/learning_curves_rf_mlp.png)

*Figure 11 — Train and validation F1 against training rows, RF (left) and MLP (right).*

The MLP's validation F1 rises monotonically through the final doubling (0.745 at 1.8k rows to
0.858 at 9.7k) with a small train–val gap — low variance, moderate bias, and a curve that has not
flattened: more blocks would still buy performance. The RF carries a persistent 0.10–0.13 gap at
every size — the variance-dominated profile — and its full-data validation score (0.81) is what
the MLP reaches with half the blocks. The compressibility echo from the CNC project: at 18 % of
the training blocks the MLP is already within 0.11 of its full-data score — most of the signal
lives in a small fraction of the data here, as it lived in a small fraction of the features there.

## 10. The Cross-Family Pareto

Scores are not commensurable across tasks (F1, cm error, patrol return), so each platform task
gets its own latency–score panel. Latency is re-measured from the verified checkpoints under one
convention — single observation, CPU, `timeit` × 2,000, including each model's own preprocessing —
because recorded values came from different weeks on different machine states (the same MLP
measures 0.135 ms in Week 3 and 0.027 ms now); no ranking flips under re-measurement. The RL score
is the floor–ceiling normalised return; privileged policies are excluded. The assembled table is
`data/pareto_points.csv`.

![Figure 12](image/W07_Pareto_and_Optimisation/pareto_three_panels.png)

*Figure 12 — Latency (log) against the task score, one panel per platform task; ringed points are
non-dominated; dashed lines are the platform gates.*

**Aido Rover classification: frontier {MLP, Transformer}.** Every other model is dominated —
RandomForest doubly so, 25–320× slower than the neural models at mid-pack F1. The full 15-pair
per-fold significance matrix (deferred from Week 3) shows exactly four significant pairs, all
Transformer wins — over RF (`p = 0.011`), 1D-CNN (`0.004`), LSTM (`0.047`), GRU (`0.0498`) — and
**Transformer vs MLP not among them** (`p = 0.34`). With the Week-3 capacity control, the honest
summary is one top equivalence class {Transformer, MLP} explained by capacity, and inside it the
axes decide: the MLP is 12.6× faster. **Deployment recommendation: MLP**, Transformer as the
interpretability alternate — unchanged from Week 3, now with the whole matrix behind it.

**Aido Humanoid trajectory: frontier {CV, Linear, MLP, LSTM}.** The axes trade smoothly from the
CV floor (0.003 ms, 2.77 cm) to the accuracy end — **LSTM, 1.44 cm at 0.11 ms, the deployment
recommendation** — and the Transformer is the one dominated point, beaten by the LSTM on both axes:
the Week-4 verdict restated geometrically.

**Aido Rover patrol: frontier {REINFORCE+baseline, PPO}, both BC-initialised.** DQN sits below
zero — its return is under the deployable rule-policy floor. PPO leads on the canonical layout
(normalised 0.33 at 0.12 ms) with the Week-6 caveat attached: on all five held-out layouts the
group-relative variant wins, so this frontier ordering is a property of the terrain. Every point
in every panel clears its gate by two or more orders of magnitude — the latency budget prices
model choice on these tasks; it does not constrain it.

## 11. Optimisation — Shrinking the RandomForest

No model misses its gate, so the plan's premise (optimise a near-frontier model that fails its
constraint) has no instance. Two real framings replace it: cost — the RF is the portfolio's
slowest model, and inference cost is battery on a patrol robot — and the Aido Rover's ≤10 ms
breach-detection tier, where the premise is nearly literal: the full forest measures 9.1 ms this
week (7.86 ms in Week 2), consuming ~91 % of that budget — one machine-state fluctuation from
failing it.

![Figure 13](image/W07_Pareto_and_Optimisation/rf_reduction.png)

*Figure 13 — Test F1 against latency for depth/width-reduced forests; the dashed line is the 10 ms
breach tier.*

| config          | test F1 | AUC    | latency | share of 10 ms tier | SHAP top-5 overlap |
| --------------- | ------- | ------ | ------- | ------------------- | ------------------ |
| 200 × d10 (full)| 0.7359  | 0.9668 | 9.07 ms | 91 %                | 5/5 (reference)    |
| **50 × d6**     | 0.7153  | 0.9573 | 2.52 ms | 25 %                | 5/5                |
| 25 × d10        | 0.6957  | 0.9653 | 1.41 ms | 14 %                | 5/5                |

The pick is 50 trees × depth 6: −0.021 F1 (inside the full model's own ±0.048 fold spread) for a
3.6× speed-up, and the SHAP top-5 components are identical in every configuration tried — the
shrunken model scores like the full one and explains itself with the same features. The trade is
acceptable on both framings; it does not change the deployment recommendation, which the MLP holds
on both axes even against the 2.5 ms forest.

## 12. Calibration Against the STUM Target

The PIC 2.0 STUM class consumes scores through fixed σ thresholds, so beyond discrimination the
deployment question is calibration.

| model       | ECE (raw) | T (val-fitted) | ECE (temperature-scaled) |
| ----------- | --------- | -------------- | ------------------------ |
| MLP         | 0.025     | 1.07           | 0.024                    |
| Transformer | 0.041     | 1.16           | 0.032                    |

Both meet the class's < 0.10 target on-distribution, and the **MLP beats the deployed 0.031
benchmark with no calibration layer at all**; temperature scaling is near-inert for it and buys the
Transformer a quarter of its miscalibration. Read together with the 0.36–0.95 threshold swings
across fold rotations, the calibration problem on this data is not marginal reliability but
between-block transfer of operating points — which is what per-site calibration exists for, and
what the PIC 2.0 analysis names as the STUM class's priority experiment.

## 13. Limitations and the Week-8 Handoff

**The attention-on-misses result rests on two fault episodes.** The one-sided claim (localisation
succeeds in a fully missed episode) is safe; any detected-vs-missed effect size needs more stuck
episodes, which means regenerating test-fold data — a Week-8 decision.

**The RL saliency is one seed on one world** by design: it explains the evaluated policy, it does
not rank methods. The cross-model back-projection for the RF is magnitude-only and correspondingly
weak; the cross-model claim rests on the top-of-ranking agreement, not the projection statistic.

**Senpai is synthetic with an invented class-conditional structure** (documented in the data
card); its external validity is a data-collection question, not a modelling one.

**The learning curves say more data still pays** for the deployment-recommended MLP — the
capstone's "future work" has a measured direction: more blocks, not more capacity. And the
stuck-type recall fix has an address (the decision head, §7), which is the highest-value candidate
for the Week-8 demo's "what next" slide.
