Hongyu LIU
InGen Dynamics - ML & NN Analyst Intern, August 2026

---

# Methodology Report — How Every Result in This Project Is Produced

**Scope:** the supervised, sequence, trajectory and reinforcement-learning results of Weeks 1–7,
stated as the protocols that generate them. Numbers appear only where they justify a protocol
choice. The classical model family is RandomForest alone — one model, a deliberate reduction of the
plan's five — and every claim about "classical" models is scoped to it.

## 1. Data and Preprocessing

**One dynamics core, two consumers.** All Rover data — the offline 9-channel sensor stream and the
online RL environments — comes from a single world-dynamics module (`shared_modules/rover_world.py`).
The Week-5 environment passes a bit-exact replay gate against the Week-2 offline table, so offline
and online results are about the same physics, not two implementations that resemble each other.
Sensor ranges, fault mechanisms and the 85/15 normal/anomaly calibration are set from the platform
documents, and the label is withheld from every deployable model's observation.

**Feature space.** The tabular models consume a 40-dimensional matrix built by one shared function
(`shared_modules/features.py`): 9 raw channels with absolute GPS replaced by per-step deltas, 25 FFT
descriptors (5 spectral features × 5 channels, 50-step window at 10 Hz), and 6 engineered
cross-channel physical features (inter-wheel torque dispersion, stall ratio, and their rolling
statistics). The cross-channel features exist because the fault mechanism is described
sensor-against-sensor; adding them was worth +0.27 F1 in Week 2, more than any model change that
week. Sequence models consume 11-channel windows (9 sensor + the 2 instantaneous physical channels)
at w ∈ {10, 20, 50}; rolling statistics are excluded there because the network sees the history the
rolling window would summarise.

**The canonical split.** Rows within a contiguous block are serially dependent, so row-level
splitting — stratified or not — leaks. The stream is segmented into 23 blocks and assigned by
`StratifiedGroupKFold` into 7 folds; validation and test folds were chosen from pre-training
structural statistics, and the first 50 rows of every block are purged so no window straddles a
boundary. Post-purge: 9,734 train / 2,215 val / 1,901 test (70.3/16.0/13.7, anomaly share 16 % in
each). Every classification and sequence model in the project reads this one assignment file.

**The other datasets.** Fari interaction quality is 3,000 independent rows (5 features, binary
label with a known generative weight vector), split 70/15/15 stratified — row independence is what
makes the plain split leakage-safe there. The Senpai learner-state set (2,000 × 5, three classes)
is generated in Week 7 under a fixed-difficulty IRT-3PL assessment scenario with every
distributional choice recorded in an in-notebook data card; its irreducible error is measured by a
Bayes rule on the generating latent ability (accuracy 0.81), so model scores have a ceiling to be
read against. Humanoid trajectories are 5,000 sequences split 70/15/15 by sequence.

## 2. Supervised Model Comparison

**Two-phase protocol.** Architecture and hyperparameter decisions are made on the canonical fold
only, with three seeds where a decision needed a spread. The chosen configuration is then locked
and retrained across all seven fold rotations — test = fold k, val = fold (k+1) mod 7, scaler and
class weights refit per rotation, decision threshold re-tuned on that rotation's validation fold —
and the ledger records mean ± std with the per-fold columns. Design exploration and headline
estimation never share a fold.

**Winners need paired tests.** Fold scores are paired across models, so comparisons use paired
t-tests over the seven rotations. The full 15-pair matrix gives four significant results —
Transformer over RandomForest (`p = 0.011`), 1D-CNN (`p = 0.004`), LSTM (`p = 0.047`) and GRU
(`p = 0.0498`) — and eleven non-significant ones, including Transformer vs MLP (`p = 0.34`). With
the Week-3 capacity control (a parameter-matched Transformer-S falls to GRU's level), the defensible
summary is a single top equivalence class {Transformer, MLP} whose membership is explained by
capacity, not attention. Within-class selection is by latency, which is why the MLP is the Rover
deployment recommendation.

**The fragile part is the operating point.** Val-tuned thresholds swing between 0.36 and 0.95
across rotations while AUC stays within 0.963–0.994: discrimination is stable, calibration is not.
Any deployed threshold therefore needs per-site calibration, and threshold-dependent F1 is treated
as the noisier of the two headline metrics throughout (see also §5 on the 1D-CNN).

## 3. Sequence and Attention Methodology

Sequence models are compared at matched window length (w = 50) under the same rotation protocol,
with a latency-vs-window sweep (w ∈ {10, 20, 50}) reported per architecture because the recurrent
models scale linearly and attention quadratically in w; at these window lengths the crossover is
thousands of steps away, so window choice is governed by detection quality alone.

Attention is used as an interpretability lens, not an accuracy argument. The Week-3 analysis
extracts per-head attention maps and a batched attention-received statistic over all test windows:
median fault/normal attention selectivity is 5.3× (above 1 in 98 % of 267 eligible anomalous
windows) against 1.15× for the LSTM hidden-state norm — attention localises the anomaly evidence
the recurrent summary blurs. The Week-4 trajectory transformer's five learned horizon queries show
attention centre-of-mass moving from history step 6.7 (first waypoint) to 3.1 (fifth), the
geometric statement that far predictions are set by earlier context. Week 7 adds SHAP on the
classical and MLP models across three tasks and gradient saliency on the RL policies, so every
model family carries an attribution method suited to its structure.

## 4. RL Evaluation Protocol

**Seeds and worlds are separated.** Every learned family trains on 5 seeds and is evaluated
deterministically on 10 fixed evaluation worlds at the full 9,600-step patrol horizon; the paired
vector for any comparison is the per-world mean across train seeds. Ten worlds is the minimum that
can produce significance: a two-sided Wilcoxon signed-rank on n pairs cannot return p below 2/2ⁿ,
so n = 5 has a floor of 0.0625 and n = 10 of 0.002. Every "A beats B" claim carries that test;
near-ties are reported as not significant by name. Option-level and multi-agent artifacts store
per-seed-per-world returns so the same test applies to them.

**Brackets, not bare returns.** Every table is read against two fixed references evaluated on the
same worlds: a deployable rule policy that navigates but cannot alert (floor, 3,931) and a
privileged expert reading the ground-truth label (ceiling, 11,575). Policies whose controllers read
the label — the option-level semi-MDP arms, the privileged IRL basis — are reported in the
privileged bracket and excluded from deployment comparisons. Because episode lengths differ by 3×
across policies, return-per-step and event-conditioned rates (P(alert | anomaly), false alerts per
1,000 normal steps, P(reroute | single blockage)) lead the tables; total return is reported
alongside.

**Generalisation is part of the protocol.** Week 6 showed five seeds on one map calling a tie that
five held-out layouts decide the other way, so seed-0 policies are re-evaluated on the five other
validated map layouts and the sweep is recorded in the ledger. A ranking that exists only on the
training layout is treated as a property of the terrain, not the method.

## 5. Reproducibility

**Refits against the record.** No Week 2–4 notebook persisted a fitted model, so Week 7 refits all
of them from documented configurations (`shared_modules/refit.py`) and verifies against the
published metrics before any downstream analysis. Twelve of thirteen models reproduce every
published digit — threshold, F1, AUC, confusion matrix, epochs run and best epoch — and are saved
under `saved_models/` with a manifest carrying config, metrics and verification deltas; loading a
checkpoint returns predictions identical to the fresh fit over the full test set.

**The thirteenth model is a finding.** The 1D-CNN does not reproduce under its original training
path: cuDNN's convolution backward accumulates gradients atomically, and eight runs of the
unchanged code give test F1 from 0.683 to 0.738 (the published 0.7317 sits near the 75th
percentile) while AUC stays at 0.9585 ± 0.0010. The module pins `torch.backends.cudnn.deterministic`
— which leaves the other seven torch models bit-identical — and records the pinned result
(F1 0.6977) with the spread documented. The general lesson matches §2: threshold-dependent F1
carries hardware-level run noise that AUC does not, and a reported F1 without a seed/backend
statement is only accurate to about ±0.03 for this architecture.

**Seed discipline.** Supervised: 7 fold rotations (design ablations on 3 seeds). Trajectory: 5
training seeds (42–46) per neural model. RL: 5 train seeds per family (references and BC are
deterministic single fits), 10 evaluation worlds. Behaviour cloning retrains deterministically from
the offline table and reproduces its published per-world returns exactly. Every notebook documents
its seed at the top; the harness records per-run seeds, configs and curves in JSON metas.

## 6. Deployment Feasibility

**One latency convention.** Recorded latencies were measured in different weeks on different
machine states (the same MLP measures 0.135 ms in Week 3 and 0.027 ms in Week 7; the full
RandomForest 7.86 ms and 9.07 ms). Week 7 therefore re-measures every family from the verified
checkpoints under one convention — single observation, CPU, `timeit` over 2,000 calls, including
each model's own preprocessing — and treats recorded values as order-of-magnitude context. No
ranking flips under re-measurement.

**The Pareto is three panels, not one chart.** F1, trajectory error in cm and normalised patrol
return are not commensurable, so each platform task gets its own latency-score panel and its own
frontier. Results: Rover classification frontier {MLP, Transformer} with the MLP recommended
(§2); Humanoid trajectory frontier {CV, Linear, MLP, LSTM} with LSTM the accuracy end (1.44 cm at
0.11 ms) and the Transformer dominated; Rover patrol frontier {REINFORCE+baseline, PPO}, both
BC-initialised, with the Week-6 caveat that the group-relative variant wins every held-out layout —
and the best value-based policy scoring below the deployable floor. Every point in every panel
clears its platform gate by at least two orders of magnitude: on these tasks the latency budget
prices model choice rather than constraining it.

**Optimisation under a met constraint.** Nothing in the portfolio misses its gate, so the
optimisation experiment targets the two real framings: cost (the RandomForest is 25–320× slower
than the neural models) and the Aido Rover's ≤10 ms breach-detection tier, of which the full
forest consumes ~91 % as measured this week — one machine-state fluctuation from failing it.
Shrinking to 50 trees × depth 6 gives F1 0.7359 → 0.7153 (−0.021, inside the model's own ±0.048
fold spread), AUC 0.9668 → 0.9573, latency 9.07 → 2.52 ms, and an unchanged SHAP top-5 — the
reduced model explains its decisions with the same features. The trade is judged acceptable on
both framings; it does not change the deployment recommendation, which the MLP holds on both axes.
