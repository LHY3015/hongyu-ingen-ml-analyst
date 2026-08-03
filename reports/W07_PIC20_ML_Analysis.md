Hongyu LIU  
InGen Dynamics - ML & NN Analyst Intern, August 2026

---

# PIC 2.0 Model-Class Analysis — Findings, Methodology, Readiness

**Scope:** the six PIC 2.0 model classes, each scored against the experimental evidence this
project produced (Weeks 1–7: classical/NN/sequence baselines, transformer and trajectory models,
value-based and policy-gradient RL, IRL/hierarchical, multi-agent). Readiness is a 1–5 statement of
how much validated evidence exists for deploying that class on InGen platforms — a score for the
evidence, not for the idea. Config parameters quoted for each class come from the public Origami
architecture page and the platform engineering documents, the public page authoritative where they
differ; experimental numbers come from this repo's ledgers.

| class | anchor result from this project | readiness |
| --- | --- | --- |
| GRPO | warm start is the precondition; training-layout leaderboards select the wrong policy | 3 |
| SEOM | training-time penalties deliver the safety property in full (depletion 26 % → 0) | 3 |
| HTD-IRL | reward recovery is bounded by demonstrations; hierarchy effects significant | 3 |
| STUM | the deployment-recommended MLP already meets the internal ECE target (0.025 vs 0.031); the risk is calibration transfer | 3 |
| AMDC | cross-channel engineered features carry the attribution mass in every model family | 2 |
| CRL-MRS | node-level coordination has ~30 updates per 1M steps to learn from — a structural null | 1 |

## GRPO — Group Relative Policy Optimisation

**Finding.** Three results, in order of consequence. First, on the rover patrol task every
from-scratch on-policy run with a clipped surrogate and bootstrapped critic collapses to
episode-termination farming (PPO 156, GRPO-style 214 against a floor of 3,931), while the same code
from a behaviour-cloning warm start reaches 5,182–6,421: the PIC 2.0 pattern of starting GRPO from
an SFT checkpoint is not an optimisation but a precondition. Second, the group-relative baseline
behaves as designed — its centring removes the shared trajectory component, its constrained variant
holds false alarms at 89 per 1,000 normal steps against PPO's 241 — and it ties PPO on the training
layout (`p = 0.084`) while winning all five held-out layouts by 1,286–2,300. A model-selection
leaderboard computed on the training distribution would have picked the wrong policy. Third,
policy-network inference is 0.027–0.12 ms — the class's cost lives entirely in training and
sampling, not serving.

**Methodology.** Paired evaluation on fixed world sets (Wilcoxon over ten common worlds),
behavioural event rates alongside return, a KL leash to the reference policy, and a held-out-layout
sweep as a standard column — the last one being the single highest-information addition for a class
whose products (curriculum sequencing on Senpai, patrol replanning on Rover) will always run off
their training distribution.

**Readiness: 3.** The class runs, its variance behaviour is instrumented and understood, and its
failure mode (collapse without a prior) is reproducible and diagnosed. What is missing for 4 is
layout-robust selection at scale: the held-out sweep used one train seed per family.

**Priority experiment.** Re-run the layout sweep with all five train seeds per family and adopt
held-out-layout return as the selection metric; on the product side this is the difference between
a curriculum policy that works for the pilot cohort and one that works for the next school.

## SEOM — Self-Supervised Ethical Oversight Mechanism

**Finding.** Two independent constraint-through-objective results behave exactly as the class
specifies. The per-step low-SoC penalty eliminates battery depletion entirely (26 % of episodes to
0) and cuts time under the 20 % threshold by 70 %, at a return cost that is the declared price of
ranking safety above patrol continuation — and the result replicates at both decision levels. The
GRPO variant's KL constraint to its reference is the same mechanism aimed at behaviour drift, and
holds the false-alarm rate at a third of unconstrained PPO's. Both match the class's gradient form
(`∇L_GRPO − λ∇L_SEOM`): the constraint shapes training, no runtime filter involved.

**Methodology.** Ablation pairs (penalty on/off) with reachability checked first — the Week-6
low-SoC ablation initially measured nothing because the penalised region was unreachable from the
training reset distribution, which is the class's central methodological trap: a safety term that
never fires in training is untested, not satisfied.

**Readiness: 3.** The mechanism is validated twice with clean effect sizes. Missing for higher: a
λ sweep mapping the safety-return frontier — the product line spans λ = 5.0 to 10.0, Rover at 5.0
and Fari eldercare at 10.0 — and any evidence on rule *sets* (the products run 10–12 concurrent
rules per platform, 46 in total; this project tested one at a time).

**Priority experiment.** The λ sweep on the low-SoC term, reporting the depletion-rate/return
frontier — the direct analogue of tuning force-gate strictness on Humanoid/Senpai.

## HTD-IRL — Hierarchical Task Decomposition via Inverse RL

**Finding.** The two halves of the class were tested separately and both returned structural
lessons. Inverse RL: with the true reward known, a privileged feature basis that contains it
exactly still recovers it at Spearman 0.31 from logged expert demonstrations, against 0.97 from a
soft-optimal demonstrator with the same code — identifiability is a property of the demonstrations,
not the fitter, and a near-deterministic expert (the product plan's 800 inspection demos will be
one) leaves the reward under-identified while still supporting a strong policy (11,285 against an
11,575 ceiling). Hierarchy: raising the decision level from 10 Hz to route nodes moves rerouting
from 0.34 to 0.82, with both factorial effects individually significant — the discount horizon at
`p = 0.014/0.002` (250k/1M) and the decision budget at `p = 0.027/0.037` (γ = 0.99/1.0) — and the
discount also collapsing seed spread (±586 → ±121 at 1M steps). The caveat that bounds deployment claims: the option
controllers are privileged, so the hierarchy results live in the non-deployable bracket.

**Methodology.** Score IRL against known rewards wherever a simulator exists (the positive-control
design), and evaluate hierarchies as budget × discount factorials with paired tests — the two
mechanisms are conflated in any single-arm comparison.

**Readiness: 3.** Decomposition value is demonstrated with significance; reward-recovery limits
are mapped. The missing piece is end-to-end deployability: sensor-driven option controllers.

**Priority experiment.** Replace the privileged option controllers' alert rule with the Week-3
sensor-window classifier and re-run the factorial — one experiment that moves the entire hierarchy
result into the deployable bracket and directly prototypes the class's leaf-node → GRPO structure.

## STUM — Spatiotemporal Uncertainty Model

**Finding.** Measured on the canonical test fold against the class's internal calibration target
of ECE 0.031: the deployment-recommended MLP meets it with no calibration layer at all (0.025 raw),
while the Transformer misses it raw (0.041) and lands just above it after validation-fitted
temperature scaling (0.032, T = 1.16; the MLP's T = 1.07 is near-inert) — the target is tight
enough to separate the two score producers this project has. The risk
the project's other evidence exposes is calibration *transfer*: across the seven fold rotations
every model holds AUC within ~0.03 while val-tuned operating thresholds swing 0.36–0.95, and the
1D-CNN's threshold-dependent F1 varies ±0.03 from cuDNN nondeterminism alone while AUC moves
±0.001. A class whose product behaviour switches at fixed σ thresholds — autonomous below 0.30,
cautious to 0.60, human escalation above it — inherits that gap: marginal reliability is fine,
operating points do not travel between blocks.

**Methodology.** Report discrimination and calibration as separate quantities; ECE alongside AUC
for any score a downstream rule consumes; per-site threshold calibration as a deployment step, not
a training artifact.

**Readiness: 3.** The on-distribution half of the class's requirement is measured — and met by the
deployment-recommended model; what is uncharacterised is the between-site half, which the threshold
swings say is the binding one.

**Priority experiment.** Cross-fold calibration transfer: fit the calibration layer on one fold
rotation and measure ECE on the other six — the direct test of whether a σ threshold set at one
site is trustworthy at the next, which is how the product deploys.

## AMDC — Adaptive Multi-Domain Calibration

**Finding.** The attribution work makes the class's core claim concrete: signal quality upstream of
the model dominates model choice. The engineered cross-channel features — inter-wheel torque
dispersion and stall ratio, the software analogue of AMDC's cross-sensor consistency checks — were
worth +0.27 F1 in Week 2, and Week 7's SHAP shows they carry the attribution mass in both model
families: three of the five selected PCA components are dominated by `inter_wheel_std` terms, and
the MLP ranks the same feature first in its native 40-D space. The RL saliency repeats the pattern
(the alert head is a `torque_max` detector). Meanwhile whole-model swaps between statistically tied
alternatives move F1 by less than the fold spread.

**Methodology.** Attribution-guided feature engineering: use SHAP/saliency to identify which
engineered channels the models actually consume, and invest calibration effort there.

**Readiness: 2.** The evidence is a strong analogy — no sensor-drift injection, no runtime
recalibration (`c_k` at 100 Hz) was tested on this data.

**Priority experiment.** Drift injection: perturb per-sensor gain/offset in the world core at
documented drift magnitudes (thermal drift 0.08 °C per °C of ambient), measure F1 degradation with and
without a recalibration step, and check whether SHAP attributions shift — a direct test of whether
the features the models depend on are the ones drift attacks.

## CRL-MRS — Cooperative RL for Multi-Robot Systems

**Finding.** A structural null with a measured cause. Two rovers with independent PPO learners are
indistinguishable under a shared team reward and a counterfactual difference reward on every
coordination metric (return 3,819 ± 434 vs 3,840 ± 430, coverage 0.735 vs 0.733), and the cause is
arithmetic: at node-level decisions an episode holds ~9 decisions per rover, so 1M environment
steps buy ~30 policy updates — nothing for a counterfactual credit signal to attribute. Two
supporting measurements bound future designs: LiDAR mutual visibility spans 25 m of a 960 m loop
(2.6 %), so coordination state must travel over comms, not sensing; and coordination metrics are
trustworthy only after termination parity — before the stuck-timeout fix, team returns were
dominated by a halted-rover artifact.

**Methodology.** Before comparing credit-assignment schemes, verify the interaction budget can
distinguish them (decisions and updates per run are now recorded in every meta); express
coordination as coverage/redundancy against random and scripted brackets.

**Readiness: 1.** The class's central question — does structured credit assignment beat a shared
reward — is untested at any budget where it could show. The fleet-scale claims (3–12 robots on
Rover, up to 47 on Sentinel) are untouched.

**Priority experiment.** Insert a decision level between 10 Hz and route nodes (e.g. 100-step
macro-actions) or raise the interaction budget by an order of magnitude, then re-run
shared-vs-difference; the first configuration in which the two schemes separate is the class's
minimum viable experiment.

## The Dependency Chain, Read Against the Evidence

The platform documents wire the classes in sequence: AMDC conditions the signal, STUM prices its
uncertainty, GRPO acts on it under SEOM's penalties, HTD-IRL supplies the decomposition, CRL-MRS
coordinates the fleet. The project's evidence orders the risk the same way the readiness column
does: the upstream links (signal quality, calibration) are where models were won or lost on this
data — every family's attribution concentrates on engineered signal-consistency features, and
operating points are the fragile quantity — while the downstream links (hierarchical
decomposition, constrained on-policy training) already have working, instrumented prototypes. The
weakest link is the last one: fleet coordination currently has no experimental regime in which it
can be measured at all, which is why its next experiment is about creating that regime rather than
about the algorithm.
