# Week 06 ML Log — Policy-Gradient, GRPO-Style, Inverse/Hierarchical and Multi-Agent RL

## What was done

- **Evaluation protocol rebuilt** (`shared_modules/rl_eval.py`). Five train seeds per family evaluated on ten
  fixed evaluation world seeds, with the paired vector taken as the per-world mean across train
  seeds and every comparison decided by a Wilcoxon signed-rank test over the ten common worlds. The
  count of ten is forced: a two-sided signed-rank test on `n` pairs cannot return `p < 2/2ⁿ`, so at
  Week 5's `n = 5` the floor was 0.0625 and no comparison could have reached significance at all.
  The two bracket policies, previously duplicated closures inside two Week-5 notebooks, are now one
  definition with a reproduction gate asserting they still return Week 5's published figures.
- **Policy-gradient implementations from scratch** (`rl/pg_algos.py`): REINFORCE, REINFORCE with a
  learned value baseline, and a GRPO-style group-relative variant, all instrumented per update with
  advantage variance, half-batch gradient agreement and KL.
- **Node-level semi-MDP** (`rl/rover_options_env.py`) with four option controllers and exact
  `Σ γᵗ r` / `γᵏ` accumulation, plus a tabular high-level learner.
- **Inverse RL** (`rl/irl.py`): maximum-entropy IRL on a tabular discretisation of the logged expert,
  run with privileged and observation-only feature sets and validated against the reward that
  actually generated the data.
- **Two-rover PettingZoo environment** (`rl/rover_multiagent_env.py`) with independent PPO learners
  under a shared team reward and a counterfactual difference reward.

## Results

### The Week-5 budget question, closed

| DQN configuration | return | episode len | P(reroute\|block) | false alerts /1k |
| --- | --- | --- | --- | --- |
| 80k, Week-5 config | 2,651 ± 480 | 6,418 | 0.012 | 165 |
| 250k, same config, ε anneal 0.3 → 0.5 | 2,048 ± 648 | 5,983 | 0.087 | 319 |

Paired over the ten common worlds the difference is −603 with `p = 0.084`, not significant. Tripling
the budget did not improve return, roughly doubled the false-alarm rate, and left rerouting an order of
magnitude below the rule policy's 1.00 — with most of the increase contributed by a single seed
(0.297 against 0.007–0.061 for the other four). Week 5's reading of the navigational failure as
structural rather than budget-limited stands.

Two protocol checks passed on the way: the bracket policies reproduce 4,072 ± 516 and 12,199 ± 736
exactly, and the Week-5 DQN checkpoints re-evaluated on Week 5's own five worlds return 1,563 ± 517,
the published number to the digit. The same checkpoints score 3,740 ± 520 on the five added worlds,
so absolute returns are comparable only within a fixed world set.

### From-scratch on-policy training collapses, but only for the trust-region methods

| family | return | episode len | P(reroute\|block) | false alerts /1k |
| --- | --- | --- | --- | --- |
| PPO, 250k | 156 ± 17 | 309 | — | 0 |
| GRPO-style, 250k | 214 ± 33 | 322 | — | 27 |
| PPO, 250k, γ = 0.999 | 135 ± 1 | 332 | — | 0.3 |
| REINFORCE, 250k | 1,579 ± 789 | 3,163 | 0.014 | 137 |
| REINFORCE + baseline, 250k | 1,329 ± 694 | 3,430 | 0.077 | 76 |
| REINFORCE + baseline, BC warm start | 5,969 ± 692 | 8,658 | **0.819** | 104 |
| GRPO-style, BC warm start | 5,182 ± 585 | 8,736 | 0.650 | **89** |
| PPO, BC warm start | **6,421 ± 659** | 7,733 | 0.344 | 241 |

PPO and the GRPO-style variant both converge to patrolling for ~310 steps and then taking the
terminating action to bank a small positive return; two seeds produce bit-identical trajectories,
which is what a deterministic rule on a deterministic world gives. A random initial policy earns about
−0.78 per step, so ending the episode is genuinely better than continuing, and once the policy shortens
its episodes it stops collecting the long-horizon patrol data that would overturn that estimate. DQN is
immune through replay and ε-greedy. So is plain REINFORCE, at 1,579 — the two methods that fall into
the trap are the two with a clipped surrogate and a bootstrapped critic, and the high-variance
estimator's updates keep the policy moving through the region where terminating looks locally optimal.

The same code from a behaviour-cloning warm start reaches 6,421, which rules out an implementation
defect and makes the warm start the precondition for on-policy training here rather than a refinement
of it. It also invalidated two experiments as originally specified — the γ = 0.999 probe and the
low-SoC ablation measure nothing when the agent never survives past 320 steps — and both were re-run
from the warm start.

Paired over the ten common worlds: PPO (BC-init) beats BC by 1,530 (`p = 0.037`), REINFORCE + baseline
by 1,078 (`p = 0.0020`), while the GRPO-style variant's +291 is **not significant** (`p = 0.19`), and
PPO's +1,239 over GRPO is **also not significant** (`p = 0.084`). The γ = 0.999 probe is a clean null
(Δ +317, `p = 0.49`): a blockage at the 150 m detection threshold is ~1,500 steps away at 0.1 m/step,
so a 1,000-step horizon still does not reach it.

### The canonical layout selects the wrong policy

| policy | map 6 (canonical) | map 1 | map 2 | map 4 | map 5 | map 7 |
| --- | --- | --- | --- | --- | --- | --- |
| DQN 250k | 2,461 | **−316** | 443 | 167 | 539 | 118 |
| PPO, BC-init | **6,421** | 4,234 | 4,893 | 5,066 | 3,186 | 3,767 |
| GRPO-style, BC-init | 5,182 | **5,807** | **7,193** | **6,352** | **4,902** | **5,448** |

Tied on the training layout, the group-relative policy wins on all five held-out layouts by
1,286–2,300 while PPO loses 21–50 %, and DQN does not transfer at all. The ordering matches the
behavioural split: PPO's canonical advantage comes from alerting at 241 false alarms per 1,000 normal
steps, a policy tuned to where the faults are on this terrain, and its fine-tuning drops rerouting from
the BC prior's 0.84 to 0.34. The GRPO implementation carries a KL penalty to the frozen BC reference
and keeps 0.65. This was the check the Week-2 design report flagged and no week had run.

### The low-SoC ablation needed a configuration change to be reachable at all

Discharge is about 0.004 % of SoC per step, so crossing the 20 % threshold from a full battery takes
roughly 19,500 steps against a 9,600-step evaluation loop, and training resets drawing SoC from
U(40, 100) under a 2,400-step cap bottom out near 30 %. Measured `low_soc_steps` was 0 for every flat
family. The ablation runs from a lowered dispatch charge of U(18, 30) with a matching evaluation start.

| configuration | return | steps below 20 % SoC | ended by docking | ended flat |
| --- | --- | --- | --- | --- |
| flat, BC-init, penalty 0.0 | 5,190 ± 729 | 41,093 | 0.26 | **0.26** |
| flat, BC-init, penalty 1.5 | 998 ± 669 | 12,257 | **1.00** | **0.00** |
| option level, penalty 0.0 | 7,863 ± 142 | 3,218 | 0.04 | — |
| option level, penalty 1.5 | 3,350 ± 129 | 1,244 | **0.96** | — |

The prescription works once the region is reachable, and both levels agree. On the flat environment the
penalty eliminates battery depletion entirely — 26 % of episodes to none — and cuts time below the
threshold by 70 %; at option level, where docking is one decision, it takes docking from 4 % to 96 %.
Return falls in both, which is the price of a reward that puts battery safety above patrol continuation
rather than a failure. The control arm reads as a deployment statement: without the penalty the policy
spends 62 % of every episode below 20 % SoC and runs one episode in four to actual depletion.

Two statistics needed care. A per-environment-step docking rate is ~0.001 for every arm, since `dock`
is one decision against thousands of low-SoC steps. And the episode-level rate alone is not enough
either — the from-scratch arms show 1.00 while never reaching the low-SoC region, because their
collapse *is* an early terminating action.

### IRL identifiability is a property of the demonstrations

| | privileged features | observation-only | positive control |
| --- | --- | --- | --- |
| Spearman vs the true reward | 0.309 | 0.080 | **0.970** |
| online return, 10 worlds | 11,285 ± 2,267 | 2,423 ± 1,266 | — |

The privileged basis reproduces `compute_reward` to 2×10⁻¹⁶ when handed the true weights, so the
shortfall is not expressiveness. The logged behaviour policy is a rule with ~3.6 % ε-exploration that
alerts on 8,239 of 8,549 anomaly steps, so feature-expectation matching has almost no evidence about
actions the expert never takes; the true reward spreads those four actions over 8.0 points and the fit
collapses them into a band 0.75 wide. Refitting against a soft-optimal demonstrator on the same MDP
with the same code gives 0.970. Converging the optimiser harder makes the correlation *worse*
(0.49 → 0.31 as the feature-expectation error falls from 0.105 to 0.022), which is what an unidentified
problem looks like.

Separately: the privileged reward correlates 0.31 with the truth yet its optimal policy returns 11,285
against a ceiling of 11,575 — reward recovery and policy recovery are different objectives. The
observation-only run falls below the deployable floor at 2,423, with alert recall 0.118 against
behaviour cloning's 0.50 on the same information, on 25 parameters against the privileged basis's 14.

### What a semi-MDP actually shortens

A main sweep edge is 160 m, which is 1,700–1,900 environment steps; only the short connectors are
~600. The option backup carries `γᵏ` at that `k`, which is of order 10⁻⁸, so an option-level learner at
γ = 0.99 optimises exactly the flat objective. What the hierarchy changes is credit-assignment
distance: the difference between a clean edge and one whose blockage matters is +134.4 against +116.6
in intra-option discounted return while the undiscounted sums are 3,128 and 3,127, and the option
learner attaches that difference to one backup where the flat learner propagates it through ~1,700.

Budget and discount were therefore crossed rather than confounded — an option spans 600–1,900 steps, so
250k steps buy only ~266 decisions and 1M buy ~1,060.

| arm | decisions | return | P(reroute\|block) | ended stuck |
| --- | --- | --- | --- | --- |
| scripted option policy (no learning) | — | 10,853 ± 2,291 | 0.52 | 0.60 |
| γ = 0.99, 250k | ~266 | 10,044 ± 2,066 | 0.49 | 0.68 |
| γ = 0.99, 1M | ~1,060 | 11,118 ± 586 | 0.58 | 0.53 |
| γ = 1.0, 250k | ~266 | 11,276 ± 586 | 0.74 | 0.50 |
| γ = 1.0, 1M | ~1,060 | **12,151 ± 141** | **0.80** | **0.40** |

Both effects are real and close to additive — the discount is worth +1,232 at the small budget and
+1,033 at the large one, the budget +1,074 at γ = 0.99 and +875 at γ = 1.0 — and the discount effect
surviving at four times the budget is what rules out a sample-efficiency artefact. On rerouting they
separate cleanly: budget +0.09, discount +0.25, both 0.80, against 0.04–0.09 for DQN in the flat MDP.
The discount also collapses the seed spread: at 250k the γ = 0.99 arm splits into two of five seeds at
7,560 and three above 11,000 for ±2,066, while every γ = 1.0 seed lands between 10,528 and 12,237 for
±586, and the larger budget narrows it to ±141. Week 5's account of one mechanism behind three failures was one mechanism too few.

### Mutual visibility between rovers spans 2.6 % of the loop

Driving the same seeded episode twice with and without the partner circle — `dynamic_obstacles`
consumes no RNG draw, so the runs are otherwise bit-identical — gives certain detection between 2 and
20 m, 0.65 at 20–25 m and exactly zero beyond 25 m. The cause is that `cast_min` returns the minimum
over a forward 120° fan and the site boundary sits 20 m outside the route, so the no-partner geometric
return never exceeds 23.1 m anywhere: the partner is invisible unless nearer than that wall echo.
Detection *falls* to 0.65 below 2 m because the ray origin enters the partner circle and the
intersection is rejected, and the exchange is not symmetric — the follower sees the leader, the leader
sees nothing. Twenty-five metres on a 960 m loop cannot support a decision about which rover covers
which stretch, so the observation carries an explicit global partner block.

### The group baseline trades gradient agreement for centring

| method | updates/seed | half-batch cosine | ‖g₁−g₂‖ / ‖ḡ‖ | advantage variance |
| --- | --- | --- | --- | --- |
| REINFORCE | 62 | 0.977 | 0.23 | 2.92 |
| REINFORCE + value baseline | 62 | 0.995 | **0.11** | 2.98 |
| GRPO-style, BC warm start | 127 | 0.265 | 1.79 | 1.00 by construction |

Every update splits its own transitions into two disjoint halves and compares the two policy gradients.
The value baseline halves the disagreement (0.23 → 0.11) while the marginal advantage variance is
unchanged (2.92 against 2.98), so its benefit shows up as agreement between independent halves rather
than as a narrower spread of `G_t`. GRPO's advantage is group-standardised, so its variance of 1.00 is
a definition and the informative quantity is the pre-standardisation group return spread, 6.0. Its low
cosine follows from the same centring: removing the component every trajectory shares is exactly what
holds an uncentred estimator's cosine near 1 while carrying no information about which action was
better. PPO runs through SB3 with no per-update instrumentation, so the comparison is against
REINFORCE's learned value baseline rather than PPO's.

### The multi-agent comparison is a null, and the decision rate explains it

| | shared team reward | difference reward |
| --- | --- | --- |
| team return | −2,522 ± 2,089 | −2,386 ± 1,741 |
| loop coverage fraction | 0.735 ± 0.012 | 0.735 ± 0.012 |
| redundantly covered edges | 1.44 | 1.42 |
| decisions per episode | 5.4 | 5.4 |

The two credit-assignment schemes are indistinguishable on every coordination metric. An option spans
~1,200 environment steps, so a 9,600-step patrol holds ~5.4 decisions per rover; 1M training steps buy
~560 decisions and, at 32 per rollout, ~17 policy updates. A difference reward is a counterfactual on
an agent's contribution, and with 17 updates neither learner has a contribution to attribute. Coverage
sits between random (0.425) and scripted (0.825), where a barely-trained learner should be. The
obstacle for the CRL-MRS direction is therefore the decision rate, three orders of magnitude below the
control rate at node granularity.

### Latency

Single-step inference over 2,000 `timeit` calls: scripted rule policy and the tabular option lookup
0.001 ms, the 2×64 policy nets 0.027 ms, the BC MLP 0.028 ms, DQN 0.066 ms, PPO 0.115 ms. The two
paths not already known from Phase B are the hierarchical one, a high-level selection roughly once a
minute plus the 10 Hz controller, at 0.002 ms for a step running both levels, and the two-rover one at
0.030 ms for two forward passes per decision. Every path clears the 100 ms patrol gate by three to
five orders of magnitude, so the RL cost sits entirely in training.

## Deliverables Completed

- `shared_modules/rl_eval.py` — evaluation protocol, static observation normalisation, bracket policies, Week-5 reproduction gate
- `rl/pg_algos.py` — REINFORCE, +baseline, GRPO-style with per-update variance instrumentation
- `rl/rover_options_env.py` — node-level semi-MDP over the single-agent environment
- `rl/rover_multiagent_env.py` — PettingZoo two-rover cooperative patrol environment
- `rl/irl.py` — maximum-entropy IRL with privileged and observation-only feature sets
- `rl/harness/{train,evaluate,run_sweep}.py` — training and evaluation entry points, kept as scripts because the 96 jobs need process-level fan-out across cores; the notebooks load what they wrote
- `W06_PolicyGradient_RL.ipynb`, `W06_IRL_Hierarchical.ipynb`, `W06_MultiAgent_RL.ipynb`
- `W06_RL_Advanced_Report.md`; `rl/rl_results.csv` extended with the protocol and event-rate columns
