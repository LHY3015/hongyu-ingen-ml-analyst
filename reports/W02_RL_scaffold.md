# RL Scaffold: Design Report

## 1. World-core suitability

The shared world core (`rover_world.py`) is used as-is for RL. Its two-layer
architecture (`world.step` physics core vs. a deferred `env.step` MDP wrapper) is the correct shape
for this task: the physics is validated and frozen, while reward/observation design can keep iterating in
W05 without touching data generation. 

---

## 2. State space — 9-D

| #    | Feature                                                 | Rationale                                |
| ---- | ------------------------------------------------------- | ---------------------------------------- |
| 0–2 | torque_mean, torque_max, torque_std                     | Window-level wheel-effort summary        |
| 3    | lidar_mean (m)                                          | Obstacle proximity                       |
| 4    | soc_slope                                               | Discharge rate over the look-back window |
| 5    | battery_soc                                             | Current charge                           |
| 6–8 | route_progress, next_main_block_dist, branch_block_dist | Position summary                         |

`anomaly_score` (a label-free torque-derived score) is deliberately excluded: it would duplicate information
already in `torque_std` (feature 2), and `reports/W02_Feature_Analysis_Report.md` establishes that torque
statistics are terrain-confounded rather than clean fault indicators — a derived score built on the same
statistic would inherit the same confound.

Absolute GPS is excluded for the same position-leakage reason it is excluded from the anomaly classifier's
features (fixed terrain → faults recur at fixed coordinates). `route_progress` plus the two block distances
serve as the position summary instead.

---

## 3. Reward function

Reward is a `(label, action)` lookup table, conditioned on three context flags — `halted`, `main_blocked`,
`rough_terrain` — because the same nominal action is correct in one context and wrong in another, which a
flat table cannot express:

| Context                                     | Action   | Reward | Rationale                                                                                                              |
| ------------------------------------------- | -------- | ------ | ---------------------------------------------------------------------------------------------------------------------- |
| normal, halted                              | continue | −0.5  | Idle at an unresolved blockage is not productive patrol (vs. +1.0 while genuinely patrolling)                          |
| normal, main_blocked                        | reroute  | 0.0    | A justified detour around a real obstruction is not penalised the same as gratuitous rerouting (−0.3 otherwise)       |
| normal, rough_terrain (torque_mean > 30 Nm) | slow     | 0.0    | Proactive slow-down on high-slip terrain is the physically correct response, not unnecessary caution (−0.1 otherwise) |

`anomaly + slow` (+1.5) outranks `anomaly + reroute` (0.0): anomalies in this world core are a mechanical
self-fault (slip/stuck) — `reroute` has no physical effect on a self-fault, while `slow` does (it scales
down the fault-trigger probability via `speed_factor` inside `RoverWorld.step`, a real mitigating effect).

Shaping (+2.0 when `battery_soc < 20%` and `action == return-to-base`) matches the Aido Rover product
documentation's auto-dock threshold ("auto-dock at <20% SoC").

Discount factor: 0.99, anchored to the PIC 2.0 GRPO product-config range (γ=0.975–0.995 across InGen
platforms, product docs §6) rather than an arbitrary pick. Unlike the thresholds in §4 — calibrated against
this dataset — gamma has zero effect on `rover_transitions.csv` itself; it only matters once a value-based
method (Q-learning/DQN) trains on this offline data in W05, which may tune it further.

---

## 4. Scripted behaviour policy

The policy is a priority-ordered rule table (`POLICY_RULES`, a list of `(description, condition, action, forced_done)` rows evaluated in order), mirroring `NORMAL_REWARD`/`ANOMALY_REWARD`'s table style.

### 4.1 Alert trigger: ground-truth label, not a torque threshold

`raise-alert` on a real fault uses the ground-truth `anomaly_label` at generation time, not an observable
torque-based proxy: torque statistics are terrain-confounded (`reports/W02_Feature_Analysis_Report.md`) — a
torque-threshold trigger, measured against the actual blockage-enabled rollout, produces a ~27% false-alert
rate on normal rough-terrain windows. The scripted policy is a behaviour-cloning expert (ground truth is
available at generation time; a deployment classifier would not have it), matching the world-core design's
"alert on fault" intent.

### 4.2 Reroute / full-block trigger distance: 150 m

`next_main_block_dist` / `branch_block_dist` report the full remaining distance to a blockage along a
hypothetical path, available as soon as the blockage is anywhere on that path — not only once the rover is
physically close. The reroute decision must be resolved before the rover commits to entering the blocked
edge, while it may still be a full edge-length away (branch edges run 53–160 m, with blocks typically 60–90 m
into them), so the trigger distance is set generously — 150 m, near the sensor's lookahead limit. Validated
against the actual rollout: halted fraction ≈0%, vs. 79% for a rover with no reroute logic engaged at all.

### 4.3 Full block: escalation by time

Blockages are static within an episode — nothing in this core clears one. A genuine full block (main AND
branch both blocked) is a dead end for a priority-only policy, so it is handled as one condition escalated
by time rather than two independent rules: `raise-alert` is logged for the first `STUCK_TIMEOUT=80` steps
(reporting the double blockage), and at step 80 the policy forces `return-to-base`, ending the episode. This
bounds how much of an episode's step budget a persistent full block can consume.

### 4.4 Slow trigger: window torque_mean > 30 Nm, isolating wet-grass/mud

The product docs describe the Aido Rover slowing specifically on **wet grass/mud** with slip compensation
(product docs §1) — not on dry grass or gravel — and the world core gives `slow` a real mitigating effect
(↓ fault-trigger probability via `speed_factor`). Terrain-conditional torque, calibrated on the actual
8-episode generation (reroute + slow both active), shows a clean separation:

| Terrain   | True occupancy (length-weighted) | Window torque_mean p25–p90 | Slip weight |
| --------- | -------------------------------- | --------------------------- | ----------- |
| asphalt   | ~40–58%                         | 5.8–6.0 Nm                 | 0.00        |
| gravel    | ~4–17%                          | 17.4–23.1 Nm               | 0.10        |
| dry_grass | ~8–24%                          | 25.2–27.5 Nm               | 0.30        |
| wet_grass | ~13–20%                         | 36.9–40.0 Nm               | 0.70        |
| mud       | ~1–15%                          | 42.5–54.8 Nm               | 1.00        |

(Ranges span the main-loop vs. branch-edge measurements and the 8-episode aggregate; branch/detour edges run
somewhat rougher than the main loop, consistent with the design note that a detour should not be
"systematically cleaner" than the main route.) dry_grass's p25–p90 band (25.2–27.5 Nm) and wet_grass's p25
(36.9 Nm) leave a clean ~9 Nm gap; 30 Nm sits in it, so `slow` triggers only for the two genuinely high-slip
terrains, matching the product docs' language rather than reacting to any terrain deviation. Checked after
the reroute condition and before the default, so it never overrides a real blockage response. Reward is
symmetrically conditioned on the same boolean (§3).

Verified on the actual generation: per-step `slow` share is 16.1%, with 929 event-level onsets — on the same
order as `reroute` (836) and `raise-alert` (1,034), not a rare action.

---

## 5. Episode structure

### 5.1 Episode termination is tied to the scripted decision, not the explored action

`done` is derived from `forced_done or (scripted_action == 4)` — the policy's decision before any ε-explore
override — and the exploration pool itself excludes `return-to-base` (`{continue, slow, reroute, raise-alert}` only). Every `done=True` row therefore reflects a genuine scripted decision (low battery, an
unresolved full block, or a completed loop), never a 1-in-20 exploratory action that happened to land on
"abort mission".

### 5.2 Episode cap: an explicit step counter, not RoverWorld's total_steps

`EP_CAP=9600` (one full 960 m loop) is enforced via an explicit `ep_len` counter that forces `done=True`
once reached — `RoverWorld`'s `total_steps` constructor argument only affects the `ambient_temp` seasonal
sine period and does not itself stop an episode. Every episode therefore ends at a well-defined point (one
full loop, or earlier for a genuine low-battery/full-block reason), and the next episode draws a fresh
blockage/fault seed.

### 5.3 Dataset size: 48,000 rows

`rover_transitions.csv`'s row count is not fixed by the weekly plan (only `synthetic_rover_data.csv`'s
15,000 rows is a plan requirement). 48,000 rows (~5 full-loop episodes before any early aborts) is sized for
enough independent blockage/fault-seed draws to reliably exercise both the full-block and low-battery
branches — a full-block-with-no-escape configuration occurs in only ~4/10 independently drawn seeds, so a
small episode budget risks never observing it. The actual 8-episode generation: all 5 actions represented,
1/8 episodes reaching `battery_soc < 20%`, 96.4% correct-response rate on true anomalies.

### 5.4 Initial battery_soc randomisation

Episode 0 starts at 100% (canonical full-charge patrol, for a clean baseline trace); subsequent episodes
draw `Uniform(40, 100)%` — set directly on the `RoverWorld` instance after `reset()` (`w.soc = ...`, no core
code change) — reflecting a fleet that doesn't always dispatch fully charged. The 40% lower bound still
leaves enough runway to plausibly cross the 20% auto-dock threshold within one loop's discharge (~30–38
points/loop measured), while avoiding an unrealistic near-empty dispatch.

### 5.5 Per-step counts vs. event counts

Per-step action counts conflate how often an action starts with how long it's sustained: a single reroute
event is logged once per row for its entire ~150 m approach, and a fault event once per row for its full
10–80-step duration. The stats cell therefore also reports event-level counts (onsets — a new streak
whenever the action differs from the previous row within the same episode) alongside the per-step
distribution, so a frequently-sustained action isn't mistaken for a frequently-chosen one.

---

## 6. Known limitations

- The stuck-counter and ground-truth-label conditions make the behaviour policy non-Markov in the 9-D state
  alone (it uses privileged label information and short history) — valid for a policy used to log data, not
  implied to be a state-only function usable at deployment.
- Blockages are static within an episode and `raise-alert` cannot clear one in this core — a genuine full
  block is necessarily a scripted dead end resolved only by the stuck-timeout abort. Suggested for the W05
  wrapper: let `raise-alert` clear a blockage after N steps (simulating an operator response), so the action
  has a learnable positive effect in full-block episodes instead of always ending in abort.
- `route_progress` is an odometer-based approximation (cumulative distance mod loop length) and drifts
  slightly from true DAG position after a branch detour.

---

## 7. Downstream usage

- `rover_transitions.csv` → Behavioural Cloning & offline RL (W05_Offline_RL_BC.ipynb)
- `rl/mdp_schema.md` → Gymnasium env construction (W05_RL_Environment.ipynb) — same world core, same
  state/action/reward definitions, so offline data and the online env are directly comparable
- `rover_windows.npz` (raw + FFT view) → Sequence model inputs (W03_Sequence_Models_RNN_vs_Transformer.ipynb)
