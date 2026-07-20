# Week 05 ML Log — RL Foundations: MDP, Gymnasium Env & Value-Based RL

## What was done

- Gymnasium environment `rl/rover_env.py` wrapping the shared world core — 9-D observation, 5 discrete actions, the `mdp_schema.md` reward function as the canonical implementation. Reward conditioning is cached at state `s` and applied before the world advances (the offline generator's convention), the 50-step window is warmed up with exactly 50 `continue` steps in `reset()`, and the 2,400-step training cap is a `TimeLimit` truncation so SB3 bootstraps at the cap. Training resets randomise world seed + initial SoC ~ U(40,100); evaluation is the deterministic full 9,600-step loop.
- Offline↔online replay-alignment gate: replayed the offline episode-0 action sequence through the env from the same seed — reward matched to 2.8×10⁻¹⁷ and next-state to float32 precision over 5,931 transitions. Env also passes SB3 `check_env`.
- Value-based RL: tabular Q-learning (96-state observation discretisation, `torque_std` axis added so *detect→alert* is representable) and DQN (SB3, MlpPolicy [64,64], 5 seeds, 80k steps) with seed-banded learning curves; reward-shaping ablation on the energy-penalty weight; single-step inference latency.
- Offline RL & behaviour cloning from the Week-2 transition table (episode-level split): class-weighted MLP behaviour cloning + neural Fitted Q-Iteration, evaluated online against two reference brackets — a deployable label-blind rule policy (floor) and a privileged label-aware expert (ceiling).
- Results ledger `rl/rl_results.csv` (separate schema from the classification `model_ledger.csv`); trained policies saved under `rl/saved_policies/`.

## Results

Full-loop return (mean ± std, 5 seeds), deployable policies in bold:

| Policy                          | Return                 | Episode len | Note                          |
| ------------------------------- | ---------------------- | ----------- | ----------------------------- |
| random (no-abort)               | −894 ± 444           | 5,258       | floor                         |
| tabular Q (96-state)            | −320 ± 735           | 3,472       | intuition only, high variance |
| FQI (offline, K=60)             | 1,394 ± 525           | 3,910       | horizon-limited, over-alerts  |
| DQN (online, 5-seed)            | 1,527 ± 1,217         | 3,634       | over-alerts, never reroutes   |
| **scripted, label-blind** | 4,072 ± 516           | 9,530       | deployable rule floor         |
| **BC (offline)**          | **6,287 ± 689** | 9,478       | strongest deployable          |
| expert, label-aware (ceiling)   | 12,199 ± 736          | 9,367       | privileged reference          |

- **BC is the best deployable policy, framed as privileged-information distillation, not "beating the expert."** BC's teacher is the label-aware expert (12,199), well above BC's 6,287; BC recovers 27% of the label-blind→privileged gap using sensors alone — the rule policy cannot alert on faults at all, and BC learns to from the torque window (alert recall 0.50, reroute recall 0.90).
- **DQN over-alerts rather than navigating.** Event-conditioned stats: P(alert|anomaly) 0.56 but 213 false alarms per 1k normal steps, P(reroute|single-block) 0.03, P(slow|rough) 0.02 — total-return parity with references was a mirage (offsetting true-alert bonuses vs false-alarm penalties), so reactive-vs-navigational competence was read from these conditionals, not the scalar. The reroute failure drives it into full-block dead ends (all eval episodes end via stuck-timeout at 79–89% SoC). The curve plateaus by 80k (~33 episodes), so it is a budget/exploration-limited local optimum, not a claimed fundamental value-based limit. 0% return-to-base is correct at ~80% SoC, not a deficiency.
- **FQI failure is horizon truncation, not distribution shift.** At 12 iterations mean max-Q is 35 (vs ~+95 full horizon) and reroute is chosen in 0 test states; extending to 60 lifts max-Q to 146 and recovers reroute (520 states) — confirmed directly. Residual Q-inflation above +95 is the max-operator over-estimation term (what CQL/BCQ target), but horizon is the dominant lever here.
- **Latency:** tabular 0.002 ms, DQN 0.071 ms, BC 0.30 ms — all PASS the 100 ms gate by 3–5 orders of magnitude; the RL cost is entirely in training.

## The reward-shaping change that most altered the learned policy

The required reflection: across the energy-penalty ablation (ew ∈ {0.0, 0.05, 0.2}), the return differences were within seed variance and not significant, so no shaping weight *significantly* moved the return. The change that most altered **observable behaviour** was removing the energy penalty entirely (ew = 0): `slow` usage jumped to 51.3% from 39.4% at the canonical weight. The mechanism is direct — with no per-step energy cost, over-cautious slowing carries no downside, so the value-maximiser slows more freely; restoring the penalty prices the extra steps a slow-down costs and pulls the policy back toward `continue`. The deeper reward-design finding is separate and more important than the ablation: under the base reward, normal patrol pays +0.95/step, so the return-maximising behaviour is to patrol until the battery forces termination and the +2.0 low-SoC return-to-base shaping is far too weak to induce the 20% auto-dock the product spec calls for — every trained agent uses return-to-base 0% of the time. For a coverage-patrol objective that is defensible, but it is exactly the kind of reward-engineering gap RL surfaces that supervised learning never does.

## Deliverables Completed

- `rl/rover_env.py` — Gymnasium env (canonical MDP implementation); passes replay-alignment gate + SB3 `check_env`
- `W05_RL_Environment.ipynb` — spaces, random-rollout validation, reward sanity checks, alignment gate, `alert_clears_block` variant
- `W05_Value_Based_RL.ipynb` — tabular Q + DQN (5 seeds, seed bands), event-conditioned stats, reward-shaping ablation, latency
- `W05_Offline_RL_BC.ipynb` — behaviour cloning (full mandatory metrics) + Fitted Q-Iteration with the horizon diagnostic; offline-vs-online comparison bracketed by the two reference policies
- `W05_RL_Foundations_Report.md` — MDP formulation, env + alignment, reward design, value-based + offline results, GRPO connection, limitations
- `rl/rl_results.csv` — RL results ledger; `rl/saved_policies/` — DQN checkpoints, tabular Q table
