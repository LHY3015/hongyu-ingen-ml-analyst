# Rover World Core: Design Report

Brief design report for the synthetic Aido Rover data generator
All scales are anchored to the Aido Rover product documentation.

---

## 1. What it is

**Stepable world core** — `world.step(action) → 10 channels + label + info` , that generates physically-coupled sensor stream and is reused by downstream tasks: W2 anomaly detection & sequence models, and W5–6 reinforcement learning.

**Core idea:** A 2D map + a path-planned pose drive GPS & LiDAR; torque/power drives the battery; terrain/slip drives anomalies. Channels data are derived from this world, not sampled as independent random streams — so cross-channel correlations (e.g. torque spike ↔ faster SoC drop) arise from the physics, not from hand-coded coupling.

**Run knobs:** `SEED` (episode RNG: faults, noise, blockage positions),
`MAP_SEED` (map/route/terrain layout), `BLOCKAGES` (on/off), `N_STEPS`. `HAZARD` is auto-calibrated per map.

---

## 2. Designs & rationale

| Decision                                                                                      | Rationale                                                                                                                                                                                                                                                                                          |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Two-layer architecture**: `world.step` (physics core) vs `env.step` (MDP wrapper) | The core alone makes the sensor stream (no obs/reward needed). The wrapper adds obs/reward/done, so reward can be iterated without touching data generation.                                                                                                                                      |
| **Forward simulation** (author map+pose → synthesize sensors), not SLAM                | We control the ground truth, each sensor naturally receives a data stream as the steps process.                                                                                                                                                                                                   |
| **Geometric primitives** (circles/segments/polys + waypoint DAG)                        | Exact, cheap analytic ray-cast; a graph fits the discrete`reroute` action.                                                                                                                                                                                                                       |
| **Motion along DAG edges**; heading = edge direction, 0 m turning radius                | No follower / no capture radius, so the path lies exactly on the planned route.                                                                                                                                                                                                                   |
| **LiDAR  takes forward 120°/120-ray MIN** distance                                   | Proximity semantics, multimodal distribution, richer structure than normal random.                                                                                                                                                                                                                |
| **Battery discharge torque-coupled, compressed time coinffient**                        | Real rate (~0.0003%/step) is invisible in 25 min; α compresses time while keeping torque↔battery physically consistent (faults raise torque → faster drain).                                                                                                                                    |
| **Anomaly = mechanical SELF-fault** (slip / stuck), ego-centric, no location            | Matches a fault-detection task; slip = one wheel spikes + others shed (anti-correlation), stuck = all wheels high + displacement≈0.                                                                                                                                                               |
| **severity s∼Beta(2,5)** latent; **label = 1 for any fault**                     | Weak faults sink into noise (realistic incipient-fault detection).                                                                                                                                                                                                                                |
| **Statistical (not deterministic) fault↔signal correlation** + shared confounders      | Normal rough terrain also raises torque+SoC,  no label leakage.                                                                                                                                                                                                                                  |
| **Fault cooldown (~15 steps)**                                                          | Prevents a "slip-trap" (a fault slows the rover inside a high-slip patch → re-triggers unboundedly); caps a single anomaly burst ≈ max fault duration.                                                                                                                                           |
| **`slow` scales the fault-trigger probability by `speed_factor`**                   | Real-world slowing is a risk/safety action (slip compensation at reduced speed), No-op at`speed_factor=1.0` (i.e. `continue`)                                                                                                                                                                  |
| **`reroute` = get around a TEMPORARY BLOCK, then rejoin**                             | Blockage is LiDAR-visible & external → decoupled from the invisible self-fault, so`reroute` is a learnable sensor/map-driven decision, not a memorized reflex.                                                                                                                                  |
| **incline = 0** (no elevation layer)                                                    | Terrain zones already provide the torque/SoC confounder; a continuous elevation field isn't needed.                                                                                                                                                                                                |
| **Fixed topology per map + per-episode events**                                         | One`MAP_SEED` → one fixed map; blockages+faults (episode-rng) vary per episode → closed-loop reactive policy, not open-loop memorization. Different `MAP_SEED` are used to train+eval **separately** and check reward/policy robustness across layouts (not cross-map generalization). |

---

## 3. Map & Route Generation

The map is procedurally generated from `MAP_SEED` (different seeds → different valid layouts):

- Boustrophedon main coverage cycle — `N_SWEEPS` horizontal sweeps + edge connectors.
- One rejoining detour per sweep — an apex bows into the interior, then returns to the
  line past the apex. So `reroute` is a general skill exercised at several branch nodes,
  not a single-spot reflex.
- Mixed-shape terrain patches (circles/rects/triangles, bounded size → short crossings)
  on both main and branch edges, weighted toward rougher terrain. Placing terrain on
  branches too means a detour is not systematically cleaner than the main route → `reroute`
  is a genuine cost/benefit choice, not an always-win.
- Static obstacles placed by rejection-sampling so they clear every route/reroute edge
  (the rover, which follows the graph exactly, never drives into one) — they only shape the LiDAR fan.
- Per-map hazard calibration (§4): terrain exposure varies by seed, so `HAZARD` is searched
  per map to land the anomaly rate in [14%, 16%]. At least the map used for the W2 detection
  dataset must be 85/15-calibratable (default `MAP_SEED=6` calibrates to 15.0% at HAZARD=0.05).

**Temporary blockages (default OFF).** When `BLOCKAGES=True`, a per-episode seeded obstacle
may sit on a branchable edge; it is LiDAR-visible and halts a rover that drives into it.
Escalation ladder: main edge blocked + branch clear → `reroute`; both blocked → `raise-alert`;
low SoC → `return-to-base`. (The edge the rover starts already committed to is never blocked,
since it can't be rerouted mid-edge.) Blockages are OFF for the W2 detection stream and ON for W5 RL.

---

## 4. Usage per task

| Task (week)                                             | How to run                                                                                                                                                                                      | Artifact                             |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **Anomaly detection** (W2–3: RF, MLP, 1D-CNN)    | `RoverWorld(hazard=HAZARD, seed, map_seed)` (blockages OFF); drive `step(0)`; dump 10 channels + label                                                                                      | `synthetic_rover_data.csv` (85/15) |
| **Sequence models** (W3–4: LSTM/GRU/Transformer) | same stream → sliding-window tensor + windowed FFT view                                                                                                                                        | `rover_windows.npz`                |
| **Offline RL / BC** (W5)                          | `RoverWorld(..., blockages=True)`; interactive rollout with a scripted behaviour policy (reroute on block, alert on fault/full-block, return-to-base on low SoC, ε-explore); log (s,a,r,s′) | `rover_transitions.csv`            |
| **Online RL** (W5 DQN / W6 PPO, GRPO-style)       | wrap the core in a Gymnasium`env.step` (obs/reward/done), blockages ON                                                                                                                        | policy checkpoints                   |
| **Multi-agent** (W6)                              | two`RoverWorld` on the shared map; pass each rover as the other's `dynamic_obstacles` so they see each other                                                                                | PettingZoo env                       |
| **Reward-robustness check**                       | run the whole train+eval on several`MAP_SEED`s separately; if a reward only "works" on one layout it is over-fit                                                                              | —                                   |

Notes: state for RL = window summary (torque mean/max/std, lidar mean, SoC-slope) + `battery_soc`

+ `position_summary` (built from `info`: progress + next-edge blockage distances). **Absolute GPS
  is excluded from anomaly-classifier features** (fixed terrain → faults at fixed coords → position leakage).

---

## 5. Outputs & channels

**Channels (10) — the sensor stream + label (→ CSV / model features):**

| #    | channel                  | meaning                            | scale / model                                                                            |
| ---- | ------------------------ | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| 1–2 | `gps_lat`, `gps_lon` | position                           | pose → lat/lon + RTK ±2 cm noise + multipath near structures                           |
| 3    | `lidar_distance`       | forward nearest obstacle distance | 0–200 m (VLP-32); ray-cast MIN + σ(r)=σ0+k·r + AR(1) noise + 200 m no-return dropout |
| 4    | `battery_soc`          | state of charge (%)                | 100 → ~44% / episode; torque-coupled, compressed-time (α)                              |
| 5–8 | `torque_0..3`          | per-wheel torque (Nm)              | ~tens of Nm, terrain-driven; anomaly = slip anti-corr / stuck all-up                     |
| 9    | `ambient_temp`         | ambient temperature (°C)          | 28 + 3·sin + noise                                                                      |
| 10   | `anomaly_label`        | fault active (0/1)                 | mechanical self-fault (slip/stuck)                                                       |

A blockage only **lowers `lidar_distance`** (it is a visible obstacle) — it adds no new channel.

**`info` (ground-truth / diagnostics — not sensor-observable; feed the W5 wrapper, not the classifier):**
`terrain`, `fault`, `sev`, `x`, `y`, `geo` (pre-noise LiDAR), `node`, `target`, `route_progress`,
`next_main_block_dist`, `branch_block_dist`, `halted`. The wrapper assembles `position_summary`
from these.

---

## 6. Validation

Default configuration: `MAP_SEED=6`, `SEED=42`, `N_STEPS=15,000`, `continue` policy.

- auto-calibrated **HAZARD=0.05 → anomaly 15.05%** (in the 14–16% band); SoC 100 → **43.5%**
- LiDAR min/med/max = **6.1 / 23.1 / 200.0 m**; torque normal **14.0** / anomaly **39.7** Nm
- power normal/anomaly ratio **1.40**
- terrain visited: asphalt/dry_grass/wet_grass/mud/gravel; slip 1369 / stuck 889
- blockages OFF, byte-identical to the pre-generator single-map core on the same layout;
  blockage demo: `continue` halts at the first blocked leg (reaches nodes 0–2), `reroute` avoids
  every blockage (0 halts, visits all 16 nodes incl. detour nodes)
- generator sanity: seeds 1,2,4,5,6,7 all calibratable to [14–16%]; every route/reroute edge is
  clear of all static obstacles by construction.
