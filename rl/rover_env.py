"""Gymnasium environment for the Aido Rover patrol / anomaly-response MDP (Week 5).

Wraps the shared world-dynamics core `shared_modules.rover_world.RoverWorld` so the online
env and the Week-2 offline transition table (`data/rover_transitions.csv`) share identical
dynamics. The state extraction and reward function here are the canonical implementations of
the MDP schema (`rl/mdp_schema.md`); the W02 scaffolding notebook holds an equivalent copy
used to generate the offline table, and `tests`/notebook replay-alignment checks guarantee the
two agree step-for-step.

Reward-timing convention (must match the offline generator exactly): the reward for taking
action `a` is conditioned on the label / halted / block-distances observed at the CURRENT state
`s`, BEFORE `a` executes. The env therefore caches those conditioning variables at reset / after
each step and applies them on the next `step(a)` call, then advances the world to `s'`.
"""

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shared_modules.rover_world import RoverWorld

# --- MDP constants (mirror W02_Sequence_and_RL_Scaffolding.ipynb) -----------------------------
WINDOW = 50                      # look-back for the torque/lidar/soc window features
MAP_SEED = 6
HAZARD = 0.05
NEAR = 150.0                     # m, main/branch "blocked ahead" threshold (reward + policy)
STUCK_TIMEOUT = 80              # steps stuck at a full block -> forced episode end
LOW_SOC = 20.0                   # %, auto-dock threshold
ENERGY_PER_STEP = 0.05
ROUGH_TERRAIN_TORQUE = 30.0     # Nm, window torque_mean threshold isolating high-slip terrain
FULL_LOOP_STEPS = 9600          # one full 960 m main-loop circuit
SOC_INIT_RANGE = (40.0, 100.0)  # per-episode dispatch charge for randomized training resets

ACTION_NAMES = {0: 'continue', 1: 'slow', 2: 'reroute', 3: 'raise-alert', 4: 'return-to-base'}
STATE_COLS = ['torque_mean', 'torque_max', 'torque_std', 'lidar_mean', 'soc_slope',
              'battery_soc', 'route_progress', 'next_main_block_dist', 'branch_block_dist']

NORMAL_REWARD = {0: 1.0, 1: -0.1, 2: -0.3, 3: -1.5, 4: -3.0}
ANOMALY_REWARD = {0: -5.0, 1: 1.5, 2: 0.0, 3: 5.0, 4: 3.0}


def extract_state(tb, lb, sb, info) -> np.ndarray:
    """9-D state vector from rolling window buffers (torque, lidar, soc) + world-core info."""
    tarr = np.asarray(tb)                       # [WINDOW, 4]
    return np.array([
        tarr.mean(),
        tarr.max(),
        tarr.std(),
        np.mean(lb),
        (sb[-1] - sb[0]) / len(sb),
        sb[-1],
        info['route_progress'],
        info['next_main_block_dist'],
        info['branch_block_dist'],
    ], dtype=np.float64)


def compute_reward(label, action, halted, main_blocked, rough_terrain, battery_soc,
                   energy_per_step=ENERGY_PER_STEP, shaping_low_soc=True) -> float:
    """Reward table from mdp_schema.md, conditioned on the state `s` the action is taken from.

    `energy_per_step` and `shaping_low_soc` are exposed for the reward-shaping ablation; the
    defaults reproduce the offline generator exactly.
    """
    if label == 0:
        if action == 0 and halted:
            base = -0.5                          # idle at a blockage is not productive
        elif action == 2 and main_blocked:
            base = 0.0                           # justified detour, not penalised
        elif action == 1 and rough_terrain:
            base = 0.0                           # justified proactive slow-down
        else:
            base = NORMAL_REWARD[action]
    else:
        base = ANOMALY_REWARD[action]
    shaping = 2.0 if (shaping_low_soc and battery_soc < LOW_SOC and action == 4) else 0.0
    return base + shaping - energy_per_step


class RoverPatrolEnv(gym.Env):
    """Single-agent Gymnasium env over the rover patrol MDP.

    Parameters
    ----------
    randomize_reset : bool
        Training mode. Each `reset()` draws a fresh world seed and an initial SoC ~ U(40,100)
        (mirrors the offline generator), giving blockage-layout and battery diversity. When
        False (evaluation / replay), the env is deterministic: fixed `seed`, SoC = 100.
    energy_weight, shaping_low_soc : reward-shaping ablation knobs (see compute_reward).
    alert_clears_block : if True, a sustained `raise-alert` clears the blockage ahead after
        `alert_clear_steps` steps (simulated operator response). OFF by default so the online
        dynamics stay identical to the offline table; ON data is NOT comparable to the offline
        transitions (clearing a block also changes the block-distance features).
    """

    metadata = {'render_modes': []}

    def __init__(self, randomize_reset=False, seed=42, energy_weight=ENERGY_PER_STEP,
                 shaping_low_soc=True, alert_clears_block=False, alert_clear_steps=40,
                 hazard=HAZARD, map_seed=MAP_SEED):
        super().__init__()
        self.randomize_reset = randomize_reset
        self._base_seed = seed
        self.energy_weight = energy_weight
        self.shaping_low_soc = shaping_low_soc
        self.alert_clears_block = alert_clears_block
        self.alert_clear_steps = alert_clear_steps
        self.hazard = hazard
        self.map_seed = map_seed

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self._ep_counter = 0
        self.world = None

    # -- warm-up + conditioning cache ---------------------------------------------------------
    def _fill_window(self):
        """Roll `WINDOW` steps of action 0 to prime the buffers, exactly as the offline generator."""
        self._tb = deque(maxlen=WINDOW)
        self._lb = deque(maxlen=WINDOW)
        self._sb = deque(maxlen=WINDOW)
        info = None
        for _ in range(WINDOW):
            r = self.world.step(0)
            self._tb.append([r['torque_0'], r['torque_1'], r['torque_2'], r['torque_3']])
            self._lb.append(r['lidar_distance'])
            self._sb.append(r['battery_soc'])
            info = r['info']
        self._last_label = int(r['anomaly_label'])
        self._cache_conditioning(info)
        return extract_state(self._tb, self._lb, self._sb, info)

    def _cache_conditioning(self, info):
        """Cache the reward-conditioning variables for the CURRENT state (used on the next step)."""
        state = extract_state(self._tb, self._lb, self._sb, info)
        self._halted = bool(info['halted'])
        self._main_blocked = state[STATE_COLS.index('next_main_block_dist')] < NEAR
        self._branch_blocked = state[STATE_COLS.index('branch_block_dist')] < NEAR
        self._rough = state[STATE_COLS.index('torque_mean')] > ROUGH_TERRAIN_TORQUE
        self._soc = state[STATE_COLS.index('battery_soc')]

    def _maybe_clear_block(self):
        """alert_clears_block wrapper surgery on world.blockages (ON variant only)."""
        if not self.world.blockages:
            return
        edge = (self.world.node, self.world.target)
        if edge in self.world.blockages:
            del self.world.blockages[edge]

    # -- gym API ------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_reset:
            world_seed = self._base_seed + 1_000 + self._ep_counter
            init_soc = float(np.random.default_rng(world_seed + 555).uniform(*SOC_INIT_RANGE))
        else:
            world_seed = self._base_seed if seed is None else seed
            init_soc = 100.0
        self._ep_counter += 1

        self.world = RoverWorld(hazard_intensity=self.hazard, seed=world_seed,
                                total_steps=FULL_LOOP_STEPS, map_seed=self.map_seed, blockages=True)
        self.world.soc = init_soc
        self._stuck_ctr = 0
        self._alert_ctr = 0
        obs = self._fill_window().astype(np.float32)
        return obs, {'label': self._last_label, 'world_seed': world_seed}

    def step(self, action):
        action = int(action)
        # (1) reward from the CURRENT state's cached conditioning, BEFORE the world advances
        reward = compute_reward(self._last_label, action, self._halted, self._main_blocked,
                                self._rough, self._soc, energy_per_step=self.energy_weight,
                                shaping_low_soc=self.shaping_low_soc)

        # (2) return-to-base ends the mission; alert-clears-block bookkeeping
        rtb = (action == 4)
        if self.alert_clears_block:
            self._alert_ctr = self._alert_ctr + 1 if action == 3 else 0
            if self._alert_ctr >= self.alert_clear_steps:
                self._maybe_clear_block()
                self._alert_ctr = 0

        # (3) advance the world one step under the chosen action
        r = self.world.step(action)
        self._tb.append([r['torque_0'], r['torque_1'], r['torque_2'], r['torque_3']])
        self._lb.append(r['lidar_distance'])
        self._sb.append(r['battery_soc'])
        info = r['info']
        self._last_label = int(r['anomaly_label'])
        self._cache_conditioning(info)
        obs = extract_state(self._tb, self._lb, self._sb, info).astype(np.float32)

        # (4) stuck-at-full-block timeout (both branches blocked and halted)
        if self._halted and self._main_blocked and self._branch_blocked:
            self._stuck_ctr += 1
        else:
            self._stuck_ctr = 0

        soc_depleted = self._soc <= 0.0
        stuck_out = self._stuck_ctr >= STUCK_TIMEOUT
        terminated = bool(rtb or soc_depleted or stuck_out)
        info_out = {'label': self._last_label, 'action_name': ACTION_NAMES[action],
                    'halted': self._halted, 'terminated_reason':
                    ('return_to_base' if rtb else 'soc_depleted' if soc_depleted
                     else 'stuck_timeout' if stuck_out else None)}
        return obs, float(reward), terminated, False, info_out
