"""Node-level options (semi-MDP) wrapper over the flat rover patrol MDP.

Why decisions move to the nodes
-------------------------------
The flat env decides at 10 Hz, but the only decision that changes the route is the one taken on
the single step that crosses a node: `RoverWorld.step` reads `action == 2` at the crossing and
switches `target` to `reroute[node]` there and nowhere else. A main sweep edge is 160 m ~ 1,600
environment steps, so the flat agent emits ~1,600 actions per edge and exactly one of them is
about routing. With gamma = 0.99 the effective horizon is 1/(1 - gamma) = 100 steps, while the
cost of driving into a blocked dead end lands 400-1,200 steps past the branch node (blockages are
placed 25-75 % along the 160 m edge) and so reaches the decision discounted by 0.99^400..0.99^1200
= 1.8e-2..6e-6. It cannot compete with the +1.0/step paid for continuing, and the Week-5 DQN
reroutes on 3 % of single-blockage encounters. Deciding at route nodes leaves the reward, the
observation and the dynamics untouched and changes only the decision rate, so the blockage is one
decision away instead of ~1,000 steps away.

Reward, and what "exact SMDP" does and does not fix
---------------------------------------------------
An option that runs k environment steps collecting r_0..r_{k-1} returns sum_t gamma^t r_t, and the
discount owed to the next option's value is gamma^k (`info['gamma_k']`, `info['option_k']`). That
is standard SMDP accumulation, and being exact about it preserves the flat objective rather than
repairing it: gamma^k for k ~ 1,600 is 1e-7, so an SMDP learner at gamma = 0.99 optimises the same
criterion the flat agent already optimises and inherits the same indifference to a cost 1,000 steps
out. The intra-option sum saturates for the same reason -- a clean 1,600-step edge is worth
~0.95/(1 - gamma) = +95 and an edge that halts 400 steps in is worth ~+92, a 3-point gap where the
undiscounted gap is ~1,500 points. Options buy credit assignment (one backup per edge instead of
1,600), not horizon.

Lengthening the horizon is a separate knob, and there are two honest settings:

* ``gamma = 1.0`` with the episode capped at `max_env_steps` -- finite-horizon undiscounted, which
  is the criterion the "the detour is worth ~95 points" statement is actually made in. `gamma_k`
  is then 1.0 and the SMDP accumulation is a plain sum, still exact.
* a per-decision ``gamma_hi`` applied at option granularity, ignoring k. This is an approximation
  (it treats a 533-step connector and a 1,600-step sweep as equally long) but it is the only thing
  a fixed-gamma learner such as SB3 PPO can consume, and 1/(1 - 0.99) = 100 *decisions* is ~100
  edges of horizon. `python -m rl.harness.train --kind options --learner ppo` takes this path; the
  default `smdp` learner applies gamma^k exactly and needs `--gamma 1.0` to see past one edge.

Options
-------
0 `patrol-edge`   drive the current edge to the next node under flat action 0
1 `reroute-branch` same, plus flat action 2 on the crossing step so the world takes `reroute[]`
2 `slow-traverse` same, at half speed (flat action 1) for the whole edge
3 `dock`          flat action 4 once; the episode ends

All three driving options share one controller: raise-alert (flat 3) while the inner env's cached
`_last_label` is 1, slow (flat 1) while `_rough` is set, continue otherwise. Keeping the alert rule
identical across options leaves routing and speed as the only difference between them.

An option ends on the node crossing, so `reroute-branch` chosen at node p takes the branch at the
node q that ENDS the edge -- the decision runs one edge ahead of the node it affects. The blockage
it is deciding about therefore sits 93-173 m away at decision time (the 53 m connector into q plus
the 40-120 m the world places the blockage into the 160 m sweep edge past q) instead of the
40-120 m a flat agent sees standing on q, and `next_main_block_dist` saturates at the 200 m LiDAR
lookahead, so the rule threshold of 150 m resolves ~70 % of the placements. Nothing here depends on
the 200 m cap being generous: the branch node reachable only across a 160 m edge is node 0, and the
world never blocks the edge the rover starts committed to.

Limitations to carry into any result
------------------------------------
The controller reads `_last_label`, which is privileged state -- it is not a function of the 9-D
observation, so per-option alert precision is an artefact of the controller and not a learned
quantity. More broadly the option controllers embed the hand-written rule policy's competence, so a
good result here shows that the horizon was the binding constraint; it does not show that RL
discovered navigation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from shared_modules.rover_world import DT, V_CRUISE
from rl.rover_env import RoverPatrolEnv, LOW_SOC, NEAR, ROUGH_TERRAIN_TORQUE, WINDOW
from shared_modules.rl_eval import GAMMA, OBS_MEAN, OBS_STD, EVENT_KEYS, normalize

OPT_PATROL, OPT_REROUTE, OPT_SLOW, OPT_DOCK = 0, 1, 2, 3
OPTION_NAMES = {OPT_PATROL: 'patrol-edge', OPT_REROUTE: 'reroute-branch',
                OPT_SLOW: 'slow-traverse', OPT_DOCK: 'dock'}

# Guard on a halted rover that will never reach its node. It has to clear the longest legitimate
# traverse: `slow-traverse` over a 160 m sweep edge is 3,200 steps at 0.05 m/step. In practice it
# is a safety net that does not fire, because a rover that halts at a blockage is ended first by
# the inner env's 80-step stuck timeout: `RoverWorld._dist_to_blockage` starts its walk from the
# edge already committed to, so once the rover is on a blocked edge the main and branch lookaheads
# report the same blockage and the "both blocked" stuck condition holds even for a single block.
MAX_OPTION_STEPS = 4_000

# A step advances at most V_CRUISE * DT = 0.1 m (faults only ever reduce it), so any step that will
# cross the node satisfies this test before it executes.
CROSS_EPS = V_CRUISE * DT


def about_to_cross(world):
    """Whether the next world step consumes the node, which is where a reroute flag is read.

    The 1e-9 slack covers the rounding accumulated over the ~1,600 `dist_into` additions of a
    160 m sweep edge: without it the crossing step can land one ulp beyond the threshold and the
    reroute is never emitted.
    """
    return (world.seg_len - world.dist_into) <= CROSS_EPS + 1e-9


class RoverOptionsEnv(gym.Env):
    """One `step()` is one option execution on the flat `RoverPatrolEnv` it owns.

    Parameters
    ----------
    randomize_reset, seed, **env_kw
        Forwarded to `RoverPatrolEnv` (`energy_weight`, `low_soc_penalty`, `soc_init_range`,
        `map_seed`, `hazard`, ...), so reset semantics are the flat env's: fresh world seed and
        SoC ~ U(soc_init_range) per episode when True, fixed seed and full battery when False.
    gamma : intra-option discount used for both the returned reward and `info['gamma_k']`.
    max_env_steps : episode truncation in ENVIRONMENT steps, not option decisions. Gymnasium's
        `TimeLimit` counts `step()` calls, which here are options, so the cap lives in the env.
    """

    metadata = {'render_modes': []}

    def __init__(self, randomize_reset=False, seed=42, norm=True, gamma=GAMMA,
                 max_option_steps=MAX_OPTION_STEPS, max_env_steps=None, **env_kw):
        super().__init__()
        self.env = RoverPatrolEnv(randomize_reset=randomize_reset, seed=seed, **env_kw)
        self.norm = norm
        self.gamma = gamma
        self.max_option_steps = max_option_steps
        self.max_env_steps = max_env_steps

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.raw_obs = None
        self.total_env_steps = 0          # cumulative across episodes; the training budget counter
        self._n_env = 0                   # within the current episode

    # -- helpers ------------------------------------------------------------------------------
    def _obs(self, raw):
        self.raw_obs = np.asarray(raw, np.float64)
        return normalize(raw) if self.norm else np.asarray(raw, np.float32)

    def _low_level(self, option):
        """Flat action for the current environment step under the running option."""
        if option == OPT_DOCK:
            return 4
        if option == OPT_REROUTE and about_to_cross(self.env.world):
            return 2
        if self.env._last_label == 1:
            return 3
        if option == OPT_SLOW or self.env._rough:
            return 1
        return 0

    # -- gym API ------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed)
        # the flat reset rolls WINDOW steps of action 0 to prime the feature buffers; they are real
        # environment steps and are charged to the budget so an episode can never cost ~nothing
        self.total_env_steps += WINDOW
        self._n_env = 0
        return self._obs(obs), dict(world_seed=info['world_seed'], label=info['label'],
                                    node=self.env.world.node)

    def step(self, option):
        option = int(option)
        node0 = self.env.world.node
        ev = {k: 0 for k in EVENT_KEYS}
        k, raw_sum, disc, g = 0, 0.0, 0.0, 1.0
        terminated = truncated = guard_hit = False
        reason = None

        while True:
            base = self.env
            label, soc = base._last_label, base._soc
            main_b, branch_b, rough = base._main_blocked, base._branch_blocked, base._rough
            a = self._low_level(option)

            # per-environment-step event counts, same keys and same pre-step conditioning as
            # `shared_modules.rl_eval.rollout`, so option-level rates are comparable with the
            # flat table
            if label == 1:
                ev['anomaly'] += 1
                ev['alert_on_anomaly'] += (a == 3)
            else:
                ev['normal'] += 1
                ev['alert_on_normal'] += (a == 3)
            if main_b and not branch_b:
                ev['single_block'] += 1
                ev['reroute_on_block'] += (option == OPT_REROUTE)
            if rough:
                ev['rough'] += 1
                ev['slow_on_rough'] += (a == 1)
            if soc < LOW_SOC:
                ev['low_soc'] += 1
                ev['dock'] += (option == OPT_DOCK)

            obs, r, term, _, step_info = self.env.step(a)
            k += 1
            self._n_env += 1
            self.total_env_steps += 1
            raw_sum += r
            disc += g * r
            g *= self.gamma

            if term:
                terminated, reason = True, step_info['terminated_reason']
                break
            if self.env.world.node != node0:
                break
            if self.max_env_steps is not None and self._n_env >= self.max_env_steps:
                truncated, reason = True, 'env_step_cap'
                break
            if k >= self.max_option_steps:
                guard_hit, truncated, reason = True, True, 'option_guard'
                break

        info = dict(option=option, option_name=OPTION_NAMES[option], option_k=k,
                    gamma_k=float(self.gamma ** k), reward_sum=float(raw_sum),
                    n_env_steps=self._n_env, total_env_steps=self.total_env_steps,
                    guard_hit=guard_hit, terminated_reason=reason,
                    prev_node=node0, node=self.env.world.node, events=ev)
        return self._obs(obs), float(disc), terminated, truncated, info

    def close(self):
        self.env.close()


def scripted_option_policy(obs_raw):
    """Rule policy at option granularity -- `shared_modules.rl_eval.scripted_blind` with the same
    thresholds, re-expressed over the four options. Reroute is the choice when the main route ahead
    is blocked and the branch is clear; with a full block there is nothing to route around, so it
    keeps patrolling and lets the inner env's stuck timeout end the episode."""
    o = np.asarray(obs_raw, float)
    torque_mean, soc = o[0], o[5]
    main_d, branch_d = o[7], o[8]
    if soc < LOW_SOC:
        return OPT_DOCK
    if main_d < NEAR and branch_d >= NEAR:
        return OPT_REROUTE
    if torque_mean > ROUGH_TERRAIN_TORQUE:
        return OPT_SLOW
    return OPT_PATROL


def make_scripted_option_predict(norm=True):
    if not norm:
        return scripted_option_policy
    return lambda obs: scripted_option_policy(np.asarray(obs, float) * OBS_STD + OBS_MEAN)
