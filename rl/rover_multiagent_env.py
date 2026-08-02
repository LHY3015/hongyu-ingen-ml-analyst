"""PettingZoo ParallelEnv for two cooperative patrol rovers sharing one Aido Rover site (Week 6).

Two `shared_modules.rover_world.RoverWorld` instances are built on the same `map_seed` and the
same per-episode world seed, so terrain, static obstacles and the temporary-blockage layout are
one physical site seen by both rovers. Each rover is handed to the other as a
`dynamic_obstacles` circle on every 10 Hz tick, which is the world core's only multi-agent hook:
the partner enters the LiDAR ray-cast and, through the same `geo`, the GPS-multipath term.

That hook is perception only, and three gaps have to be closed in this wrapper:

* the core sets `halted` from `self.blockages` alone, so two rovers pass through each other with
  no consequence — proximity is charged as a reward term here rather than as new dynamics, since
  the core is frozen and shared with the Week-2 offline pipelines;
* LiDAR reaches 200 m while the main loop is 960 m, so mutual visibility is a *local* signal.
  Cooperative coverage needs the partner's global position, so the observation carries an
  explicit partner block on top of the LiDAR channel;
* the core has no coverage state at all, so the shared objective (which loop edges have been
  patrolled, and how recently) lives here.

**Decision granularity is node-level.** One `step()` is one *option* per rover, executed inline
as a run of 10 Hz world steps. Coordination on this site is a routing/sector problem — which
rover takes which half of the loop, who detours around a blockage — and every decision of that
kind is made at a node. Running the policy at 10 Hz would multiply training cost by ~1,200 with
no extra coordination content. The mutual `dynamic_obstacles` exchange still happens on every
10 Hz tick *inside* option execution, so the Week-2 design mechanism is realised, not discarded.

**Synchronisation.** A `step()` returns only when *both* rovers' options have terminated; a rover
that finishes early keeps driving under flat action 0 until its partner finishes. The alternative
— return the moment either option ends and resume the partner's partial option on the next call —
gives tighter control but discards the action sampled for the resuming agent, which breaks the
log-prob/advantage attribution that independent PPO learners rely on. The cost of the choice made
here is that the early finisher travels uncontrolled for the remainder of the macro-step, and may
cross further nodes: pairing a 533-step vertical edge against a 3,200-step slow traverse of a
160 m sweep leaves the fast rover 267 m — two uncommanded node crossings — of autopilot.

Option -> flat-action mapping (flat actions are the Week-5 MDP actions in `rl/rover_env.py`):

| option | name           | flat actions emitted                                          |
| ------ | -------------- | ------------------------------------------------------------- |
| 0      | patrol-edge    | 0 until the next node crossing                                |
| 1      | reroute-branch | 0, then 2 once the crossing is within one cruise step         |
| 2      | slow-traverse  | 1 until the next node crossing                                |
| 3      | dock           | 4 once; the agent terminates                                  |

`reroute-branch` emits flat 2 only on the crossing step because the core reads the reroute flag
exactly when it consumes a node (`target = reroute[node]`); holding action 2 for the whole edge
would be behaviourally identical but would charge the -0.3 reroute cost on every 10 Hz step.
"""

from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from shared_modules.rover_world import RoverWorld
from rl.rover_env import (WINDOW, MAP_SEED, HAZARD, NEAR, LOW_SOC, ROUGH_TERRAIN_TORQUE,
                          STUCK_TIMEOUT, FULL_LOOP_STEPS, STATE_COLS, extract_state,
                          compute_reward)
from rl.rover_options_env import about_to_cross
from shared_modules.rl_eval import normalize, TRAIN_CAP, FULL_LOOP, EVENT_KEYS

ROVER_RADIUS = 0.6              # m, LiDAR circle each rover presents to the other
MAX_OPTION_STEPS = 3_500        # stall guard: 160 m (longest main edge) at 0.5 m/s = 3,200 steps
OPTION_NAMES = {0: 'patrol-edge', 1: 'reroute-branch', 2: 'slow-traverse', 3: 'dock'}
AGENTS = ['rover_0', 'rover_1']

I_RP = STATE_COLS.index('route_progress')
I_SOC = STATE_COLS.index('battery_soc')
I_TORQUE = STATE_COLS.index('torque_mean')
I_MAIN = STATE_COLS.index('next_main_block_dist')
I_BRANCH = STATE_COLS.index('branch_block_dist')


class _RoverAgent:
    """One rover: its world, the rolling window buffers, and the cached reward conditioning.

    Mirrors the buffer/conditioning half of `rl.rover_env.RoverPatrolEnv` so the flat reward and
    the 9-D state are computed by the same code path as the single-agent runs.
    """

    def __init__(self, world, start_node, start_arc):
        self.world = world
        _place(world, start_node)
        # route_progress is (cum_dist % loop) / loop, so a rover placed part-way round the loop
        # must start with the matching arc length already banked or its progress -- and hence the
        # partner block below -- would read as if it were at node 0.
        world.cum_dist = start_arc
        self.start_arc = start_arc
        self.terminated = False
        self.term_reason = None
        self.halted_steps = 0
        self.stuck_ctr = 0
        self.anomaly_steps = 0
        self.events = {k: 0 for k in EVENT_KEYS}

    def warm_start(self):
        self.tb, self.lb, self.sb = (deque(maxlen=WINDOW) for _ in range(3))

    def push(self, r):
        self.tb.append([r['torque_0'], r['torque_1'], r['torque_2'], r['torque_3']])
        self.lb.append(r['lidar_distance'])
        self.sb.append(r['battery_soc'])
        self.info = r['info']
        self.label = int(r['anomaly_label'])
        self.anomaly_steps += self.label
        self.halted_steps += bool(self.info['halted'])

    def cache(self):
        s = extract_state(self.tb, self.lb, self.sb, self.info)
        self.state = s
        self.halted = bool(self.info['halted'])
        self.main_blocked = s[I_MAIN] < NEAR
        self.branch_blocked = s[I_BRANCH] < NEAR
        self.rough = s[I_TORQUE] > ROUGH_TERRAIN_TORQUE
        self.soc = s[I_SOC]

    @property
    def dist_m(self):
        return self.world.cum_dist - self.start_arc

    def observe(self, partner):
        """9-D own state (normalised) ++ [partner progress, partner SoC/100, signed loop gap]."""
        gap = float(partner.state[I_RP] - self.state[I_RP])
        gap = (gap + 0.5) % 1.0 - 0.5                      # shortest signed way round the loop
        tail = np.array([partner.state[I_RP], partner.soc / 100.0, gap], dtype=np.float32)
        return np.concatenate([normalize(self.state), tail]).astype(np.float32)


def _place(world, node):
    """Move a freshly constructed world to `node` (post-construction reseat, pose recomputed)."""
    world.node = node
    world.target = world.m['route'][node]
    world.dist_into = 0.0
    world.seg_len = world._elen(node, world.target)
    world._update_pose()


def _main_cycle(world):
    """Main-route edge list and the arc length of each main node, walking `route` from node 0."""
    edges, arcs, n, cum = [], {0: 0.0}, 0, 0.0
    while True:
        nx = world.m['route'][n]
        edges.append((n, nx))
        cum += world._elen(n, nx)
        n = nx
        if n == 0:
            break
        arcs[n] = cum
    return edges, arcs


class RoverMultiAgentEnv(ParallelEnv):
    """Two-rover cooperative patrol over the Week-5 rover MDP, at node-level decision granularity.

    Parameters
    ----------
    reward_mode : {'shared', 'difference'}
        `'shared'` gives both rovers the identical team reward — the credit-assignment setting,
        in which a rover cannot tell which part of the reward it caused. `'difference'` gives
        each rover the team reward minus the team reward that would have accrued without its own
        contribution (coverage traversals and its half of the proximity pair removed).
    coverage_weight, redundant_penalty, stale_threshold
        A main-loop edge traversal scores `+coverage_weight` if the edge has not been patrolled
        for more than `stale_threshold` environment steps, and `-redundant_penalty` otherwise.
        At cruise a rover cannot return to its own edge inside 9,600 steps, so with the default
        2,000-step threshold a redundancy charge is always the partner's recent pass.
    proximity_penalty, proximity_m
        Charged per 10 Hz step on which the two rovers are within `proximity_m` of each other.
        The world core has no inter-rover collision, so this is what makes bunching costly.
    randomize_reset
        Training mode: each reset draws a fresh world seed (blockage layout + fault stream).
    max_env_steps
        Shared truncation horizon in 10 Hz steps; defaults to 2,400 in training mode and 9,600
        (one full main loop) otherwise.
    start_nodes
        Main-route nodes the two rovers start from. `None` puts rover_0 at node 0 and rover_1 at
        the main node closest to half a loop away, so the pair does not start superimposed.
    """

    metadata = {'render_modes': [], 'name': 'rover_multiagent_v0'}

    def __init__(self, reward_mode='shared', coverage_weight=5.0, redundant_penalty=2.0,
                 stale_threshold=2000, proximity_penalty=0.02, proximity_m=5.0,
                 randomize_reset=False, seed=42, max_env_steps=None, start_nodes=None,
                 max_option_steps=MAX_OPTION_STEPS, hazard=HAZARD, map_seed=MAP_SEED,
                 init_soc=100.0):
        if reward_mode not in ('shared', 'difference'):
            raise ValueError(f'unknown reward_mode {reward_mode!r}')
        self.reward_mode = reward_mode
        self.coverage_weight = coverage_weight
        self.redundant_penalty = redundant_penalty
        self.stale_threshold = stale_threshold
        self.proximity_penalty = proximity_penalty
        self.proximity_m = proximity_m
        self.randomize_reset = randomize_reset
        self._base_seed = seed
        self.max_env_steps = max_env_steps or (TRAIN_CAP if randomize_reset else FULL_LOOP)
        self.max_option_steps = max_option_steps
        self.hazard = hazard
        self.map_seed = map_seed
        self.init_soc = init_soc
        self._start_nodes = start_nodes

        self.possible_agents = list(AGENTS)
        self.agents = []
        self._obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self._act_space = spaces.Discrete(4)
        self._ep = 0
        self.ag = {}

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    # -- construction ---------------------------------------------------------------------
    def _new_world(self, world_seed):
        w = RoverWorld(hazard_intensity=self.hazard, seed=world_seed,
                       total_steps=FULL_LOOP_STEPS, map_seed=self.map_seed, blockages=True)
        w.soc = self.init_soc
        return w

    def reset(self, seed=None, options=None):
        if seed is not None:
            world_seed = int(seed)
        elif self.randomize_reset:
            world_seed = self._base_seed + 1_000 + self._ep
        else:
            world_seed = self._base_seed
        self._ep += 1
        self.world_seed = world_seed

        worlds = [self._new_world(world_seed) for _ in AGENTS]
        self.main_edges, arcs = _main_cycle(worlds[0])
        if self._start_nodes is None:
            half = worlds[0].main_loop_len / 2
            far = min(arcs, key=lambda n: abs(arcs[n] - half))
            starts = [0, far]
        else:
            starts = list(self._start_nodes)

        # The core refuses to block the edge it starts committed to, because a mid-edge blockage
        # cannot be rerouted around once entered. Rover_1 starts on a different edge, so extend
        # that rule to both start edges -- and to both worlds, so the two rovers keep seeing the
        # same physical blockage set.
        drop = {(s, worlds[0].m['route'][s]) for s in starts}
        for w in worlds:
            for e in drop:
                w.blockages.pop(e, None)

        self.ag = {aid: _RoverAgent(w, s, arcs[s])
                   for aid, w, s in zip(AGENTS, worlds, starts)}
        self.agents = list(AGENTS)
        self.env_step = 0
        self.coverage = {e: -10 ** 9 for e in self.main_edges}
        self.covered = set()
        self.redundant_edges = 0
        self.proximity_steps = 0

        for a in self.ag.values():
            a.warm_start()
        for _ in range(WINDOW):
            self._tick({aid: 0 for aid in AGENTS})
        for a in self.ag.values():
            a.cache()
        self.env_step = 0                      # horizon counts decision-phase steps only
        self.proximity_steps = 0

        obs = {aid: self.ag[aid].observe(self.ag[self._partner(aid)]) for aid in self.agents}
        infos = {aid: self._agent_info(aid, None, 0) for aid in self.agents}
        return obs, infos

    def _partner(self, aid):
        return AGENTS[1 - AGENTS.index(aid)]

    def _disable_exchange(self):
        """Run without the partner circle, for the paired visibility probe below."""
        self._no_exchange = True

    # -- 10 Hz execution ------------------------------------------------------------------
    def _tick(self, flat_actions):
        """Advance every live world one 10 Hz step, exchanging positions as dynamic obstacles.

        Poses are snapshotted before the tick so the exchange is symmetric: neither rover sees the
        other's post-step position. A rover that has docked leaves the loop and stops being an
        obstacle, otherwise it would sit as a phantom return on its partner's LiDAR for the rest
        of the episode.
        """
        pos = {aid: (a.world.x, a.world.y) for aid, a in self.ag.items() if not a.terminated}
        crossings = []
        for aid, flat in flat_actions.items():
            a = self.ag[aid]
            if a.terminated:
                continue
            dyn = ([] if getattr(self, '_no_exchange', False) else
                   [(x, y, ROVER_RADIUS) for o, (x, y) in pos.items() if o != aid])
            prev_node = a.world.node
            a.push(a.world.step(flat, dynamic_obstacles=dyn))
            if a.world.node != prev_node:
                crossings.append((aid, (prev_node, a.world.node)))
        self.env_step += 1
        live = [a for a in self.ag.values() if not a.terminated]
        if len(live) == 2:
            d = np.hypot(live[0].world.x - live[1].world.x, live[0].world.y - live[1].world.y)
            if d < self.proximity_m:
                self.proximity_steps += 1
                return crossings, True
        return crossings, False

    def _flat_action(self, a, opt):
        if opt == 1:
            # the core consumes the reroute flag on the step that crosses the node
            return 2 if about_to_cross(a.world) else 0
        return {0: 0, 2: 1, 3: 4}[opt]

    # -- coverage bookkeeping -------------------------------------------------------------
    def _score_coverage(self, traversals, cov):
        """Apply `(env_step, edge)` traversals to a coverage map; returns (reward, n_redundant)."""
        r, redundant = 0.0, 0
        for t, edge in traversals:
            if t - cov[edge] > self.stale_threshold:
                r += self.coverage_weight
            else:
                r -= self.redundant_penalty
                redundant += 1
            cov[edge] = t
        return r, redundant

    # -- parallel API ---------------------------------------------------------------------
    def step(self, actions):
        acting = list(self.agents)
        opts = {aid: int(actions[aid]) for aid in acting}
        for aid in acting:
            self._count_event(aid, opts[aid])

        ctrl = {aid: {'done': False, 'n': 0} for aid in acting}
        flat_ret = {aid: 0.0 for aid in acting}
        traversals = []                       # (env_step, edge) in time order, all agents
        by_agent = {aid: [] for aid in acting}
        prox_steps = 0
        truncated_now = False

        while True:
            flat = {}
            for aid in acting:
                a = self.ag[aid]
                if a.terminated:
                    continue
                f = 0 if ctrl[aid]['done'] else self._flat_action(a, opts[aid])
                # reward conditions on the state the action is taken FROM (Week-5 timing)
                flat_ret[aid] += compute_reward(a.label, f, a.halted, a.main_blocked,
                                                a.rough, a.soc)
                flat[aid] = f
                ctrl[aid]['n'] += 1

            crossings, close = self._tick(flat)
            prox_steps += close
            for aid, edge in crossings:
                if edge in self.coverage:
                    traversals.append((self.env_step, edge))
                    by_agent[aid].append((self.env_step, edge))
                ctrl[aid]['done'] = True      # option ends at the node it was driving toward

            for aid in acting:
                a = self.ag[aid]
                if a.terminated:
                    continue
                a.cache()
                # same stuck-at-full-block condition and timeout as the flat env: a rover
                # committed onto a blocked edge cannot reroute out of it, and would otherwise
                # stand still paying -0.5/step to the horizon
                a.stuck_ctr = (a.stuck_ctr + 1
                               if (a.halted and a.main_blocked and a.branch_blocked) else 0)
                if flat.get(aid) == 4:
                    a.terminated, a.term_reason, ctrl[aid]['done'] = True, 'dock', True
                elif a.soc <= 0.0:
                    a.terminated, a.term_reason, ctrl[aid]['done'] = True, 'soc_depleted', True
                elif a.stuck_ctr >= STUCK_TIMEOUT:
                    a.terminated, a.term_reason, ctrl[aid]['done'] = True, 'stuck_timeout', True
                elif ctrl[aid]['n'] >= self.max_option_steps:
                    ctrl[aid]['done'] = True

            if self.env_step >= self.max_env_steps:
                truncated_now = True
                break
            if all(ctrl[aid]['done'] or self.ag[aid].terminated for aid in acting):
                break

        # coverage is applied once, from the real joint pass; the counterfactuals below replay a
        # snapshot so the episode's coverage state stays single-valued
        snapshot = dict(self.coverage)
        cov_r, n_red = self._score_coverage(traversals, self.coverage)
        self.redundant_edges += n_red
        self.covered.update(e for _, e in traversals)
        prox_r = -self.proximity_penalty * prox_steps
        team = sum(flat_ret.values()) + cov_r + prox_r

        if self.reward_mode == 'shared':
            rewards = {aid: team for aid in acting}
        else:
            rewards = {}
            for aid in acting:
                other = [o for o in acting if o != aid]
                cf_cov, _ = self._score_coverage(
                    sorted(t for o in other for t in by_agent[o]), dict(snapshot))
                cf_team = sum(flat_ret[o] for o in other) + cf_cov   # no pair left -> no proximity
                rewards[aid] = team - cf_team

        term = {aid: self.ag[aid].terminated for aid in acting}
        trunc = {aid: bool(truncated_now and not term[aid]) for aid in acting}
        obs = {aid: self.ag[aid].observe(self.ag[self._partner(aid)]) for aid in acting}
        infos = {aid: self._agent_info(aid, opts[aid], ctrl[aid]['n'],
                                       team=team, cov_r=cov_r, prox_r=prox_r,
                                       prox_steps=prox_steps) for aid in acting}
        self.agents = [aid for aid in acting if not (term[aid] or trunc[aid])]
        return obs, rewards, term, trunc, infos

    # -- reporting ------------------------------------------------------------------------
    def _count_event(self, aid, opt):
        """Decision-level event-conditioned counters, mirroring `shared_modules.rl_eval.rollout`.

        No option emits flat action 3, so the alert counters are structurally zero; they are kept
        so the dict pools straight into `shared_modules.rl_eval.event_rates`.
        """
        a, e = self.ag[aid], self.ag[aid].events
        if a.label == 1:
            e['anomaly'] += 1
        else:
            e['normal'] += 1
        if a.main_blocked and not a.branch_blocked:
            e['single_block'] += 1
            e['reroute_on_block'] += (opt == 1)
        if a.rough:
            e['rough'] += 1
            e['slow_on_rough'] += (opt == 2)
        if a.soc < LOW_SOC:
            e['low_soc'] += 1
            e['dock'] += (opt == 3)

    def _roster_entry(self, aid):
        a = self.ag[aid]
        return dict(dist_m=float(a.dist_m), soc=float(a.soc),
                    route_progress=float(a.state[I_RP]), halted_steps=int(a.halted_steps),
                    anomaly_steps=int(a.anomaly_steps), terminated_reason=a.term_reason,
                    events={k: int(v) for k, v in a.events.items()})

    def _agent_info(self, aid, opt, n_steps, team=0.0, cov_r=0.0, prox_r=0.0, prox_steps=0):
        # the roster carries every rover's episode totals, including one that has already docked
        # and been dropped from `self.agents` -- otherwise its distance and event counts would be
        # unrecoverable from the final step's info
        return dict(
            option=opt, option_name=None if opt is None else OPTION_NAMES[opt],
            option_env_steps=int(n_steps), **self._roster_entry(aid),
            team=dict(env_step=int(self.env_step), world_seed=int(self.world_seed),
                      n_main_edges=len(self.main_edges), covered_edges=len(self.covered),
                      coverage_frac=len(self.covered) / len(self.main_edges),
                      redundant_edges=int(self.redundant_edges),
                      proximity_steps=int(self.proximity_steps),
                      step_proximity_steps=int(prox_steps), team_reward=float(team),
                      coverage_term=float(cov_r), proximity_term=float(prox_r),
                      roster={a: self._roster_entry(a) for a in self.possible_agents}))

    def render(self):
        pass

    def close(self):
        pass


def mutual_visibility_probe(world_seed=0, n_steps=2400, start_nodes=(0, 0)):
    """Measure what the `dynamic_obstacles` exchange actually contributes to the LiDAR channel.

    The same episode is driven twice from the same seed with the same action sequence, once with
    each rover injected into the other's ray-cast and once without. `dynamic_obstacles` consumes no
    RNG draw, so the two runs are otherwise bit-identical and every difference is attributable to
    the partner.

    The measurement matters because `cast_min` returns the minimum over the 120 degree fan and the
    site boundary sits 20 m outside the route, so the no-partner geometric return is ~23 m rather
    than the 200 m sensor ceiling: the partner is only detectable once it is nearer than the
    ever-present wall return. There is also a blind spot below `ROVER_RADIUS`, where the ray origin
    lies inside the partner circle and the intersection is rejected.

    The separation is swept deliberately rather than left to chance: both rovers start at the same
    node, one driving at cruise and the other at half speed, so the gap opens from 0 to an edge
    length across the run. Sharing a world seed keeps the two RNG streams in lockstep, so rovers
    started identically and driven identically stay exactly superimposed and no range is sampled at
    all; the deployed configuration places them half a loop apart, where they never approach.

    Geometric return and measured LiDAR are recorded separately: the geometry is what the partner
    changes, the measurement is what a policy actually reads once AR(1) range noise and the
    max-range dropout have been applied.

    The channel recorded is the *following* rover's. `cast_min` sweeps a forward 120 degree fan, so
    the exchange is not symmetric: a rover behind its partner can see it, a rover ahead of its
    partner cannot see anything at all. Recording the leader's channel returns a detection rate of
    zero at every separation.
    """
    def run(exchange):
        env = RoverMultiAgentEnv(reward_mode='shared', max_env_steps=n_steps,
                                 start_nodes=list(start_nodes))
        if not exchange:
            env._disable_exchange()
        env.reset(seed=world_seed)
        lidar, geo, sep = [], [], []
        drive = {AGENTS[0]: 0, AGENTS[1]: 1}          # cruise against half speed
        while env.agents and env.env_step < n_steps:
            env._tick({aid: drive[aid] for aid in env.agents})
            a0, a1 = env.ag[AGENTS[0]], env.ag[AGENTS[1]]
            lidar.append(float(a1.lb[-1]))            # the follower's channel
            geo.append(float(a1.info['geo']))
            sep.append(float(np.hypot(a0.world.x - a1.world.x, a0.world.y - a1.world.y)))
        env.close()
        return np.array(lidar), np.array(geo), np.array(sep)

    on, geo_on, sep = run(True)
    off, geo_off, _ = run(False)
    k = min(len(on), len(off))
    on, off, geo_on, geo_off, sep = on[:k], off[:k], geo_on[:k], geo_off[:k], sep[:k]
    detected = geo_off - geo_on > 0.5

    bins = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 1e9)]
    summary = []
    for lo, hi in bins:
        m = (sep >= lo) & (sep < hi)
        if not m.any():
            continue
        summary.append(dict(separation=f'{lo}-{hi if hi < 1e9 else "inf"} m', steps=int(m.sum()),
                            p_detect_geometric=round(float(detected[m].mean()), 3),
                            geo_off=round(float(geo_off[m].mean()), 1),
                            geo_on=round(float(geo_on[m].mean()), 1),
                            lidar_on=round(float(on[m].mean()), 1),
                            mean_drop=round(float((geo_off[m] - geo_on[m]).mean()), 1)))
    return dict(steps=np.arange(k), lidar_on=on, lidar_off=off,
                geo_on=geo_on, geo_off=geo_off, separation=sep, summary=summary,
                no_partner_geometric=dict(mean=round(float(geo_off.mean()), 1),
                                          median=round(float(np.median(geo_off)), 1),
                                          max=round(float(geo_off.max()), 1)),
                no_partner_measured=dict(mean=round(float(off.mean()), 1),
                                         median=round(float(np.median(off)), 1),
                                         max=round(float(off.max()), 1)))
