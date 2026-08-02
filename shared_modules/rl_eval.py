"""Shared RL evaluation protocol, observation normalisation and reference policies.

Two levels of randomness are kept separate throughout Week 6:

* **train seed** — algorithm randomness (network init, exploration, minibatch order)
* **eval world seed** — environment randomness (fault stream and, via ``seed + 10007`` inside
  the world core, the per-episode blockage layout)

Week-5 results mixed the two, so every headline comparison here reports the mean over train
seeds evaluated on the same fixed set of `EVAL_WORLD_SEEDS`, and any "A beats B" claim is
backed by a paired Wilcoxon signed-rank test across those common worlds.

The observation is normalised by fixed statistics taken from the Week-2 offline transition
table rather than by a running `VecNormalize`: on-policy policy gradients need a scaled
observation (the raw 9-D vector mixes torque ~14-40 Nm with LiDAR ~200 m and SoC 0-100 %),
and fixed statistics keep behaviour-cloning features, the PPO/GRPO observation and the
saved checkpoints mutually compatible.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from rl.rover_env import (RoverPatrolEnv, ACTION_NAMES, STATE_COLS, NEAR, ROUGH_TERRAIN_TORQUE,
                          LOW_SOC, STUCK_TIMEOUT)

ROOT = Path(__file__).resolve().parents[1]
POL_DIR = ROOT / 'rl' / 'saved_policies'
ART_DIR = ROOT / 'rl' / 'w6_artifacts'
POL_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)

GAMMA = 0.99
TRAIN_CAP = 2_400              # training-episode truncation (Week-5 convention)
FULL_LOOP = 9_600              # deterministic full-loop evaluation horizon
W6_TIMESTEPS = 250_000         # common budget for the flat-env four-way comparison
# The first five are Week 5's evaluation worlds, so the Week-5 reference numbers are recoverable
# as a subset. The count is ten rather than five because a two-sided Wilcoxon signed-rank test on
# n paired worlds cannot return a p-value below 2/2^n: at n = 5 the floor is 0.0625, so no
# comparison can reach significance however large the effect. Ten worlds put the floor at 0.002.
EVAL_WORLD_SEEDS = [0, 1, 2, 3, 4, 101, 202, 303, 404, 505]
W5_EVAL_SEEDS = [0, 1, 2, 3, 4]
CROSS_MAP_SEEDS = [1, 2, 4, 5, 7]   # W02-validated layouts other than the canonical MAP_SEED=6
N_TRAIN_SEEDS = 5
CURVE_SEEDS = [101, 202, 303]      # mid-training learning-curve worlds (final eval uses all 10)

# Event-conditioned counters. Defined once here because the flat rollout, the options env and the
# multi-agent env all pool their counts through `event_rates`, which indexes this exact key list.
EVENT_KEYS = ['anomaly', 'alert_on_anomaly', 'normal', 'alert_on_normal', 'single_block',
              'reroute_on_block', 'rough', 'slow_on_rough', 'low_soc', 'dock']


# --- observation normalisation ------------------------------------------------------------
def _fit_obs_stats():
    df = pd.read_csv(ROOT / 'data' / 'rover_transitions.csv',
                     usecols=[f's_{c}' for c in STATE_COLS])
    x = df[[f's_{c}' for c in STATE_COLS]].to_numpy(np.float64)
    return x.mean(0), x.std(0) + 1e-6


_STATS_PATH = ART_DIR / 'obs_stats.npz'
if _STATS_PATH.exists():
    _d = np.load(_STATS_PATH)
    OBS_MEAN, OBS_STD = _d['mean'], _d['std']
else:
    OBS_MEAN, OBS_STD = _fit_obs_stats()
    np.savez(_STATS_PATH, mean=OBS_MEAN, std=OBS_STD)


class NormalizeObs(gym.ObservationWrapper):
    """Static affine rescaling of the 9-D observation; a bijection, so the MDP is unchanged."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,),
                                                dtype=np.float32)

    def observation(self, obs):
        return ((obs - OBS_MEAN) / OBS_STD).astype(np.float32)


def normalize(x):
    return ((np.asarray(x, np.float64) - OBS_MEAN) / OBS_STD).astype(np.float32)


# --- env construction ---------------------------------------------------------------------
def make_train_env(seed, norm=True, cap=TRAIN_CAP, **kw):
    env = RoverPatrolEnv(randomize_reset=True, seed=seed, **kw)
    env = TimeLimit(env, max_episode_steps=cap)
    return NormalizeObs(env) if norm else env


def make_eval_env(world_seed, norm=True, horizon=FULL_LOOP, **kw):
    env = RoverPatrolEnv(randomize_reset=False, seed=world_seed, **kw)
    env = TimeLimit(env, max_episode_steps=horizon)
    return NormalizeObs(env) if norm else env


# --- evaluation ---------------------------------------------------------------------------
def _obs_is_normalized(env):
    """Whether the wrapper chain rescales the observation, so a raw copy can be recovered."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, NormalizeObs):
            return True
        env = env.env
    return False


TRACE_KEYS = ['obs_raw', 'obs_norm', 'action', 'reward', 'label', 'halted', 'main_blocked',
              'branch_blocked', 'rough', 'soc']


def rollout(predict, env, gamma=None, env_aware=False, record=False):
    """Run one deterministic episode. Returns total return, length and event-conditioned counts.

    `env_aware` calls `predict(obs, base)` instead of `predict(obs)`, which is how the privileged
    label-aware expert reference reads the ground-truth anomaly label the observation withholds.

    `record` additionally returns `(stats, trace)`, a dict of per-step arrays: the observation in
    both frames (raw physical units and the rescaling the policy is actually fed), the action, the
    reward, and the state flags the event counters condition on. Input-gradient and value-landscape
    work needs the per-step trajectory, and re-implementing this loop is how the two drift apart.
    """
    obs, info = env.reset()
    if hasattr(predict, 'reset'):
        predict.reset()          # the scripted references carry a per-episode stuck counter
    base = env.unwrapped
    total, disc, n, g = 0.0, 0.0, 0, 1.0
    acts = np.zeros(5, int)
    ev = {k: 0 for k in EVENT_KEYS}
    tr = {k: [] for k in TRACE_KEYS}
    normed = _obs_is_normalized(env)
    while True:
        label = base._last_label
        main_b, branch_b, rough = base._main_blocked, base._branch_blocked, base._rough
        soc = base._soc
        a = int(predict(obs, base) if env_aware else predict(obs))
        acts[a] += 1
        if label == 1:
            ev['anomaly'] += 1
            ev['alert_on_anomaly'] += (a == 3)
        else:
            ev['normal'] += 1
            ev['alert_on_normal'] += (a == 3)
        if main_b and not branch_b:
            ev['single_block'] += 1
            ev['reroute_on_block'] += (a == 2)
        if rough:
            ev['rough'] += 1
            ev['slow_on_rough'] += (a == 1)
        if soc < LOW_SOC:
            ev['low_soc'] += 1
            ev['dock'] += (a == 4)
        if record:
            o = np.asarray(obs, np.float64)
            tr['obs_raw'].append(o * OBS_STD + OBS_MEAN if normed else o)
            tr['obs_norm'].append(normalize(o) if not normed else o)
            tr['action'].append(a)
            tr['label'].append(label)
            tr['halted'].append(base._halted)
            tr['main_blocked'].append(main_b)
            tr['branch_blocked'].append(branch_b)
            tr['rough'].append(rough)
            tr['soc'].append(soc)
        obs, r, term, trunc, info = env.step(a)
        total += r
        disc += g * r
        g *= (gamma or GAMMA)
        n += 1
        if record:
            tr['reward'].append(r)
        if term or trunc:
            break
    stats = dict(ret=total, disc_ret=disc, length=n, ret_per_step=total / n,
                 terminal_soc=soc, actions=acts,
                 terminated_reason=info.get('terminated_reason'), **ev)
    if record:
        return stats, {k: np.asarray(v) for k, v in tr.items()}
    return stats


def event_rates(rows):
    """Aggregate `rollout` dicts into the Week-6 headline event-conditioned rates."""
    s = {k: sum(r[k] for r in rows) for k in EVENT_KEYS}
    d = lambda a, b: (s[a] / s[b]) if s[b] else np.nan
    return dict(p_alert_anomaly=d('alert_on_anomaly', 'anomaly'),
                false_alerts_per_1k=1000 * d('alert_on_normal', 'normal'),
                p_reroute_block=d('reroute_on_block', 'single_block'),
                p_slow_rough=d('slow_on_rough', 'rough'),
                dock_rate=d('dock', 'low_soc'), low_soc_steps=s['low_soc'])


def evaluate(predict, world_seeds=None, norm=True, horizon=FULL_LOOP, env_kw=None,
             env_aware=False, map_seed=None):
    """Evaluate one policy over the fixed eval worlds. Returns (per-world rows, summary)."""
    world_seeds = world_seeds or EVAL_WORLD_SEEDS
    env_kw = dict(env_kw or {})
    if map_seed is not None:
        env_kw['map_seed'] = map_seed
    rows = []
    for ws in world_seeds:
        env = make_eval_env(ws, norm=norm, horizon=horizon, **env_kw)
        rows.append(rollout(predict, env, env_aware=env_aware))
        env.close()
    rets = np.array([r['ret'] for r in rows])
    rps = np.array([r['ret_per_step'] for r in rows])
    lens = np.array([r['length'] for r in rows])
    summary = dict(return_mean=rets.mean(), return_std=rets.std(),
                   ret_per_step_mean=rps.mean(), ret_per_step_std=rps.std(),
                   len_mean=lens.mean(), **event_rates(rows))
    return rows, summary


def paired_test(a, b):
    """Wilcoxon signed-rank over the common eval worlds; `a`, `b` are per-world returns."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return dict(delta=0.0, p=1.0, significant=False)
    stat, p = wilcoxon(a, b)
    return dict(delta=float((a - b).mean()), p=float(p), significant=bool(p < 0.05))


# --- reference policies -------------------------------------------------------------------
# Both brackets are carried over verbatim from Week 5 so the Week-6 table is anchored to the same
# floor and ceiling. They are stateful: the stuck counter makes the policy abort a full-block dead
# end rather than alert into it indefinitely, which is why the label-blind rule policy shows ~1
# false alert per 1k normal steps and not two orders of magnitude more.
class ScriptedBlind:
    """Deployable rule policy: observation only, so it navigates but never alerts on a fault."""

    def __init__(self, norm=True):
        self.norm = norm
        self.stuck = 0

    def reset(self):
        self.stuck = 0

    def raw(self, obs):
        o = np.asarray(obs, float)
        return o * OBS_STD + OBS_MEAN if self.norm else o

    def __call__(self, obs):
        o = self.raw(obs)
        tmean, soc, main_d, branch_d = o[0], o[5], o[7], o[8]
        halted = main_d < 2.0
        full_block = main_d < NEAR and branch_d < NEAR
        self.stuck = self.stuck + 1 if (halted and full_block) else 0
        if soc < LOW_SOC:
            return 4
        if self.stuck >= STUCK_TIMEOUT:
            return 4
        if halted and full_block:
            return 3
        if main_d < NEAR and branch_d >= NEAR:
            return 2
        if tmean > ROUGH_TERRAIN_TORQUE:
            return 1
        return 0


class ExpertAware(ScriptedBlind):
    """Privileged generation-time expert: also alerts on the ground-truth label. The ceiling."""

    def __call__(self, obs, base):
        o = self.raw(obs)
        tmean, soc, main_d, branch_d = o[0], o[5], o[7], o[8]
        halted = main_d < 2.0
        full_block = main_d < NEAR and branch_d < NEAR
        self.stuck = self.stuck + 1 if (halted and full_block) else 0
        if soc < LOW_SOC:
            return 4
        if self.stuck >= STUCK_TIMEOUT:
            return 4
        if halted and full_block:
            return 3
        if base._last_label == 1:
            return 3
        if main_d < NEAR and branch_d >= NEAR:
            return 2
        if tmean > ROUGH_TERRAIN_TORQUE:
            return 1
        return 0


def make_scripted_predict(norm=True):
    return ScriptedBlind(norm=norm)


W5_REFERENCE = {'scripted_blind': (4072.02, 516.10, 9529.6),
                'expert_aware': (12199.24, 736.02, 9367.2)}


def verify_w5_reference_gate(tol=1.0):
    """Check the two bracket policies against the numbers Week 5 published.

    Week 5 defined them as closures inside two notebooks rather than in an importable module, so
    the definitions here are a reconstruction. Reproducing the published return, spread and
    episode length on Week 5's own evaluation worlds is what makes the Week-6 table comparable
    to the Week-5 one; without it the floor and ceiling would silently drift.
    """
    out = {}
    for name, predict, aware in [('scripted_blind', ScriptedBlind(), False),
                                 ('expert_aware', ExpertAware(), True)]:
        _, s = evaluate(predict, world_seeds=W5_EVAL_SEEDS, env_aware=aware)
        want = W5_REFERENCE[name]
        got = (s['return_mean'], s['return_std'], s['len_mean'])
        ok = all(abs(g - w) <= tol for g, w in zip(got, want))
        out[name] = dict(published=want, reproduced=tuple(round(g, 2) for g in got), ok=ok)
        assert ok, f'{name} does not reproduce Week 5: {got} vs {want}'
    return out


def scripted_blind(obs_raw):
    """Stateless convenience form on the raw observation, for driving an env to a start state."""
    return ScriptedBlind(norm=False)(obs_raw)


# --- ledger -------------------------------------------------------------------------------
LEDGER = ROOT / 'rl' / 'rl_results.csv'


def append_ledger(rows):
    """Append Week-6 rows, adding the new columns without disturbing the Week-5 rows."""
    old = pd.read_csv(LEDGER)
    new = pd.DataFrame(rows)
    out = pd.concat([old, new], ignore_index=True, sort=False)
    out.to_csv(LEDGER, index=False)
    return out
