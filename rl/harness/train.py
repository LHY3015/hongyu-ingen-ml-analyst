"""Week-6 training entry points, merged behind one CLI: `python -m rl.harness.train --kind ...`.

Each `--kind` reproduces one of the original per-family trainers unchanged:

  python -m rl.harness.train --kind flat --algo dqn --seed 0 --tag dqn_raw --no-norm --exploration-fraction 0.5
  python -m rl.harness.train --kind flat --algo ppo --seed 0 --tag ppo --bc-init
  python -m rl.harness.train --kind pg --algo reinforce --seed 0 --tag reinforce
  python -m rl.harness.train --kind pg --algo grpo --seed 0 --tag grpo_bc --bc-init
  python -m rl.harness.train --kind options --seed 0 --tag opt_smdp
  python -m rl.harness.train --kind options --seed 0 --tag opt_ppo --learner ppo
  python3 -m rl.harness.train --kind marl --seed 0 --tag marl_shared --reward-mode shared
  python3 -m rl.harness.train --kind marl --seed 0 --tag marl_diff --reward-mode difference

--kind flat: train one flat-env agent (DQN or PPO) and save the checkpoint + learning curve.
One process per (algo, config, train seed) so the Week-6 sweep parallelises across cores.
Networks are [64, 64] MLPs, so training runs on CPU: the GPU transfer overhead exceeds the
forward/backward cost at this size and the bottleneck is the 312 step/s environment.

--kind pg: train one from-scratch policy-gradient agent (REINFORCE, +baseline, or GRPO), same
caching and artefact layout as `--kind flat`, so the two families can be swept and read back
through the same code. `--bc-init` fits the action head on the Week-2 offline transitions before
RL. For GRPO the behaviour-cloned network is also the frozen KL reference, matching the way GRPO
anchors on the SFT checkpoint it starts from; `rl.pg_algos.grpo` gets this by default from the
initial policy.

--kind options: train one node-level options agent on `rl.rover_options_env` and save the policy
+ curve. `--steps` is a budget in ENVIRONMENT steps, not option decisions, so a run here is
directly comparable to `--kind flat --steps 250000`. That equivalence is also what forces the
shape of the default learner: an option is 530-3,100 environment steps, so a 250k budget is only
~250 high-level decisions. Two learners are provided:

* ``--learner smdp`` (default) -- tabular Q over a 12-cell discretisation of the routing-relevant
  observation, updated with the exact semi-MDP target ``R + gamma^k max_o' Q(s', o')`` where k is
  the option's environment-step count. Visit-count averaging (alpha = 1/n) is used instead of a
  fixed rate because with ~250 samples the estimator has to spend every one of them.
* ``--learner ppo`` -- SB3 PPO at a fixed per-decision ``--gamma-hi``, ignoring k. This is an
  approximation of the SMDP objective, and the reason to keep it: gamma^k at gamma = 0.99 and
  k ~ 1,600 is 1e-7, so the exact learner sees nothing past the current edge unless it is run at
  ``--gamma 1.0``, while gamma_hi = 0.99 per decision is ~100 edges of horizon. PPO's rollout
  buffer is sized down to the decision budget (n_steps = 64), which still only buys ~3 updates at
  250k environment steps; a PPO comparison worth reading needs ``--steps`` in the millions.

--kind marl: train two independent PPO learners on the cooperative 2-rover patrol env and save
both. Independent learners: each rover runs its own actor-critic over its own 12-D observation
and treats the partner as part of the environment. That is the setting the shared-vs-difference
reward comparison is about -- with a shared team reward neither learner can attribute the reward
to its own option, and the difference reward is the counterfactual that removes the ambiguity.

PPO is written out here rather than taken from Stable-Baselines3: SB3 consumes a single
Gymnasium env, and the usual PettingZoo bridge (supersuit's `pettingzoo_env_to_vec_env`) is not
installed. The network and the hyperparameters match the flat-env PPO above (2x64 tanh trunk,
lr 3e-4, clip 0.2, GAE lambda 0.95, 10 epochs, entropy 0.01) so the two Week-6 arms differ in the
environment and the reward, not in the optimiser.

`--steps` counts 10 Hz ENVIRONMENT steps, matching the flat-env budget. Note that a decision is
taken per option, and an option runs to the next node -- ~1,200 environment steps on this map --
so the decision count is roughly `steps / 1200` per rover; see the header note in
`rl/rover_multiagent_env.py` on decision granularity.
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from shared_modules.rl_eval import (POL_DIR, ART_DIR, ROOT, GAMMA, TRAIN_CAP, FULL_LOOP,
                                    W6_TIMESTEPS, EVAL_WORLD_SEEDS, OBS_MEAN, OBS_STD, normalize,
                                    event_rates)
from rl.rover_env import STATE_COLS, LOW_SOC, NEAR
from rl.pg_algos import PGPolicy, reinforce, grpo, CURVE_SEEDS as PG_CURVE_SEEDS
from shared_modules.rover_world import LIDAR_MAX
from rl.rover_options_env import RoverOptionsEnv, EVENT_KEYS, MAX_OPTION_STEPS
from rl.rover_multiagent_env import RoverMultiAgentEnv, AGENTS, OPTION_NAMES

CURVE_SEEDS = [101, 202, 303]      # mid-training learning-curve worlds (final eval uses all 10)
N_OPTIONS = 4
N_STATES = 12


# === --kind flat (formerly rl/train_flat.py) ================================================
def bc_pretrain_flat(model, epochs=40, device='cpu'):
    """Supervised warm start of the PPO policy head on the Week-2 offline (s, a) pairs.

    The rover analogue of the SFT checkpoint that GRPO starts from in the PIC 2.0 configuration.
    Only the action head is fitted; the value head stays random, which is what the caller's
    critic-warmup phase then repairs before the policy is allowed to move.
    """
    df = pd.read_csv(ROOT / 'data' / 'rover_transitions.csv')
    X = normalize(df[[f's_{c}' for c in STATE_COLS]].to_numpy(np.float64))
    y = df['action'].to_numpy(np.int64)
    counts = np.bincount(y, minlength=5).astype(float)
    w = counts.sum() / (5 * np.clip(counts, 1, None))

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, device=device)
    cw = torch.tensor(w, dtype=torch.float32, device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=cw)
    params = list(model.policy.mlp_extractor.policy_net.parameters()) + \
        list(model.policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            lat = model.policy.mlp_extractor.forward_actor(model.policy.extract_features(Xt[idx]))
            opt.zero_grad()
            lossf(model.policy.action_net(lat), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        lat = model.policy.mlp_extractor.forward_actor(model.policy.extract_features(Xt[:20000]))
        acc = (model.policy.action_net(lat).argmax(1) == yt[:20000]).float().mean().item()
    return acc


def _main_flat(argv):
    from shared_modules.rl_eval import make_train_env, make_eval_env

    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['dqn', 'ppo'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--steps', type=int, default=W6_TIMESTEPS)
    p.add_argument('--gamma', type=float, default=GAMMA)
    p.add_argument('--no-norm', action='store_true')
    p.add_argument('--bc-init', action='store_true')
    p.add_argument('--critic-warmup', type=int, default=10_000)
    p.add_argument('--exploration-fraction', type=float, default=0.3)
    p.add_argument('--energy-weight', type=float, default=0.05)
    p.add_argument('--low-soc-penalty', type=float, default=0.0)
    p.add_argument('--soc-init-lo', type=float, default=40.0)
    p.add_argument('--soc-init-hi', type=float, default=100.0)
    p.add_argument('--eval-freq', type=int, default=25_000)
    a = p.parse_args(argv)

    tag = f'{a.tag}_s{a.seed}'
    zip_path = POL_DIR / f'{tag}.zip'
    meta_path = ART_DIR / f'{tag}_meta.json'
    if zip_path.exists() and meta_path.exists():
        print(f'{tag}: cached'); return

    norm = not a.no_norm
    env_kw = dict(energy_weight=a.energy_weight, low_soc_penalty=a.low_soc_penalty,
                  soc_init_range=(a.soc_init_lo, a.soc_init_hi))
    torch.set_num_threads(1)
    t0 = time.time()

    env = Monitor(make_train_env(a.seed, norm=norm, **env_kw))
    eval_env = Monitor(make_eval_env(CURVE_SEEDS[0], norm=norm, horizon=TRAIN_CAP, **env_kw))
    log_dir = ART_DIR / f'curve_{tag}'
    cb = EvalCallback(eval_env, log_path=str(log_dir), eval_freq=a.eval_freq,
                      n_eval_episodes=3, deterministic=True, verbose=0)

    if a.algo == 'dqn':
        model = DQN('MlpPolicy', env, learning_rate=5e-4, buffer_size=50_000,
                    learning_starts=2_000, batch_size=128, gamma=a.gamma, train_freq=4,
                    gradient_steps=1, target_update_interval=1_000,
                    exploration_fraction=a.exploration_fraction, exploration_final_eps=0.05,
                    policy_kwargs=dict(net_arch=[64, 64]), device='cpu', seed=a.seed, verbose=0)
    else:
        model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048, batch_size=256,
                    n_epochs=10, gamma=a.gamma, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]),
                    device='cpu', seed=a.seed, verbose=0)

    bc_acc = None
    if a.bc_init:
        if a.algo != 'ppo':
            raise SystemExit('--bc-init applies to PPO only')
        bc_acc = bc_pretrain_flat(model)
        # The value head is still random, so the first advantages are noise and would wash the
        # BC prior straight out. Freeze the policy while the critic catches up.
        for prm in list(model.policy.mlp_extractor.policy_net.parameters()) + \
                list(model.policy.action_net.parameters()):
            prm.requires_grad_(False)
        model.learn(a.critic_warmup, reset_num_timesteps=False, progress_bar=False)
        for prm in list(model.policy.mlp_extractor.policy_net.parameters()) + \
                list(model.policy.action_net.parameters()):
            prm.requires_grad_(True)

    model.learn(a.steps, callback=cb, reset_num_timesteps=False, progress_bar=False)
    model.save(zip_path)

    curve = {}
    npz = log_dir / 'evaluations.npz'
    if npz.exists():
        d = np.load(npz)
        curve = dict(timesteps=d['timesteps'].tolist(), results=d['results'].mean(1).tolist())
    meta = dict(tag=tag, algo=a.algo, seed=a.seed, steps=a.steps, gamma=a.gamma, norm=norm,
                bc_init=a.bc_init, bc_acc=bc_acc, energy_weight=a.energy_weight,
                low_soc_penalty=a.low_soc_penalty,
                soc_init_range=[a.soc_init_lo, a.soc_init_hi],
                exploration_fraction=a.exploration_fraction,
                minutes=round((time.time() - t0) / 60, 2), curve=curve)
    meta_path.write_text(json.dumps(meta))
    print(f'{tag}: done in {meta["minutes"]} min')


# === --kind pg (formerly rl/train_pg.py) =====================================================
def bc_pretrain_pg(policy, epochs=40, batch=512, lr=1e-3):
    """Class-weighted supervised fit of the action head on the Week-2 offline (s, a) pairs.

    The five actions are wildly unbalanced in the scripted-expert table (continue dominates),
    so unweighted cross-entropy converges to a constant-`continue` policy that carries no prior
    about rerouting or docking at all.
    """
    df = pd.read_csv(ROOT / 'data' / 'rover_transitions.csv')
    X = normalize(df[[f's_{c}' for c in STATE_COLS]].to_numpy(np.float64))
    y = df['action'].to_numpy(np.int64)
    counts = np.bincount(y, minlength=5).astype(float)
    w = counts.sum() / (5 * np.clip(counts, 1, None))

    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y.copy())
    lossf = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(policy.policy_parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            lossf(policy.logits(Xt[idx]), yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        return float((policy.logits(Xt).argmax(1) == yt).float().mean())


def _main_pg(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['reinforce', 'reinforce_baseline', 'grpo'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--steps', type=int, default=W6_TIMESTEPS)
    p.add_argument('--bc-init', action='store_true')
    p.add_argument('--kl-coef', type=float, default=0.02)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--segment-len', type=int, default=256)
    p.add_argument('--start-mix', type=float, default=0.5)
    p.add_argument('--eval-freq', type=int, default=25_000)
    a = p.parse_args(argv)

    tag = f'{a.tag}_s{a.seed}'
    meta_path = ART_DIR / f'{tag}_meta.json'
    if meta_path.exists():
        print(f'{tag}: cached'); return

    torch.set_num_threads(1)
    t0 = time.time()
    torch.manual_seed(a.seed)
    policy = PGPolicy(with_value=(a.algo == 'reinforce_baseline'))
    bc_acc = bc_pretrain_pg(policy) if a.bc_init else None

    hp = dict(lr=3e-4, eval_freq=a.eval_freq)
    if a.algo == 'grpo':
        hp.update(group_size=a.group_size, segment_len=a.segment_len, start_mix=a.start_mix,
                  kl_coef=a.kl_coef, clip_range=0.2, n_epochs=10, minibatch=256,
                  search_cap=3000)
        policy, updates, curve = grpo(a.seed, a.steps, group_size=a.group_size,
                                      segment_len=a.segment_len, start_mix=a.start_mix,
                                      kl_coef=a.kl_coef, lr=hp['lr'], gamma=GAMMA,
                                      eval_freq=a.eval_freq, policy=policy)
    else:
        baseline = (a.algo == 'reinforce_baseline')
        hp.update(batch_steps=4096, baseline=baseline)
        if baseline:
            hp.update(value_lr=1e-3, value_epochs=10, value_batch=256)
        policy, updates, curve = reinforce(a.seed, a.steps, baseline=baseline, lr=hp['lr'],
                                           gamma=GAMMA, eval_freq=a.eval_freq, policy=policy)

    torch.save(policy.state_dict(), POL_DIR / f'{tag}.pt')
    meta = dict(tag=tag, algo=a.algo, seed=a.seed, steps=a.steps, gamma=GAMMA, norm=True,
                bc_init=a.bc_init, bc_acc=bc_acc, curve_seeds=PG_CURVE_SEEDS, hyperparams=hp,
                minutes=round((time.time() - t0) / 60, 2), curve=curve, updates=updates)
    meta_path.write_text(json.dumps(meta))
    print(f'{tag}: done in {meta["minutes"]} min, {len(updates)} updates, '
          f'final curve {curve["results"][-1]:.1f}')


# === --kind options (formerly rl/train_options.py) ===========================================
def discretize(obs_raw):
    """Routing-relevant cell of the 9-D observation: (main-route block, branch, battery).

    The three main-route bins are `< NEAR` (inside the rule policy's 150 m trigger), visible but
    beyond it, and nothing inside the 200 m LiDAR lookahead. The middle bin exists because an
    option decides the branch taken at the END of the edge it is about to drive, so the blockage
    it is deciding about sits one edge away -- 93-173 m rather than the 40-120 m a flat agent sees
    from the branch node itself.
    """
    o = np.asarray(obs_raw, float)
    b_main = 0 if o[7] < NEAR else (1 if o[7] < LIDAR_MAX - 1.0 else 2)
    b_branch = int(o[8] >= NEAR)
    b_soc = int(o[5] < LOW_SOC)
    return (b_main * 2 + b_branch) * 2 + b_soc


def make_q_predict(q, norm=True):
    def predict(obs):
        raw = np.asarray(obs, float) * OBS_STD + OBS_MEAN if norm else obs
        return int(np.argmax(q[discretize(raw)]))
    return predict


# --- evaluation ----------------------------------------------------------------------------
def rollout_options(predict, env):
    """One deterministic option-level episode. `ret` is the undiscounted flat-reward sum, so it is
    on the same scale as the returns in `rl_eval.rollout`; `disc_ret` chains gamma^k across
    options and therefore equals the flat discounted return of the same trajectory."""
    obs, _ = env.reset()
    ev = {k: 0 for k in EVENT_KEYS}
    opts = np.zeros(N_OPTIONS, int)
    ret, disc, g, ks, guards = 0.0, 0.0, 1.0, [], 0
    while True:
        a = int(predict(obs))
        opts[a] += 1
        obs, r, term, trunc, info = env.step(a)
        for k in EVENT_KEYS:
            ev[k] += info['events'][k]
        ret += info['reward_sum']
        disc += g * r
        g *= info['gamma_k']
        ks.append(info['option_k'])
        guards += info['guard_hit']
        if term or trunc:
            break
    return dict(ret=ret, disc_ret=disc, length=info['n_env_steps'], n_options=len(ks),
                option_k_mean=float(np.mean(ks)), guards=guards, options=opts,
                terminated_reason=info['terminated_reason'], **ev)


def evaluate_options(predict, world_seeds, horizon=FULL_LOOP, **env_kw):
    rows = [rollout_options(predict, RoverOptionsEnv(randomize_reset=False, seed=ws,
                                                     max_env_steps=horizon, **env_kw))
            for ws in world_seeds]
    s = {k: sum(r[k] for r in rows) for k in EVENT_KEYS}
    rets = np.array([r['ret'] for r in rows])
    return dict(return_mean=float(rets.mean()), return_std=float(rets.std()),
                len_mean=float(np.mean([r['length'] for r in rows])),
                n_options_mean=float(np.mean([r['n_options'] for r in rows])),
                option_k_mean=float(np.mean([r['option_k_mean'] for r in rows])),
                p_reroute_block=(s['reroute_on_block'] / s['single_block']
                                 if s['single_block'] else float('nan')),
                p_alert_anomaly=(s['alert_on_anomaly'] / s['anomaly']
                                 if s['anomaly'] else float('nan')),
                dock_rate=(s['dock'] / s['low_soc']) if s['low_soc'] else float('nan'),
                guards=sum(r['guards'] for r in rows))


# --- learners ------------------------------------------------------------------------------
def train_smdp(a, env, env_kw, curve):
    q = np.zeros((N_STATES, N_OPTIONS))
    cnt = np.zeros((N_STATES, N_OPTIONS))
    rng = np.random.default_rng(a.seed)
    anneal = max(a.exploration_fraction * a.steps, 1.0)
    next_eval = a.eval_freq
    n_dec = 0

    env.reset()
    s = discretize(env.raw_obs)
    while env.total_env_steps < a.steps:
        eps = max(0.05, 1.0 - (1.0 - 0.05) * env.total_env_steps / anneal)
        o = int(rng.integers(N_OPTIONS)) if rng.random() < eps else int(np.argmax(q[s]))
        _, r, term, trunc, info = env.step(o)
        s2 = discretize(env.raw_obs)
        target = r + (0.0 if term else info['gamma_k'] * q[s2].max())
        cnt[s, o] += 1
        q[s, o] += (target - q[s, o]) / cnt[s, o]
        s = s2
        n_dec += 1
        if term or trunc:
            env.reset()
            s = discretize(env.raw_obs)
        if env.total_env_steps >= next_eval:
            curve.append(eval_point(env.total_env_steps, make_q_predict(q, a.norm), a, env_kw))
            next_eval += a.eval_freq
    return q, cnt, n_dec


def train_ppo(a, env, env_kw, curve):
    from stable_baselines3.common.callbacks import BaseCallback

    class BudgetCallback(BaseCallback):
        """SB3 counts option decisions; the budget is in environment steps, so the stop condition
        and the curve schedule both read the env's own environment-step counter."""

        def __init__(self):
            super().__init__()
            self.next_eval = a.eval_freq

        def _on_step(self):
            n = env.total_env_steps
            if n >= self.next_eval:
                curve.append(eval_point(n, make_ppo_predict(self.model), a, env_kw))
                self.next_eval += a.eval_freq
            return n < a.steps

    model = PPO('MlpPolicy', Monitor(env), learning_rate=3e-4, n_steps=64, batch_size=16,
                n_epochs=10, gamma=a.gamma_hi, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[64, 64]), device='cpu', seed=a.seed, verbose=0)
    model.learn(10 ** 9, callback=BudgetCallback(), reset_num_timesteps=False,
                progress_bar=False)
    return model, model.num_timesteps


def make_ppo_predict(model):
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def eval_point(n_env, predict, a, env_kw):
    s = evaluate_options(predict, CURVE_SEEDS[:1], horizon=a.curve_horizon, **env_kw)
    # a curve world with no single blockage leaves p_reroute undefined; JSON null rather than NaN
    # keeps the meta files loadable by strict parsers
    pr = s['p_reroute_block']
    return dict(timestep=int(n_env), result=s['return_mean'],
                p_reroute=(float(pr) if np.isfinite(pr) else None),
                n_options=s['n_options_mean'])


def _main_options(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--learner', choices=['smdp', 'ppo'], default='smdp')
    p.add_argument('--steps', type=int, default=W6_TIMESTEPS,
                   help='budget in environment steps (includes the 50-step reset warm-ups)')
    p.add_argument('--gamma', type=float, default=GAMMA,
                   help='intra-option discount; for --learner smdp it is also the exact gamma^k '
                        'inter-option discount')
    p.add_argument('--gamma-hi', type=float, default=GAMMA,
                   help='--learner ppo only: fixed per-decision discount, k ignored')
    p.add_argument('--no-norm', action='store_true')
    p.add_argument('--exploration-fraction', type=float, default=0.5)
    p.add_argument('--energy-weight', type=float, default=0.05)
    p.add_argument('--low-soc-penalty', type=float, default=0.0)
    p.add_argument('--soc-init-lo', type=float, default=40.0)
    p.add_argument('--soc-init-hi', type=float, default=100.0)
    p.add_argument('--max-option-steps', type=int, default=MAX_OPTION_STEPS)
    # the flat runs truncate training episodes at 2,400 env steps; that is ~3 option decisions and
    # usually stops before the first branch node whose main edge is blocked, so options train on a
    # full loop instead
    p.add_argument('--train-cap', type=int, default=FULL_LOOP)
    p.add_argument('--eval-freq', type=int, default=50_000)
    p.add_argument('--curve-horizon', type=int, default=FULL_LOOP)
    a = p.parse_args(argv)
    a.norm = not a.no_norm

    tag = f'{a.tag}_s{a.seed}'
    ckpt = POL_DIR / (f'{tag}.zip' if a.learner == 'ppo' else f'{tag}.npz')
    meta_path = ART_DIR / f'{tag}_meta.json'
    if ckpt.exists() and meta_path.exists():
        print(f'{tag}: cached'); return

    env_kw = dict(norm=a.norm, gamma=a.gamma, max_option_steps=a.max_option_steps,
                  energy_weight=a.energy_weight, low_soc_penalty=a.low_soc_penalty,
                  soc_init_range=(a.soc_init_lo, a.soc_init_hi))
    t0 = time.time()
    env = RoverOptionsEnv(randomize_reset=True, seed=a.seed, max_env_steps=a.train_cap, **env_kw)
    curve = []

    if a.learner == 'smdp':
        q, cnt, n_dec = train_smdp(a, env, env_kw, curve)
        np.savez(ckpt, q=q, counts=cnt)
        predict = make_q_predict(q, a.norm)
        extra = dict(q=q.tolist(), visit_counts=cnt.tolist())
    else:
        torch.set_num_threads(1)
        model, n_dec = train_ppo(a, env, env_kw, curve)
        model.save(ckpt)
        predict = make_ppo_predict(model)
        extra = {}

    curve.append(eval_point(env.total_env_steps, predict, a, env_kw))
    meta = dict(tag=tag, learner=a.learner, seed=a.seed, steps=a.steps,
                env_steps_used=env.total_env_steps, decisions=n_dec, gamma=a.gamma,
                gamma_hi=(a.gamma_hi if a.learner == 'ppo' else None), norm=a.norm,
                exploration_fraction=a.exploration_fraction, energy_weight=a.energy_weight,
                low_soc_penalty=a.low_soc_penalty,
                soc_init_range=[a.soc_init_lo, a.soc_init_hi], train_cap=a.train_cap,
                max_option_steps=a.max_option_steps, eval_freq=a.eval_freq,
                curve_horizon=a.curve_horizon,
                curve=dict(timesteps=[c['timestep'] for c in curve],
                           results=[c['result'] for c in curve],
                           p_reroute=[c['p_reroute'] for c in curve],
                           n_options=[c['n_options'] for c in curve]),
                minutes=round((time.time() - t0) / 60, 2), **extra)
    meta_path.write_text(json.dumps(meta))
    print(f'{tag}: done in {meta["minutes"]} min '
          f'({n_dec} decisions over {env.total_env_steps} env steps)')


# === --kind marl (formerly rl/train_marl.py) =================================================
CURVE_SEED = EVAL_WORLD_SEEDS[0]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_act)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits, v = self(torch.as_tensor(obs, dtype=torch.float32))
        d = Categorical(logits=logits)
        a = logits.argmax(-1) if deterministic else d.sample()
        return int(a), float(d.log_prob(a)), float(v)

    @torch.no_grad()
    def value(self, obs):
        return float(self(torch.as_tensor(obs, dtype=torch.float32))[1])


class Buffer:
    """One rover's on-policy rollout. `nextval` is stored per transition so episode boundaries
    (termination -> 0, truncation -> V(s')) need no special-casing in the GAE sweep."""

    KEYS = ['obs', 'act', 'logp', 'val', 'rew', 'done', 'nextval']

    def __init__(self):
        self.clear()

    def clear(self):
        self.d = {k: [] for k in self.KEYS}

    def add(self, **kw):
        for k in self.KEYS:
            self.d[k].append(kw[k])

    def __len__(self):
        return len(self.d['act'])

    def tensors(self, gamma, lam):
        val = np.array(self.d['val'], np.float32)
        rew = np.array(self.d['rew'], np.float32)
        done = np.array(self.d['done'], np.float32)
        nextval = np.array(self.d['nextval'], np.float32)
        adv = np.zeros_like(rew)
        run = 0.0
        for t in range(len(rew) - 1, -1, -1):
            delta = rew[t] + gamma * nextval[t] - val[t]
            run = delta + gamma * lam * (1.0 - done[t]) * run
            adv[t] = run
        return (torch.as_tensor(np.array(self.d['obs'], np.float32)),
                torch.as_tensor(np.array(self.d['act'], np.int64)),
                torch.as_tensor(np.array(self.d['logp'], np.float32)),
                torch.as_tensor(adv), torch.as_tensor(adv + val))


def ppo_update(net, opt, buf, a):
    obs, act, logp_old, adv, ret = buf.tensors(a.gamma, a.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(act)
    losses = []
    for _ in range(a.n_epochs):
        for i in range(0, n, a.batch_size):
            idx = torch.randperm(n)[:a.batch_size] if a.batch_size < n else torch.arange(n)
            logits, v = net(obs[idx])
            d = Categorical(logits=logits)
            ratio = torch.exp(d.log_prob(act[idx]) - logp_old[idx])
            pg = -torch.min(ratio * adv[idx],
                            torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * adv[idx]).mean()
            loss = pg + a.vf_coef * ((v - ret[idx]) ** 2).mean() - a.ent_coef * d.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
            opt.step()
            losses.append(loss.detach().item())
            if a.batch_size >= n:
                break
    return float(np.mean(losses))


def run_episode(nets, env, world_seed=None, deterministic=True):
    """One episode under the given policies; returns the team return and the coordination info."""
    obs, _ = env.reset(seed=world_seed)
    team_ret, n_dec, last = 0.0, 0, None
    while env.agents:
        acts = {aid: nets[aid].act(obs[aid], deterministic=deterministic)[0] for aid in env.agents}
        obs, _, term, trunc, info = env.step(acts)
        aid0 = next(iter(info))
        team_ret += info[aid0]['team']['team_reward']
        last = info
        n_dec += 1
    team = last[next(iter(last))]['team']
    roster = team['roster']
    return dict(team_return=team_ret, decisions=n_dec,
                coverage_frac=team['coverage_frac'], covered_edges=team['covered_edges'],
                redundant_edges=team['redundant_edges'],
                proximity_steps=team['proximity_steps'], env_steps=team['env_step'],
                dist_m={aid: roster[aid]['dist_m'] for aid in roster},
                halted_steps={aid: roster[aid]['halted_steps'] for aid in roster},
                events={aid: roster[aid]['events'] for aid in roster})


def make_env(a, train):
    return RoverMultiAgentEnv(
        reward_mode=a.reward_mode, coverage_weight=a.coverage_weight,
        redundant_penalty=a.redundant_penalty, stale_threshold=a.stale_threshold,
        proximity_penalty=a.proximity_penalty, proximity_m=a.proximity_m,
        randomize_reset=train, seed=a.seed, init_soc=a.init_soc,
        max_env_steps=a.train_cap if train else a.eval_horizon)


def _main_marl(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--steps', type=int, default=W6_TIMESTEPS, help='10 Hz environment steps')
    p.add_argument('--reward-mode', choices=['shared', 'difference'], default='shared')
    p.add_argument('--n-steps', type=int, default=64, help='option decisions per rollout')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--n-epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=GAMMA)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--coverage-weight', type=float, default=5.0)
    p.add_argument('--redundant-penalty', type=float, default=2.0)
    p.add_argument('--stale-threshold', type=int, default=2000)
    p.add_argument('--proximity-penalty', type=float, default=0.02)
    p.add_argument('--proximity-m', type=float, default=5.0)
    p.add_argument('--init-soc', type=float, default=100.0)
    p.add_argument('--train-cap', type=int, default=TRAIN_CAP)
    p.add_argument('--eval-horizon', type=int, default=FULL_LOOP)
    p.add_argument('--eval-freq', type=int, default=50_000, help='environment steps between'
                   ' learning-curve evaluations')
    a = p.parse_args(argv)

    tag = f'{a.tag}_s{a.seed}'
    ckpt = POL_DIR / f'{tag}.pt'
    meta_path = ART_DIR / f'{tag}_meta.json'
    if ckpt.exists() and meta_path.exists():
        print(f'{tag}: cached'); return

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()

    env = make_env(a, train=True)
    eval_env = make_env(a, train=False)
    obs_dim = env.observation_space(AGENTS[0]).shape[0]
    n_act = env.action_space(AGENTS[0]).n
    nets = {aid: ActorCritic(obs_dim, n_act) for aid in AGENTS}
    opts = {aid: torch.optim.Adam(nets[aid].parameters(), lr=a.lr) for aid in AGENTS}
    bufs = {aid: Buffer() for aid in AGENTS}

    env_steps, decisions, updates = 0, {aid: 0 for aid in AGENTS}, 0
    curve = dict(env_steps=[], team_return=[], coverage_frac=[], decisions=[])
    next_eval = a.eval_freq
    obs, _ = env.reset()
    prev_env_step = 0

    while env_steps < a.steps:
        # --- collect ---------------------------------------------------------------------
        while min(len(b) for b in bufs.values()) < a.n_steps and env_steps < a.steps:
            if not env.agents:
                obs, _ = env.reset()
                prev_env_step = 0
            acts, cache = {}, {}
            for aid in env.agents:
                act, logp, val = nets[aid].act(obs[aid])
                acts[aid] = act
                cache[aid] = (obs[aid], act, logp, val)
            nxt, rew, term, trunc, _ = env.step(acts)
            env_steps += env.env_step - prev_env_step
            prev_env_step = env.env_step
            for aid, (o, act, logp, val) in cache.items():
                done = term[aid] or trunc[aid]
                nv = 0.0 if term[aid] else nets[aid].value(nxt[aid])
                bufs[aid].add(obs=o, act=act, logp=logp, val=val, rew=rew[aid],
                              done=float(done), nextval=nv)
                decisions[aid] += 1
            obs = nxt

        # --- update ----------------------------------------------------------------------
        if all(len(b) > 1 for b in bufs.values()):
            for aid in AGENTS:
                ppo_update(nets[aid], opts[aid], bufs[aid], a)
            updates += 1
        for b in bufs.values():
            b.clear()

        if env_steps >= next_eval or env_steps >= a.steps:
            r = run_episode(nets, eval_env, world_seed=CURVE_SEED)
            curve['env_steps'].append(env_steps)
            curve['team_return'].append(r['team_return'])
            curve['coverage_frac'].append(r['coverage_frac'])
            curve['decisions'].append(r['decisions'])
            next_eval = env_steps + a.eval_freq

    torch.save({aid: nets[aid].state_dict() for aid in AGENTS}, ckpt)

    # --- final evaluation over the fixed Week-6 worlds -------------------------------------
    rows = [run_episode(nets, eval_env, world_seed=ws) for ws in EVAL_WORLD_SEEDS]
    agg = lambda k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
    ret_m, ret_s = agg('team_return')
    ev = event_rates([r['events'][aid] for r in rows for aid in AGENTS])
    evaluation = dict(
        world_seeds=EVAL_WORLD_SEEDS, team_return_mean=ret_m, team_return_std=ret_s,
        coverage_frac_mean=agg('coverage_frac')[0], coverage_frac_std=agg('coverage_frac')[1],
        redundant_edges_mean=agg('redundant_edges')[0],
        proximity_steps_mean=agg('proximity_steps')[0],
        decisions_mean=agg('decisions')[0], env_steps_mean=agg('env_steps')[0],
        dist_m_mean={aid: float(np.mean([r['dist_m'][aid] for r in rows])) for aid in AGENTS},
        halted_steps_mean={aid: float(np.mean([r['halted_steps'][aid] for r in rows]))
                           for aid in AGENTS},
        **{k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
           for k, v in ev.items()})

    meta = dict(tag=tag, algo='ippo', seed=a.seed, reward_mode=a.reward_mode,
                steps_requested=a.steps, env_steps=env_steps, decisions=decisions,
                updates=updates, options=OPTION_NAMES, obs_dim=obs_dim, net_arch=[64, 64],
                n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=a.n_epochs, lr=a.lr,
                gamma=a.gamma, gae_lambda=a.gae_lambda, clip=a.clip, ent_coef=a.ent_coef,
                vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm,
                env=dict(coverage_weight=a.coverage_weight,
                         redundant_penalty=a.redundant_penalty,
                         stale_threshold=a.stale_threshold,
                         proximity_penalty=a.proximity_penalty, proximity_m=a.proximity_m,
                         init_soc=a.init_soc, train_cap=a.train_cap,
                         eval_horizon=a.eval_horizon,
                         max_option_steps=env.max_option_steps, map_seed=env.map_seed,
                         hazard=env.hazard),
                curve=curve, eval=evaluation,
                minutes=round((time.time() - t0) / 60, 2))
    meta_path.write_text(json.dumps(meta))
    print(f'{tag}: done in {meta["minutes"]} min, {env_steps} env steps, '
          f'{sum(decisions.values())} decisions, {updates} updates, '
          f'team return {ret_m:.1f} +- {ret_s:.1f}, '
          f'coverage {evaluation["coverage_frac_mean"]:.2f}')


def load_agents(tag, seed, obs_dim=12, n_act=4):
    """Reload both rovers' independent policies from one checkpoint, for evaluation and latency."""
    ckpt = POL_DIR / f'{tag}_s{seed}.pt'
    if not ckpt.exists():
        return None
    state = torch.load(ckpt, map_location='cpu')
    nets = {}
    for aid in AGENTS:
        net = ActorCritic(obs_dim, n_act)
        net.load_state_dict(state[aid])
        net.eval()
        net.obs_dim = obs_dim
        nets[aid] = net
    return nets


# === dispatch =================================================================================
_KINDS = {'flat': _main_flat, 'pg': _main_pg, 'options': _main_options, 'marl': _main_marl}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--kind', choices=list(_KINDS), required=True)
    ns, rest = p.parse_known_args(argv)
    _KINDS[ns.kind](rest)


if __name__ == '__main__':
    main()
