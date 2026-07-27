"""From-scratch policy-gradient algorithms for the Week-6 rover study.

Three estimators over the same network and the same environment, so the only thing that
differs between them is how the advantage is built:

* `reinforce` — Monte-Carlo return-to-go, no baseline. The textbook estimator, kept exactly
  as textbook (`-(logp * G).mean()`, no entropy bonus, no advantage normalisation) so that its
  variance is the measured quantity rather than something tuning has hidden.
* `reinforce_baseline` — the same estimator minus a learned `V(s)`; the state-dependent
  baseline is the only change.
* `grpo` — group-relative: one start state, `group_size` stochastic trajectories from it, and
  a single scalar advantage per trajectory obtained by standardising the group's discounted
  returns. No critic anywhere.

Capacity is matched to the SB3 PPO baseline (`MlpPolicy`, `net_arch=[64, 64]`): a 2x64 tanh
trunk with a linear action head, plus — where a critic exists — a second, separate 2x64 tanh
trunk with a linear value head. The orthogonal init (gain sqrt(2) on the trunk, 0.01 on the
action head) is copied from SB3 as well, since the near-uniform initial policy it produces is
what keeps the first REINFORCE updates from collapsing onto one action.

**Group construction.** The G trajectories of an update must share a start state, otherwise
the group mean is a baseline for a mixture of states and the "relative" part is meaningless.
The rover env holds only numpy arrays and a `np.random.Generator`, so the start state is
reached once and `copy.deepcopy` gives each group member an independent continuation of it;
`assert_deepcopy_determinism` checks the copies replay bit-exactly before any training starts.

**Constant advantage.** `A_i` is broadcast unchanged to every timestep of trajectory `i`. That
is the structural difference from PPO — no per-timestep, state-dependent baseline, hence no
credit assignment inside the segment — and it is the property the comparison is about, so it
is preserved rather than repaired.

**Prompt distribution.** A `segment_len`-step segment carries no bootstrap, so its effective
horizon is `segment_len` steps. On an ordinary episode start the next blockage is 600-900
steps ahead, which is outside that horizon: every group member would collect the same
event-free reward stream, the group returns would differ only by sensor noise, and the run
would be a null result designed in from the start. `start_mix` therefore routes a fraction of
the updates to an *event-proximal* start, reached by driving the scripted rule policy forward
until a blockage is within `NEAR` or SoC drops below 25 %.

**Budget accounting.** Only the group's own transitions count against the step budget, so the
2048 steps per update line up with PPO's `n_steps=2048`. The steps spent searching for an
event-proximal start are logged separately as `search_steps` — they are real interactions but
carry no gradient, and folding them into the budget would make the two curves incomparable.

**Instrumentation.** Every update records `grad_noise`: the update's own transitions are split
into two disjoint halves and the policy-gradient vector of each is compared by cosine
similarity and relative L2 gap. A return curve cannot separate "this estimator is biased" from
"this estimator is too noisy to move the parameters"; the disagreement between two halves of
one batch can.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

from shared_modules.rl_eval import (GAMMA, TRAIN_CAP, make_train_env, make_eval_env, rollout,
                          make_scripted_predict)

CURVE_SEEDS = [101, 202, 303]      # mid-training learning-curve worlds (final eval uses all 10)
EVENT_SOC = 25.0                   # event-proximal search also accepts a near-flat battery


# --- network ------------------------------------------------------------------------------
def _trunk(obs_dim, hidden):
    return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh())


def _ortho(module, gain):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class PGPolicy(nn.Module):
    """2x64 tanh trunk + linear action head; `with_value` adds SB3's separate critic trunk."""

    def __init__(self, obs_dim=9, n_actions=5, hidden=64, with_value=False):
        super().__init__()
        self.pi_trunk = _trunk(obs_dim, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.v_trunk = _trunk(obs_dim, hidden) if with_value else None
        self.value_head = nn.Linear(hidden, 1) if with_value else None
        _ortho(self.pi_trunk, np.sqrt(2))
        _ortho(self.action_head, 0.01)
        if with_value:
            _ortho(self.v_trunk, np.sqrt(2))
            _ortho(self.value_head, 1.0)

    def logits(self, obs):
        return self.action_head(self.pi_trunk(obs))

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.logits(obs))

    def value(self, obs):
        return self.value_head(self.v_trunk(obs)).squeeze(-1)

    def policy_parameters(self):
        return list(self.pi_trunk.parameters()) + list(self.action_head.parameters())

    def value_parameters(self):
        return list(self.v_trunk.parameters()) + list(self.value_head.parameters())

    @torch.no_grad()
    def act(self, obs):
        """Sample an action for on-policy collection."""
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        return int(torch.distributions.Categorical(logits=self.logits(x)).sample())

    @torch.no_grad()
    def predict(self, obs):
        """Greedy action — the `deterministic=True` policy that gets evaluated."""
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        return int(self.logits(x).argmax())


# --- estimator plumbing -------------------------------------------------------------------
def _return_to_go(rewards, dones, gamma):
    """Discounted return-to-go inside each episode segment.

    A batch that ends mid-episode is treated as ending there — no bootstrap value is available
    without a critic, so the tail of the batch is a truncated Monte-Carlo return.
    """
    g = np.zeros(len(rewards), np.float64)
    run = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        if dones[t]:
            run = 0.0
        run = rewards[t] + gamma * run
        g[t] = run
    return g


def _cat_kl(logits_p, logits_q):
    """KL(p || q) summed over the 5 actions — exact, not the single-sample k3 estimator."""
    logp = torch.log_softmax(logits_p, -1)
    logq = torch.log_softmax(logits_q, -1)
    return (logp.exp() * (logp - logq)).sum(-1)


def _grad_noise(policy, obs, acts, adv, rng):
    """Agreement between the policy gradients of two disjoint halves of one update's batch.

    The halves are drawn by permutation rather than by time order, so the number measures the
    estimator's own variance and not the temporal correlation of a patrol episode.
    """
    n = len(obs) // 2
    if n < 2:
        return float('nan'), float('nan')
    idx = rng.permutation(len(obs))
    params = policy.policy_parameters()
    gs = []
    for part in (idx[:n], idx[n:2 * n]):
        sel = torch.as_tensor(np.asarray(part))
        loss = -(policy.dist(obs[sel]).log_prob(acts[sel]) * adv[sel]).mean()
        g = torch.autograd.grad(loss, params)
        gs.append(torch.cat([x.reshape(-1) for x in g]))
    g1, g2 = gs
    cos = float(torch.nn.functional.cosine_similarity(g1, g2, dim=0))
    rel = float((g1 - g2).norm() / (0.5 * (g1 + g2).norm() + 1e-12))
    return cos, rel


class _Curve:
    """Greedy-policy return on the three curve worlds, sampled every `eval_freq` env steps."""

    def __init__(self, policy, eval_freq, env_kw, seeds=CURVE_SEEDS):
        self.policy, self.eval_freq, self.env_kw, self.seeds = policy, eval_freq, env_kw, seeds
        self.timesteps, self.results, self._last = [], [], 0

    def evaluate(self, t):
        rets = []
        for ws in self.seeds:
            env = make_eval_env(ws, horizon=TRAIN_CAP, **self.env_kw)
            rets.append(rollout(self.policy.predict, env)['ret'])
            env.close()
        self.timesteps.append(int(t))
        self.results.append(float(np.mean(rets)))
        self._last = t

    def maybe(self, t):
        if t - self._last >= self.eval_freq:
            self.evaluate(t)

    def as_dict(self):
        return dict(timesteps=self.timesteps, results=self.results)


class _Stream:
    """Continuing on-policy collector: a batch spans episode boundaries and resets on done."""

    def __init__(self, env):
        self.env = env
        self.obs, _ = env.reset()

    def step(self, action):
        obs, r, term, trunc, _ = self.env.step(action)
        done = bool(term or trunc)
        self.obs = self.env.reset()[0] if done else obs
        return float(r), done


# --- REINFORCE ----------------------------------------------------------------------------
def reinforce(seed, steps, baseline=False, batch_steps=4096, lr=3e-4, gamma=GAMMA,
              value_lr=1e-3, value_epochs=10, value_batch=256, eval_freq=25_000,
              env_kw=None, policy=None):
    """Monte-Carlo policy gradient, optionally with a learned state-value baseline.

    The critic gets its own optimiser and several passes per update: at one full-batch step per
    4096 transitions it would see ~60 updates over the whole budget and never become a baseline
    at all, which would make `baseline=True` an expensive way to rerun `baseline=False`.
    """
    env_kw = env_kw or {}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_train_env(seed, **env_kw)
    policy = policy or PGPolicy(with_value=baseline)
    if baseline and policy.v_trunk is None:
        raise ValueError('baseline=True needs a policy built with with_value=True')

    opt = torch.optim.Adam(policy.policy_parameters(), lr=lr)
    vopt = torch.optim.Adam(policy.value_parameters(), lr=value_lr) if baseline else None
    stream = _Stream(env)
    curve = _Curve(policy, eval_freq, env_kw)
    updates = []
    t = 0

    while t < steps:
        ob, ac, rw, dn = [], [], [], []
        for _ in range(batch_steps):
            o = stream.obs
            a = policy.act(o)
            r, done = stream.step(a)
            ob.append(o); ac.append(a); rw.append(r); dn.append(done)
        t += batch_steps

        obs = torch.as_tensor(np.asarray(ob), dtype=torch.float32)
        acts = torch.as_tensor(np.asarray(ac), dtype=torch.int64)
        G = torch.as_tensor(_return_to_go(rw, dn, gamma), dtype=torch.float32)
        with torch.no_grad():
            old_logits = policy.logits(obs)
            entropy = float(torch.distributions.Categorical(logits=old_logits).entropy().mean())
            adv = G - policy.value(obs) if baseline else G

        cos, rel = _grad_noise(policy, obs, acts, adv, rng)

        loss = -(policy.dist(obs).log_prob(acts) * adv).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        vloss = None
        if baseline:
            for _ in range(value_epochs):
                perm = torch.randperm(len(obs))
                for i in range(0, len(obs), value_batch):
                    idx = perm[i:i + value_batch]
                    vl = ((policy.value(obs[idx]) - G[idx]) ** 2).mean()
                    vopt.zero_grad(set_to_none=True)
                    vl.backward()
                    vopt.step()
            vloss = float(vl.detach())

        with torch.no_grad():
            kl = float(_cat_kl(policy.logits(obs), old_logits).mean())

        updates.append(dict(timesteps=t, mean_return=float(G.mean()),
                            reward_per_step=float(np.mean(rw)),
                            advantage_var=float(adv.var()), grad_cos=cos, grad_rel_l2=rel,
                            kl=kl, entropy=entropy, loss=float(loss.detach()), value_loss=vloss,
                            episodes=int(np.sum(dn))))
        curve.maybe(t)

    curve.evaluate(t)
    env.close()
    return policy, updates, curve.as_dict()


# --- GRPO ---------------------------------------------------------------------------------
def assert_deepcopy_determinism(env, n=40):
    """Two deepcopies of the same env must replay one action sequence identically.

    The group construction rests on this: if the copies diverged, the members would not share a
    start state and the group-relative advantage would compare unrelated rollouts.
    """
    env.reset()
    a, b = copy.deepcopy(env), copy.deepcopy(env)
    acts = np.random.default_rng(0).integers(0, 4, n)
    for k in acts:
        (o1, r1, t1, _, _), (o2, r2, t2, _, _) = a.step(int(k)), b.step(int(k))
        assert r1 == r2 and t1 == t2 and np.array_equal(o1, o2), 'env deepcopy is not exact'
        if t1:
            break


def _event_proximal_start(env, search_cap):
    """Drive the scripted rule policy forward until a blockage or a low battery is in reach."""
    predict = make_scripted_predict(norm=True)
    obs, _ = env.reset()
    used = 0
    for _ in range(search_cap):
        base = env.unwrapped
        if base._main_blocked or base._soc < EVENT_SOC:
            return obs, True, used
        obs, _, term, trunc, _ = env.step(int(predict(obs)))
        used += 1
        if term or trunc:
            obs, _ = env.reset()
    return obs, False, used


def grpo(seed, steps, group_size=8, segment_len=256, start_mix=0.5, kl_coef=0.02,
         clip_range=0.2, n_epochs=10, minibatch=256, lr=3e-4, gamma=GAMMA, search_cap=3000,
         eval_freq=25_000, env_kw=None, policy=None, ref_policy=None):
    """Group-relative policy optimisation, simplified to the discrete rover MDP.

    `start_mix` is the probability that an update starts from an *event-proximal* state rather
    than from a fresh episode start (see the module docstring for why a pure fresh-start prompt
    distribution cannot work at this segment length).

    `ref_policy` defaults to a frozen copy of the initial policy, so a BC-initialised run keeps
    its behaviour-cloned checkpoint as the KL reference — the rover analogue of GRPO anchoring
    on the SFT model it was started from.
    """
    env_kw = env_kw or {}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_train_env(seed, **env_kw)
    policy = policy or PGPolicy()
    ref = ref_policy or copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)

    assert_deepcopy_determinism(env)
    opt = torch.optim.Adam(policy.policy_parameters(), lr=lr)
    curve = _Curve(policy, eval_freq, env_kw)
    updates = []
    t = 0

    while t < steps:
        event = bool(rng.random() < start_mix)
        if event:
            obs0, found, search = _event_proximal_start(env, search_cap)
        else:
            obs0, _ = env.reset()
            found, search = False, 0

        # One start state, `group_size` independent continuations of it. The TimeLimit budget
        # left at the start state is shared by all members, so a shortened segment shortens the
        # whole group and leaves the within-group comparison intact.
        ob, ac, rets, lens, raw = [], [], [], [], 0.0
        for _ in range(group_size):
            member = copy.deepcopy(env)
            o, disc, g = obs0, 0.0, 1.0
            k = 0
            for _ in range(segment_len):
                a = policy.act(o)
                o2, r, term, trunc, _ = member.step(a)
                ob.append(o); ac.append(a)
                disc += g * r
                raw += r
                g *= gamma
                k += 1
                o = o2
                if term or trunc:
                    break
            member.close()
            rets.append(disc)
            lens.append(k)

        R = np.asarray(rets)
        A = (R - R.mean()) / (R.std() + 1e-8)
        adv_flat = np.repeat(A, lens)          # one scalar per trajectory, held over its steps
        n = int(np.sum(lens))
        t += n

        obs = torch.as_tensor(np.asarray(ob), dtype=torch.float32)
        acts = torch.as_tensor(np.asarray(ac), dtype=torch.int64)
        adv = torch.as_tensor(adv_flat, dtype=torch.float32)
        with torch.no_grad():
            old_logits = policy.logits(obs)
            old_logp = torch.distributions.Categorical(logits=old_logits).log_prob(acts)
            ref_logits = ref.logits(obs)
            entropy = float(torch.distributions.Categorical(logits=old_logits).entropy().mean())

        cos, rel = _grad_noise(policy, obs, acts, adv, rng)

        for _ in range(n_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, minibatch):
                idx = perm[i:i + minibatch]
                logits = policy.logits(obs[idx])
                logp = torch.distributions.Categorical(logits=logits).log_prob(acts[idx])
                ratio = (logp - old_logp[idx]).exp()
                a_i = adv[idx]
                surr = -torch.min(ratio * a_i,
                                  ratio.clamp(1 - clip_range, 1 + clip_range) * a_i).mean()
                kl_pen = _cat_kl(logits, ref_logits[idx]).mean()
                opt.zero_grad(set_to_none=True)
                (surr + kl_coef * kl_pen).backward()
                opt.step()

        with torch.no_grad():
            kl = float(_cat_kl(policy.logits(obs), ref_logits).mean())
            kl_step = float(_cat_kl(policy.logits(obs), old_logits).mean())

        updates.append(dict(timesteps=t, mean_return=float(R.mean()),
                            group_return_std=float(R.std()),
                            reward_per_step=float(raw / max(n, 1)),
                            advantage_var=float(adv.var()), grad_cos=cos, grad_rel_l2=rel,
                            kl=kl, kl_step=kl_step, entropy=entropy, event_start=event,
                            event_found=found, search_steps=search,
                            mean_segment_len=float(np.mean(lens))))
        curve.maybe(t)

    curve.evaluate(t)
    env.close()
    return policy, updates, curve.as_dict()


def load_policy_predict(path, obs_dim=9, n_actions=5):
    """Greedy predict callable from a saved `PGPolicy` state_dict, for the evaluation pass."""
    state = torch.load(path, map_location='cpu')
    net = PGPolicy(obs_dim=obs_dim, n_actions=n_actions,
                   with_value=any(k.startswith('v_trunk') for k in state))
    net.load_state_dict(state)
    net.eval()
    return net.predict
