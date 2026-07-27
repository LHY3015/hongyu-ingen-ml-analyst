"""Maximum-entropy inverse RL on the Week-2 rover demonstrations, scored against the true reward.

Why this setup is unusually checkable
-------------------------------------
An IRL demonstration normally ends at "the recovered reward looks plausible", because the reward
that produced the demonstrations is unknown. Here it is known exactly and it is a small table:
`rl.rover_env.compute_reward` is the function the offline generator and the online env both call.
Its structure is, for `label == 0`, `NORMAL_REWARD[action]` with three overrides — `(action 0,
halted)` pays -0.5, `(action 2, main_blocked)` pays 0.0, `(action 1, rough_terrain)` pays 0.0 —
for `label == 1`, `ANOMALY_REWARD[action]`; then `+2.0` when `battery_soc < LOW_SOC` and the action
is `return-to-base`, and `-ENERGY_PER_STEP` on every step. That is *exactly linear* in indicator
features over `(label, action, halted, main_blocked, rough, soc < LOW_SOC)`, so a feature set built
from those indicators contains the true reward as a single weight vector and recovery can be scored
rather than eyeballed.

Identifiability, and what it costs here
---------------------------------------
A linear reward is recoverable only up to a positive affine transformation: adding a constant to
`r(s, a)` for all `(s, a)` leaves every policy and every soft-Bellman backup unchanged, and scaling
`w` by `c > 0` only sharpens the soft-optimal policy, so a MaxEnt fit to a near-deterministic
expert inflates the reward scale instead of converging on the true magnitude — here the
least-squares fit of `r_true` against `r_w` comes out at slope 0.588. Pearson and Spearman
correlation are themselves invariant to that transformation, so they are the natural scores here
and no separate "after removing the constant and scale" number exists — the quantity that does
change is the residual, so `validate` additionally fits `r_true ≈ alpha * r_w + beta` by least
squares and reports `alpha`, `beta`, R² and the normalised RMSE of that fit.

The affine freedom is the textbook caveat. The binding one on this data set is narrower and shows
up in the numbers: `phi_priv` contains the true reward exactly (`TRUE_W_PRIV` reproduces
`true_reward_table` to 2e-16), yet the fit recovers it only to Pearson 0.458. The reason is the
demonstrator. The logged expert is a deterministic privileged rule: of its 8,549 anomaly steps it
raises an alert on 8,239 and takes `slow`, `reroute`, `continue` and `return-to-base` on 101, 114,
94 and 1. The true reward spreads those four over 8.0 points (`continue` -5.05, `slow` +1.45,
`reroute` -0.05, `return-to-base` +2.95); the fit collapses them into a band 0.75 wide, because
"the expert essentially never does this" is all 48,000 demonstration steps say about any of them.
`low_soc:return-to-base` is worse: the eight episodes contain one low-SoC step and no dock, so the
+2.0 docking bonus comes back as 0.01 — not approximately, but with no evidence at all.

Two checks separate this from a broken fitter. `positive_control` refits with the logged expert
replaced by the soft-optimal policy under the true reward on the same MDP — the behaviour MaxEnt
actually assumes, and one that takes every action with positive probability — and recovers
Pearson 0.947 / Spearman 0.970, so the code, the occupancy solve and the feature set are sound.
Its one remaining miss is `low_soc:return-to-base`, still 0.00: no demonstrator can fix a region
`mu_0` gives no mass to. And driving the optimiser harder makes the score *worse*, not better: at
`lr = 1.0, n_iter = 800` the feature-expectation error falls to 0.022 and Spearman to 0.31, where
a deliberately under-converged fit (`lr = 0.3, n_iter = 200`, error 0.105) sits at 0.49. The
defaults below are the best-converged setting, not the best-scoring one.

Discretisation
--------------
Seven axes, all reconstructed from the state columns the way `rover_env._cache_conditioning` does:

| axis             | values | rule                                                          |
| ---------------- | ------ | ------------------------------------------------------------- |
| `label`          | 2      | `actual_label` (offline) / `env._last_label` (online)          |
| `halted`         | 2      | `next_main_block_dist < 2.0`                                   |
| `main_blocked`   | 2      | `next_main_block_dist < NEAR`                                  |
| `branch_blocked` | 2      | `branch_block_dist < NEAR`                                     |
| `rough`          | 2      | `torque_mean > ROUGH_TERRAIN_TORQUE`                           |
| `low_soc`        | 2      | `battery_soc < LOW_SOC`                                        |
| `tstd_bin`       | 3      | `torque_std` against the two class medians of the offline table |

192 index cells, of which 57 are visited by the eight demonstration episodes; the tabular MDP is
built over those 57. `halted` is not a state column — the Week-5 reference policies approximate it
as `next_main_block_dist < 2.0`, and the same approximation is used here. It is not exact: the
world halts on the step the clearance is actually consumed, so 30 of the 48,000 logged rows
disagree. `build_tabular_mdp` recomputes `compute_reward` from the reconstructed axes and returns
the fraction of rows whose logged reward it reproduces (0.99938) as `reward_match`.

The `tstd_bin` edges are the medians of `s_torque_std` under each label (0.535 normal, 2.765
anomaly), the three-way generalisation of the Week-5 tabular split, which used the midpoint of the
same two numbers. Nothing is hand-picked.

The two feature sets
--------------------
`phi_priv` uses the ground-truth `label` indicator crossed with the action one-hot, plus the three
label-0 context overrides and the low-SoC dock bonus: 14 features, and the true reward is one of
them (`TRUE_W_PRIV`). `phi_obs` replaces *every* occurrence of the label indicator with the
observable `tstd_bin` one-hot — the anomaly signature is torque *variability*, not mean, since
terrain confounds the mean and wet grass baselines above 30 Nm. That substitution leaves `phi_obs`
with 25 features against `phi_priv`'s 14 — more parameters, not more reach, since none of them is
a function of the label. Its degradation therefore cannot be blamed on capacity.

The expert is privileged: it alerts on the ground-truth label, which no function of the nine sensor
channels reproduces. So `phi_obs` cannot express the rule it is imitating, and the same gap that
capped Week-5 behaviour cloning's raise-alert recall at 0.50 caps what IRL recovers here. The
`phi_priv` / `phi_obs` contrast is what this module is for; the absolute level of either is set by
the identifiability limit above, so the two are only meaningful read against each other and
against the positive control.

Occupancy and matching
----------------------
Ziebart-style: soft value iteration under `r_w`, discounted state-action visitation of the induced
soft policy, gradient step on `w` toward the expert's empirical feature expectation, repeat. The
occupancy is solved as the linear system `(I - gamma * M^T) mu = mu_0` rather than rolled forward,
which is exact and costs one 57x57 solve.

`mu_0` is the empirical state marginal over all 48,000 demonstration steps, not the distribution of
episode starts. The demonstrations are eight 6,000-step episodes while `1/(1 - gamma) = 100`, so an
episode-start `mu_0` would confine the matched occupancy to the first ~1.7 % of each episode. With
the marginal instead, the fit is self-consistent: if the learner's policy equals the expert's and
that marginal is stationary under it, the normalised occupancy reproduces the empirical joint
exactly and the gradient vanishes.

Terminal transitions are handled by letting `P[s, a, :]` sum to less than one — the missing mass is
the probability of ending the episode, so it contributes zero continuation value and drops out of
the occupancy automatically. `(s, a)` pairs with no logged data (101 of the 285) get a self-loop.

A Gaussian prior (`l2`) keeps `‖w‖` finite. Without it the objective has no maximum: the expert is
deterministic, no softmax policy matches a deterministic one exactly, and the fit improves
monotonically as the weights are scaled up.

Deriving a policy
-----------------
`make_predict` solves the same tabular MDP for the hard-optimal policy under `r_w` and looks the
action up by cell. The `phi_priv` policy reads the label, so it is evaluated with `env_aware=True`
alongside the `ExpertAware` ceiling. The `phi_obs` policy must not, so its action comes from the
label-marginalised Q-values, `Q_o(a) = sum_l P_expert(l | o) Q(l, o, a)` — a QMDP approximation,
weighted by offline visitation, which reads nothing the deployed rover cannot measure. Cells absent
from the demonstrations back off to the nearest known cell by Hamming distance over the axes, ties
broken by expert visitation; the predictor counts how often that happens.

Results
-------
`python -m rl.irl`, gamma = 0.99, 800 gradient steps at lr = 1.0 with l2 = 1e-3. Soft value
iteration converges in ~2,210 sweeps to a Bellman residual of 1.0e-9 at every gradient step, and
the final gradient norm is 4e-4 (`priv`) / 6e-4 (`obs`).

|                                        | `phi_priv`   | `phi_obs`     | control (`phi_priv`) |
| -------------------------------------- | ------------ | ------------- | -------------------- |
| features                               | 14           | 25            | 14                   |
| Pearson `r_w` vs `r_true`, 285 cells   | **0.458**    | **-0.023**    | 0.947                |
| Spearman                               | **0.309**    | **0.080**     | 0.970                |
| Pearson, the 184 demonstrated cells    | 0.559        | -0.008        |                      |
| affine fit `alpha`                     | 0.588        | -0.033        |                      |
| online return, 10 worlds               | **11,285**   | **2,423**     |                      |
| alert recall                           | 1.000        | 0.118         |                      |
| false alerts per 1k normal steps       | 0.18         | 103.3         |                      |
| reroute rate on a single blockage      | 0.638        | 0.327         |                      |

Floor `ScriptedBlind` 3,930.6 +/- 615.9, ceiling `ExpertAware` 11,574.6 +/- 1,343.2, both over the
same ten worlds and the same 9,600-step horizon.

The `phi_priv` policy lands at 96.2 % of the floor-to-ceiling range and inside the bracket, which
is the strongest statement available here: whatever the recovered reward gets wrong about
never-demonstrated actions does not matter, because the optimal policy under it almost reproduces
the privileged expert. The `phi_obs` policy lands *below* the floor at 2,423. It is not merely a
weaker imitator: it alerts on 103 of every 1,000 normal steps, trading +1.00 for -1.50 each time,
and catches 12 % of anomalies in exchange. Its episodes also end at 6,009 steps against the
floor's 9,009, because it takes the branch on only 33 % of single blockages and the rest run into
the stuck timeout. A reward fitted to explain label-driven alerting with a feature that does not
carry the label produces a policy worse than not trying to alert at all.

Both recovered policies select `return-to-base` zero times in 145,000 evaluation steps, which is
the same conclusion Week 5 reached from DQN, FQI and behaviour cloning: -3.05 against +0.95/step
means docking is nowhere the argmax, so the recovered reward reproduces the property rather than
missing it.

Usage
-----
    python -m rl.irl

Results are cached in `rl/w6_artifacts/irl_{priv,obs}.json`, `irl_control_priv.json` and
`irl_brackets.json`; delete those to refit, or pass `force=True`.
"""

import json
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

from rl.rover_env import (ACTION_NAMES, STATE_COLS, NEAR, ROUGH_TERRAIN_TORQUE, LOW_SOC,
                          ENERGY_PER_STEP, NORMAL_REWARD, ANOMALY_REWARD, compute_reward)
from shared_modules.rl_eval import (ROOT, ART_DIR, GAMMA, EVAL_WORLD_SEEDS, evaluate,
                          ScriptedBlind, ExpertAware)

N_ACTIONS = len(ACTION_NAMES)
HALT_DIST = 2.0                  # Week-5 stand-in for the world's `halted` flag
AXES = ('label', 'halted', 'main_blocked', 'branch_blocked', 'rough', 'low_soc', 'tstd_bin')
AXIS_SIZES = (2, 2, 2, 2, 2, 2, 3)
N_CELLS = int(np.prod(AXIS_SIZES))
OBS_AXES = tuple(i for i, name in enumerate(AXES) if name != 'label')   # label-free projection

TRANSITIONS = ROOT / 'data' / 'rover_transitions.csv'

# The weight vector `phi_priv` needs to reproduce `compute_reward` exactly, in the feature order
# `_features` emits. The per-action entries carry the -ENERGY_PER_STEP charge; the three override
# entries carry the difference between the override and the normal-table value they replace.
TRUE_W_PRIV = np.array(
    [NORMAL_REWARD[a] - ENERGY_PER_STEP for a in range(N_ACTIONS)] +
    [ANOMALY_REWARD[a] - ENERGY_PER_STEP for a in range(N_ACTIONS)] +
    [-0.5 - NORMAL_REWARD[0], 0.0 - NORMAL_REWARD[2], 0.0 - NORMAL_REWARD[1], 2.0])


# --- discretisation -------------------------------------------------------------------------
def tstd_edges(df):
    """`torque_std` bin edges: the per-label medians of the offline table."""
    return np.array([df.loc[df.actual_label == 0, 's_torque_std'].median(),
                     df.loc[df.actual_label == 1, 's_torque_std'].median()], dtype=np.float64)


def state_axes(x, label, edges):
    """Axis matrix (N, 7) for raw 9-D states `x` in `STATE_COLS` order plus their labels."""
    x = np.atleast_2d(np.asarray(x, np.float64))
    col = lambda name: x[:, STATE_COLS.index(name)]
    main_d, branch_d = col('next_main_block_dist'), col('branch_block_dist')
    return np.stack([np.broadcast_to(np.asarray(label, int), (len(x),)),
                     (main_d < HALT_DIST).astype(int),
                     (main_d < NEAR).astype(int),
                     (branch_d < NEAR).astype(int),
                     (col('torque_mean') > ROUGH_TERRAIN_TORQUE).astype(int),
                     (col('battery_soc') < LOW_SOC).astype(int),
                     np.digitize(col('torque_std'), edges)], axis=1)


def cell_index(axes):
    return np.ravel_multi_index(np.asarray(axes, int).T, AXIS_SIZES)


# --- tabular MDP ----------------------------------------------------------------------------
def build_tabular_mdp(df=None, gamma=GAMMA):
    """Empirical tabular MDP over the visited cells of the discretisation.

    Returns `P` (S, A, S) with row sums below one where episodes end, the expert's empirical
    state-action joint, the occupancy start distribution, per-cell context for scoring against
    `compute_reward`, and the reward-reconstruction match rate.
    """
    if df is None:
        df = pd.read_csv(TRANSITIONS)
    edges = tstd_edges(df)
    label = df['actual_label'].to_numpy(int)
    xs = df[[f's_{c}' for c in STATE_COLS]].to_numpy(np.float64)
    xn = df[[f'ns_{c}' for c in STATE_COLS]].to_numpy(np.float64)
    # `ns_*` has no label column, but row i+1 of the same episode is that successor, so its
    # `actual_label` is the exact label of `s'`; only each episode's last row has to carry over
    ep = df['episode_id'].to_numpy()
    label_next = np.where(np.append(ep[1:] == ep[:-1], False), np.append(label[1:], label[-1]), label)
    ax_s = state_axes(xs, label, edges)
    ax_n = state_axes(xn, label_next, edges)

    c_s, c_n = cell_index(ax_s), cell_index(ax_n)
    states = np.unique(np.concatenate([c_s, c_n]))
    lookup = np.full(N_CELLS, -1, int)
    lookup[states] = np.arange(len(states))
    S = len(states)
    i_s, i_n = lookup[c_s], lookup[c_n]
    act = df['action'].to_numpy(int)
    done = df['done'].to_numpy(bool)

    counts = np.zeros((S, N_ACTIONS, S))
    np.add.at(counts, (i_s[~done], act[~done], i_n[~done]), 1.0)
    term = np.zeros((S, N_ACTIONS))
    np.add.at(term, (i_s[done], act[done]), 1.0)
    total = counts.sum(2) + term
    P = np.zeros((S, N_ACTIONS, S))
    seen = total > 0
    P[seen] = counts[seen] / total[seen][:, None]
    # unobserved (s, a): the model has nothing to say, so the pair is a self-loop rather than an
    # absorbing sink, which would make every unexplored action look terminal and therefore free
    for s, a in zip(*np.where(~seen)):
        P[s, a, s] = 1.0

    n_rows = np.zeros((S, N_ACTIONS))
    np.add.at(n_rows, (i_s, act), 1.0)
    expert_visit = n_rows / n_rows.sum()

    axes = np.array(np.unravel_index(states, AXIS_SIZES)).T
    soc = df['s_battery_soc'].to_numpy(np.float64)
    soc_median = np.array([np.median(soc[i_s == s]) if (i_s == s).any()
                           else (LOW_SOC / 2 if axes[s, 5] else 100.0) for s in range(S)])

    r_hat = np.array([compute_reward(l, a, bool(h), bool(m), bool(g), sc) for l, a, h, m, g, sc in
                      zip(label, act, ax_s[:, 1], ax_s[:, 2], ax_s[:, 4], soc)])
    reward_match = float(np.mean(np.abs(r_hat - df['reward'].to_numpy(np.float64)) < 1e-9))

    return dict(edges=edges, states=states, lookup=lookup, axes=axes, P=P, term=term / np.maximum(total, 1),
                expert_visit=expert_visit, mu0=expert_visit.sum(1), n_rows=n_rows,
                soc_median=soc_median, gamma=gamma, n_states=S, n_cells=N_CELLS,
                n_rows_total=int(len(df)), n_sa_seen=int(seen.sum()), reward_match=reward_match)


def true_reward_table(mdp):
    """`compute_reward` evaluated on every (cell, action) of the MDP.

    The discretisation axes *are* the reward's conditioning variables, so only `battery_soc` is
    coarsened; the cell's median SoC stands in for it and enters `compute_reward` solely through
    the `< LOW_SOC` comparison, which the `low_soc` axis already fixes.
    """
    ax, soc = mdp['axes'], mdp['soc_median']
    return np.array([[compute_reward(ax[s, 0], a, bool(ax[s, 1]), bool(ax[s, 2]), bool(ax[s, 4]),
                                     soc[s]) for a in range(N_ACTIONS)]
                     for s in range(mdp['n_states'])])


# --- feature sets ---------------------------------------------------------------------------
def _features(mdp, feature_set):
    """Indicator feature tensor (S, A, F) and the feature names."""
    ax = mdp['axes']
    S = mdp['n_states']
    label, halted, main_b, rough, low_soc = ax[:, 0], ax[:, 1], ax[:, 2], ax[:, 4], ax[:, 5]
    if feature_set == 'priv':
        group, gname = label, ['normal', 'anomaly']
    elif feature_set == 'obs':
        group, gname = ax[:, 6], ['tstd_lo', 'tstd_mid', 'tstd_hi']
    else:
        raise ValueError(f'unknown feature_set {feature_set!r}')
    G = len(gname)

    cols, names = [], []
    for g in range(G):
        in_g = (group == g).astype(float)
        for a in range(N_ACTIONS):
            f = np.zeros((S, N_ACTIONS))
            f[:, a] = in_g
            cols.append(f)
            names.append(f'{gname[g]}:{ACTION_NAMES[a]}')
    # the three overrides `compute_reward` applies inside the label-0 branch; under `obs` the
    # group index replaces the label everywhere it appears, hence one copy per bucket
    for g in range(G):
        in_g = (group == g).astype(float)
        for a, cond, cname in [(0, halted, 'halted'), (2, main_b, 'main_blocked'),
                               (1, rough, 'rough')]:
            f = np.zeros((S, N_ACTIONS))
            f[:, a] = in_g * cond
            cols.append(f)
            names.append(f'{gname[g]}:{ACTION_NAMES[a]}:{cname}')
        if feature_set == 'priv':
            break            # the true reward applies the overrides only in the label-0 branch
    f = np.zeros((S, N_ACTIONS))
    f[:, 4] = low_soc
    cols.append(f)
    names.append('low_soc:return-to-base')
    return np.stack(cols, axis=2), names


# --- solvers --------------------------------------------------------------------------------
def soft_value_iteration(R, P, gamma, tol=1e-9, max_iter=20_000):
    """Ziebart soft backup: V(s) = logsumexp_a [ r(s,a) + gamma * sum_s' P(s'|s,a) V(s') ]."""
    S, A = R.shape
    Pf = P.reshape(S * A, S)
    V = np.zeros(S)
    resid = np.inf
    for it in range(1, max_iter + 1):
        Q = R + gamma * Pf.dot(V).reshape(S, A)
        Vn = logsumexp(Q, axis=1)
        resid = float(np.abs(Vn - V).max())
        V = Vn
        if resid < tol:
            break
    Q = R + gamma * Pf.dot(V).reshape(S, A)
    return V, Q, np.exp(Q - logsumexp(Q, axis=1)[:, None]), it, resid


def hard_value_iteration(R, P, gamma, tol=1e-9, max_iter=20_000):
    S, A = R.shape
    Pf = P.reshape(S * A, S)
    V = np.zeros(S)
    resid = np.inf
    for it in range(1, max_iter + 1):
        Q = R + gamma * Pf.dot(V).reshape(S, A)
        Vn = Q.max(1)
        resid = float(np.abs(Vn - V).max())
        V = Vn
        if resid < tol:
            break
    Q = R + gamma * Pf.dot(V).reshape(S, A)
    return V, Q, it, resid


def occupancy(P, pi, mu0, gamma):
    """Discounted state-action visitation of `pi`, normalised to sum to one.

    Solved directly: mu = mu0 + gamma * M^T mu with M(s, s') = sum_a pi(a|s) P(s'|s, a).
    """
    S, A = pi.shape
    M = np.einsum('sa,sap->sp', pi, P)
    mu = np.linalg.solve(np.eye(S) - gamma * M.T, mu0)
    D = mu[:, None] * pi
    return D / D.sum()


# --- MaxEnt IRL -----------------------------------------------------------------------------
def _fit(phi, f_expert, P, mu0, gamma, n_iter, lr, l2):
    """Gradient ascent on the MaxEnt log-likelihood: grad = f_expert - f_learner(w) - l2 * w."""
    w = np.zeros(phi.shape[2])
    hist = dict(grad_norm=[], fe_error=[], w_norm=[], svi_iters=[], svi_resid=[])
    for _ in range(n_iter):
        _, _, pi, svi_it, svi_res = soft_value_iteration(phi.dot(w), P, gamma)
        f_learner = np.einsum('sa,saf->f', occupancy(P, pi, mu0, gamma), phi)
        grad = f_expert - f_learner - l2 * w
        w = w + lr * grad
        hist['grad_norm'].append(float(np.linalg.norm(grad)))
        hist['fe_error'].append(float(np.abs(f_expert - f_learner).sum()))
        hist['w_norm'].append(float(np.linalg.norm(w)))
        hist['svi_iters'].append(int(svi_it))
        hist['svi_resid'].append(float(svi_res))
    return w, hist


def run_maxent_irl(mdp, feature_set='priv', n_iter=800, lr=1.0, l2=1e-3, force=False):
    """Fit `w` so the soft-optimal policy under `r_w = w . phi` matches the expert's feature
    expectations. Returns the weights, the per-iteration diagnostics and the reward table."""
    path = ART_DIR / f'irl_{feature_set}.json'
    if path.exists() and not force:
        cached = json.loads(path.read_text())
        cached['w'] = np.array(cached['w'])
        cached['reward'] = np.array(cached['reward'])
        return cached

    phi, names = _features(mdp, feature_set)
    f_expert = np.einsum('sa,saf->f', mdp['expert_visit'], phi)
    t0 = time.time()
    w, hist = _fit(phi, f_expert, mdp['P'], mdp['mu0'], mdp['gamma'], n_iter, lr, l2)
    out = dict(feature_set=feature_set, feature_names=names, w=w, reward=phi.dot(w),
               f_expert=f_expert.tolist(), n_iter=n_iter, lr=lr, l2=l2, gamma=mdp['gamma'],
               n_features=len(names), history=hist, seconds=round(time.time() - t0, 1))
    _write(path, out)
    return out


def positive_control(mdp, feature_set='priv', n_iter=800, lr=1.0, l2=1e-3, force=False):
    """Refit with the logged expert replaced by the soft-optimal policy under the TRUE reward.

    Two things can make the logged-expert fit miss the true reward: an error in the fitting code,
    and a demonstration set that does not pin the reward down. This synthetic expert is exactly
    the behaviour MaxEnt assumes and it visits every action with positive probability, so its
    recovery score is the ceiling the fitter can reach on this MDP and feature set. Whatever the
    logged-expert fit falls short of that ceiling by is a property of the demonstrations.
    """
    path = ART_DIR / f'irl_control_{feature_set}.json'
    if path.exists() and not force:
        return json.loads(path.read_text())

    phi, names = _features(mdp, feature_set)
    P, gamma, mu0 = mdp['P'], mdp['gamma'], mdp['mu0']
    _, _, pi_true, _, _ = soft_value_iteration(true_reward_table(mdp), P, gamma)
    f_expert = np.einsum('sa,saf->f', occupancy(P, pi_true, mu0, gamma), phi)
    w, hist = _fit(phi, f_expert, P, mu0, gamma, n_iter, lr, l2)

    r_w, r_true = phi.dot(w).ravel(), true_reward_table(mdp).ravel()
    out = dict(feature_set=feature_set, feature_names=names, w=w.tolist(),
               pearson=float(pearsonr(r_w, r_true)[0]), spearman=float(spearmanr(r_w, r_true)[0]),
               final_grad_norm=hist['grad_norm'][-1], final_fe_error=hist['fe_error'][-1],
               n_iter=n_iter, lr=lr, l2=l2)
    path.write_text(json.dumps(out))
    return out


def _write(path, out):
    serialisable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()}
    path.write_text(json.dumps(serialisable))


# --- policy extraction ----------------------------------------------------------------------
def make_predict(result, mdp):
    """Greedy policy under the recovered reward, as a callable on a raw 9-D observation.

    Accepts `(obs)` or `(obs, base)` so it drops into `w6_common.rollout` either way, but the
    `priv` policy needs the second argument: `result['feature_set'] == 'priv'` is the case that
    must be evaluated with `env_aware=True`, since its cell includes the ground-truth label.
    """
    P, gamma = mdp['P'], mdp['gamma']
    _, Q, _, _ = hard_value_iteration(np.asarray(result['reward']), P, gamma)
    ax = mdp['axes']
    priv = result['feature_set'] == 'priv'

    if priv:
        keys = ax
        action = Q.argmax(1)
        weight = mdp['expert_visit'].sum(1)
    else:
        # collapse the label axis by expert-weighted averaging of the Q-values (QMDP): the
        # observable policy may not condition on a variable the rover cannot measure
        keys, inv = np.unique(ax[:, OBS_AXES], axis=0, return_inverse=True)
        vis = mdp['expert_visit'].sum(1)
        num = np.zeros((len(keys), N_ACTIONS))
        den = np.zeros(len(keys))
        np.add.at(num, inv, vis[:, None] * Q)
        np.add.at(den, inv, vis)
        # cells with no demonstration mass fall back to unweighted averaging
        flat = np.zeros((len(keys), N_ACTIONS))
        np.add.at(flat, inv, Q)
        cnt = np.bincount(inv, minlength=len(keys)).astype(float)
        action = np.where(den[:, None] > 0, num / np.maximum(den, 1e-12)[:, None],
                          flat / cnt[:, None]).argmax(1)
        weight = den

    edges = mdp['edges']
    order = np.lexsort((-weight,))            # nearest-cell ties go to the better-supported cell
    keys_o, action_o = keys[order], action[order]

    def predict(obs, base=None):
        label = int(base._last_label) if priv else 0
        a = state_axes(np.asarray(obs, np.float64), label, edges)[0]
        key = a if priv else a[list(OBS_AXES)]
        hit = np.flatnonzero((keys_o == key).all(1))
        if hit.size:
            return int(action_o[hit[0]])
        predict.n_fallback += 1
        return int(action_o[np.abs(keys_o - key).astype(bool).sum(1).argmin()])

    predict.n_fallback = 0
    return predict


# --- validation -----------------------------------------------------------------------------
def _bracket_returns(force=False):
    """Floor (`ScriptedBlind`) and ceiling (`ExpertAware`) on the ten evaluation worlds."""
    path = ART_DIR / 'irl_brackets.json'
    if path.exists() and not force:
        return json.loads(path.read_text())
    out = {}
    for name, predict, aware in [('scripted_blind', ScriptedBlind(norm=False), False),
                                 ('expert_aware', ExpertAware(norm=False), True)]:
        rows, s = evaluate(predict, world_seeds=EVAL_WORLD_SEEDS, norm=False, env_aware=aware)
        out[name] = dict(return_mean=float(s['return_mean']), return_std=float(s['return_std']),
                         len_mean=float(s['len_mean']),
                         per_world=[float(r['ret']) for r in rows])
    path.write_text(json.dumps(out))
    return out


def validate(result, mdp, force=False):
    """Score the recovered reward two ways: against the true reward table, and online.

    (1) Pearson / Spearman between `r_w` and `compute_reward` over every (cell, action) of the
        MDP, uniformly and weighted by expert visitation, plus the least-squares affine fit that
        the identifiability freedom leaves open.
    (2) Return of the greedy policy under `r_w`, measured under the TRUE reward in the real
        environment over `EVAL_WORLD_SEEDS`, against the scripted floor and privileged ceiling.
    """
    path = ART_DIR / f'irl_{result["feature_set"]}.json'
    cached = json.loads(path.read_text()) if path.exists() else {}
    if 'validation' in cached and not force:
        return cached['validation']

    r_w = np.asarray(result['reward']).ravel()
    r_true = true_reward_table(mdp).ravel()
    vis = mdp['expert_visit'].ravel()
    alpha, beta = np.polyfit(r_w, r_true, 1)
    resid = r_true - (alpha * r_w + beta)
    wm = lambda x, y: float(np.cov(x, y, aweights=vis)[0, 1] /
                            np.sqrt(np.cov(x, x, aweights=vis)[0, 1] * np.cov(y, y, aweights=vis)[0, 1]))
    sup = vis > 0
    reward_metrics = dict(
        pearson=float(pearsonr(r_w, r_true)[0]), spearman=float(spearmanr(r_w, r_true)[0]),
        pearson_supported=float(pearsonr(r_w[sup], r_true[sup])[0]),
        spearman_supported=float(spearmanr(r_w[sup], r_true[sup])[0]),
        pearson_visit_weighted=wm(r_w, r_true),
        affine_alpha=float(alpha), affine_beta=float(beta),
        affine_r2=float(1 - resid.var() / r_true.var()),
        affine_nrmse=float(np.sqrt((resid ** 2).mean()) / (r_true.max() - r_true.min())),
        n_sa=int(r_w.size), n_sa_supported=int(sup.sum()))

    predict = make_predict(result, mdp)
    aware = result['feature_set'] == 'priv'
    rows, summary = evaluate(predict, world_seeds=EVAL_WORLD_SEEDS, norm=False, env_aware=aware)
    br = _bracket_returns()
    ret = float(summary['return_mean'])
    floor, ceil = br['scripted_blind']['return_mean'], br['expert_aware']['return_mean']
    policy_metrics = dict(
        return_mean=ret, return_std=float(summary['return_std']),
        len_mean=float(summary['len_mean']),
        per_world=[float(r['ret']) for r in rows],
        floor=floor, ceiling=ceil,
        gap_to_ceiling=ceil - ret, frac_of_range=(ret - floor) / (ceil - floor),
        bracketed=bool(floor <= ret <= ceil),
        p_alert_anomaly=float(summary['p_alert_anomaly']),
        false_alerts_per_1k=float(summary['false_alerts_per_1k']),
        p_reroute_block=float(summary['p_reroute_block']),
        p_slow_rough=float(summary['p_slow_rough']),
        n_fallback_cells=int(predict.n_fallback), env_aware=aware,
        actions={ACTION_NAMES[a]: int(sum(r['actions'][a] for r in rows))
                 for a in range(N_ACTIONS)})

    out = dict(reward=reward_metrics, policy=policy_metrics,
               soft_vi_iters=result['history']['svi_iters'][-1],
               soft_vi_residual=result['history']['svi_resid'][-1],
               final_grad_norm=result['history']['grad_norm'][-1],
               final_fe_error=result['history']['fe_error'][-1])
    cached['validation'] = out
    path.write_text(json.dumps(cached))
    return out


# --- entry point ----------------------------------------------------------------------------
def main():
    mdp = build_tabular_mdp()
    print(f'MDP  {mdp["n_states"]} visited cells of {mdp["n_cells"]}  '
          f'{mdp["n_sa_seen"]}/{mdp["n_states"] * N_ACTIONS} (s,a) supported  '
          f'reward reconstruction {mdp["reward_match"]:.5f}')

    ctrl = positive_control(mdp)
    print(f'positive control (soft-optimal expert, phi_priv)  pearson {ctrl["pearson"]:.4f}  '
          f'spearman {ctrl["spearman"]:.4f}  final |grad| {ctrl["final_grad_norm"]:.4f}')

    results = {}
    for fs in ('priv', 'obs'):
        res = run_maxent_irl(mdp, feature_set=fs)
        val = validate(res, mdp)
        results[fs] = (res, val)
        r, p = val['reward'], val['policy']
        print(f'\n[{fs}]  {res["n_features"]} features  '
              f'soft VI {val["soft_vi_iters"]} iters, residual {val["soft_vi_residual"]:.2e}  '
              f'final |grad| {val["final_grad_norm"]:.4f}  FE error {val["final_fe_error"]:.4f}')
        print(f'  reward   pearson {r["pearson"]:.4f}  spearman {r["spearman"]:.4f}  '
              f'(supported cells only: {r["pearson_supported"]:.4f} / {r["spearman_supported"]:.4f}'
              f'; visitation-weighted pearson {r["pearson_visit_weighted"]:.4f})')
        print(f'           affine fit r_true = {r["affine_alpha"]:.4f} r_w + {r["affine_beta"]:.4f}  '
              f'R2 {r["affine_r2"]:.4f}  nRMSE {r["affine_nrmse"]:.4f}')
        print(f'  policy   return {p["return_mean"]:.1f} +/- {p["return_std"]:.1f}  '
              f'len {p["len_mean"]:.0f}  bracketed {p["bracketed"]}  '
              f'{100 * p["frac_of_range"]:.1f} % of floor-ceiling range')
        print(f'           alert recall {p["p_alert_anomaly"]:.3f}  '
              f'false alerts/1k {p["false_alerts_per_1k"]:.2f}  '
              f'fallback cells {p["n_fallback_cells"]}')

    (rp, vp), (ro, vo) = results['priv'], results['obs']
    print('\n--- privileged vs observable -------------------------------------------------')
    print(f'{"":22s}{"phi_priv":>12s}{"phi_obs":>12s}')
    for label, key in [('spearman', 'spearman'), ('pearson', 'pearson'), ('affine R2', 'affine_r2')]:
        print(f'{label:22s}{vp["reward"][key]:12.4f}{vo["reward"][key]:12.4f}')
    for label, key in [('online return', 'return_mean'), ('alert recall', 'p_alert_anomaly'),
                       ('% of range', 'frac_of_range')]:
        print(f'{label:22s}{vp["policy"][key]:12.4f}{vo["policy"][key]:12.4f}')
    print(f'{"floor (scripted)":22s}{vp["policy"]["floor"]:12.1f}')
    print(f'{"ceiling (expert)":22s}{vp["policy"]["ceiling"]:12.1f}')
    return mdp, results, ctrl


if __name__ == '__main__':
    main()
