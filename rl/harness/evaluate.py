"""Week-6 evaluation entry points, merged behind one CLI: `python -m rl.harness.evaluate --mode ...`.

  python -m rl.harness.evaluate --mode one --tag dqn250k --seed 0
  python -m rl.harness.evaluate --mode options --tag options_smdp
  python -m rl.harness.evaluate --mode all --cross-map

--mode one: evaluate one (family, train seed) on the ten fixed worlds and write a partial result.
Split out from `--mode all` because a full evaluation pass is ~480k environment steps per family
and the environment runs at ~300 steps/s, so the pass has to fan out across cores the same way
training does.

--mode options: evaluate node-level option policies on the ten fixed worlds. Kept separate from
`--mode one` because the option-level policy acts on a different environment: one call selects an
option that runs for 600-1,900 environment steps, so the event-conditioned counters have to be
accumulated inside the option rather than per policy call.

--mode all: fan out `--mode one` over every checkpoint, then merge and test. The protocol
separates the two randomness sources Week 5 conflated. A family is trained on `N_TRAIN_SEEDS`
train seeds; each checkpoint is evaluated on all ten `EVAL_WORLD_SEEDS`. The paired vector for a
family is its per-world mean across train seeds, so "A beats B" is decided by a Wilcoxon
signed-rank test over the ten common worlds rather than by overlapping error bars.

Raw total return is reported but is not the headline: Week-5 policies terminated at anywhere
between 3.6k and 9.5k steps, so return per step and the event-conditioned rates carry the
comparison instead.
"""

import argparse
import itertools
import json
import subprocess
import sys
import timeit
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from shared_modules.rl_eval import (POL_DIR, ART_DIR, ROOT, N_TRAIN_SEEDS, evaluate, event_rates,
                                    make_scripted_predict, ExpertAware, normalize, paired_test,
                                    verify_w5_reference_gate, EVAL_WORLD_SEEDS, CROSS_MAP_SEEDS,
                                    FULL_LOOP)
from rl.rover_env import STATE_COLS
from rl.rover_options_env import RoverOptionsEnv, make_scripted_option_predict, OPT_REROUTE

# === --mode one (formerly rl/eval_one.py) ====================================================
# Which env kwargs each family was trained under; evaluation must match, or the low-SoC families
# would be scored on a full battery they never see.
FAMILY_ENV_KW = {
    'ppo_lowsoc_c0': dict(low_soc_penalty=0.0, soc_init_range=(18.0, 30.0)),
    'ppo_lowsoc_c15': dict(low_soc_penalty=1.5, soc_init_range=(18.0, 30.0)),
    'ppo_lowsoc_c0_bcinit': dict(low_soc_penalty=0.0, soc_init_range=(18.0, 30.0)),
    'ppo_lowsoc_c15_bcinit': dict(low_soc_penalty=1.5, soc_init_range=(18.0, 30.0)),
}
# The Week-5 DQN and its Week-6 budget re-run both take the unnormalised observation, so the
# 80k-vs-250k comparison changes only the training budget and the exploration anneal.
FAMILY_NORM = {'dqn250k_raw': False, 'dqn_ew0.05': False, 'dqn_ew0.0': False, 'dqn_ew0.2': False}


def load_policy(tag, seed):
    zip_path = POL_DIR / f'{tag}_s{seed}.zip'
    if zip_path.exists():
        from stable_baselines3 import DQN, PPO
        cls = DQN if tag.startswith('dqn') else PPO
        model = cls.load(zip_path, device='cpu')
        return lambda obs: model.predict(obs, deterministic=True)[0]
    pt_path = POL_DIR / f'{tag}_s{seed}.pt'
    if pt_path.exists():
        from rl.pg_algos import load_policy_predict
        try:
            return load_policy_predict(pt_path)
        except RuntimeError:
            return None      # a multi-agent checkpoint: two policies keyed by agent id
    return None


def _main_one(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--cross-map', action='store_true')
    a = p.parse_args(argv)

    torch.set_num_threads(1)
    out_path = ART_DIR / f'eval_{a.tag}_s{a.seed}.json'
    if out_path.exists():
        print(f'{a.tag}_s{a.seed}: cached'); return

    predict = load_policy(a.tag, a.seed)
    if predict is None:
        raise SystemExit(f'no checkpoint for {a.tag}_s{a.seed}')

    norm = FAMILY_NORM.get(a.tag, True)
    env_kw = FAMILY_ENV_KW.get(a.tag, {})
    rows, summary = evaluate(predict, norm=norm, env_kw=env_kw)
    reasons = [r['terminated_reason'] for r in rows]
    res = dict(tag=a.tag, seed=a.seed,
               per_world=[r['ret'] for r in rows],
               per_world_rps=[r['ret_per_step'] for r in rows],
               lengths=[r['length'] for r in rows],
               terminal_soc=[r['terminal_soc'] for r in rows],
               actions=np.sum([r['actions'] for r in rows], 0).tolist(),
               # per-episode termination: the per-step docking rate is ~1/2000 whatever the policy
               # does, since one dock decision ends thousands of low-SoC steps
               dock_episode_rate=float(np.mean([x == 'return_to_base' for x in reasons])),
               soc_depleted_rate=float(np.mean([x == 'soc_depleted' for x in reasons])),
               stuck_rate=float(np.mean([x == 'stuck_timeout' for x in reasons])),
               **{k: (None if v != v else float(v)) for k, v in event_rates(rows).items()})

    if a.cross_map:
        # Five worlds per layout rather than ten: this asks whether a policy transfers to a
        # different procedural layout at all, not for a precise return on each one.
        cross = {}
        for ms in CROSS_MAP_SEEDS:
            _, s = evaluate(predict, norm=norm, env_kw=env_kw, map_seed=ms,
                            world_seeds=EVAL_WORLD_SEEDS[:5])
            cross[str(ms)] = dict(return_mean=float(s['return_mean']),
                                  ret_per_step_mean=float(s['ret_per_step_mean']),
                                  p_reroute_block=None if s['p_reroute_block'] != s['p_reroute_block']
                                  else float(s['p_reroute_block']))
        res['cross_map'] = cross

    out_path.write_text(json.dumps(res))
    print(f'{a.tag}_s{a.seed}: ret {np.mean(res["per_world"]):.0f} '
          f'reroute {res["p_reroute_block"]}')


# === --mode options (formerly rl/eval_options.py) ============================================
OPTIONS_FAMILY_ENV_KW = {
    'options_lowsoc_c0': dict(low_soc_penalty=0.0, soc_init_range=(18.0, 30.0)),
    'options_lowsoc_c15': dict(low_soc_penalty=1.5, soc_init_range=(18.0, 30.0)),
}


def load_table_policy(tag, seed):
    """Greedy option selection from the tabular high-level Q, using the trainer's discretiser."""
    path = POL_DIR / f'{tag}_s{seed}.npz'
    if not path.exists():
        return None
    from rl.harness.train import make_q_predict
    return make_q_predict(np.load(path)['q'], norm=True)


def measure_option_latency(predict, env_kw):
    """Both levels of the hierarchical decision path, in ms.

    A deployed option policy pays the high-level lookup once per node and the option controller
    on every 10 Hz tick, so the step the latency gate has to clear is the one where both fire.
    The controller is timed on `reroute-branch`, its longest branch (the crossing test as well as
    the alert and rough-terrain tests).
    """
    env = RoverOptionsEnv(randomize_reset=False, seed=EVAL_WORLD_SEEDS[0], **env_kw)
    env.reset()
    high = measure_latency(predict)
    low = 1000.0 * timeit.timeit(lambda: env._low_level(OPT_REROUTE), number=5000) / 5000
    env.close()
    return dict(latency_high_level_ms=high, latency_low_level_ms=low,
                latency_ms=high + low)


def _main_options(argv):
    from rl.harness.train import evaluate_options, rollout_options

    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--seeds', type=int, default=N_TRAIN_SEEDS)
    a = p.parse_args(argv)

    env_kw = OPTIONS_FAMILY_ENV_KW.get(a.tag, {})
    def one(predict):
        """Summary plus the per-episode termination breakdown `evaluate_options` does not keep.

        A per-environment-step docking rate is meaningless at option granularity: `dock` is one
        decision against the thousands of low-SoC environment steps it terminates, so the ratio is
        ~1/2000 whatever the policy does. Whether the episode *ended* by docking is the quantity
        the low-SoC ablation is actually about.
        """
        s = evaluate_options(predict, EVAL_WORLD_SEEDS, **env_kw)
        rows = [rollout_options(predict, RoverOptionsEnv(randomize_reset=False, seed=ws,
                                                         max_env_steps=FULL_LOOP, **env_kw))
                for ws in EVAL_WORLD_SEEDS]
        reasons = [r['terminated_reason'] for r in rows]
        s['dock_episode_rate'] = float(np.mean([r == 'return_to_base' for r in reasons]))
        s['soc_depleted_rate'] = float(np.mean([r == 'soc_depleted' for r in reasons]))
        s['stuck_rate'] = float(np.mean([r == 'stuck_timeout' for r in reasons]))
        s['low_soc_steps'] = float(np.mean([r['low_soc'] for r in rows]))
        # per-world returns, kept so a paired test over the ten common worlds can be run from the
        # artefact rather than only from the aggregate
        s['per_world'] = [float(r['ret']) for r in rows]
        return s

    if a.tag == 'scripted_option':
        first = make_scripted_option_predict()
        summaries, seeds = [one(first)], [0]
    else:
        summaries, seeds, first = [], [], None
        for s in range(a.seeds):
            predict = load_table_policy(a.tag, s)
            if predict is None:
                continue
            first = first or predict
            summaries.append(one(predict))
            seeds.append(s)
    if not summaries:
        raise SystemExit(f'no checkpoints for {a.tag}')

    keys = [k for k in summaries[0] if isinstance(summaries[0][k], (int, float))]
    agg = {k: float(np.nanmean([s[k] for s in summaries])) for k in keys}
    per_seed = [s['return_mean'] for s in summaries]
    # `evaluate_options` reports return_std across the ten evaluation worlds within one train seed.
    # Averaging that would put a different quantity in the same column as `merge_family`, which
    # reports the spread across train seeds. Overwrite it so both paths mean the same thing, and
    # keep the world-level spread under its own name.
    # A fixed policy has no seed dimension, so its across-world spread stays the reported one,
    # matching how the flat reference policies are reported.
    agg['return_std_across_worlds'] = agg['return_std']
    if len(per_seed) > 1:
        agg['return_std'] = float(np.std(per_seed))
    out = dict(tag=a.tag, n_train_seeds=len(summaries), seeds=seeds,
               per_seed_return=per_seed, world_seeds=EVAL_WORLD_SEEDS,
               per_seed_per_world=[s['per_world'] for s in summaries],
               **measure_option_latency(first, env_kw), **agg)
    (ART_DIR / f'evalopt_{a.tag}.json').write_text(json.dumps(out))
    print(f"{a.tag}: return {agg['return_mean']:.0f} "
          f"P(reroute|block) {agg.get('p_reroute_block')} "
          f"dock-ended episodes {agg['dock_episode_rate']:.2f} "
          f"low-SoC steps {agg['low_soc_steps']:.0f}")


# === --mode all (formerly rl/eval_all.py) ====================================================
RESULTS = ART_DIR / 'w6_eval.json'
MAX_PAR = 10


def train_bc(device='cpu', seed=42):
    """Retrain the Week-5 behaviour-cloning policy on the Week-2 table (it was never saved).

    Seeded so the offline reference is the same number on every evaluation pass; without it the
    bracket moves by ~15 return between runs and the paired tests against it are not reproducible.
    """
    import torch.nn as nn
    torch.manual_seed(seed)
    df = pd.read_csv(ROOT / 'data' / 'rover_transitions.csv')
    X = normalize(df[[f's_{c}' for c in STATE_COLS]].to_numpy(np.float64))
    y = df['action'].to_numpy(np.int64)
    counts = np.bincount(y, minlength=5).astype(float)
    w = counts.sum() / (5 * np.clip(counts, 1, None))
    net = nn.Sequential(nn.Linear(9, 128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 5)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, device=device)
    net.train()
    for _ in range(60):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad(); lossf(net(Xt[idx]), yt[idx]).backward(); opt.step()
    net.eval()   # Dropout must be off or argmax inference is stochastic

    def predict(obs):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32, device=device)[None]).argmax())
    return predict


def measure_latency(predict, n=2000):
    obs = np.zeros(9, dtype=np.float32)
    return 1000.0 * timeit.timeit(lambda: predict(obs), number=n) / n


def family_kind(tag):
    """Which environment a family's checkpoints act on, read from the run's own meta JSON.

    Deciding this from the tag would be wrong for any tag that does not carry its environment in
    its name -- the options trainer's documented example is `opt_ppo`, whose checkpoint is a
    4-action SB3 zip that the flat fan-out below would happily score on the 5-action flat env.
    Returns None for a checkpoint with no meta, which is how the Week-5 runs present.
    """
    metas = sorted(ART_DIR.glob(f'{tag}_s[0-9]_meta.json'))
    if not metas:
        return None
    m = json.loads(metas[0].read_text())
    if 'learner' in m:                  # --kind options writes the high-level learner name
        return 'options'
    if m.get('algo') == 'ippo' or 'reward_mode' in m:
        return 'marl'
    return 'flat'


def discover_families():
    """All Week-6 families, plus the Week-5 canonical DQN so the budget comparison is paired."""
    tags = {f.stem.rsplit('_s', 1)[0] for f in POL_DIR.glob('*_s[0-9].zip')}
    tags |= {f.stem.rsplit('_s', 1)[0] for f in POL_DIR.glob('*_s[0-9].pt')}
    # options and multi-agent families act on different environments and are evaluated by
    # `--mode options` and the multi-agent trainer's own evaluation block, not by the flat-env
    # fan-out here.
    keep = {t for t in tags if not t.startswith('smoke') and family_kind(t) == 'flat'}
    if 'dqn_ew0.05' in tags:
        keep.add('dqn_ew0.05')          # Week-5 80k canonical run, the budget-comparison arm
    return sorted(keep)


# The cross-layout check answers whether a policy transfers off the canonical layout at all, so it
# runs on one representative of each family type rather than on every configuration.
CROSS_MAP_FAMILIES = ['dqn250k', 'ppo250k_bcinit', 'grpo250k_bcinit']


def fan_out(families, cross_map):
    jobs = [['python3', '-m', 'rl.harness.evaluate', '--mode', 'one', '--tag', t, '--seed', str(s)] +
            (['--cross-map'] if cross_map and s == 0 and t in CROSS_MAP_FAMILIES else [])
            for t, s in itertools.product(families, range(N_TRAIN_SEEDS))
            if load_policy(t, s) is not None]

    def one(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print('FAIL', cmd[-3:], r.stderr[-800:], flush=True)
        else:
            print(r.stdout.strip(), flush=True)
        return r.returncode
    with ThreadPoolExecutor(max_workers=MAX_PAR) as ex:
        list(ex.map(one, jobs))
    return len(jobs)


def merge_family(tag):
    parts = sorted(ART_DIR.glob(f'eval_{tag}_s[0-9].json'))
    if not parts:
        return None
    ds = [json.loads(p.read_text()) for p in parts]
    per_seed = np.array([d['per_world'] for d in ds])          # [n_train_seed, 10]
    rps = np.array([d['per_world_rps'] for d in ds])
    lens = np.array([d['lengths'] for d in ds])
    rate_keys = ['p_alert_anomaly', 'false_alerts_per_1k', 'p_reroute_block', 'p_slow_rough',
                 'dock_rate', 'low_soc_steps', 'dock_episode_rate', 'soc_depleted_rate',
                 'stuck_rate']
    rates = {k: float(np.nanmean([d[k] for d in ds if d.get(k) is not None]))
             if any(d.get(k) is not None for d in ds) else None for k in rate_keys}
    predict = load_policy(tag, ds[0]['seed'])
    out = dict(tag=tag, n_train_seeds=len(ds), n_worlds=per_seed.shape[1],
               per_world=per_seed.mean(0).tolist(), per_seed=per_seed.tolist(),
               return_mean=float(per_seed.mean()), return_std=float(per_seed.mean(1).std()),
               ret_per_step_mean=float(rps.mean()), ret_per_step_std=float(rps.mean(1).std()),
               len_mean=float(lens.mean()),
               actions=np.sum([d['actions'] for d in ds], 0).tolist(),
               terminal_soc=float(np.mean([d['terminal_soc'] for d in ds])),
               latency_ms=measure_latency(predict) if predict else None, **rates)
    cross = [d['cross_map'] for d in ds if 'cross_map' in d]
    if cross:
        out['cross_map'] = cross[0]
    return out


def eval_reference(name, predict, env_aware=False):
    rows, s = evaluate(predict, env_aware=env_aware)
    per_world = [r['ret'] for r in rows]
    return dict(tag=name, n_train_seeds=1, n_worlds=len(rows), per_world=per_world,
                per_seed=[per_world], return_mean=float(np.mean(per_world)),
                return_std=float(np.std(per_world)),
                ret_per_step_mean=float(s['ret_per_step_mean']),
                ret_per_step_std=float(s['ret_per_step_std']), len_mean=float(s['len_mean']),
                actions=np.sum([r['actions'] for r in rows], 0).tolist(),
                terminal_soc=float(np.mean([r['terminal_soc'] for r in rows])),
                latency_ms=None if env_aware else measure_latency(predict),
                **{k: (None if v != v else float(v)) for k, v in event_rates(rows).items()})


def _main_all(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--families', nargs='*', default=None)
    p.add_argument('--cross-map', action='store_true')
    p.add_argument('--skip-fanout', action='store_true')
    a = p.parse_args(argv)
    torch.set_num_threads(1)

    print('W5 reference gate:', verify_w5_reference_gate(), flush=True)

    families = a.families or discover_families()
    print('families:', families, flush=True)
    if not a.skip_fanout:
        print('evaluated', fan_out(families, a.cross_map), 'checkpoints', flush=True)

    out = {}
    rng = np.random.default_rng(0)
    # the random floor samples the four non-terminating actions: uniform over all five draws
    # `return-to-base` within a few steps, so the episode ends before the policy is a floor for
    # anything
    for name, predict, aware in [('random', lambda o: rng.integers(0, 4), False),
                                 ('scripted_blind', make_scripted_predict(), False),
                                 ('bc_offline', train_bc(), False),
                                 ('expert_aware', ExpertAware(), True)]:
        out[name] = eval_reference(name, predict, env_aware=aware)
        print(name, round(out[name]['return_mean']), flush=True)

    for tag in families:
        m = merge_family(tag)
        if m:
            out[tag] = m
            print(tag, round(m['return_mean']), 'reroute', m['p_reroute_block'], flush=True)

    keys = [k for k in out if out[k]['n_worlds'] == 10]
    out['_paired'] = {f'{x}|{y}': paired_test(out[x]['per_world'], out[y]['per_world'])
                      for i, x in enumerate(keys) for y in keys[i + 1:]}
    RESULTS.write_text(json.dumps(out, indent=1))
    print('wrote', RESULTS)


# === dispatch =================================================================================
_MODES = {'one': _main_one, 'options': _main_options, 'all': _main_all}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--mode', choices=list(_MODES), required=True)
    ns, rest = p.parse_known_args(argv)
    _MODES[ns.mode](rest)


if __name__ == '__main__':
    main()
