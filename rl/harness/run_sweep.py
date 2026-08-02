"""Run a list of training jobs with a bounded process pool.

Each worker is a single-threaded SB3 process costing ~376 MB PSS (its RSS reads ~700 MB, but
libtorch's mapped pages are shared and counted once per process). On a 15 GB host with ~4 GB
of desktop processes resident, memory allows ~14 concurrent workers; the cap below is set
lower to leave cores free for the environment-building work running alongside the sweep.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from shared_modules.rl_eval import N_TRAIN_SEEDS, W6_TIMESTEPS

MAX_PAR = 10

FLAT_CONFIGS = [
    # E1a — Week-5 DQN config, only the budget and the exploration anneal change, so the
    # "under-trained or converged" question is answered without a second confound.
    ('dqn250k_raw', ['--algo', 'dqn', '--no-norm', '--exploration-fraction', '0.5']),
    # E1b — same budget on the normalised observation: the like-for-like baseline for PPO/GRPO.
    ('dqn250k', ['--algo', 'dqn', '--exploration-fraction', '0.5']),
    ('ppo250k', ['--algo', 'ppo']),
    ('ppo250k_bcinit', ['--algo', 'ppo', '--bc-init']),
    # E6 — widened discount horizon (1/(1-gamma) = 1000 steps) as a mechanism probe.
    ('ppo250k_g999', ['--algo', 'ppo', '--gamma', '0.999']),
    # E8 — the low-SoC region is only reachable from a lowered dispatch charge: at ~0.004 %/step
    # a 100 % start never crosses 20 % inside the 9,600-step loop.
    ('ppo_lowsoc_c0', ['--algo', 'ppo', '--low-soc-penalty', '0.0',
                       '--soc-init-lo', '18', '--soc-init-hi', '30']),
    ('ppo_lowsoc_c15', ['--algo', 'ppo', '--low-soc-penalty', '1.5',
                        '--soc-init-lo', '18', '--soc-init-hi', '30']),
]


PG_CONFIGS = [
    ('reinforce', ['--algo', 'reinforce']),
    ('reinforce_baseline', ['--algo', 'reinforce_baseline']),
    ('reinforce_baseline_bcinit', ['--algo', 'reinforce_baseline', '--bc-init']),
    # From-scratch GRPO is run without the event-proximal start search: an untrained policy takes
    # the terminating action early, so its segments are tens of steps long, the update count
    # explodes, and each search would cost 1.4-3k environment steps for a segment too short to
    # contain the blockage it was searching for. The BC-initialised arm keeps the search, because
    # only there are the segments long enough for it to pay off.
    ('grpo250k', ['--algo', 'grpo', '--start-mix', '0.0']),
    ('grpo250k_bcinit', ['--algo', 'grpo', '--bc-init', '--start-mix', '0.5']),
]

# A sweep edge is 1,700-1,900 steps, so gamma^k at the option boundary is ~1e-8 and an SMDP at
# gamma = 0.99 optimises exactly the flat objective. The two arms separate what the hierarchy
# changes: gamma = 0.99 holds the objective fixed and isolates credit assignment, gamma = 1.0 uses
# the episode cap as the horizon and lets the terminal cost become visible.
OPTIONS_CONFIGS = [
    ('options_smdp', ['--gamma', '0.99']),
    ('options_smdp_g1', ['--gamma', '1.0']),
    # Docking is one decision away at the option level, so the reward ablation that is structurally
    # unreachable in the flat MDP can be tested here.
    ('options_lowsoc_c0', ['--gamma', '1.0', '--low-soc-penalty', '0.0',
                           '--soc-init-lo', '18', '--soc-init-hi', '30']),
    ('options_lowsoc_c15', ['--gamma', '1.0', '--low-soc-penalty', '1.5',
                            '--soc-init-lo', '18', '--soc-init-hi', '30']),
]

# From-scratch PPO converges to "patrol briefly, then take the terminating action" on every flat
# variant tried, including gamma=0.999 and both low-SoC settings, so the discount probe and the
# reward ablation measure nothing when started from random weights. Both are repeated from the BC
# warm start, which is the only initialisation on this environment under which on-policy training
# stays in the patrolling regime long enough for the manipulated variable to matter.
FLAT_BCINIT_CONFIGS = [
    ('ppo_g999_bcinit', ['--algo', 'ppo', '--bc-init', '--gamma', '0.999']),
    ('ppo_lowsoc_c0_bcinit', ['--algo', 'ppo', '--bc-init', '--low-soc-penalty', '0.0',
                              '--soc-init-lo', '18', '--soc-init-hi', '30']),
    ('ppo_lowsoc_c15_bcinit', ['--algo', 'ppo', '--bc-init', '--low-soc-penalty', '1.5',
                               '--soc-init-lo', '18', '--soc-init-hi', '30']),
]

# 250k environment steps buy only ~260 option decisions, and the visit counts show most cells of the
# 12-state table never reached. This arm gives the high-level learner ~1,060 decisions to separate
# "the hierarchy does not help" from "the learner had nothing to learn from".
OPTIONS_LONG_CONFIGS = [
    ('options_smdp_1m', ['--gamma', '0.99', '--steps', '1000000']),
    ('options_smdp_g1_1m', ['--gamma', '1.0', '--steps', '1000000']),
]

# Three measured calibrations, all of which the first parameterisation got wrong:
#  - a coverage event at weight 5 is 0.5 % of the +-1,100 per-decision patrol reward, so the
#    coordination signal was invisible; 500 makes the two commensurate.
#  - with the rovers half a loop apart the second pass over an edge arrives ~4,800 steps later, so
#    a 2,000-step staleness threshold never fired the redundancy penalty at all.
#  - one option spans ~1,200 environment steps, so 250k steps buy ~208 decisions per rover and
#    three PPO updates. 1M steps with 32 decisions per rollout gives ~26 updates, which is still
#    few and is reported as such.
MARL_COMMON = ['--steps', '1000000', '--n-steps', '32', '--coverage-weight', '500',
               '--redundant-penalty', '200', '--stale-threshold', '5000']
MARL_CONFIGS = [
    ('marl_shared', ['--reward-mode', 'shared'] + MARL_COMMON),
    ('marl_difference', ['--reward-mode', 'difference'] + MARL_COMMON),
]

MODULES = {'flat': 'rl.harness.train', 'pg': 'rl.harness.train',
           'options': 'rl.harness.train', 'marl': 'rl.harness.train'}
KINDS = {'flat': 'flat', 'pg': 'pg', 'options': 'options', 'marl': 'marl'}


def build_jobs(configs, module='rl.harness.train', kind='flat', steps=W6_TIMESTEPS,
               seeds=N_TRAIN_SEEDS):
    jobs = []
    for tag, extra in configs:
        for s in range(seeds):
            jobs.append(['python3', '-m', module, '--kind', kind, '--tag', tag, '--seed', str(s),
                         '--steps', str(steps)] + extra)
    return jobs


def run(jobs, max_par=MAX_PAR):
    t0 = time.time()
    done = [0]

    def one(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        done[0] += 1
        label = cmd[cmd.index('--tag') + 1] + '_s' + cmd[cmd.index('--seed') + 1]
        status = 'ok' if r.returncode == 0 else f'FAIL\n{r.stderr[-1500:]}'
        print(f'[{done[0]}/{len(jobs)}] {label} {status} '
              f'({(time.time() - t0) / 60:.1f} min elapsed)', flush=True)
        return r.returncode

    with ThreadPoolExecutor(max_workers=max_par) as ex:
        codes = list(ex.map(one, jobs))
    print(f'sweep finished: {sum(c == 0 for c in codes)}/{len(jobs)} ok, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return codes


SWEEPS = {'flat': FLAT_CONFIGS, 'flat_bcinit': FLAT_BCINIT_CONFIGS, 'pg': PG_CONFIGS,
          'options': OPTIONS_CONFIGS, 'options_long': OPTIONS_LONG_CONFIGS, 'marl': MARL_CONFIGS}
MODULES['flat_bcinit'] = 'rl.harness.train'
MODULES['options_long'] = 'rl.harness.train'
KINDS['flat_bcinit'] = 'flat'
KINDS['options_long'] = 'options'
# The `--steps` these carry in their own argument list overrides the sweep default.
SWEEP_SEEDS = {'options_long': 5}

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'flat'
    par = int(sys.argv[2]) if len(sys.argv) > 2 else MAX_PAR
    if which not in SWEEPS:
        raise SystemExit(f'unknown sweep {which}')
    run(build_jobs(SWEEPS[which], module=MODULES[which], kind=KINDS[which],
                   seeds=SWEEP_SEEDS.get(which, N_TRAIN_SEEDS)), max_par=par)
