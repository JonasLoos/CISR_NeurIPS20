import os
from functools import partial
import multiprocessing as mp
import time
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from src.envs.frozen_lake.frozen_maps import MAPS
from src.envs.frozen_lake.utils import plot_map
from src.teacher.flake_approx.config import MAP_NAME, INTERVENTION_MODES, NUMBER_OF_TRIALS

from src.teacher.flake_approx.deploy_teacher_policy import deploy_policy, \
    plot_deployment_metric, OpenLoopTeacher
from src.teacher.flake_approx.teacher_env import create_teacher_env, \
    small_base_cenv_fn
from src.teacher.frozen_single_switch_utils import SingleSwitchPolicy
from src.teacher.NonStationaryBanditPolicy import NonStationaryBanditPolicy
from src.utils.plotting import cm2inches, set_figure_params


def plot_comparison(log_dir, modes, t):
    text_width = cm2inches(13.968)  # Text width in cm
    figsize = (text_width / 2, text_width / 3.5)
    set_figure_params(fontsize=7)

    # Fix plotting when using command line on Mac
    plt.rcParams['pdf.fonttype'] = 42  # type: ignore

    metric = ['successes', 'training_failures', 'averarge_returns']
    metric_summary = np.zeros((len(modes), len(metric)), dtype=float)

    log_dir = os.path.join(log_dir, t)

    for i, subdir in enumerate(modes):
        if subdir == 'Trained':
            label = 'Optimized'
        elif subdir == 'Original':
            label = 'No interv.'
        else:
            label = subdir
        if os.path.isdir(os.path.join(log_dir, subdir)):
            for j, metric_name in enumerate(metric):
                fig = plt.figure(j, figsize=figsize)
                mu = plot_deployment_metric(os.path.join(log_dir, subdir),
                                            metric=metric_name, fig=fig,
                                            label=label, legend=True)
                metric_summary[i, j] = mu

    if not os.path.isdir(log_dir):
        raise Exception('The teacher has to be evaluated before plotting (--evaluate)')

    np.savez(os.path.join(log_dir, 'metrics_summary.npz'),
             metric_summary=metric_summary)
    for j, metric_name in enumerate(metric):
        plt.figure(j)
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(log_dir, metric_name + '.pdf'), format='pdf',
                    transparent=True)
        plt.close(j)


def run_comparision(log_dir, teacher_dir, modes, t):
    env_f = partial(create_teacher_env)
    env_f_original = partial(create_teacher_env, original=True)
    env_f_single_switch = partial(create_teacher_env, obs_from_training=True)
    env_f_stationary_bandit = partial(create_teacher_env, non_stationary_bandit=True)

    log_dir = os.path.join(log_dir, t)

    start_time = time.time()
    process_pool = mp.Pool()
    for mode in modes:
        if mode == 'SR2':
            model = OpenLoopTeacher([1])
        elif mode in ['SR1', 'Original']:
            model = OpenLoopTeacher([0])
        elif mode == 'HR':
            model = OpenLoopTeacher([2])
        elif mode == 'Bandit':
            model = NonStationaryBanditPolicy(3, 10)
        elif mode == 'Trained':
            model = SingleSwitchPolicy.load(os.path.join(teacher_dir, 'trained_teacher'))
        else:
            teacher_module = importlib.import_module("src.teacher.flake_approx.deploy_teacher_policy")
            teacher_class = getattr(teacher_module, mode + 'Teacher')
            model = teacher_class(range(3, 1003))
        
        for i in range(NUMBER_OF_TRIALS):
            log_tmp = os.path.join(log_dir, mode, f'experiment{i}')
            if mode == 'Original':
                teacher_env = env_f_original
            elif mode == 'Trained':
                teacher_env = env_f_single_switch
            elif mode == 'Bandit':
                teacher_env = env_f_stationary_bandit
            else:
                teacher_env = env_f
            process_name = f'{mode}-{i}'
            process_pool.apply_async(deploy_policy,
                           args=[model, log_tmp, teacher_env,
                                 small_base_cenv_fn, process_name])
    process_pool.close()
    process_pool.join()
    print(f'[run_comparison] time elapsed: {time.time() - start_time}')


def run_bandits(log_dir):
    """
    Run n_rounds x n_trials students with bandit teacher from "Teacher-student
    curriculum learning" by Matiisen et al. and stores the results in log_dir.
    """
    env_f_stationary_bandit = partial(create_teacher_env,
                                      non_stationary_bandit=True)
    n_rounds = 3  # Number of times n_trials students are run
    n_trials = 10  # Number of students to run in parallel
    start_time = time.time()

    model = NonStationaryBanditPolicy(3, 10)

    for j in range(n_rounds):
        print(f'--------Running {j}th round of bandits----------')
        processes = []
        n_existing_experiments = len(os.listdir(log_dir))
        for i in range(n_trials):
            log_tmp = os.path.join(log_dir, f'experiment{i + n_existing_experiments}')
            teacher_env = env_f_stationary_bandit
            p = mp.Process(target=deploy_policy,
                           args=[model, log_tmp, teacher_env,
                                 small_base_cenv_fn])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    print(f'[run_bandits] time elapsed: {time.time() - start_time}')


def get_metric_summary(log_dir, t):
    log_dir = os.path.join(log_dir, t)
    return np.load(os.path.join(log_dir, 'metrics_summary.npz'))['metric_summary']


def print_latex_table(mu, std):
    table = []
    for mu_row, std_row in zip(mu, std):
        line = []
        for j in range(len(mu_row)):
            line.append(f'${mu_row[j]:.3f}\\pm{std_row[j]:.3f}$')
        table.append(line)
    print()
    print(tabulate(table, tablefmt="latex_raw"))
    print()


def main():
    # initialize paths
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir, 'results',
                               'flake')
    log_dir = os.path.join(results_dir, 'teacher_comparison')
    base_teacher_dir = os.path.join(results_dir, 'teacher_training')

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the comparison for the pre-trained teachers against the baselines')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Run the comparison between a pre-trained teacher and the baselines')
    parser.add_argument("--teacher_dir", nargs="*", type=str, default=[],
                        help='Directory(ies) containing the teacher to plot or evaluate (assumed to be in result/flake/teacher_training)')

    args = parser.parse_args()

    # load teachers
    teachers = []
    for t in args.teacher_dir:
        if os.path.isdir(os.path.join(base_teacher_dir, t)):
            teachers.append(t)
        else:
            print(f'Could not find teacher {t} in {base_teacher_dir}')
    # Use default teacher is none is given
    if len(teachers) == 0:
        teachers = ['03_06_20__11_46_57']

    if args.evaluate:
        # evaluate teachers
        for t in teachers:
            print(f'Evaluating teacher {t}')
            teacher_dir = os.path.join(base_teacher_dir, t)
            run_comparision(log_dir, teacher_dir, INTERVENTION_MODES, t)

    if args.plot:
        # plot teachers
        for t in teachers:
            print(f'Plotting teacher {t}')
            teacher_dir = os.path.join(base_teacher_dir, t)
            plot_comparison(log_dir, INTERVENTION_MODES, t)

        # plot map
        plot_map(MAPS[MAP_NAME], legend=True)
        plt.savefig(os.path.join(log_dir, 'map.pdf'))

        # Print table
        metrics_statistics = np.array([
            get_metric_summary(log_dir, t)
            for t in teachers
        ])
        mu = metrics_statistics.mean(axis=0)
        std = metrics_statistics.std(axis=0) / metrics_statistics.shape[0]**.5
        print_latex_table(mu, std)

if __name__ == '__main__':
    main()
