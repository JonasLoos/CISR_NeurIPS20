from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import time
from itertools import cycle
from functools import partial
from src.teacher.NonStationaryBanditPolicy import NonStationaryBanditPolicy
from src.teacher.flake_approx.config import ORIGINAL_INTERVENTION_MODES

from src.teacher.flake_approx.teacher_env import create_teacher_env
from src.envs.frozen_lake.utils import plot_trajectories, deploy

import tensorflow as tf

from src.teacher.frozen_single_switch_utils import SingleSwitchPolicy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__all__ = ['deploy_policy', 'plot_deployment_metric', 'OpenLoopTeacher']


def deploy_policy(policy, log_dir, env_f, deployment_env_fn=None, process_name='?'):
    os.makedirs(log_dir, exist_ok=True)
    teacher_env = env_f()
    obs_t = teacher_env.reset()
    student = teacher_env.student

    n_steps = int(teacher_env.time_steps_lim / teacher_env.learning_steps) + 1
    successes = np.zeros(n_steps, dtype=float)
    training_failures = np.zeros(n_steps, dtype=float)
    averarge_returns = np.zeros(n_steps, dtype=float)
    policy_actions = np.zeros(n_steps, dtype=int)
    teacher_rewards = np.zeros(n_steps, dtype=float)
    teacher_observations = np.zeros((obs_t.size, n_steps), dtype=float)

    for i in range(n_steps):
        if not isinstance(policy, (OpenLoopTeacher, NonStationaryBanditPolicy, SingleSwitchPolicy)):
            params = dict(
                n_steps = n_steps,
                learning_steps = teacher_env.learning_steps,
                steps_teacher_episode = teacher_env.steps_teacher_episode,
                student_training_episodes_current_env = teacher_env.student_training_episodes_current_env,
                i = i,
                logger = teacher_env.evaluation_logger,
                steps_per_episode = teacher_env.steps_per_episode
            )
            a, _ = policy.predict(obs_t, params=params)
        else:
            a, _ = policy.predict(obs_t)
        policy_actions[i] = a
        obs_t, teacher_rewards[i], done, _ = teacher_env.step(a)
        teacher_observations[:, i] = obs_t
        if hasattr(policy, 'add_point'):  # For non stationary bandit policy
            policy.add_point(a, teacher_rewards[i])
        env = teacher_env.final_env if deployment_env_fn is None else deployment_env_fn()
        # env = teacher_env.actions[a]()
        succ, avg_r, avg_r_succ, traj = deploy(student, env, 10000)
        successes[i], averarge_returns[i] = succ, avg_r
        training_failures[i] = teacher_env.student_failures

        print(f'process {process_name:13} - training step {i+1}/{n_steps} -> success: {succ:.4f}  avg_ret: '
              f'{avg_r:.4f}  avg_ret_succ: {avg_r_succ:.4f}  action {a}')
        plot_trajectories(traj, env.desc.shape)
        plt.savefig(os.path.join(log_dir, f'trajectories{i}.pdf'),format='pdf')
        # plot_networks(student.br, env.desc.shape)
        plt.close()
        if done:
            break
    else:
        # this point should not be reached if the code is correct
        raise Exception(f'Teacher was not done after `n_steps` ({policy}, {teacher_env})')

    # save logged data
    np.savez(os.path.join(log_dir, 'results.npz'),
             successes=successes, averarge_returns=averarge_returns,
             teacher_rewards=teacher_rewards, training_failures=training_failures,
             policy_actions=policy_actions)

    # plot returns and successes
    plt.figure()
    _, axes = plt.subplots(1, 3)
    metrics = [successes, averarge_returns, teacher_rewards]
    titles = ['successes', 'student ret', 'teacher ret']
    for ax, metric, title in zip(axes, metrics, titles):  # type: ignore
        if title == 'teacher ret':
            ax.plot(np.cumsum(metric))
        else:
            ax.plot(metric)
        ax.set_title(title)
    plt.savefig(os.path.join(log_dir, 'results.pdf'), format='pdf')

    # plot teacher observations
    plt.figure()
    for o_num, o in enumerate(teacher_observations):
        plt.plot(o, label=o_num)
    plt.legend()
    plt.title('teacher observations')
    plt.xlabel('curriculum steps')
    plt.savefig(os.path.join(log_dir, 'teacher_observations.pdf'))


def plot_deployment_metric(log_dir, metric, ax=None, fig=None, label=None, legend=True):
    # setup figure and axis
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = plt.gca()
    plot_intervention_changes = label in ['Optimized',]  # only plot horizontal line for Optimized

    # load data
    returns = []
    policy_actions = []
    for subdir in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, subdir)):
            try:
                data = np.load(os.path.join(log_dir, subdir, 'results.npz'))
                returns.append(data[metric])
                try:
                    policy_actions.append(data['policy_actions'])
                except KeyError:
                    # don't plot intervention changes if they cannot be loaded
                    plot_intervention_changes = False
            except FileNotFoundError:
                    pass
    returns = np.array(returns)

    # check if data was found
    if len(returns) == 0:
        print(f'[plot_deployment_metric] Couldn\'t find entries for {metric} in {os.path.basename(log_dir)}')
        return np.nan

    # fix data for teacher_rewards
    if metric == 'teacher_rewards':
        returns = np.cumsum(returns, axis=1)

    # compute mean and std
    mu = np.mean(returns, axis=0)
    print(f'{log_dir.split("/")[-1]} - {metric} final mean -> {mu[-1]}')
    std = np.std(returns, axis=0) / np.sqrt(returns.shape[0])  # type: ignore

    # plot
    if label is None:
        label = log_dir.split('/')[-1].replace('_', ' ')
<<<<<<< HEAD
    if plot_intervention_changes:
        # plot a vertical line at the median intervention change index
        n = len(policy_actions[0])
        counts = defaultdict(lambda: [0]*n)
        for tmp in policy_actions:
            for i in range(len(tmp)-1):
                if tmp[i] != tmp[i+1]:
                    counts[(tmp[i],tmp[i+1])][i] += 1
        for tmp in counts.values():
            median_index = [x for k,v in zip(range(n),tmp) for x in [k]*v][len(policy_actions)//2]
            plt.axvline(x=median_index, color='lightgray', ls='--')
    ax.plot(mu, label=label)
    ax.fill_between(np.arange(mu.size), mu-std, mu+std, alpha=0.5)
=======

    mode = log_dir.split('/')[-1]
    if mode in ORIGINAL_INTERVENTION_MODES:
        ax.plot(mu, label=label, linestyle='dashed')
        ax.fill_between(np.arange(mu.size), mu-std, mu+std, alpha=0.2)
    else:
        ax.plot(mu, label=label)
        ax.fill_between(np.arange(mu.size), mu-std, mu+std, alpha=0.5)
>>>>>>> master
    if legend:
        plt.legend(bbox_to_anchor=(0,-0.4,1,0.2), loc="upper left",
                mode="expand", borderaxespad=0, ncol=3, frameon=False)
    # Ticks
    plt.tick_params(axis='both',
                    which='both',
                    bottom=True,
                    left=True)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Curriculum steps')
    plt.ylabel(metric.replace('_', ' '))
    plt.tight_layout(pad=0.2)

    return mu [-1]


class OpenLoopTeacher(object):
    """
    Dummy teacher that cycles through a sequence of actions
    """
    def __init__(self, action_sequence):
        self.actions = cycle(action_sequence)

    def predict(self, obs):
        return next(self.actions), None


class IncrementalTeacher(object):
    """
    Incremental heuristic teacher that increases the buffer size on each intervention
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        action = self.interventions_counter
        self.interventions_counter += 1
        return self.actions[action], None


class DoubleIncrementalTeacher(object):
    """
    Incremental heuristic teacher that increases the buffer size on each intervention
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        action = self.interventions_counter
        self.interventions_counter += 1
        return self.actions[2 * action], None


class HalfIncrementalTeacher(object):
    """
    Incremental heuristic teacher that increases the buffer size on each intervention
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        action = self.interventions_counter
        self.interventions_counter += 1
        return self.actions[int(np.ceil(0.5 * action))], None


class QuarterIncrementalTeacher(object):
    """
    Incremental heuristic teacher that increases the buffer size on each intervention
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        action = self.interventions_counter
        self.interventions_counter += 1
        return self.actions[int(np.ceil(0.25 * action))], None


class EighthIncrementalTeacher(object):
    """
    Incremental heuristic teacher that increases the buffer size on each intervention
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        action = self.interventions_counter
        self.interventions_counter += 1
        return self.actions[int(np.ceil(0.125 * action))], None


class SixteenthIncrementalTeacher(object):
    """
    Incremental heuristic teacher that increases the buffer size on each intervention
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        action = self.interventions_counter
        self.interventions_counter += 1
        return self.actions[int(np.ceil(0.0625 * action))], None


class TwentyPercentTeacher(object):
    """
    Heuristic teacher that goes back twenty percent of the number of student training episodes in the current env
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence

    def predict(self, obs, params=None):
        action = int(np.ceil(0.2 * params['student_training_episodes_current_env']))
        return self.actions[action], None


class RandomTeacher(object):
    """
    Random teacher that goes back a random number of steps
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence

    def predict(self, obs, params=None):
        upper_limit = params['steps_teacher_episode']
        action = np.random.randint(1, upper_limit + 2)
        return self.actions[action], None


class EvaluationTeacher(object):
    def __init__(self, action_sequence):
        self.actions = action_sequence

    def predict(self, obs, params=None):
        evaluation_logger = params['logger']
        info = evaluation_logger.get_counters()
        lengths = info['lengths']
        return self.actions[lengths[-1]], None


class ExpTeacher(object):
    """
    Teacher that goes exponentially with the number of interventions
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence
        self.interventions_counter = 0

    def predict(self, obs, params=None):
        intervention = self.interventions_counter
        action = int(np.ceil(1.3 ** intervention))
        self.interventions_counter += 1
        return self.actions[action], None


class StepsEpisodeTeacher(object):
    """
    Teacher that goes back the steps per episode
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence

    def predict(self, obs, params=None):
        steps_per_episode = params['steps_per_episode']
        return self.actions[steps_per_episode], None


class StepsTeacher(object):
    """
    Teacher that goes back the current teacher steps in the episode
    """
    def __init__(self, action_sequence):
        self.actions = action_sequence

    def predict(self, obs, params=None):
        steps_teacher_episode = params['steps_teacher_episode']
        return self.actions[steps_teacher_episode], None


class Back(object):
    """
    Teacher that goes back a constant number of steps
    """
    def __init__(self, action_sequence, steps=None):
        self.actions = action_sequence
        self.steps = steps

    def predict(self, obs, params=None):
        return self.actions[self.steps - 1], None


if __name__ == '__main__':
    n = 5
    small_map = True
    experiment_name = 'cnn_small_env_3_interventions'
    actions_sequences = [[0] * 11,
                         [1] * 6 + [2] * 5]

    policy_names = ['zeros', '1x6_2x5']
    env_f = partial(create_teacher_env)
    t = time.time()
    for actions, pi_name in zip(actions_sequences, policy_names):
        processes = []
        for i in range(n):
            name = os.path.join(os.path.abspath('.'), experiment_name,
                                pi_name, f'experiment_{i}')
            pi = OpenLoopTeacher(actions)
            p = mp.Process(target=deploy_policy, args=[pi, name, env_f])
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    print(f'Elapsed {time.time() - t}')

    # Plot successes and teacher reward

    plt.figure()
    exp_dir = os.path.join(os.path.abspath('.'), experiment_name)
    for subdir in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, subdir)):
             plot_deployment_metric(os.path.join(exp_dir, subdir),
                                    metric='teacher_rewards')
    plt.savefig(os.path.join(os.path.abspath('.'), experiment_name,
                             'policy_copmarison.pdf'), format='pdf')

    plt.figure()
    exp_dir = os.path.join(os.path.abspath('.'), experiment_name)
    for subdir in os.listdir(exp_dir):
        if os.path.isdir(os.path.join(exp_dir, subdir)):
            plot_deployment_metric(os.path.join(exp_dir, subdir),
                                   metric='successes')
    plt.savefig(os.path.join(os.path.abspath('.'), experiment_name,
                             'successes.pdf'), format='pdf')
