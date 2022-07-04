import os
from typing import Any, Callable, Optional
import numpy as np
import time
import multiprocessing as mp
from functools import partial
import argparse
import matplotlib.pyplot as plt
from src.envs import CMDP
from src.utils.plotting import cm2inches, set_figure_params
import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.a2c.utils import linear

import tensorflow as tf

from gym.envs.classic_control.cartpole import CartPoleEnv

from src.teacher.cart_pole.deploy_teacher_policy import plot_deployment_metric, OpenLoopTeacher
from src.teacher.cart_pole.teacher_env import SingleSwitchPolicy
from src.students import LagrangianStudent, identity_transfer
from src.online_learning import ExponetiatedGradient
from src.teacher import create_intervention

class CustomCartPole(CartPoleEnv):
    # TODO
    pass


def my_small_mlp(inp, **kwargs):
    activ = tf.nn.relu
    return activ(linear(inp, 'fc1', n_hidden=4, init_scale=np.sqrt(2)))


def create_teacher_env(new_br_kwargs={}, new_online_kwargs={},
                       original=False):
    # Student definition
    br_kwargs = dict(policy=MlpPolicy, verbose=0, n_steps=128,
                     ent_coef=0.05, cliprange=0.2, learning_rate=1e-3,
                     noptepochs=9,
                     policy_kwargs={'mlp_extractor': my_small_mlp})
    br_kwargs.update(new_br_kwargs)

    # Define online kwargs
    online_kwargs = dict(B=0.5, eta=1.0)
    online_kwargs.update(new_online_kwargs)

    student_cls = LagrangianStudent
    n_envs = 4
    use_sub_proc_env = False
    student_default_kwargs = {'env': None,
                              'br_algo': PPO2,
                              'online_algo': ExponetiatedGradient,
                              'br_kwargs': br_kwargs,
                              'online_kwargs': online_kwargs,
                              'lagrangian_ronuds': 2,
                              'curriculum_transfer': identity_transfer,
                              'br_uses_vec_env': True,
                              'use_sub_proc_env': use_sub_proc_env,
                              'n_envs': n_envs,
                              }
    student_ranges_dict = {}

    # Teacher interventions
    if original:
        # To preserve the teacher env interface while training in the
        # original environment, we introduce a dummy intervention
        # condition that is always False.
        def dummy_intervention(**kwargs):
            return 0
        _, test_env = make_base_small_cenvs()
        intervention = create_intervention(
            base_cenv=small_base_cenv_fn,
            interventions=[dummy_intervention], taus=[0], buf_size=0,
            use_vec=True, avg_constraint=True)
        interventions = [intervention]
    else:
        interventions, test_env = make_base_small_cenvs()

    # config
    learning_steps = 4800 * 2
    time_steps_lim = learning_steps * 10
    test_episode_timeout = 200
    test_episode_number = 5

    return SmallFrozenTeacherEnv(student_cls=student_cls,
                        student_default_kwargs=student_default_kwargs,
                        interventions=interventions,
                        final_env=test_env,
                        logger_cls=FrozenLakeEvaluationLogger,
                        student_ranges_dict=student_ranges_dict,
                        learning_steps=learning_steps,
                        test_episode_number=test_episode_number,
                        test_episode_timeout=test_episode_timeout,
                        time_steps_lim=time_steps_lim,
                        normalize_obs=False)



def deploy(model, env : gym.Env, timesteps : int = 1000):
    """
    Deploy an agent in a cart pole environemnt (MDP or CMDP) and measures
    performance.

    The performance is measured in terms of success rate, average return and
    average return conditioned on success.

    Parameters
    ----------
    model: stable_baselines model
    env: gym.env
        cart pole environment
    timesteps: int

    Returns
    -------
    success_ratio: float
    avg_return: float
    avg_return_success: float
    trajectories: list of ints
        list of trajectories visited during deployment
    """

    obs = env.reset()
    reward_sum, length, successes, n_episodes = (0.0, 0, 0, 0)
    returns, returns_success, trajectories, trajectory = ([], [], [], [])

    for _ in range(timesteps):
        action, _ = model.predict(obs, deterministic=False)
        if isinstance(env, CMDP):
            obs, reward, g, done, info = env.step(action)
        else:
            obs, reward, done, info = env.step(action)
        reward_sum += reward
        length += 1
        trajectory.append(env.s)
        if done:
            success = info['next_state_type'] == 'G'
            successes += float(success)
            returns.append(reward_sum)
            if success:
                returns_success.append(reward_sum)
            length = 0
            reward_sum = 0.0
            n_episodes += 1
            obs = env.reset()
            trajectories.append(trajectory)
            trajectory = []
    if trajectory:
        trajectories.append(trajectory)
    if n_episodes == 0:
        n_episodes = 1
        returns.append(reward_sum)
    success_ratio = successes / n_episodes
    avg_return = np.mean(returns)
    avg_return_success = np.mean(returns_success)
    return success_ratio, avg_return, avg_return_success, trajectories



def deploy_policy(policy, log_dir : str, env_f, deployment_env_fn : Optional[Callable] = None):
    os.makedirs(log_dir, exist_ok=True)
    teacher_env = env_f()
    obs_t = teacher_env.reset()
    student = teacher_env.student

    n_steps = int(teacher_env.time_steps_lim / teacher_env.learning_steps) + 1
    successes = np.zeros(n_steps, dtype=float)
    training_failures = np.zeros(n_steps, dtype=float)
    averarge_returns = np.zeros(n_steps, dtype=float)
    teacher_rewards = np.zeros(n_steps, dtype=float)
    teacher_observations = np.zeros((obs_t.size, n_steps), dtype=float)

    for i in range(n_steps):
        a, _ = policy.predict(obs_t)
        obs_t, teacher_rewards[i], done, _ = teacher_env.step(a)
        teacher_observations[:, i] = obs_t
        env = teacher_env.final_env if deployment_env_fn is None else deployment_env_fn()
        # env = teacher_env.actions[a]()
        succ, avg_r, avg_r_succ, traj = deploy(student, env, 10000)
        successes[i], averarge_returns[i] = succ, avg_r
        training_failures[i] = teacher_env.student_failures

        i += 1
        if done:
            break

    # Plot successes and returns
    np.savez(os.path.join(log_dir, 'results.npz'),
             successes=successes, averarge_returns=averarge_returns,
             teacher_rewards=teacher_rewards, training_failures=training_failures)
    plt.figure()
    f, axes = plt.subplots(1, 3)
    metrics = [successes, averarge_returns, teacher_rewards]
    titles = ['successes', 'student ret', 'teacher ret']
    for a, metric, title in zip(axes, metrics, titles):
        if title == 'teacher ret':
            a.plot(np.cumsum(metric))
        else:
            a.plot(metric)
        a.set_title(title)

    plt.savefig(os.path.join(log_dir, 'results.pdf'), format='pdf')

    plt.figure()
    for o_num, o in enumerate(teacher_observations):
        plt.plot(o, label=o_num)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'teacher_observations.pdf'),
                format='pdf')


def plot_comparison(log_dir : str, teacher_dir : str) -> None:
    ''' Plot a comparison between different teachers and save the result
    '''
    text_width = cm2inches(13.968)  # Text width in cm
    figsize = (text_width / 2, text_width / 3.5)
    set_figure_params(fontsize=7)

    # Fix plotting when using command line on Mac
    plt.rcParams['pdf.fonttype'] = 42

    metric = ['successes', 'training_failures', 'averarge_returns']
    metric_summary = np.zeros((len(envs), len(metric)), dtype=float)
    teacher = SingleSwitchPolicy.load(os.path.join(teacher_dir,
                                                   'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)

    for i, env in enumerate(envs):
        label = env['label']
        if os.path.isdir(os.path.join(log_dir, label)):
            for j, metric_name in enumerate(metric):
                fig = plt.figure(j, figsize=figsize)
                mu = plot_deployment_metric(os.path.join(log_dir, label),
                                            metric=metric_name, fig=fig,
                                            label=label, legend=True)
                metric_summary[i, j] = mu
        else:
            print(f'[plot_comparison] `{os.path.join(log_dir, label)}` does not exist. Plotting skipped.')

    np.savez(os.path.join(log_dir, 'metrics_summary.npz'),
             metric_summary=metric_summary)
    for j, metric_name in enumerate(metric):
        plt.figure(j)
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(log_dir, metric_name + '.pdf'), format='pdf',
                    transparent=True)
        plt.close(j)


def run_comparision(log_dir : str, teacher_dir : str) -> None:
    cmdp = CMDP(
        env=CustomCartPole(),
        constraints=None,  # TODO: adjust
        constraints_values=None,
        n_constraints=0,
        avg_constraint=True
    )
    teacher = SingleSwitchPolicy.load(os.path.join(teacher_dir,
                                                   'trained_teacher'))
    log_dir = os.path.join(log_dir, teacher.name)

    n_trials = 10
    start_time = time.time()
    for env in envs:
        processes = []
        for i in range(n_trials):
            log_tmp = os.path.join(log_dir, env['name'], f'experiment{i}')
            p = mp.Process(target=deploy_policy,
                           args=[env['model'](), log_tmp, env['teacher_env'],
                                 cmdp])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    print(f'elapsed {time.time() - start_time}')



# define available environments
envs : "list[dict[str,Any]]" = [{
    'name': 'Trained',
    'model': lambda: SingleSwitchPolicy.load(os.path.join(teacher_dir,'trained_teacher')),
    'teacher_env': partial(create_teacher_env, obs_from_training=True),
},{
    'name': 'Original',
    'model': lambda: OpenLoopTeacher([0]),
    'teacher_env': partial(create_teacher_env, original=True),
}]



if __name__ == '__main__':
    
    # compute paths
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
        # we don't have a default one yet
        raise Exception('no teacher given')

    # evaluate
    if args.evaluate:
        for t in teachers:
            print(f'Evaluating teacher {t}')
            teacher_dir = os.path.join(base_teacher_dir, t)
            run_comparision(log_dir, teacher_dir)

    # plot
    if args.plot:
        for t in teachers:
            print(f'Plotting teacher {t}')
            teacher_dir = os.path.join(base_teacher_dir, t)
            plot_comparison(log_dir, teacher_dir)
