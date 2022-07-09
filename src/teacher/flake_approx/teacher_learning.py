import os
import time
from datetime import datetime
import numpy as np
import GPy
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel
from src.teacher.flake_approx.config import MAP_NAME
from src.teacher.frozen_single_switch_utils import evaluate_single_switch_policy, \
    SingleSwitchPolicy
from src.teacher.flake_approx.teacher_env import create_teacher_env, \
    small_base_cenv_fn


def main(interventions = (0,1,2)):
    n_interv = len(interventions)

    if n_interv == 2:
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 6)},
                  {'name': 'var_2', 'type': 'continuous', 'domain': (0, 0.5)}]
        kern = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=[1., 0.05],
                            ARD=True)
        model = GPModel(kernel=kern, noise_var=0.1, max_iters=0)

        teacher_env = create_teacher_env(obs_from_training=True)
        student_final_env = small_base_cenv_fn()

        def bo_objective(thresholds):
            thresholds = np.array(thresholds)
            if thresholds.ndim == 2:
                thresholds = thresholds[0]
            policy = SingleSwitchPolicy(thresholds)
            return evaluate_single_switch_policy(policy, teacher_env, student_final_env)

    elif n_interv == 3:
        domain = [
            # threshold limits
            {'name': 'var_1', 'type': 'continuous', 'domain': (-0.5, 5.5)},
            {'name': 'var_2', 'type': 'continuous', 'domain': (0, 0.2)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (-0.5, 5.5)},
            {'name': 'var_4', 'type': 'continuous', 'domain': (0, 0.2)},
            # teacher interventions
            {'name': 'var_5', 'type': 'discrete', 'domain': (0, 1, 2)},
            {'name': 'var_6', 'type': 'discrete', 'domain': (0, 1, 2)},
            {'name': 'var_7', 'type': 'discrete', 'domain': (0, 1, 2)}
        ]

        kern = GPy.kern.RBF(input_dim=7, variance=1,
                            lengthscale=[1., 0.05, 1, 0.05, 0.5, 0.5, 0.5],
                            ARD=True)
        kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(1, 1),
                                    np.array([0, 2]))
        kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(0.05, 0.02),
                                    np.array([1, 3]))
        kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(0.2, 0.2),
                                    np.array([4, 5, 6]))
        kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 0.2))
        model = GPModel(kernel=kern, noise_var=0.05, max_iters=1000)

        teacher_env = create_teacher_env(obs_from_training=True)
        student_final_env = small_base_cenv_fn()

        def init_teaching_policy(params, name=None):
            params = np.squeeze(np.array(params))
            thresholds = params[:4]
            thresholds = thresholds.reshape(2, 2)
            available_actions = params[4:].astype(np.int64)
            policy = SingleSwitchPolicy(thresholds, available_actions, name=name)
            return policy

        def bo_objective(params):
            policy = init_teaching_policy(params)
            return evaluate_single_switch_policy(policy, teacher_env,
                                                 student_final_env)

    elif n_interv > 3:
        # init domain of teacher parameters to optimize
        domain = [
            # threshold limits
            x for i in range(0, 2*n_interv-2, 2) for x in [
                {'name': f'var_{i}', 'type': 'continuous', 'domain': (-0.5, 5.5)},
                {'name': f'var_{i+1}', 'type': 'continuous', 'domain': (0, 0.2)},
            ]
        ] + [
            # teacher interventions
            {'name': f'var_{i}', 'type': 'discrete', 'domain': interventions}
            for i in range(2*n_interv-2, 3*n_interv-2)
        ]

        # init RBF kernel
        kern = GPy.kern.RBF(
            input_dim = len(domain),
            variance = 1,
            lengthscale = [1., 0.05]*(n_interv-1) + [0.5]*n_interv,
            ARD = True
        )

        # TODO: maybe add priors?

        # init model
        # TODO: maybe adjust parameters
        model = GPModel(kernel=kern, noise_var=0.05, max_iters=1000)

        # create environments
        teacher_env = create_teacher_env(obs_from_training=True)
        student_final_env = small_base_cenv_fn()

        def init_teaching_policy(params, name=None):
            params = np.squeeze(np.array(params))
            thresholds = params[:2*n_interv-2]
            thresholds = thresholds.reshape(n_interv-1, 2)
            available_actions = params[2*n_interv-2:].astype(np.int64)
            policy = SingleSwitchPolicy(thresholds, available_actions, name=name)
            return policy

        def bo_objective(params):
            policy = init_teaching_policy(params)
            return evaluate_single_switch_policy(policy, teacher_env,
                                                 student_final_env)

    else:
        raise ValueError(f'Unexpected number of intrventions. Expected >= 2, but got {n_interv}')

    # Logging dir
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir, 'results',
                               'flake')
    base_dir = os.path.join(results_dir, 'teacher_training', MAP_NAME)
    os.makedirs(base_dir, exist_ok=True)

    my_bo = BayesianOptimization(bo_objective,
                                 domain=domain,
                                 initial_design_numdata=10,
                                 initial_design_type='random',
                                 acquisition_type='LCB',
                                 maximize=True,
                                 normalize_Y=True,
                                 model_update_interval=1,
                                 model=model)

    my_bo.suggest_next_locations()  # Creates the GP model
    my_bo.model.model['Gaussian_noise.variance'].set_prior(
        GPy.priors.Gamma.from_EV(0.01, 0.1))

    start_time = time.time()
    my_bo.run_optimization(20,
                           report_file=os.path.join(base_dir, 'bo_report.txt'),
                           evaluations_file=os.path.join(base_dir,
                                                         'bo_evaluations.csv'),
                           models_file=os.path.join(base_dir, 'bo_model.csv'))
    print(f'Optimization complete in {time.time() - start_time:.2f} s')
    print(f'Optimal threshold: {my_bo.x_opt}')
    print(f'Optimal return: {my_bo.fx_opt}')
    np.savez(os.path.join(base_dir, 'solution.npz'), xopt=my_bo.x_opt,
             fxopt=my_bo.fx_opt)
    trained_policy = init_teaching_policy(my_bo.x_opt)
    save_path = os.path.join(base_dir, 'trained_teacher')
    trained_policy.save(save_path)


if __name__ == '__main__':
    main()
    # main((4,5,6,7))
