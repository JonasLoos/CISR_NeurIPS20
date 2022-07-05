'''
Config file for the frozen lake teacher

other files should import the constants defined here
'''


# Map name from src/envs/frozen_lake/frozen_maps.py
# The paper used 'small'
MAP_NAME = '5x5_simple'


# Number of trials (students run in parallel) to get a good average during mode evaluation
# This might not incude the bandit policy, see `compare_teachers.run_bandits`
# The paper used 10
NUMBER_OF_TRIALS = 10


# Interventions / modes
# The paper used ['Trained', 'SR1', 'SR2', 'HR', 'Original', 'Bandit']
# Custom modes: ['Halfway', 'Incremental']
INTERVENTION_MODES = ['Halfway', 'Trained', 'Incremental']


# INTERVENTIONS : "list[dict[str,Any]]" = [{
#     'name': 'Trained',
#     'label': '',
#     'model': lambda: SingleSwitchPolicy.load(os.path.join(teacher_dir,'trained_teacher')),
#     'teacher_env': partial(create_teacher_env, obs_from_training=True),
# },{
#     'name': 'Original',
#     'label': '',
#     'model': lambda: OpenLoopTeacher([0]),
#     'teacher_env': partial(create_teacher_env, original=True),
# }]
