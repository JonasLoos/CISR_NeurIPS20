'''
Config file for the frozen lake teacher

other files should import the constants defined here
'''

from typing import Any


# map name from src/envs/frozen_lake/frozen_maps.py
MAP_NAME = '5x5_simple'


# TODO: WIP
# interventions / modes
INTERVENTION_MODES = ['Halfway', 'Trained', 'Incremental'] # ['Trained', 'SR1', 'SR2', 'HR', 'Original', 'Bandit']

# INTERVENTIONS : "list[dict[str,Any]]" = [{
#     'name': 'Trained',
#     'label': '',
#     'model': lambda: SingleSwitchPolicy.load(os.path.join(teacher_dir,'trained_teacher')),
#     'teacher_env': partial(create_teacher_env, obs_from_training=True),
# },{
#     'name': 'Original',
#     'model': lambda: OpenLoopTeacher([0]),
#     'teacher_env': partial(create_teacher_env, original=True),
# }]
