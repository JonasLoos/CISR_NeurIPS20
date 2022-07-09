'''
Config file for the frozen lake teacher

Other files should import the constants defined here.
Some settings can additionally be specified by command line, which takes priority.
'''


# The name of the folder where the comparison results (from evaluate and plot) should be stored
# If it is empty, a name is created from the settings
COMPARISON_FOLDER = ''


# Map name from src/envs/frozen_lake/frozen_maps.py
# The paper used 'small'
MAP_NAME = '16x16'


# Number of trials (students trained per mode) to get a good average during mode evaluation
# This might not incude the bandit policy, see `compare_teachers.run_bandits`
# The paper used 10
NUMBER_OF_TRIALS = 1


# Number of (curriculum) steps
# Training is done for N_STEPS+1, to account for the zeroth step
# The paper used 10
N_STEPS = 15


# Interventions / modes
# The paper used the following modes
ORIGINAL_INTERVENTION_MODES= ['Trained', 'SR1', 'SR2', 'HR', 'Original', 'Bandit']


# Custom modes: ['Halfway', 'Incremental']
INTERVENTION_MODES = ['Trained', 'SR1', 'QuarterIncremental']


# Teacher to use (from results/flake/teacher_training)
# The paper default was ['03_06_20__11_46_57']
TEACHER_DIRS = ['16x16_3,4,5,6']
