'''
Config file for the frozen lake teacher

other files should import the constants defined here
'''


# Map name from src/envs/frozen_lake/frozen_maps.py
# The paper used 'small'
MAP_NAME = 'small'


# Number of trials (students trained per mode) to get a good average during mode evaluation
# This might not incude the bandit policy, see `compare_teachers.run_bandits`
# The paper used 10
NUMBER_OF_TRIALS = 10


# Interventions / modes
# The paper used ['Trained', 'SR1', 'SR2', 'HR', 'Original', 'Bandit']
# Custom modes: ['Halfway', 'Incremental']
INTERVENTION_MODES = ['Halfway', 'Trained', 'Incremental']
