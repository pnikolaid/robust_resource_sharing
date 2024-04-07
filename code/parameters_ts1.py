from parameters import test_scenario

import os
from datetime import datetime
from zoneinfo import ZoneInfo


# Directories
parent_directory = os.path.dirname(os.getcwd())
os.chdir(parent_directory + "\\data")   # working directory
plot_directory = parent_directory + "\\plots\\"
test_scenario_plot_directory = plot_directory + "ts" + str(test_scenario) + "\\"
code_directory = parent_directory + "\\code\\"


# Define Zone and Frequency
zones = ['I', 'I']
freqs = ['1815', '2650']

# Each zone  + freq pair defines a new Network Slice
slices = len(zones)

# Provisioning parameters
epsilon = 0.01/slices
delta = 0.01/slices

# Define data collection period for each file
days = 5

# Slice 1
start_time1 = datetime(2020, 5, 25, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time1 = datetime(2020, 5, 25, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 2
start_time2 = datetime(2020, 5, 18, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time2 = datetime(2020, 5, 18, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Define P_H
P_H = [0.9, 0.9]

start_times = [start_time1, start_time2]
end_times = [end_time1, end_time2]

# Define multiplexing layer
mu = 2

# Define target bitrate in Mbps for each slice
Rc = [1, 1]

# Define aggregation in seconds
T = 10

# Define space aggregation parameters
round_step_u = 10   # number of users N in range k*round_step_u<N <= (k+1)*round_step_u mapped to the state value k
round_step_m = 5    # similarly for MCS value
round_step_w = 10   # similarly for PRBs

# Sample default window size
sample_size_n = list(range(50, 300, 50))

# Set false alarm threshold
a = 0.01

# Anomaly parameters
anomalous_slices = [0]   # anomalous_slice indices
# remove the lowest β% user states in MC
low_states_removal_ratio = round(5/6, 2)   # please also set parameter to 4/6 and to 5/6, and re-run f_regular_phase.py
start_anomaly = max(sample_size_n)
