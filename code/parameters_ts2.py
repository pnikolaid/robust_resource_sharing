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

# ------------- re-run all files from a to g if any of the below changes

# Define Zone and Frequency
zones = ['I', 'I']
freqs = ['796', '1815']

# Each zone  + freq pair defines a new Network Slice
slices = len(zones)

# Define data collection period for each file
days = 3

# Slice 1
start_time1 = datetime(2020, 6, 17, 14, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time1 = datetime(2020, 6, 17, 19, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 2
start_time2 = datetime(2020, 5, 24, 14, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time2 = datetime(2020, 5, 24, 19, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# -------------

# Provisioning parameters
epsilon = 0.01/slices
delta = 0.01/slices

# Define P_H (re-run files e-g if changed)
P_H = [0.9, 0.9]

start_times = [start_time1, start_time2]
end_times = [end_time1, end_time2]

# Define multiplexing layer (re-run files d-g if changed)
mu = 2

# Define target bitrate in Mbps for each slice (re-run files d-g if changed)
Rc = [1, 2]

# Define aggregation in seconds (equivalent to D in the paper) (re-run files c-g if changed)
T = 10

# Define space aggregation parameters (re-run files d-g if changed)
round_step_u = 10   # number of users N in range k*round_step_u<N <= (k+1)*round_step_u mapped to the state value k
round_step_m = 5    # similarly for MCS value
round_step_w = 10   # similarly for PRBs

# Sample default window size (re-run files f-g if changed)
sample_size_n = list(range(50, 300, 50))
# sample_size_n = [1, 10, 20, 30, 40, 50, 100, 150, 200]

# Set false alarm threshold (re-run files f-g if changed)
a = 0.01

# Anomaly parameters (re-run files f-g if changed)
anomaly_matrix = [[0], [1]]
start_anomaly = max(sample_size_n)
