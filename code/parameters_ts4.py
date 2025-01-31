from parameters import test_scenario

import os
from datetime import datetime
from zoneinfo import ZoneInfo


# Directories
parent_directory = os.path.dirname(os.getcwd())
rep_directory = os.path.dirname(parent_directory)
data_directory = os.path.join(rep_directory, "data")
os.chdir(data_directory)   # working directory

plot_directory = os.path.join(parent_directory, "plots")
test_scenario_plot_directory = os.path.join(plot_directory, f"ts{test_scenario}")
if not os.path.exists(test_scenario_plot_directory):
    os.makedirs(test_scenario_plot_directory)

code_directory = os.path.join(parent_directory, "code")

# ------------- re-run all files from a to g if any of the below changes

# Define Zone and Frequency
zones = ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']
freqs = ['796', '796', '796', '1815', '1815', '1815', '1815', '1815', '1815', '2650', '2650', '2650']

# Each zone  + freq pair defines a new Network Slice
slices = len(zones)

# Define data collection period for each file
days = 3

# Slice 1
start_time1 = datetime(2020, 6, 15, 9, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time1 = datetime(2020, 6, 15, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 2
start_time2 = datetime(2020, 6, 16, 9, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time2 = datetime(2020, 6, 16, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 3
start_time3 = datetime(2020, 6, 17, 9, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time3 = datetime(2020, 6, 17, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 4
start_time4 = datetime(2020, 5, 11, 9, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time4 = datetime(2020, 5, 11, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 5
start_time5 = datetime(2020, 5, 12, 9, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time5 = datetime(2020, 5, 12, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 6
start_time6 = datetime(2020, 5, 13, 9, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time6 = datetime(2020, 5, 13, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 7
start_time7 = datetime(2020, 5, 25, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time7 = datetime(2020, 5, 25, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 8
start_time8 = datetime(2020, 5, 26, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time8 = datetime(2020, 5, 26, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 9
start_time9 = datetime(2020, 5, 27, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time9 = datetime(2020, 5, 27, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 10
start_time10 = datetime(2020, 5, 18, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time10 = datetime(2020, 5, 18, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 11
start_time11 = datetime(2020, 5, 18, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time11 = datetime(2020, 5, 18, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

# Slice 12
start_time12 = datetime(2020, 5, 18, 17, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)
# print(datetime.isoweekday(start_time1))   # 1 means Monday
end_time12 = datetime(2020, 5, 18, 22, 00, 0, tzinfo=ZoneInfo(key='Europe/Madrid'))  # (year, month, day, h, min, sec)

start_times = [start_time1, start_time2, start_time3, start_time4, start_time5, start_time6, start_time7, start_time8, start_time9, start_time10, start_time11, start_time12]
end_times = [end_time1, end_time2, end_time3, end_time4, end_time5, end_time6, end_time7, end_time8, end_time9, end_time10, end_time11, end_time12]


# -------------
# Provisioning parameters (re-run files e-g if changed)
epsilon = 0.01/slices
delta = 0.01/slices

# Define P_H (re-run files e-g if changed)
P_H = 12 * [0.9]

# Define multiplexing layer (re-run files d-g if changed)
mu = 2

# Define target bitrate in Mbps for each slice (re-run files d-g if changed)
Rc = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1.5, 1.5, 1.5]

# Define aggregation in seconds (equivalent to D in the paper) (re-run files c-g if changed)
T = 10

# Define space aggregation parameters (re-run files d-g if changed)
round_step_u = 10   # number of users N in range k*round_step_u<N <= (k+1)*round_step_u mapped to the state value k
round_step_m = 5    # similarly for MCS value
round_step_w = 10   # similarly for PRBs

# Sample default window size (re-run files f-g if changed)
sample_size_n = [200]
#sample_size_n = list(range(50, 300, 50))
# sample_size_n = [1, 10, 20, 30, 40, 50, 100, 150, 200]

# Set false alarm threshold (re-run files f-g if changed)
a = 0.01

# Anomaly parameters (re-run files f-g if changed)
anomaly_matrix = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
start_anomaly = max(sample_size_n)
