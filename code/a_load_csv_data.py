from parameters import slices, zones, freqs, start_times, end_times, days, test_scenario

import csv
import pickle
import numpy as np
from datetime import timedelta

for data_index in range(slices):
    zone = zones[data_index]
    freq = freqs[data_index]
    first_start_time = start_times[data_index]
    first_end_time = end_times[data_index]

    users_file = "users_" + zone + freq + "_s.csv"
    file = open(users_file, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    data.pop(0)
    data = [[float(item[2]), int(item[1])] for item in data]   # [timestamp, users]
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Processing slice with data {zone}-{freq}")
    for i in range(days):
        start_time = first_start_time + timedelta(days=i)
        end_time = first_end_time + timedelta(days=i)

        print("-------------------------------------------------------------------------------------------------------")
        print(f"Day {i+1}/{days}")
        print(f"Target period: {start_time} to {end_time}")

        # Consider 1-second slots
        l_point = int(start_time.timestamp())
        new_data = []
        expected_entries = int(end_time.timestamp()) - int(start_time.timestamp())
        print(f"Expected total entries: {expected_entries}")
        alert = 0
        resume = 0
        while l_point < int(end_time.timestamp()):
            temp_list = []
            for k in range(resume, len(data)):
                item = data[k]
                if l_point <= item[0] < l_point + 1:
                    temp_list.append(item)
                elif item[0] >= l_point + 1:
                    break
            if temp_list:
                temp_array = np.array(temp_list)
                avg_u = int(np.mean(temp_array[:, 1]))
            else:
                if new_data:
                    avg_u = new_data[-1][1]  # consider previous data
                else:
                    avg_u = -1
                alert += 1
            new_data.append([l_point, avg_u])
            l_point += 1
            resume = k

        # Remove the -1
        for k in range(len(new_data)):
            if new_data[k][1] == -1:
                for j in range(k + 1, len(new_data)):
                    if new_data[j][1] != -1:
                        new_data[k][1] = new_data[j][1]
                        break

        actual_entries = len(new_data)

        print(f"Actual entries: {actual_entries}")
        print(f"Ratio: {actual_entries/expected_entries}")
        print(f"Alert Ratio: {alert/expected_entries}")
        print(f"Sample line: {new_data[0]}")

        if i == 0:
            with open(f"ts{test_scenario}_" + zone + freq + '-RRC-users' + '.pkl', 'wb') as f:
                pickle.dump(new_data, f)
        else:
            with open(f"ts{test_scenario}_" + zone + freq + '-RRC-users' + '.pkl', 'ab') as f:
                pickle.dump(new_data, f)
