from parameters import code_directory, round_step_u, round_step_m, round_step_w, slices, zones, freqs, Rc, T, \
    test_scenario

import bisect
import pickle
import numpy as np
import math
from datetime import datetime
from zoneinfo import ZoneInfo


def similar_values(olist, step):
    min_val = math.ceil(np.min(olist) / step)
    max_val = math.ceil(np.max(olist) / step)
    rlist = [step * x for x in range(min_val, max_val + 1)]

    nlist = [min(bisect.bisect_left(rlist, item), len(rlist) - 1) for item in olist]
    nlist = [rlist[item] for item in nlist]
    return nlist, rlist


def compute_demands(list_of_users, list_of_mcs, bitrate):
    # transform MCS to I_TBS
    i_tbs = []
    for x in list_of_mcs:
        if x <= 9:
            i_tbs.append(x)
        elif x <= 16:
            i_tbs.append(x - 1)
        elif x <= 26:
            i_tbs.append(x - 2)
        else:
            i_tbs.append(26)

    # Add required PRBs
    with open(code_directory + '3GPP-table.pkl', 'rb') as f:
        table = pickle.load(f)

    table = np.array(table)

    # transform table to Mbps to match the bitrate in Mbps
    table = table / 1000

    demands = []
    # find required PRBs
    for k in range(len(i_tbs)):
        itbs = int(i_tbs[k])
        tbs_row = table[itbs]
        W = bisect.bisect_left(tbs_row, bitrate) + 1
        total_demand = W * list_of_users[k]
        demands.append(total_demand)

    return demands


def big_processing_function(list1_dl, RRC_users, target_bitrate, trial_data):

    if trial_data:
        string = "trial"
    else:
        string = "regular"

    print(f"list1_dl entries: {len(list1_dl)}")
    first_t = datetime.fromtimestamp(list1_dl[0][0], ZoneInfo(key='Europe/Madrid'))
    last_t = datetime.fromtimestamp(list1_dl[-1][0], ZoneInfo(key='Europe/Madrid'))
    print(f"list1_dl observation period: from {first_t} to {last_t}")

    print(f"RRC_users entries: {len(RRC_users)}")
    first_t = datetime.fromtimestamp(RRC_users[0][0], ZoneInfo(key='Europe/Madrid'))
    last_t = datetime.fromtimestamp(RRC_users[-1][0], ZoneInfo(key='Europe/Madrid'))
    print(f"RRC_users observation period: from {first_t} to {last_t}")

    # Combine RRC connected users list with list1_dl list
    resume = 0
    list2_dl = []
    temp_list = []
    alerts = 0
    for item in RRC_users:
        for k in range(resume, len(list1_dl)):
            if item[0] <= list1_dl[k][0] < item[0] + T:
                temp_list.append(list1_dl[k])
            # elif list1_dl[k][0] >= item[0] + 1:  # T was originally 1
            else:
                if temp_list:
                    temp_array = np.array(temp_list)
                    avg_mcs = np.mean(temp_array[:, 2])
                    avg_PRBs = np.mean(temp_array[:, 3])
                    max_PRBs = np.max(temp_array[:, 3])
                else:
                    alerts += 1
                    temp_array = np.array(list1_dl)
                    avg_mcs = np.mean(temp_array[:, 2])
                    avg_PRBs = np.mean(temp_array[:, 3])
                    max_PRBs = np.max(temp_array[:, 3])
                break
        list2_dl.append([item[0], item[1], avg_mcs, avg_PRBs, max_PRBs])
        resume = k
        temp_list = []

    # list2_dl format: [timestamp, RRC users, avg_mcs, avg_PRBs, max_PRBs, max_users]
    print(f"list2_dl entries: {len(list2_dl)}")
    print(f"Alert Ratio: {alerts / len(list2_dl)}")

    np_new_list_dl = np.array(list2_dl)
    Users = np_new_list_dl[:, 1]
    MCS = np_new_list_dl[:, 2]
    avg_W = np_new_list_dl[:, 3]
    max_W = np_new_list_dl[:, 4]

    # Compute required PRBs given the fixed target_bitrate in Mbps
    demands = compute_demands(Users, MCS, target_bitrate)

    # Aggregate similar values
    Users, rU = similar_values(Users, round_step_u)
    print("RRC Users:", Users)
    # print(rU)

    MCS, rM = similar_values(MCS, round_step_m)
    print("I_MCS:", MCS)
    # print(rM)

    demands, rW = similar_values(demands, round_step_w)
    print("Demands:", demands)
    # print(rW)

    new_list_dl = [[Users, MCS, demands], [rU, rM, rW]]

    with open(f"ts{test_scenario}_" + zone + freq + '-DL-' + string + '.pkl', 'wb') as f:
        pickle.dump(new_list_dl, f)


# MAIN
if __name__ == "__main__":

    for data_index in range(slices):
        zone = zones[data_index]
        freq = freqs[data_index]
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Processing slice with data {zone}-{freq}")

        with open(f"ts{test_scenario}_" + zone + freq + '-DL-' + '-trial-' + 'almost' + '.pkl', 'rb') as f:
            trial_data = pickle.load(f)
            trial_list1_dl = trial_data[0]
            trial_RRC_users = trial_data[1]

        with open(f"ts{test_scenario}_" + zone + freq + '-DL-' + '-regular-' + 'almost' + '.pkl', 'rb') as f:
            regular_data = pickle.load(f)
            regular_list1_dl = trial_data[0]
            regular_RRC_users = trial_data[1]

        print("--------------------------------------------- Trial Data ----------------------------------------------")
        big_processing_function(trial_list1_dl, trial_RRC_users, Rc[data_index], True)
        print("-------------------------------------------- Regular Data ---------------------------------------------")
        big_processing_function(regular_list1_dl, regular_RRC_users, Rc[data_index], False)

        # function saves data as .pkl files with format [[Users, MCS, demands], [rU, rM, rW]]
