from parameters import slices, zones, freqs, days, T, test_scenario

import pickle
import numpy as np


# Create list that contains [timestamp, users, avg_mcs, PRBs] from
# the original list [timestamp, sfn, subframe, RNTI, direction, mcs_idx, nof_PRBs, TBS_sum]


def derive_list(olist):
    nlist = []
    timestamp = olist[0][0]
    sfn = olist[0][1]
    subframe = olist[0][2]
    direction = olist[0][4]

    avg_mcs = olist[0][5]
    users = 1
    PRBs = olist[0][6]
    for sublist in olist[1:]:
        if sublist[1] == sfn and sublist[2] == subframe and sublist[4] == direction:
            users += 1
            avg_mcs += sublist[5]
            PRBs += sublist[6]
        else:
            avg_mcs = avg_mcs / users
            nlist.append([timestamp, users, avg_mcs, PRBs])
            timestamp = sublist[0]
            sfn = sublist[1]
            subframe = sublist[2]
            direction = sublist[4]
            users = 1
            avg_mcs = sublist[5]
            PRBs = sublist[6]

    return nlist


# Create list that contains [timestamp, users, avg_mcs, avg_PRBs, max_PRBs] by aggregating L seconds from original list
def aggregate_list(olist, L):

    # initialize
    start_line = olist[0]
    nlist = []
    temp_list = [start_line]
    for item in olist[1:]:
        if item[0] < start_line[0] + L:
            temp_list.append(item)
        else:
            # temp_list is now complete, so process it
            start_time = start_line[0]
            start_users = start_line[1]
            start_avg_mcs = start_line[2]

            temp_array = np.array(temp_list)
            avg_demand = np.mean(temp_array[:, 3])
            max_demand = np.max(temp_array[:, 3])
            nlist.append([start_time, start_users, start_avg_mcs, avg_demand, max_demand])

            # reset
            temp_list = [item]
            start_line = item

    # Add final element
    start_time = start_line[0]
    start_users = start_line[1]
    start_avg_mcs = start_line[2]
    temp_array = np.array(temp_list)
    avg_demand = np.mean(temp_array[:, 3])
    max_demand = np.max(temp_array[:, 3])
    nlist.append([start_time, start_users, start_avg_mcs, avg_demand, max_demand])

    return nlist


# Create smaller list by aggregating RRC users by L seconds
def aggregate_short_list(olist, L):

    # initialize
    start_line = olist[0]
    nlist = []
    temp_list = [start_line]
    for item in olist[1:]:
        if item[0] < start_line[0] + L:
            temp_list.append(item)
        else:
            # temp_list is now complete, so process it
            start_time = start_line[0]
            start_users = start_line[1]
            nlist.append([start_time, start_users])

            # reset
            temp_list = [item]
            start_line = item

    # Add final element
    start_time = start_line[0]
    start_users = start_line[1]
    nlist.append([start_time, start_users])

    return nlist


# MAIN
for data_index in range(slices):
    zone = zones[data_index]
    freq = freqs[data_index]
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Processing slice with data {zone}-{freq}")

    # Unpickle multiple times
    trial_list1_dl = []
    regular_list1_dl = []
    f = open(f"ts{test_scenario}_" + zone + freq + '-alldata' + '.pkl', 'rb')
    i = 1
    while True:
        try:
            plist1 = pickle.load(f)
            if i < days:
                print(f"Loading {zone+freq}-alldata.pkl: Day {i}/{days} (trial phase data)")
            else:
                print(f"Loading {zone+freq}-alldata.pkl: Day {i}/{days} (final day, regular phase data)")
            plist1_dl = []
            plist1_ul = []

            # Separate list into downlink and uplink direction
            for sublist in plist1:
                if sublist[4] == 1:
                    plist1_dl.append(sublist)
                else:
                    plist1_ul.append(sublist)

            plist1_dl = derive_list(plist1_dl)  # [timestamp, users, avg_mcs, PRBs]

            # Aggregate list1_dl by T seconds
            plist1_dl = aggregate_list(plist1_dl, T)
            # print(f"plist_dl entries (after aggregation): {len(plist1_dl)}")

            if i < days:
                trial_list1_dl += plist1_dl
            else:
                regular_list1_dl += plist1_dl
            i += 1
        except EOFError:
            break

    f = open(f"ts{test_scenario}_" + zone + freq + '-RRC-users.pkl', 'rb')
    trial_RRC_users = []
    regular_RRC_users = []
    i = 1
    while True:
        try:
            # Aggregate pRRC_users list by T seconds
            pRRC_users = pickle.load(f)  # [timestamp, RRC users]
            pRRC_users = aggregate_short_list(pRRC_users, T)
            if i < days:
                print(f"Loading {zone+freq}-RRC-users.pkl: Day {i}/{days} (trial phase data)")
            else:
                print(f"Loading {zone+freq}-RRC-users.pkl: Day {i}/{days} (final day, regular phase data)")
            if i < days:
                trial_RRC_users += pRRC_users
            else:
                regular_RRC_users += pRRC_users
            i += 1
        except EOFError:
            break

    with open(f"ts{test_scenario}_" + zone + freq + '-DL-' + '-trial-' + 'almost' + '.pkl', 'wb') as f:
        pickle.dump([trial_list1_dl, trial_RRC_users], f)

    with open(f"ts{test_scenario}_" + zone + freq + '-DL-' + '-regular-' + 'almost' + '.pkl', 'wb') as f:
        pickle.dump([regular_list1_dl, regular_RRC_users], f)
