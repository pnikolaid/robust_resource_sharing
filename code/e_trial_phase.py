from parameters import delta, epsilon, slices, zones, freqs, P_H, test_scenario_plot_directory, test_scenario

import math
import pickle
import numpy
from pydtmc import MarkovChain
from collections import defaultdict
from transition_matrix_visualization import *


def sort_by_key(mydict):
    mykeys = list(mydict.keys())
    mykeys.sort()
    sorted_dict = {k: mydict[k] for k in mykeys}
    return sorted_dict


def store_as_matrix_list(dic_matrix, states):
    matrix_list = []
    for k in states:
        row = []
        for j in states:
            if (k, j) in dic_matrix:
                prob = dic_matrix[(k, j)]
            else:
                prob = 0
            row.append(prob)
        if numpy.sum(row) != 1:
            argmax = numpy.argmax(row)
            row[argmax] += 1 - numpy.sum(row)
        matrix_list.append(list(row))
    return matrix_list


def statistics(trans_matrix, state_space):
    factor = math.log(2/delta)/(2*epsilon**2)

    temp_list = trans_matrix

    temp_states = [str(element) for element in state_space]

    # first inequality
    eigenvalues, _ = numpy.linalg.eig(temp_list)
    abs_eig = numpy.absolute(eigenvalues)
    abs_eig = list(abs_eig)
    abs_eig.sort(reverse=True)
    if len(abs_eig) > 2:
        second_eig = abs_eig[1]
    else:
        second_eig = abs_eig[0]
    factor1 = (1 + second_eig) / (1 - second_eig)
    # print("factor1:", factor1)
    T1 = factor1*factor
    # print("Samples needed with method 1:", T1)

    # stationary dist
    MCi = MarkovChain(temp_list, temp_states)
    stat_dist = MCi.pi

    # second inequality
    worst_times = []
    for target in MCi.states:
        times = MCi.hitting_times(target)
        worst_time = max(times)
        worst_times.append(worst_time)
    H = max(worst_times)
    factor2 = H ** 2
    # print("factor2:", factor2)
    T2 = factor2*factor
    # print("Samples needed with method 2:", T2)
    # print("Sample ratio:", T2/T1)

    return stat_dist, T1, T2


if __name__ == "__main__":

    # Simplification 1: consider W=μ(X) (deterministic policy)
    # Z(t)=(X(t),W(t)) becomes X(t)

    # Simplification 2: each state component X_(i)^j behaves independently
    # reduction in memory: 3 matrices of size N^2 instead of a huge one of size N^6 for Zi

    # Simplification 3: set g(x)=x
    # Line 7 and lines 15-25 can be skipped
    # In line 26, g(x) is now defined as g(x)=x

    Users = []
    MCS = []
    Demands = []

    s_Users = []
    s_MCS = []
    s_Demands = []

    c_Users = []
    t_Users = []

    c_MCS = []
    t_MCS = []

    # Load all data
    sum_of_max_demands = 0
    for i in range(slices):
        zone = zones[i]
        freq = freqs[i]
        with open(f"ts{test_scenario}_" + zone + freq + '-DL-trial.pkl', 'rb') as f:
            new_list_dl = pickle.load(f)  # [RRC users, I_MCS, Demands]

        # Time series
        Users.append(new_list_dl[0][0])
        MCS.append(new_list_dl[0][1])
        Demands.append(new_list_dl[0][2])
        # State and action spaces
        s_Users.append(new_list_dl[1][0])
        s_MCS.append(new_list_dl[1][1])
        s_Demands.append(new_list_dl[1][2])

        c_Users.append(defaultdict(int)) # list containing the counts for each number of users
        t_Users.append(defaultdict(float))  # list containing the user transition matrix of each slice as dictionary

        c_MCS.append(defaultdict(int))
        t_MCS.append(defaultdict(float))

        # Store sum of maximum demands to check later statistical multiplexing gain
        sum_of_max_demands += max(Demands[i])

    pmf_total_demand = defaultdict(float)
    # Scan it sequentially as if done online
    Timeslots = len(Users[0])
    print("Total number of timeslots", Timeslots)
    for t in range(1, Timeslots):  # t ranges from 1,..., T-1
        total_demand = 0
        for i in range(slices):
            c_Users[i][Users[i][t]] += 1
            t_Users[i][(Users[i][t-1], Users[i][t])] += 1

            c_MCS[i][MCS[i][t]] += 1
            t_MCS[i][(MCS[i][t-1], MCS[i][t])] += 1
            total_demand += Demands[i][t]
        pmf_total_demand[total_demand] += 1

    # online processing done

    # normalize values
    pmf_total_demand = {key: val/(Timeslots-1) for key, val in pmf_total_demand.items()}
    pmf_total_demand = sort_by_key(pmf_total_demand)
    for i in range(slices):
        for key in t_Users[i].keys():
            initial_state = key[0]
            counts = c_Users[i][initial_state]
            t_Users[i][key] = t_Users[i][key] / counts

        for key in t_MCS[i].keys():
            initial_state = key[0]
            counts = c_MCS[i][initial_state]
            t_MCS[i][key] = t_MCS[i][key] / counts

        t_Users[i] = sort_by_key(t_Users[i])
        t_MCS[i] = sort_by_key(t_MCS[i])


    print("User transition matrix:", t_Users)
    print("MCS transition matrix:", t_MCS)
    print("Demand pmf:", pmf_total_demand)
    max_total_demand = max(pmf_total_demand.keys())

    print("Maximum of sum demand:", max_total_demand)
    print("Sum of maximum demands:", sum_of_max_demands)

    # Find provisioned bandwidths when multiplexing and when not for each slice
    target = max(P_H)
    probability = 0
    Wc = 0
    for demand in pmf_total_demand.keys():
        probability += pmf_total_demand[demand]
        Wc = demand
        if probability >= target:
            break

    print("Provisioned bandwidth (with multiplexing):", Wc)

    # Find provisioned bandwidth Wac if no multiplexing
    Wac = 0
    WH = []
    for i in range(slices):
        WH. append(numpy.quantile(Demands[i], P_H[i], method='inverted_cdf'))

    Wac = sum(WH)

    print("Provisioned bandwidth (no multiplexing):", Wac)
    print("Multiplexing gain:", Wac-Wc)

    # Visualize data
    for i in range(slices):
        filename = zones[i] + freqs[i] + "-users_MC"
        visualize_matrix(t_Users[i], test_scenario_plot_directory + "MarkovChains\\" + filename)

        filename = zones[i] + freqs[i] + "-MCS_MC"
        visualize_matrix(t_MCS[i], test_scenario_plot_directory + "MarkovChains\\" + filename)

    # Save stochastic models and provisioned bandwidth
    string = ""
    for i in range(slices):
        string += zones[i] + freqs[i]
        if i != slices-1:
            string += "_vs_"

    with open(f"ts{test_scenario}_" + string + '-DL-' + 'estimation_data' + '.pkl', 'wb') as f:
        pickle.dump([t_Users, t_MCS, Wc, WH, Wac], f)

    # Statistics of the MCs
    user_matrices = []
    mcs_matrices = []
    stochastic_models = []
    for i in range(slices):
        user_matrix = store_as_matrix_list(t_Users[i], s_Users[i])
        mcs_matrix = store_as_matrix_list(t_MCS[i], s_MCS[i])

        user_matrices.append(user_matrix)
        mcs_matrices.append(mcs_matrix)

        print(f"\n-----Statistics for NS {i}----")
        print(f"\n--User Markov Chain--")
        print("States", s_Users[i])
        if len(s_Users[i]) > 1:
            pi_U, TU1, TU2 = statistics(user_matrices[i], s_Users[i])
            pi_U = [str(round(element*100, 2)) + "%" for element in pi_U[0]]
            result = dict(zip(s_Users[i], pi_U))
            print("Stationary distribution:", result)

        print(f"\n--MCS Markov Chain--")
        print("States", s_MCS[i])
        if len(s_MCS[i]) > 1:
            pi_M, TM1, TM2 = statistics(mcs_matrices[i], s_MCS[i])
            pi_M = [str(round(element*100, 2)) + "%" for element in pi_M[0]]
            result2 = dict(zip(s_MCS[i], pi_M))
            print("Stationary distribution:", result2)

