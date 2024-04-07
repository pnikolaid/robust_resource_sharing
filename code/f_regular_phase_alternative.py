from parameters import slices, sample_size_n, start_anomaly, zones, freqs,\
    test_scenario, test_scenario_plot_directory, round_step_u, round_step_w, P_H, Rc, a, anomaly_matrix

import math
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.stats
from pydtmc import MarkovChain
from e_trial_phase import store_as_matrix_list
from ortools.algorithms import pywrapknapsack_solver
from d_compute_Z_time_series_part2 import compute_demands, similar_values
import time


def mle(data):

    counts = defaultdict(int)
    q_matrix = defaultdict(float)
    for k in range(0, len(data)):

        counts[data[k]] += 1

        if k != 0:
            q_matrix[(data[k-1], data[k])] += 1

    for key in q_matrix.keys():
        initial_state = key[0]
        state_counts = counts[initial_state]
        q_matrix[key] = q_matrix[key] / state_counts

    return q_matrix


def hypothesis_testing(u_data, m_data, u_pmatrix, m_pmatrix, u_g, m_g):

    anomaly = 0
    # first find MLE for Q matrices of U_data and M_data
    u_qmatrix = mle(u_data)
    m_qmatrix = mle(m_data)

    # Compute log-likelihood ratios
    log_l_u = 0
    log_l_m = 0

    for k in range(1, len(u_data)):

        u_transition = (u_data[k-1], u_data[k])
        m_transition = (m_data[k-1], m_data[k])

        if u_transition not in u_pmatrix.keys() or m_transition not in m_pmatrix:
            anomaly = 10
            return anomaly

        log_l_u = log_l_u + math.log(u_qmatrix[u_transition]) - math.log(u_pmatrix[u_transition])
        log_l_m = log_l_m + math.log(m_qmatrix[m_transition]) - math.log(m_pmatrix[m_transition])

    if log_l_u >= math.log(u_g):
        anomaly = 1

    # if log_l_m >= math.log(m_g):
    #    anomaly = 2

    return anomaly


def bkp(values, weights, capacity):
    weights = [weights]
    capacity = [capacity]
    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.
        KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')
    solver.Init(values, weights, capacity)
    solver.Solve()
    selected_items = []
    for i in range(len(values)):
        if solver.BestSolutionContains(i):
            selected_items.append(i)
    return selected_items


def big_multiplexing_function_over_time(t_Users, t_MCS, Wc, WH, scheme, stored_samples, dict_times):
    print("Provisioned Bandwidth:", Wc)

    # Define lists that hold n samples for hypothesis testing
    n_Users = []
    n_MCS = []
    for i in range(slices):
        n_Users.append([])
        n_MCS.append([])

    allocations = []
    deficits = [0] * slices
    v_times = []   # times that bandwidth was not enough
    decisions = [[] for i in range(slices)]  # contains vector u
    d_times = [[] for i in range(slices)]  # contains number of times that anomaly was detected for each slice
    for t in range(T):
        A = list(range(slices))  # initialize prioritized slices to be all slices (no anomaly will be detected)
        B = []  # make list empty
        allocated_bandwidths = np.zeros(slices)
        total_demand = 0
        for i in range(slices):
            total_demand += Demands[i][t]
            n_Users[i].append(Users[i][t])
            n_MCS[i].append(MCS[i][t])

            if t >= stored_samples:
                n_Users[i].pop(0)
                n_MCS[i].pop(0)

        if total_demand > Wc:
            # print(f"Time {t}: Not enough bandwidth!")
            v_times.append(t)

            # Determine prioritized slices A(t)

            # if scheme is proposed solution, use hypothesis testing to find A(t)
            if scheme == "sharing + testing":
                A = []  # reset prioritized slices
                for i in range(slices):
                    start_time = time.perf_counter_ns()
                    anomaly_detected = hypothesis_testing(n_Users[i], n_MCS[i], t_Users[i], t_MCS[i], u_gamma[i],
                                                          m_gamma[i])
                    h_time = time.perf_counter_ns() - start_time
                    h_time = h_time / (10**6)  # time in ms
                    dict_times[stored_samples].append(h_time)
                    if not anomaly_detected:
                        A.append(i)
                    else:
                        B.append(i)
                        d_times[i].append(t)

                        # if anomaly_detected == 1:
                        #     print(f"Time {t}: User anomaly detected for slice {i}")
                        # elif anomaly_detected == 2:
                        #     print(f"Time {t}: MCS anomaly detected for slice {i}")
                        # elif anomaly_detected == 10:
                        #     print(f"Time {t}: A transition not observed in trial phase has been detected for slice {i}")

            # if scheme is no sharing, slice i is served if W_i(t) <= W^H_i
            if scheme == "no sharing":
                # reset prioritized slices
                A = []
                for i in range(slices):
                    if Demands[i][t] <= WH[i]:
                        A.append(i)

            # Allocate resources between slices in A by solving Binary Knapsack Problem (32)
            A_demands = [Demands[i][t] for i in A]
            A_deficits = [deficits[i] for i in A]
            A_selected = bkp(A_deficits, A_demands, Wc)
            A_accepted = [A[item] for item in A_selected]
            for i in A_accepted:
                allocated_bandwidths[i] = Demands[i][t]
            A_rejected = [i for i in A if i not in A_accepted]  # slices in A but not in A_accepted

            WR = Wc - sum(allocated_bandwidths)

            # In case not all slices in A were accepted, split the remaining bandwidth among the slices in A_R
            if A_rejected:
                inverse_demands = [1 / Demands[i][t] for i in A_rejected]
                arg_max = np.argmax(inverse_demands)
                slice_index = A_rejected[arg_max]
                allocated_bandwidths[slice_index] = WR
            # in case all accepted slices were satisfied
            else:
                B_demands = [Demands[i][t] for i in B]
                B_deficits = [deficits[i] for i in B]
                B_selected = bkp(B_deficits, B_demands, WR)
                B_accepted = [B[item] for item in B_selected]
                for i in B_accepted:
                    allocated_bandwidths[i] = Demands[i][t]
                B_rejected = [i for i in B if i not in B_accepted]  # slices in B but not in B_accepted

                WR = Wc - sum(allocated_bandwidths)
                if B_rejected:
                    inverse_demands = [1 / Demands[i][t] for i in B_rejected]
                    arg_max = np.argmax(inverse_demands)
                    slice_index = B_rejected[arg_max]
                    allocated_bandwidths[slice_index] = WR
        else:
            # Provisioned bandwidth was enough for all
            for i in range(slices):
                allocated_bandwidths[i] = Demands[i][t]

        # Update deficits
        for i in range(slices):
            if allocated_bandwidths[i] == Demands[i][t]:
                u = 1
            else:
                u = 0
            deficits[i] = max(deficits[i] - u, 0) + P_H[i]
            decisions[i].append(u)

        allocations.append(allocated_bandwidths)

    return decisions, v_times, d_times, dict_times


def process_data(bandwidth, decisions, v_times, d_times, dict_anomaly_times, mode, scheme):
    success_ratio = [[] for i in range(slices)]
    for i in range(slices):
        success_ratio[i] = round(decisions[i].count(1) / len(decisions[i]) * 100, 2)
        print(f"NS {i} satisfied for {success_ratio[i]}%  of the time (target = {100 * P_H[i]}%)")

    print("Total number of violations:", len(v_times))
    print(f"Ratio of violations: {round(100 * len(v_times) / T, 2)}%")

    unfair_rejection_ratio = []
    fair_rejection_ratio = []

    # find detector statistics for each slice
    if scheme == "sharing + testing":

        if mode == "no anomaly":
            anomalous_slices = []
        else:
            anomalous_slices = dict_anomaly_times.keys()

        for i in range(slices):

            # Attention! pd and pfa are not the detector's power and false alarm rate
            if i not in anomalous_slices:
                print(f"Well-behaved NS {i}---")
                # pfa = len(d_times[i])/len(v_times)
                # pd = 1

            else:
                print(f"Anomalous NS {i}---")

            anomaly_times = []

            if i in anomalous_slices:
                anomaly_times = dict_anomaly_times[i]

            a_v_times = []  # times where testing happened and slice was anomalous
            w_v_times = []  # times during testing where slice was well-behaved
            for t in v_times:
                if t in anomaly_times:
                    a_v_times.append(t)  # times where the detector has the chance to detect anomaly
                else:
                    w_v_times.append(t)  # times where the detector has the chance to ignore good behavior

            pd = 0
            nfa = 0

            for t in a_v_times:
                if t in d_times[i]:
                    pd += 1

            if a_v_times:
                pd = pd / len(a_v_times)
            else:
                pd = 1  # no change to detect so print 1 by convention

            for t in w_v_times:
                if t in d_times[i]:
                    nfa += 1

            if w_v_times:
                pfa = nfa / (len(w_v_times))
            else:
                pfa = 0  # no chance to detect so print 0 by convention

            print(f"Fair rejection rate: {round(pd * 100, 2)}%")
            print(f"Unfair rejection rate: {round(pfa * 100, 2)}%")
            print(f"Number of Unfair rejections: {nfa}")
            fair_rejection_ratio.append(pd)
            unfair_rejection_ratio.append(pfa)

    results = [success_ratio, fair_rejection_ratio, unfair_rejection_ratio]
    results.insert(0, bandwidth)

    return results


def create_anomalies(t_Users, s_Users):
    return_list = []
    dict_anomaly_times = {}
    end_anomaly = T - max(sample_size_n)
    anomalous_slice = anomalous_slices[0]
    temp_Users = copy.deepcopy(or_Users)
    temp_s_Users = copy.deepcopy(s_Users)

    temp_Demands = copy.deepcopy(or_Demands)
    temp_s_Demands = copy.deepcopy(s_Demands)

    # create anomalies
    new_trans_matrix = {}
    old_trans_matrix = t_Users[anomalous_slice]
    old_states = s_Users[anomalous_slice]

    # start from n-1 states all the way to just the worst state
    for w in range(1, len(old_states)):

        low_states_removal_ratio = w / len(old_states)
        states_to_be_removed = old_states[:w]  # state space is sorted
        states_to_stay = old_states[w:]

        for count, stay_state in enumerate(states_to_stay):
            row = {}
            # find the largest transition from that state
            for k, v in old_trans_matrix.items():
                if k[0] == stay_state:
                    row[k] = v

            # sum the transitions to the deleted states  and then delete the transitions
            summed_transition = 0
            for del_state in states_to_be_removed:
                if (stay_state, del_state) in row.keys():
                    summed_transition += row[(stay_state, del_state)]
                    row.pop((stay_state, del_state))

            if row:
                # Add this summed_transition to the largest transition if there are any left
                max_key = max(row, key=row.get)
                row[max_key] += summed_transition
            else:
                print(f"Cannot create MC partition, considering 50% self loop 50% neighboring state")
                # Go to the next state with probability 1/2 and stay where you are with probability 1/2
                row[(stay_state, stay_state)] = 0.5

                if count != len(states_to_stay) - 1:
                    print(f"There exists a larger stay state so neighboring state = next state")
                    next_stay_state = states_to_stay[count + 1]
                    row[(stay_state, next_stay_state)] = 0.5
                else:
                    print(f"There does not exist a larger stay state so neighboring state = previous state")
                    prev_stay_state = states_to_stay[count - 1]
                    row[(stay_state, prev_stay_state)] = 0.5
            new_trans_matrix.update(row)

        new_states = states_to_stay

        # Generate data using new transition matrix
        matrix_as_list = store_as_matrix_list(new_trans_matrix, new_states)

        # find first time after start_anomaly after which state in new_states for smooth transition to anomalous MC

        # Initialization
        model_change_time = start_anomaly
        model_change_state = new_states[0]
        for time, value in enumerate(or_Users[anomalous_slice]):
            if time >= start_anomaly and value in new_states:
                model_change_time = time
                model_change_state = value
                break

        # Generate new sequence that starts from model_change_state
        if len(states_to_stay) > 1:
            strings = [str(i) for i in new_states]
            Markov_Chain = MarkovChain(matrix_as_list, strings)
            new_sequence = Markov_Chain.simulate(end_anomaly + 1 - model_change_time - 1, str(model_change_state), seed=32)
            new_sequence = [int(i) for i in new_sequence]
        else:
            print("Only the largest state is left, new sequence is constant")
            new_sequence = [states_to_stay[0]] * (end_anomaly + 1 - model_change_time)

        # Apply new sequence to old sequence at model_change_time
        temp_Users[anomalous_slice][model_change_time:end_anomaly + 1] = new_sequence

        # Compute anomalous demands
        temp_Demands[anomalous_slice] = compute_demands(temp_Users[anomalous_slice], or_MCS[anomalous_slice], Rc[anomalous_slice])

        # Create very high demands for checks
        # temp_Demands[anomalous_slice][model_change_time:end_anomaly+1] = [Wc_sharing] * (end_anomaly - model_change_time+1)

        # Aggregate values as before
        temp_Users[anomalous_slice], temp_s_Users[anomalous_slice] = similar_values(temp_Users[anomalous_slice],
                                                                                    round_step_u)
        temp_Demands[anomalous_slice], temp_s_Demands[anomalous_slice] = similar_values(temp_Demands[anomalous_slice],
                                                                                        round_step_w)

        print(
            f"Created anomaly for NS {anomalous_slice} starting from time {model_change_time} to time {end_anomaly}"
            f" by deleting the lowest {w} states out of the total {len(old_states)} states in the User MC"
            f" (β={100*low_states_removal_ratio}%)\n")
        anomaly_times = list(range(model_change_time, end_anomaly + 1))
        dict_anomaly_times[anomalous_slice] = anomaly_times

        temp_user_matrix = matrix_as_list
        return_item = [temp_Users, temp_Demands, temp_s_Users, temp_s_Demands, dict_anomaly_times, temp_user_matrix, states_to_stay, low_states_removal_ratio]
        return_list.append(return_item)
    return return_list


def plot_ecdf(time_series, string_label):
    res1 = scipy.stats.ecdf(time_series)
    states = res1.cdf.quantiles
    probs = res1.cdf.probabilities
    plt.step(states, probs, label=string_label)


# --------------------------MAIN ---------------------------------------------------------------------------------------
# Load stochastic models and provisioned bandwidth
string = ""
for i in range(slices):
    string += zones[i] + freqs[i]
    if i != slices-1:
        string += "_vs_"

with open(f"ts{test_scenario}_" + string + '-DL-' + 'estimation_data' + '.pkl', 'rb') as f:
    estimation_data = pickle.load(f)

t_Users = estimation_data[0]
t_MCS = estimation_data[1]
Wc_sharing = estimation_data[2]
WH = estimation_data[3]
Wc_no_sharing = estimation_data[4]

# Load all regular phase data
Users = []
MCS = []
Demands = []

s_Users = []
s_MCS = []
s_Demands = []

u_s = []
m_s = []

for i in range(slices):
    zone = zones[i]
    freq = freqs[i]

    with open(f"ts{test_scenario}_" + zone + freq + '-DL-regular.pkl', 'rb') as f:
        new_list_dl = pickle.load(f)  # [RRC users, I_MCS, Demands]

    # Time series
    Users.append(new_list_dl[0][0])
    MCS.append(new_list_dl[0][1])
    Demands.append(new_list_dl[0][2])

    # State and action spaces
    s_Users.append(new_list_dl[1][0])
    u_s.append(len(new_list_dl[1][0]))

    s_MCS.append(new_list_dl[1][1])
    m_s.append(len(new_list_dl[1][1]))

    s_Demands.append(new_list_dl[1][2])

or_Users = copy.deepcopy(Users)
or_MCS = copy.deepcopy(MCS)
or_Demands = copy.deepcopy(Demands)

T = len(Users[0])

print("Percentiles $W^H_i$:", WH)
print(f"Number of timeslots: {T}")
print(f"User space sizes: {u_s}")
print(f"MCS space sizes: {m_s}")

u_gamma = []
m_gamma = []

# Calculation of gammas for hypothesis testing
for i in range(slices):
    u_r = (u_s[i] ** 2 - u_s[i])  # degrees of freedom
    m_r = (m_s[i] ** 2 - m_s[i])

    u_gamma.append(math.exp(scipy.stats.chi2.ppf(1 - a / 4, u_r)))  # use inverse cdf of chi square
    m_gamma.append(math.exp(scipy.stats.chi2.ppf(1 - a / 4, m_r)))

print(f"User γ: {u_gamma}")
print(f"MCS γ: {m_gamma}")

beta_matrix = []
for anomaly_index, anomalous_slices in enumerate(anomaly_matrix):

    # -------------------------- Create anomalies----------------------------------------------------------------
    print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    anomalies_list = create_anomalies(t_Users, s_Users)
    beta_matrix.append([])
    for anomaly_item in anomalies_list:
        a_Users, a_Demands, _, _, dict_anomaly_times, _, _, low_states_removal_ratio = anomaly_item
        beta_matrix[anomaly_index].append(low_states_removal_ratio)

        # ------------------------- Store all simulations results in the "sim_results" dictionary ----------------------
        sim_results = {}
        dict_h_times = defaultdict(list)
        # ------------------------------------------ Start simulations ------------------------------------------------
        modes = ["no anomaly", "anomaly"]   # anomaly must be last
        schemes = ["no sharing", "sharing", "sharing + testing"]
        for mode in modes:

            # if anomaly mode
            if mode == "anomaly":
                Users = a_Users
                Demands = a_Demands
            else:
                Users = or_Users
                Demands = or_Demands

            for scheme in schemes:

                label = mode + " & " + scheme

                sim_results[label] = []

                # Set provisioned bandwidth
                if scheme == "no sharing":
                    prov_bw = Wc_no_sharing
                else:
                    prov_bw = Wc_sharing

                for n in sample_size_n:

                    # Run the whole simulation and return statistics
                    print(f"\n++++++++++++++++++++ anomalous NS = {anomalous_slices[0]} | β = {low_states_removal_ratio} | mode = {label} | sample size = {n} +++++++++++++++")
                    decisions, v_times, d_times, dict_h_times = big_multiplexing_function_over_time(t_Users, t_MCS,
                                                                                                    prov_bw, WH, scheme,
                                                                                                    n, dict_h_times)
                    # Process the statistics to get the results
                    results = process_data(prov_bw, decisions, v_times, d_times, dict_anomaly_times, mode, scheme)

                    # store the results
                    sim_results[label].append(results)

                    # if the scheme does not perform hypothesis testing, no need to consider more sample sizes
                    if scheme != "sharing + testing":
                        break

        print("----------------------------------- Print and store simulation results dictionary ---------------------")
        print(f"\n {sim_results}\n \n")

        # Store all data needed for plots

        with open(f"ts{test_scenario}_sim_results_aNS{anomalous_slices[0]}_b{low_states_removal_ratio}.pkl", 'wb') as f:
            pickle.dump(sim_results, f)

        with open(f"ts{test_scenario}_dict_h_times_{low_states_removal_ratio}.pkl", 'wb') as f:
            pickle.dump(dict_h_times, f)

print(beta_matrix)
with open(f"ts{test_scenario}_beta_matrix.pkl", 'wb') as f:
    pickle.dump(beta_matrix, f)
