from parameters import slices, zones, freqs, test_scenario,\
    test_scenario_plot_directory, sample_size_n, anomaly_matrix, P_H

from collections import defaultdict


from statistics import mean
import pickle
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tabulate import tabulate

import copy


def plot_ecdf(time_series, string_label):
    res1 = scipy.stats.ecdf(time_series)
    states_x = res1.cdf.quantiles
    probs = res1.cdf.probabilities
    plt.step(states_x, probs, label=string_label)


# Plot general traffic information for this test scenario
params = {'font.size': 20, 'font.family': 'serif', 'figure.figsize': [12, 4], 'text.usetex': False}
plt.rcParams.update(params)

# Will store all execution times of hypothesis testing
all_execution_times = defaultdict(list)

# Load beta matrix
with open(f"ts{test_scenario}_beta_matrix.pkl", 'rb') as f:
    beta_matrix = pickle.load(f)  # [RRC users, I_MCS, Demands]


for anomaly_index, anomalous_slices in enumerate(anomaly_matrix):
    betas = beta_matrix[anomaly_index]

    # The scatter plots will be saved here at some point
    fig, ax = plt.subplots()
    for low_states_removal_ratio in betas:

        modes = ["no anomaly", "anomaly"]   # anomaly must be last
        schemes = ["no sharing", "sharing", "sharing + testing"]

        # Load all data needed for plots
        Users = []
        Demands = []
        for i in range(slices):
            zone = zones[i]
            freq = freqs[i]

            with open(f"ts{test_scenario}_" + zone + freq + '-DL-regular.pkl', 'rb') as f:
                new_list_dl = pickle.load(f)  # [RRC users, I_MCS, Demands]

            # Time series
            Users.append(new_list_dl[0][0])
            Demands.append(new_list_dl[0][2])

        with open(f"ts{test_scenario}_sim_results_aNS{anomalous_slices[0]}_b{low_states_removal_ratio}.pkl", 'rb') as f:
            sim_results = pickle.load(f)

        with open(f"ts{test_scenario}_dict_h_times_{low_states_removal_ratio}.pkl", 'rb') as f:
            dict_h_times = pickle.load(f)

        # Plot User Traffic
        fig2, ax2 = plt.subplots()
        ax2.set(xlabel='Number of Connected Users', ylabel="ECDF", title=f"Test Scenario {test_scenario}: "
                                                                         f"Connected Users")

        total_U = np.zeros(len(Users[0]))
        for i in range(slices):
            original_U = Users[i]
            total_U = total_U + np.array(original_U)

            label = f"NS {i}"
            plot_ecdf(original_U, label)

            if i == slices - 1:
                label = "total"
                plot_ecdf(total_U, label)

        ax2.legend()
        fig2.savefig(test_scenario_plot_directory + "user_ecdf" + ".pdf", bbox_inches="tight")
        plt.close(fig2)

        # Plot Demand Traffic
        fig3, ax3 = plt.subplots()
        ax3.set(xlabel='Bandwidth Demand (PRBs)', ylabel="ECDF", title=f"Test Scenario {test_scenario}:"
                                                                       f" Bandwidth Demands")
        total_W = np.zeros(len(Demands[0]))
        for i in range(slices):
            original_W = Demands[i]
            total_W = total_W + np.array(original_W)

            label = f"NS {i}"
            plot_ecdf(original_W, label)

            if i == slices - 1:
                label = "total"
                plot_ecdf(total_W, label)

        ax3.legend()
        fig3.savefig(test_scenario_plot_directory + "PRB_ecdf" + ".pdf", bbox_inches="tight")
        plt.close(fig3)

        # Create Table data
        labels = ["NoSh", "Sh", "ShT"]

        columns = ["Scheme", " PRBs"]
        ac_strings = []
        for i in range(slices):
            string = "$a_" + str(i) + "$" + " (\%)"
            ac_strings.append(string)
        columns += ac_strings
        columns += ["$r^c_" + str(anomalous_slices[0]) + "$" + " (\%)"] + ["$r^w_" + str(anomalous_slices[0]) + "$" +
                                                                           " (\%)"]
        for mode in modes:
            data = []
            if mode == "no anomaly":
                print(f"\caption{{Test Scenario {test_scenario}: Results when all NSs behave normally.}}")
            else:
                print(f"\caption{{Test Scenario {test_scenario}: Results when NS ${anomalous_slices[0]}$ "
                      f"is anomalous with $\\beta={low_states_removal_ratio}$.}}")
            for i, scheme in enumerate(schemes):
                key = mode + " & " + scheme
                for idx, n in enumerate(sample_size_n):
                    if scheme != "sharing + testing":
                        success_ratios = sim_results[key][idx][1]
                        success_ratios = [int(e) for e in success_ratios]
                        bw = sim_results[key][0][0]
                        label = labels[i]
                        line = [label, bw] + success_ratios
                        if mode == "anomaly":
                            line += ["-", "-"]
                        data.append(line)
                        break
                    else:
                        label = labels[i]
                        label += f"{n}"
                        success_ratios = sim_results[key][idx][1]
                        success_ratios = [int(e) for e in success_ratios]
                        bw = sim_results[key][0][0]
                        line = [label, bw] + success_ratios

                        if mode == "anomaly":
                            fair_rej_ratio = sim_results[key][idx][2][anomalous_slices[0]]
                            fair_rej_ratio = int(100*fair_rej_ratio)
                            unfair_rej_ratio = sim_results[key][idx][3][anomalous_slices[0]]
                            unfair_rej_ratio = int(100*unfair_rej_ratio)
                            rejection_ratios = [fair_rej_ratio, unfair_rej_ratio]
                            line += rejection_ratios
                        data.append(line)

            # data = ["-" for e in data if e is None]

            # Generate LaTeX table
            latex_table = tabulate(data, headers=columns, tablefmt="latex_raw")

            print("\\vspace{-2mm}")
            # Print LaTeX table
            print(latex_table)
            print("\n\n")

        # Store all execution times
        for key in dict_h_times.keys():
            all_execution_times[key] += dict_h_times[key]

# Plot all execution times
mean_times = []
total_tests = 0
for key in all_execution_times:
    total_tests += len(all_execution_times[key])
    mean_times.append(mean(all_execution_times[key]))
keys = list(all_execution_times.keys())

text = f"Total hypothesis tests: {total_tests}"
fig1, ax1 = plt.subplots()
mean_times = [round(e, 1) for e in mean_times]
ax1.plot(keys, mean_times, marker=".")
ax1.set(xlabel="Sample size n", ylabel="Execution time (ms)", title=f"Time Complexity: Hypothesis Testing")
ax1.set_xticks(keys)
ax1.set_yticks(mean_times)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left
ax1.text(0.05, 0.95, text, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

fig1.savefig(test_scenario_plot_directory + "execution_time.pdf", bbox_inches="tight")
plt.close(fig1)

# β vs n scatterplot
for anomaly_index, anomalous_slices in enumerate(anomaly_matrix):

    betas = beta_matrix[anomaly_index]
    states = len(betas) + 1

    # The scatter plots will be saved here at some point
    fig, ax = plt.subplots()
    for low_states_removal_ratio in betas:
        states_removed = round(states * low_states_removal_ratio)

        with open(f"ts{test_scenario}_sim_results_aNS{anomalous_slices[0]}_b{low_states_removal_ratio}.pkl", 'rb') as f:
            sim_results = pickle.load(f)

        # First plot β for the sharing scheme as a reference point
        key = "anomaly" + " & " + "sharing"

        success_ratios = copy.deepcopy(sim_results[key][0][1])
        success_ratios = [e/100 for e in success_ratios]

        # Check if all SLAs of WELL-BEHAVING NSs are fulfilled
        successful = True
        for i in range(slices):
            if i not in anomalous_slices and success_ratios[i] < P_H[i]:
                successful = False
                break

        if successful:
            c = "green"
            marked = "*"
            labeled = "SLAs fulfilled"
        else:
            c = "red"
            marked = "x"
            labeled = "SLAs violated"

        # Add point to plot
        ax.scatter(states_removed, 0, marker=marked, color=c, label=labeled)

        key = "anomaly" + " & " + "sharing + testing"

        for idx, n in enumerate(sample_size_n):

            success_ratios = copy.deepcopy(sim_results[key][idx][1])
            success_ratios = [e / 100 for e in success_ratios]

            # Check if all SLAs of WELL-BEHAVING NSs are fulfilled
            successful = True
            for i in range(slices):
                if i not in anomalous_slices and success_ratios[i] < P_H[i]:
                    successful = False
                    break

            if successful:
                c = "green"
                marked = "*"
                labeled = "SLAs fulfilled"
            else:
                c = "red"
                marked = "x"
                labeled = "SLAs violated"

            # Add point to plot

            ax.scatter(states_removed, n, marker=marked, color=c, label=labeled)

    # Once all betas are completed, create figure
    ax.set(xlabel='Number of Removed States', ylabel="Sample Size n",
           title=f"Test Scenario {test_scenario}: Results when NS {anomalous_slices[0]} is anomalous")

    xlist = list(range(1, states))
    y_list = [0]
    y_list += sample_size_n

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left
    ax.text(0.05, 0.95, "n=0 → Sh scheme", transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.set_xticks(xlist)
    ax.set_yticks(y_list)
    green_star = mlines.Line2D([], [], color='green', marker='*', linestyle='None', label='SLAs of normal NSs satisfied')
    red_x = mlines.Line2D([], [], color='red', marker='x', linestyle='None', label='SLA of a normal NS violated')
    ax.legend(handles=[green_star, red_x])
    fig.savefig(test_scenario_plot_directory + f"scatter_plot_aNS{anomalous_slices[0]}" + ".pdf", bbox_inches="tight")
    plt.close(fig)

print("Plots saved in", test_scenario_plot_directory)
