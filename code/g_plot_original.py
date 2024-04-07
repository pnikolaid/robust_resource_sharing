from parameters import slices, anomalous_slices, low_states_removal_ratio, zones, freqs, test_scenario,\
    test_scenario_plot_directory, sample_size_n

from statistics import mean
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tabulate import tabulate


def plot_ecdf(time_series, string_label):
    res1 = scipy.stats.ecdf(time_series)
    states = res1.cdf.quantiles
    probs = res1.cdf.probabilities
    plt.step(states, probs, label=string_label)


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


with open(f"ts{test_scenario}_sim_results.pkl", 'rb') as f:
    sim_results = pickle.load(f)

with open(f"ts{test_scenario}_dict_h_times.pkl", 'rb') as f:
    dict_h_times = pickle.load(f)


# Plot general traffic information for this test scenario
params = {'font.size': 20, 'font.family': 'serif', 'figure.figsize': [12, 4], 'text.usetex': False}
plt.rcParams.update(params)

# Plot User Traffic
total_U = np.zeros(len(Users[0]))
for i in range(slices):
    original_U = Users[i]
    total_U = total_U + np.array(original_U)

    label = f"NS {i}"
    plot_ecdf(original_U, label)

    if i == slices - 1:
        label = "total"
        plot_ecdf(total_U, label)

plt.title(f"Test Scenario {test_scenario}: Connected Users")
plt.xlabel('Number of Connected Users')
plt.ylabel("ECDF")
plt.legend()
plt.savefig(test_scenario_plot_directory + "user_ecdf" + ".pdf", bbox_inches="tight")
plt.close()

# Plot Demand Traffic
total_W = np.zeros(len(Demands[0]))
for i in range(slices):
    original_W = Demands[i]
    total_W = total_W + np.array(original_W)

    label = f"NS {i}"
    plot_ecdf(original_W, label)

    if i == slices - 1:
        label = "total"
        plot_ecdf(total_W, label)


plt.title(f"Test Scenario {test_scenario}: Bandwidth Demands")
plt.xlabel('Bandwidth Demand (PRBs)')
plt.ylabel("ECDF")
plt.legend()
plt.savefig(test_scenario_plot_directory + "PRB_ecdf" + ".pdf", bbox_inches="tight")
plt.close()

print("\nPlots saved in", test_scenario_plot_directory)
print("")
# Create Table data
labels = ["NoSh", "Sh", "ShT"]

columns = ["Scheme", " PRBs"]
ac_strings = []
for i in range(slices):
    string = "$a_" + str(i) + "$" + " (\%)"
    ac_strings.append(string)
columns += ac_strings
columns += ["$r^c_" + str(anomalous_slices[0]) + "$" + " (\%)"] + ["$r^w_" + str(anomalous_slices[0]) + "$" + " (\%)"]
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

# Plot time complexity
mean_times = []
for key in dict_h_times.keys():
    temp_times = dict_h_times[key]
    mean_times.append(mean(temp_times))
keys = dict_h_times.keys()

plt.plot(keys, mean_times)
plt.xlabel("Sample size n")
plt.ylabel("Execution time (ms)")
plt.title(f"Test Scenario {test_scenario}: Hypothesis Testing")
plt.savefig(test_scenario_plot_directory + "execution_time.pdf", bbox_inches="tight")
plt.close()
