import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from icecream import ic

''' Function to take hypothesis testing data and plot how correct the
    algorithm was at different shot numbers and alpha levels
'''
def create_hypothesis_test_accuracy_plots(alphas, shots):

    gold_y_values = np.zeros([len(shots), len(alphas)])
    nongold_y_values = np.zeros([len(shots), len(alphas)])

    gold_error_bars = np.zeros([len(shots), len(alphas)])
    nongold_error_bars = np.zeros([len(shots), len(alphas)])

    for idx_a, alpha in enumerate(alphas):
        for idx_s, shot in enumerate(shots):
            # file names specifying information about this configuration
            golden_file_name = f"results/percents_golden_alpha_{alpha}_shots_{shot}.npy"
            standard_file_name = f"results/percents_nongolden_alpha_{alpha}_shots_{shot}.npy"

            # load in saved values
            golden_vals = np.load(golden_file_name)
            nongolden_vals = np.load(standard_file_name)

            gold_y_val = golden_vals[0] / golden_vals[1]
            nongold_y_val = nongolden_vals[0] / nongolden_vals[1]

            gold_standard_error = np.sqrt(gold_y_val*(1-gold_y_val)/golden_vals[1])
            overall_interval = st.t.interval(confidence=0.95, df=golden_vals[1]-1, loc=gold_y_val, scale=gold_standard_error)
            plus_minus = (overall_interval[1] - overall_interval[0]) / 2
            gold_error_bars[idx_s, idx_a] = plus_minus

            nongold_standard_error = np.sqrt(nongold_y_val*(1-nongold_y_val)/nongolden_vals[1])
            overall_interval = st.t.interval(confidence=0.95, df=nongolden_vals[1]-1, loc=nongold_y_val, scale=nongold_standard_error)
            plus_minus = (overall_interval[1] - overall_interval[0]) / 2
            nongold_error_bars[idx_s, idx_a] = plus_minus

            gold_y_values[idx_s, idx_a] = gold_y_val
            nongold_y_values[idx_s, idx_a] = nongold_y_val

    # Define the x-values
    x_values = np.arange(len(shots))

    # Define the colors for each alpha (comes from nord theme)
    colors = ['#BF616A', '#A3BE8C', '#5E81AC']

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each line with a different color
    for i, alpha in enumerate(alphas):
        # y = nongold_y_values[:, i]
        y = gold_y_values[:, i]
        y_err = gold_error_bars[:, i]
        ax.errorbar(shots, y, yerr=y_err, capsize=5, color=colors[i], label=f'alpha={alpha}')
    
    plt.xscale('log')

    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Shots')
    ax.set_ylabel('Correctly identified golden')

    # Show the plot
    plt.show()

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each line with a different color
    for i, alpha in enumerate(alphas):
        y = nongold_y_values[:, i]
        y_err = nongold_error_bars[:, i]
        ax.errorbar(shots, y, yerr=y_err, capsize=5, color=colors[i], label=f'alpha={alpha}')
    
    plt.xscale('log')

    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Shots')
    ax.set_ylabel('Correctly identified nongolden')

    # Show the plot
    plt.show()


def create_hypothesis_test_distance_plots(alphas, shots):

    gold_y_values = np.zeros([len(shots), len(alphas)])
    nongold_y_values = np.zeros([len(shots), len(alphas)])

    gold_error_bars = np.zeros([len(shots), len(alphas)])
    nongold_error_bars = np.zeros([len(shots), len(alphas)])

    for idx_a, alpha in enumerate(alphas):
        for idx_s, shot in enumerate(shots):
            # file names specifying information about this configuration
            golden_file_name = f"results/distances_golden_alpha_{alpha}_shots_{shot}.npy"
            standard_file_name = f"results/distances_nongolden_alpha_{alpha}_shots_{shot}.npy"

            # load in saved values
            golden_vals = np.load(golden_file_name)
            nongolden_vals = np.load(standard_file_name)

            gold_mean = np.average(golden_vals)
            nongold_mean = np.average(nongolden_vals)

            gold_y_values[idx_s, idx_a] = gold_mean
            nongold_y_values[idx_s, idx_a] = nongold_mean

            gold_standard_error = np.std(golden_vals, ddof=1) / np.sqrt(len(golden_vals))
            overall_interval = st.t.interval(confidence=0.95, df=len(golden_vals)-1, loc=gold_mean, scale=gold_standard_error)
            plus_minus = (overall_interval[1] - overall_interval[0]) / 2
            gold_error_bars[idx_s, idx_a] = plus_minus

            nongold_standard_error = np.std(nongolden_vals, ddof=1) / np.sqrt(len(nongolden_vals))
            overall_interval = st.t.interval(confidence=0.95, df=len(nongolden_vals)-1, loc=nongold_mean, scale=nongold_standard_error)
            plus_minus = (overall_interval[1] - overall_interval[0]) / 2
            nongold_error_bars[idx_s, idx_a] = plus_minus

    # Define the x-values
    x_values = np.arange(len(shots))

     # Define the colors for each alpha (comes from nord theme)
    colors = ['#BF616A', '#A3BE8C', '#5E81AC']

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each line with a different color
    for i, alpha in enumerate(alphas):
        y = gold_y_values[:, i]
        y_err = gold_error_bars[:, i]
        ax.errorbar(shots, y, yerr=y_err, capsize=5, color=colors[i], label=f'alpha={alpha}')
    
    plt.xscale('log')

    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Shots')
    ax.set_ylabel('l2 distance (golden reconstruct)')

    # Show the plot
    plt.show()

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each line with a different color
    for i, alpha in enumerate(alphas):
        y = nongold_y_values[:, i]
        y_err = nongold_error_bars[:, i]
        ax.errorbar(shots, y, yerr=y_err, capsize=5, color=colors[i], label=f'alpha={alpha}')
    
    plt.xscale('log')

    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Shots')
    ax.set_ylabel('l2 distance (nongolden reconstruct)')

    # Show the plot
    plt.show()


def create_hypothesis_test_time_plot(alphas):
    testing_y_vals = np.zeros([len(alphas)])
    standard_y_vals = np.zeros([len(alphas)])

    testing_y_errs = np.zeros([len(alphas)])
    standard_y_errs = np.zeros([len(alphas)])

    for idx, alpha in enumerate(alphas):
        ic(alpha)
        # file names specifying information about this configuration
        golden_file_name = f"results/golden_times_alpha_{alpha}.npy"
        standard_file_name = f"results/nongolden_times_alpha_{alpha}.npy"

        golden_times = np.load(golden_file_name)
        standard_times = np.load(standard_file_name)

        testing_mean = np.average(golden_times)
        standard_mean = np.average(standard_times)

        testing_y_vals[idx] = testing_mean
        standard_y_vals[idx] = standard_mean

        gold_standard_error = np.std(golden_times, ddof=1) / np.sqrt(len(golden_times))
        overall_interval = st.t.interval(confidence=0.95, df=len(golden_times)-1, loc=testing_mean, scale=gold_standard_error)
        plus_minus = (overall_interval[1] - overall_interval[0]) / 2
        testing_y_errs[idx] = plus_minus

        standard_standard_error = np.std(standard_times, ddof=1) / np.sqrt(len(standard_times))
        overall_interval = st.t.interval(confidence=0.95, df=len(standard_times)-1, loc=standard_mean, scale=standard_standard_error)
        plus_minus = (overall_interval[1] - overall_interval[0]) / 2
        standard_y_errs[idx] = plus_minus

    # Define the x-values
    x_values = np.arange(len(alphas))

    colors = ['#A3BE8C', '#5E81AC']

    # Create a figure and axis object
    fig, ax = plt.subplots()
    ax.errorbar(alphas, testing_y_vals, yerr=testing_y_errs, capsize=5, color='#BF616A', label='performed hypo. test')
    ax.errorbar(alphas, standard_y_vals, yerr=standard_y_errs, capsize=5, color='#5E81AC', label='no hypo. test')

    plt.xscale('log')
    
    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Alpha')
    ax.set_ylabel('time to complete reconstruction (s)')

    # Show the plot
    plt.show()


''' Function to take hypothesis testing data and plot how many
    golden points there were at different depths and alpha levels
'''
def create_prop_of_random_golden_plots(alphas, depths):

    prop_y_values = np.zeros([len(depths), len(alphas)])

    prop_error_bars = np.zeros([len(depths), len(alphas)])

    for idx_a, alpha in enumerate(alphas):
        for idx_d, depth in enumerate(depths):
            # file names specifying information about this configuration
            data_file_name = f"results/prop_data_alpha_{alpha}_depth_{depth}.npy"

            # load in saved values
            prop_vals = np.load(data_file_name)

            y_val = prop_vals[0] / prop_vals[1]

            standard_error = np.sqrt(y_val*(1-y_val)/prop_vals[1])
            overall_interval = st.t.interval(confidence=0.95, df=prop_vals[1]-1, loc=y_val, scale=standard_error)
            plus_minus = (overall_interval[1] - overall_interval[0]) / 2
            prop_error_bars[idx_d, idx_a] = plus_minus

            prop_y_values[idx_d, idx_a] = y_val

    # Define the x-values
    x_values = np.arange(len(depths))

    # Define the colors for each alpha (comes from nord theme)
    colors = ['#BF616A', '#A3BE8C', '#5E81AC']

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each line with a different color
    for i, alpha in enumerate(alphas):
        # y = nongold_y_values[:, i]
        y = prop_y_values[:, i]
        y_err = prop_error_bars[:, i]
        ax.errorbar(depths, y, yerr=y_err, capsize=5, color=colors[i], label=f'alpha={alpha}')

    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Depth of upstream subcircuit')
    ax.set_ylabel('Proportion identified as golden')

    # Show the plot
    plt.show()



# five_qubit_reconstructed = np.array([1.4074852085238574,
#                                         0.017868887078699325,
#                                         0.024213309596474844,
#                                         0.006091082239396866,
#                                         0.018155823945393917,
#                                         0.5269854324922746,
#                                         0.04826506825923811,
#                                         0.030888714532464178,
#                                         0.03580744717827404,
#                                         1.085320314963532])
# five_qubit_full = np.array([0.5517542628262189,
#                             0.036953282633326506,
#                             0.014257431624436339,
#                             0.00595196698567,
#                             0.009381460795516607,
#                             1.1280944173293714,
#                             0.038188494657883265,
#                             0.01557064911715274,
#                             0.04533825851015292,
#                             0.5288732310281644])

# seven_qubit_reconstructed = np.array([0.07526394884202078, 
#                                         1.0829483048454138, 
#                                         0.07958280792472788, 
#                                         0.41526155600130654, 
#                                         2.168616476483447, 
#                                         0.23079945962832776, 
#                                         0.12109898569985258, 
#                                         1.0107215383364128,
#                                         0.3663553976629582,
#                                         0.33631449447336226])
# seven_qubit_full = np.array([0.38561365905205575,
#                                 1.09490807941074,
#                                 0.10808728769875667,
#                                 1.4795683851309027,
#                                 1.3467772706328958,
#                                 0.22102615357080838,
#                                 0.17492552917486862,
#                                 16.208313364517377,
#                                 0.0450318546172423,
#                                 0.30973225630203866])

# vals_to_plot = np.array([])
# errs_to_plot = np.array([])
# labels = ["5 qubit full", "5 qubit reconstruct", "7 qubit full", "7 qubit reconstruct"]

# five_full_mean = five_qubit_full.mean()
# five_reconstruct_mean = five_qubit_reconstructed.mean()
# vals_to_plot = np.append(vals_to_plot, five_full_mean)
# vals_to_plot = np.append(vals_to_plot, five_reconstruct_mean)
# five_full_sem = five_full_mean / np.sqrt(len(five_qubit_full))
# five_reconstruct_sem = five_reconstruct_mean / np.sqrt(len(five_qubit_reconstructed))

# seven_full_mean = seven_qubit_full.mean()
# seven_reconstruct_mean = seven_qubit_reconstructed.mean()
# vals_to_plot = np.append(vals_to_plot, seven_full_mean)
# vals_to_plot = np.append(vals_to_plot, seven_reconstruct_mean)
# seven_full_sem = seven_full_mean / np.sqrt(len(seven_qubit_full))
# seven_reconstruct_sem = seven_reconstruct_mean / np.sqrt(len(seven_qubit_reconstructed))

# overall_interval = st.t.interval(confidence=0.95, df=len(five_qubit_full)-1, loc=five_full_mean, scale=five_full_sem)
# plus_minus = (overall_interval[1] - overall_interval[0]) / 2
# errs_to_plot = np.append(errs_to_plot, plus_minus)

# overall_interval = st.t.interval(confidence=0.95, df=len(five_qubit_reconstructed)-1, loc=five_reconstruct_mean, scale=five_reconstruct_sem)
# plus_minus = (overall_interval[1] - overall_interval[0]) / 2
# errs_to_plot = np.append(errs_to_plot, plus_minus)

# overall_interval = st.t.interval(confidence=0.95, df=len(seven_qubit_full)-1, loc=seven_full_mean, scale=seven_full_sem)
# plus_minus = (overall_interval[1] - overall_interval[0]) / 2
# errs_to_plot = np.append(errs_to_plot, plus_minus)

# overall_interval = st.t.interval(confidence=0.95, df=len(seven_qubit_reconstructed)-1, loc=seven_reconstruct_mean, scale=seven_reconstruct_sem)
# plus_minus = (overall_interval[1] - overall_interval[0]) / 2
# errs_to_plot = np.append(errs_to_plot, plus_minus)

# ic(vals_to_plot)
# ic(errs_to_plot)


# x_pos = np.arange(len(labels))
# colors = ['#BF616A', '#FFD700', '#BF616A', '#FFD700']
# fig, ax = plt.subplots()
# ax.bar(x_pos, vals_to_plot, yerr=errs_to_plot, align='center', color=colors, ecolor='#2E3440', capsize=10)
# ax.set_title('Comparison of weighted distances')
# ax.set_ylabel('Average weighted distance')
# ax.set_xlabel('Run type and size of device')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels)

# plt.savefig('results/comparison_of_distances.png')
# # plt.show()