import matplotlib.pyplot as plt

from reconstruct import *
from distances import *
from qiskit.circuit.random import random_circuit
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, Aer, transpile
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from icecream import ic
from main import *
import time
import random
import os.path
import numpy as np
import scipy.stats as st



''' Create a circuit that definitely does *not* have a golden cutting point
'''
def gen_random_circuit_not_golden(subcirc_size=2):
    # create some random gates for the upstream circuit
    # subcirc1 = random_circuit(subcirc_size, subcirc_size)
    subcirc1 = QuantumCircuit(subcirc_size)

    # make sure the last qubit in upstream circuit is not a golden cutting point
    # theta = random.uniform(0.1, 1.5)
    theta = 0.5
    subcirc1.rx(theta, subcirc_size-1)
    subcirc1.ry(theta, subcirc_size-1)
    # subcirc1.rz(theta, subcirc_size-1)

    subcirc1.rx(theta, 0)
    subcirc1.ry(theta, 0)

    # create the random downstream circuit
    subcirc2 = random_circuit(subcirc_size, subcirc_size)

    # create the full circuit
    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[i for i in range(subcirc_size-1, subcirc_size*2-1)], inplace=True)
    fullcirc.measure_all()

    # print(subcirc1)
    # print(subcirc2)
    # print(fullcirc)

    return fullcirc, subcirc1, subcirc2

''' Create a random circuit where only 1 axis of the bloch
    sphere on the first qubit is rotated so only 2 measurements
    must be done later

    axis: the axis to rotate about
    subcirc_size: number of qubits in each subcircuit
'''
def gen_random_circuit_specific_rotation(axis, subcirc_size=2):
    # First half of our circuit (qubits 0 and 1)
    subcirc1 = QuantumCircuit(subcirc_size)
    # Get the random value to rotate
    # theta = random.uniform(0.1, 1.5)
    theta = 0.5

    # Rotate just along the axis we want
    if axis == "X":
        subcirc1.rx(theta, [i for i in range(0, subcirc_size)])
        subcirc1.ry(theta, 0)
    elif axis == "Y":
        subcirc1.ry(theta, [i for i in range(0, subcirc_size)])
        subcirc1.rx(theta, 1)

    subcirc1_non_shared = random_circuit(subcirc_size-1, subcirc_size)
    subcirc1.compose(subcirc1_non_shared, inplace=True)

    # Create a random second half of the circuit
    subcirc2 = random_circuit(subcirc_size, subcirc_size)

    # create the full circuit
    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[i for i in range(subcirc_size-1, subcirc_size*2-1)], inplace=True)
    fullcirc.measure_all()

    # print(subcirc1)
    # print(subcirc2)
    # print(fullcirc)

    return fullcirc, subcirc1, subcirc2


''' Function to do hypothesis testing with different alpha values and
    different numbers of shots
'''
def create_hypothesis_test_data(alphas, shots, n_trials=10):

    simulator = Aer.get_backend('aer_simulator')

    for alpha in alphas:
        for shot in shots:
            # file names specifying information about this configuration
            golden_file_name = f"results/fixedaxis_golden_percents_alpha_{alpha}_shots_{shot}.npy"
            standard_file_name = f"results/fixedaxis_nongolden_percents_alpha_{alpha}_shots_{shot}.npy"

            golden_distances_file = f"results/fixedaxis_golden_distances_alpha_{alpha}_shots_{shot}.npy"
            nongolden_distances_file = f"results/fixedaxis_nongolden_distances_alpha_{alpha}_shots_{shot}.npy"
            
            # if any of those files doesn't exist, create them
            if not os.path.isfile(golden_file_name):
                # Arrays to store the timing results
                golden_vals = np.zeros([2])
                np.save(golden_file_name, golden_vals)
            if not os.path.isfile(standard_file_name):
                standard_vals = np.zeros([2])
                np.save(standard_file_name, standard_vals)
            if not os.path.isfile(golden_distances_file):
                # Arrays to store the timing results
                golden_dists = np.zeros([n_trials])
                np.save(golden_distances_file, golden_dists)
            if not os.path.isfile(nongolden_distances_file):
                nongolden_dists = np.zeros([n_trials])
                np.save(nongolden_distances_file, nongolden_dists)

            # load in saved values, allowing for stopping the program
            golden_vals = np.load(golden_file_name)
            nongolden_vals = np.load(standard_file_name)
            golden_dists = np.load(golden_distances_file)
            nongolden_dists = np.load(nongolden_distances_file)

            ic(alpha, shot, "golden")
            while golden_vals[1] < n_trials:
                circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation("X", 2)
                # device = get_least_busy_real_device()
                # pA, pB, correct = run_subcirc_axis_testing_batched(subcirc1, subcirc2, device, "X", alpha, shots=shot)
                pA, pB, correct, _ = run_subcirc_hypo_test_axis(subcirc1, subcirc2, "X", alpha, simulator, shots=shot)
                if correct:
                    golden_vals[0] = golden_vals[0] + 1
                golden_vals[1] = golden_vals[1] + 1
                np.save(golden_file_name, golden_vals)

                reconstructed = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
                # Run the full circuit on a simulator to get a "ground truth" result
                sim_circ = transpile(circ, simulator)
                job = simulator.run(sim_circ, shots=shot)
                simulator_full_counts = job.result().get_counts()
                # golden_dists[int(golden_vals[1])-1] = weighted_distance(reconstructed, simulator_full_counts)
                golden_dists[int(golden_vals[1])-1] = l2_norm_distance(reconstructed, simulator_full_counts, 3)
                np.save(golden_distances_file, golden_dists)


            ic(alpha, shot, "nongolden")
            while nongolden_vals[1] < n_trials:
                circ,subcirc1,subcirc2 = gen_random_circuit_not_golden(2)
                # device = get_least_busy_real_device()
                # pA, pB, correct = run_subcirc_axis_testing_batched(subcirc1, subcirc2, device, "none", alpha, shots=shot)
                pA, pB, correct, _ = run_subcirc_hypo_test_axis(subcirc1, subcirc2, "none", alpha, simulator, shots=shot)
                if correct:
                    nongolden_vals[0] = nongolden_vals[0] + 1
                nongolden_vals[1] = nongolden_vals[1] + 1
                np.save(standard_file_name, nongolden_vals)

                reconstructed = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
                # Run the full circuit on a simulator to get a "ground truth" result
                sim_circ = transpile(circ, simulator)
                job = simulator.run(sim_circ, shots=shot)
                simulator_full_counts = job.result().get_counts()
                # nongolden_dists[int(nongolden_vals[1])-1] = weighted_distance(reconstructed, simulator_full_counts)
                nongolden_dists[int(nongolden_vals[1])-1] = l2_norm_distance(reconstructed, simulator_full_counts, 3)
                np.save(nongolden_distances_file, nongolden_dists)


''' Function to take hypothesis testing data and plot how correct the
    algorithm was at different shot numbers and alpha levels
'''
def create_hypothesis_test_plots(alphas, shots):

    gold_y_values = np.zeros([len(shots), len(alphas)])
    nongold_y_values = np.zeros([len(shots), len(alphas)])

    gold_error_bars = np.zeros([len(shots), len(alphas)])
    nongold_error_bars = np.zeros([len(shots), len(alphas)])

    for idx_a, alpha in enumerate(alphas):
        for idx_s, shot in enumerate(shots):
            # file names specifying information about this configuration
            golden_file_name = f"results/fixedaxis_golden_percents_alpha_{alpha}_shots_{shot}.npy"
            standard_file_name = f"results/fixedaxis_nongolden_percents_alpha_{alpha}_shots_{shot}.npy"

            # load in saved values
            golden_vals = np.load(golden_file_name)
            nongolden_vals = np.load(standard_file_name)

            gold_y_val = golden_vals[0] / golden_vals[1]
            nongold_y_val = nongolden_vals[0] / nongolden_vals[1]

            gold_standard_error = np.sqrt(gold_y_val*(1-gold_y_val)/golden_vals[1])
            # ic(gold_y_val)
            # ic(1-gold_y_val)
            # ic(gold_y_val*(1-gold_y_val))
            # ic(gold_y_val*(1-gold_y_val)/golden_vals[1])
            # ic(np.sqrt(gold_y_val*(1-gold_y_val)/golden_vals[1]))
            # ic(gold_standard_error)
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

    # Define the colors for each alpha
    colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C', '#B48EAD']

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

    for idx_a, alpha in enumerate(alphas):
        for idx_s, shot in enumerate(shots):
            # file names specifying information about this configuration
            golden_file_name = f"results/fixedaxis_golden_distances_alpha_{alpha}_shots_{shot}.npy"
            standard_file_name = f"results/fixedaxis_nongolden_distances_alpha_{alpha}_shots_{shot}.npy"

            # load in saved values
            golden_vals = np.load(golden_file_name)
            nongolden_vals = np.load(standard_file_name)

            gold_y_values[idx_s, idx_a] = np.average(golden_vals)
            nongold_y_values[idx_s, idx_a] = np.average(nongolden_vals)

    # Define the x-values
    x_values = np.arange(len(shots))

    # Define the colors for each alpha
    colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C', '#B48EAD']

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot each line with a different color
    for i, alpha in enumerate(alphas):
        # y = nongold_y_values[:, i]
        y = gold_y_values[:, i]
        ax.plot(shots, y, color=colors[i], label=f'alpha={alpha}')
    
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
        ax.plot(shots, y, color=colors[i], label=f'alpha={alpha}')
    
    plt.xscale('log')

    # Add legend and labels
    ax.legend()
    ax.set_xlabel('Shots')
    ax.set_ylabel('l2 distance (nongolden reconstruct)')

    # Show the plot
    plt.show()


''' For a given axis (X,Y,Z) create subcircuits and a full circuit
    to compare the reconstruction method with a run of the full
    circuit

    Used to verify correctness of the reconstruction method
'''
def compare_golden_and_standard_fidelities(axis="X", shots=20000, run_on_real_device=False):
    # get the type of device we want to run on
    if run_on_real_device:
        device = get_least_busy_real_device()
        subcirc_size = (device.configuration().n_qubits + 1) // 2
    else:
        subcirc_size = 3
    
    # We get a simulator every time because we always need to get a "ground truth"
    simulator = Aer.get_backend('aer_simulator')
    # Get the random circuit with a specific axis of rotation
    circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis, subcirc_size)
    # reconstruct using the golden cutting method and the desired device & shots
    if run_on_real_device:
        # pA, pB, _ = run_subcirc_known_axis(subcirc1, subcirc2, axis, device, shots)
        pA, pB, _ = run_subcirc_known_axis_batched(subcirc1, subcirc2, axis, device, shots)
    else:
        pA, pB, _ = run_subcirc_known_axis(subcirc1, subcirc2, axis, simulator, shots)
    reconstructed = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())

    # remove any entry that is 0
    keys_to_delete = []
    for key, value in reconstructed.items():
        if value == 0:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del reconstructed[key]

    # Run the full circuit for comparison with reconstruction
    if run_on_real_device:
        circ_ = transpile(circ, device)
        job = device.run(circ_, shots=shots)
        ic("full circuit", job.job_id())
        hardware_full_counts = job.result().get_counts()

    # Run the full circuit on a simulator to get a "ground truth" result
    sim_circ = transpile(circ, simulator)
    job = simulator.run(sim_circ, shots=shots)
    simulator_full_counts = job.result().get_counts()

    # just testing
    # pA, pB, _ = run_subcirc(subcirc1, subcirc2, simulator, shots)
    # standard = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
    # ic(weighted_distance(standard, simulator_full_counts))
    # plot_histogram(standard, title="standard reconstruct")

    # print distances between what distributions we have and the ground truth distribution
    ic(weighted_distance(reconstructed, simulator_full_counts))
    if run_on_real_device:
        ic(weighted_distance(hardware_full_counts, simulator_full_counts))

    # Plot both of the results to see visually if they are the same
    plot_histogram(reconstructed, title=f"reconstructed for {axis}")
    plot_histogram(simulator_full_counts, title=f"full circ simulated for {axis}")
    if run_on_real_device:
        plot_histogram(hardware_full_counts, title=f"full circ real hardware for {axis}")

    plt.show()


''' This function runs both the golden method and the standard method,
    timing each to determine runtime differences.
'''
def compare_golden_and_standard_runtimes(trials=1000, max_size=5, shots=10000, run_on_real_device=False):
    max_size = max_size+1   # make max_size inclusive
    # get the type of device we want to run on
    if run_on_real_device:
        device = get_least_busy_real_device(qubits=5)
        max_size = min(max_size, (device.configuration().n_qubits + 1) // 2)
    else:
        device = Aer.get_backend('aer_simulator')
    # The axes available to us. For info on why "Z" is not an option,
    #   please see the paper
    axes = ["X", "Y"]
    # file names specifying information about the run
    golden_file_name = f"results/golden_times_{trials}_trials_{max_size}_size_{shots}_shots_{run_on_real_device}_real.npy"
    standard_file_name = f"results/standard_times_{trials}_trials_{max_size}_size_{shots}_shots_{run_on_real_device}_real.npy"
    # if either of those files doesn't exist, create them
    if not os.path.isfile(golden_file_name):
        # Arrays to store the timing results
        golden_times = np.zeros([max_size-2, trials])
        np.save(golden_file_name, golden_times)
    if not os.path.isfile(standard_file_name):
        standard_times = np.zeros([max_size-2, trials])
        np.save(standard_file_name, standard_times)

    # load arrays from the files for continued testing
    golden_times = np.load(golden_file_name)
    standard_times = np.load(standard_file_name)

    # find the first trial we're missing (used for continuing from file)
    starting_trial, starting_size = find_first_zero(golden_times, max_size, trials)
    ic(starting_trial, starting_size)
    # if we didn't find any missing trials, we're done, analyze and return
    if starting_trial == -1:
        analyze_runtime_arrays(golden_file_name, standard_file_name)
        return
    
    # 2 is the smallest valid subcircuit size, iterate from there
    for subcirc_size in range(starting_size, max_size):
        ic(subcirc_size)
        # collect multiple trials for each subcircuit size
        for trial in range(starting_trial, trials):
            # On each trial select a random axis for our rotation before the cut
            axis = random.choice(axes)
            circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis)

            # time how long it takes to run and reconstruct using the golden method
            pA, pB, execution_time = run_subcirc_known_axis(subcirc1, subcirc2, axis, device, shots)
            start = time.time()
            exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
            end = time.time()
            elapsed = (end-start) + execution_time
            golden_times[subcirc_size-2][trial] = elapsed

            # time how long it takes to run and reconstruct using the standard method
            pA, pB, execution_time = run_subcirc(subcirc1, subcirc2, device, shots)
            start = time.time()
            exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
            end = time.time()
            elapsed = (end-start) + execution_time
            standard_times[subcirc_size-2][trial] = elapsed

            np.save(golden_file_name, golden_times)
            np.save(standard_file_name, standard_times)
    
    # Now that all the data is in, analyse it
    analyze_runtime_arrays(golden_file_name, standard_file_name)

""" helper function that finds the first 0 index in our array
"""
def find_first_zero(golden_times, max_size, trials):
    for i in range(0, max_size-2):
        for j in range(0, trials):
            if golden_times[i][j] == 0.0:
                starting_trial = j
                starting_size = i+2
                return starting_trial, starting_size
    return -1, -1

""" helper function used to calculate and display statistics
    about our runtime arrays
"""
def analyze_runtime_arrays(golden_file_name, standard_file_name):
    # load arrays from the files for analysis
    golden_times = np.load(golden_file_name)
    standard_times = np.load(standard_file_name)
    # get the total number of trials
    shape = golden_times.shape
    if shape[0] == 1:
        trials = np.count_nonzero(golden_times)
    else:
        # for now, just assume that if we did multiple subcirc sizes,
        #   all sizes completed all trials
        trials = shape[1]
        ic(shape)
    ic("total trials", trials)
    golden_times = golden_times[: , 0:trials]
    standard_times = standard_times[: , 0:trials]
    # Get mean and standard error for runtime for each size of subcircuit
    golden_means = golden_times.mean(axis=1)
    golden_sems = np.std(golden_times, axis=1, ddof=1) / np.sqrt(trials)
    # repeat for standard method
    standard_means = standard_times.mean(axis=1)
    standard_sems = np.std(standard_times, axis=1, ddof=1) / np.sqrt(trials)
    # create empty arrays to store our 95% confidence intervals
    golden_interval = np.zeros([len(golden_means)])
    standard_interval = np.zeros([len(standard_means)])

    # iterate over all of the subcircuit sizes
    for i in range(len(golden_means)):
        ic(i+2)
        ic(golden_means[i], standard_means[i], golden_sems[i], standard_sems[i])

        # for both golden and standard methods, find the upper and lower bounds of
        #   the 95% confidence interval - then find what that is in terms of symmmetric +/-
        overall_interval = st.t.interval(confidence=0.95, df=trials-1, loc=golden_means[i], scale=golden_sems[i])
        plus_minus = (overall_interval[1] - overall_interval[0]) / 2
        golden_interval[i] = plus_minus

        overall_interval = st.t.interval(confidence=0.95, df=trials-1, loc=standard_means[i], scale=standard_sems[i])
        plus_minus = (overall_interval[1] - overall_interval[0]) / 2
        standard_interval[i] = plus_minus

    # create array of x-values for plot
    sizes = np.array([i for i in range(2, len(golden_means) + 2)])
    # create the plot
    fig, ax = plt.subplots()
    if len(golden_means) == 1:
        # if we only ran one subcircuit size, just display the two means as a bar chart with error bars
        labels = ["Standard", "Golden"]
        colors = ['#BF616A', '#FFD700']
        x_pos = np.arange(len(labels))
        means = [standard_means[0], golden_means[0]]
        errors = [standard_interval[0], golden_interval[0]]
        ax.bar(x_pos, means, yerr=errors, align='center', color=colors, ecolor='#2E3440', capsize=10)
        ax.set_title('Comparing runtimes on real hardware')
        ax.set_xlabel('Reconstruction type')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
    else:
        # we ran more than one subcircuit size, so display as line graph with error bars
        ax.errorbar(sizes, golden_means, yerr=golden_interval, fmt ='-o', color='#FFD700', capsize=5, ecolor='#2E3440', label='golden')
        ax.errorbar(sizes, standard_means, yerr=standard_interval, fmt ='-o', color='#BF616A', capsize=5, ecolor='#2E3440', label='standard')
        ax.set_title('Time vs subcircuit size')
        ax.set_xlabel('Subcircuit size (width)')
    ax.set_ylabel('Time (s)')
    plt.savefig('results/real_devices_5_qubit_1k_shots_50_trials_95_confidence.png')
    plt.show()

def get_least_busy_real_device(qubits=0):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    if qubits == 0:
        real_devices = provider.backends(filters=lambda x: not x.configuration().simulator)
    else:
        real_devices = provider.backends(filters=lambda x: not x.configuration().simulator and x.configuration().n_qubits == qubits)
    device = least_busy(real_devices)
    # device = provider.get_backend('ibmq_lima')
    # device = provider.get_backend('ibm_nairobi')
    # device = provider.get_backend('ibmq_quito')
    device = provider.get_backend('simulator_statevector')

    return device


def test_one():
    for _ in range(10):
        total = 0
        for _ in range(10):
            circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation("X", 2)
            # device = get_least_busy_real_device()
            pA, pB = run_subcirc_known_axis_batched_local(subcirc1, subcirc2, "X", shots=10000)

            reconstructed = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
            # We get a simulator every time because we always need to get a "ground truth"
            simulator = Aer.get_backend('aer_simulator')
            # Run the full circuit on a simulator to get a "ground truth" result
            sim_circ = transpile(circ, simulator)
            job = simulator.run(sim_circ, shots=1000)
            simulator_full_counts = job.result().get_counts()

            pA, pB, correct = run_subcirc_axis_testing_local(subcirc1, subcirc2, "X", 0.001, shots=10000)
            reconstructed_w_hypo_test = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())

            # total += weighted_distance(reconstructed, simulator_full_counts)
            total += l2_norm_distance(reconstructed_w_hypo_test, simulator_full_counts, 3)
            # ic(l2_norm_distance(reconstructed_w_hypo_test, simulator_full_counts, 3))
            # ic(l2_norm_distance(reconstructed, simulator_full_counts, 3))
        ic(total / 10)

def test_two():
    circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation("X", 2)
    # device = get_least_busy_real_device()
    pA, pB, correct = run_subcirc_axis_testing_local(subcirc1, subcirc2, "X", 0.001, shots=10000)

    reconstructed = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
    # We get a simulator every time because we always need to get a "ground truth"
    simulator = Aer.get_backend('aer_simulator')
    # Run the full circuit on a simulator to get a "ground truth" result
    sim_circ = transpile(circ, simulator)
    job = simulator.run(sim_circ, shots=1000)
    simulator_full_counts = job.result().get_counts()

    ic(weighted_distance(reconstructed, simulator_full_counts))
    ic(l2_norm_distance(reconstructed, simulator_full_counts, 3))

    keys_to_delete = []
    for key, value in reconstructed.items():
        if value == 0:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del reconstructed[key]
    
    plot_histogram(reconstructed, title=f"reconstructed")
    plot_histogram(simulator_full_counts, title=f"Simulated circuit")

    plt.show()
    

# alphas = [0.1, 0.01, 0.001]
# shots = [10, 50, 100, 500, 1000, 5000, 10000]
alphas = [0.01]
shots = [100]

# gen_random_circuit_specific_rotation("X", 3)
# compare_golden_and_standard_fidelities(axis='X', shots=10000, run_on_real_device=True)
# gen_random_circuit_not_golden(3)
# test_one()
# test_two()
create_hypothesis_test_data(alphas, shots, 100)
create_hypothesis_test_plots(alphas, shots)
create_hypothesis_test_distance_plots(alphas, shots)

# compare_golden_and_standard_runtimes()
# compare_golden_and_standard_runtimes(trials=1, max_size=2, shots=10, run_on_real_device=True)
# compare_golden_and_standard_runtimes(trials=50, max_size=2, shots=1000, run_on_real_device=True)

# analyze_runtime_arrays("results/golden_times_50_trials_3_size_1000_shots_True_real.npy", "results/standard_times_50_trials_3_size_1000_shots_True_real.npy")