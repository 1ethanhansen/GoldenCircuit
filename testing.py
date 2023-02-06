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
    theta = random.uniform(0, 6.28)

    # Rotate just along the axis we want
    if axis == "X":
        subcirc1.rx(theta, [i for i in range(0, subcirc_size)])
    elif axis == "Y":
        subcirc1.ry(theta, [i for i in range(0, subcirc_size)])

    subcirc1_non_shared = random_circuit(subcirc_size-1, subcirc_size//2)
    subcirc1.compose(subcirc1_non_shared, inplace=True)

    # Create a random second half of the circuit
    subcirc2 = random_circuit(subcirc_size, subcirc_size//2)

    # create the full circuit
    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[i for i in range(subcirc_size-1, subcirc_size*2-1)], inplace=True)
    fullcirc.measure_all()

    # print(subcirc1)
    # print(subcirc2)
    # print(fullcirc)

    return fullcirc, subcirc1, subcirc2


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
        pA, pB, _ = run_subcirc_known_axis(subcirc1, subcirc2, axis, device, shots)
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
        overall_interval = st.t.interval(confidence=0.99, df=trials-1, loc=golden_means[i], scale=golden_sems[i])
        plus_minus = (overall_interval[1] - overall_interval[0]) / 2
        golden_interval[i] = plus_minus

        overall_interval = st.t.interval(confidence=0.99, df=trials-1, loc=standard_means[i], scale=standard_sems[i])
        plus_minus = (overall_interval[1] - overall_interval[0]) / 2
        standard_interval[i] = plus_minus

    # create array of x-values for plot
    sizes = np.array([i for i in range(2, len(golden_means) + 2)])
    # create the plot
    fig, ax = plt.subplots()
    if len(golden_means) == 1:
        # if we only ran one subcircuit size, just display the two means as a bar chart with error bars
        labels = ["Standard", "Golden"]
        x_pos = np.arange(len(labels))
        means = [standard_means[0], golden_means[0]]
        errors = [standard_interval[0], golden_interval[0]]
        ax.bar(x_pos, means, yerr=errors, align='center', color='#BF616A', ecolor='#2E3440', capsize=10)
        ax.set_title('Time vs reconstruction type')
        ax.set_xlabel('Subcircuit size (width)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
    else:
        # we ran more than one subcircuit size, so display as line graph with error bars
        ax.errorbar(sizes, golden_means, yerr=golden_interval, fmt ='-o', color='#FFD700', capsize=5, ecolor='#2E3440', label='golden')
        ax.errorbar(sizes, standard_means, yerr=standard_interval, fmt ='-o', color='#BF616A', capsize=5, ecolor='#2E3440', label='standard')
        ax.set_title('Time vs subcircuit size')
        ax.set_xlabel('Subcircuit size (width)')
    ax.set_ylabel('Time (s)')
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

    return device


# gen_random_circuit_specific_rotation("X", 5)
compare_golden_and_standard_fidelities(axis='X', shots=10000, run_on_real_device=True)

# compare_golden_and_standard_runtimes()
# compare_golden_and_standard_runtimes(trials=1, max_size=2, shots=10, run_on_real_device=True)
# compare_golden_and_standard_runtimes(trials=50, max_size=2, shots=1000, run_on_real_device=True)

# analyze_runtime_arrays("results/golden_times_50_trials_3_size_1000_shots_True_real.npy", "results/standard_times_50_trials_3_size_1000_shots_True_real.npy")