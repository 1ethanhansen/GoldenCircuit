import matplotlib.pyplot as plt

from reconstruct import *
from qiskit.circuit.random import random_circuit
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, Aer, transpile
from icecream import ic
from main import *
import time
import random
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
        subcirc1.ry(theta, [0])

    subcirc1_non_shared = random_circuit(subcirc_size-1, subcirc_size//2)

    subcirc1.compose(subcirc1_non_shared, inplace=True)
        
    # Optionally add entangling gate between qubits 0 and 1
    # before the cut
    # if entangled and axis == "Z":
    #     subcirc1.cz(0, 1)
    # elif entangled and axis != "Z":
    #     subcirc1.cnot(0, 1)

    # Create a random second half of the circuit
    subcirc2 = random_circuit(subcirc_size, subcirc_size//2)

    # create the full circuit
    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.cnot(subcirc_size-1, subcirc_size)
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
def one_cut_known_axis(axis="X", shots=10000):
    # Get the random circuit with a specific axis of rotation, then reconstruct
    circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis)
    pA, pB = run_subcirc_known_axis(subcirc1, subcirc2, axis, shots)
    # pA, pB = run_subcirc(subcirc1, subcirc2, shots)
    exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())

    # Run the full circuit for comparison with reconstruction
    simulator = Aer.get_backend('aer_simulator')
    circ_ = transpile(circ, simulator)
    counts = simulator.run(circ_, shots=shots).result().get_counts()
    # print(counts)

    # Plot both of the results to see visually if they are the same
    plot_histogram(exact, title=f"reconstructed for {axis}")
    plot_histogram(counts, title=f"full circ for {axis}")

    plt.show()


''' This function runs both the golden method and the standard method,
    timing each to determine runtime differences.
'''
def compare_golden_and_standard(trials=1000, max_size=5, shots=10000):
    max_size = max_size+1   # make max_size inclusive
    # The axes available to us. For info on why "Z" is not an option,
    #   please see the paper
    axes = ["X", "Y"]
    # Arrays to store the timing results
    golden_times = np.zeros([max_size-2, trials])
    standard_times = np.zeros([max_size-2, trials])

    # 2 is the smallest valid subcircuit size, iterate from there
    for subcirc_size in range(2, max_size):
        ic(subcirc_size)
        # collect multiple trials for each subcircuit size
        for trial in range(trials):
            # On each trial select a random axis for our rotation before the cut
            axis = random.choice(axes)
            circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis)

            # time how long it takes to run and reconstruct using the golden method
            start = time.time()
            pA, pB = run_subcirc_known_axis(subcirc1, subcirc2, axis, shots)
            exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
            end = time.time()
            elapsed = end-start
            golden_times[subcirc_size-2][trial] = elapsed

            # time how long it takes to run and reconstruct using the standard method
            start = time.time()
            pA, pB = run_subcirc(subcirc1, subcirc2, shots)
            exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
            end = time.time()
            elapsed = end-start
            standard_times[subcirc_size-2][trial] = elapsed
    
    # Now that all the data is in, analyse it
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
    sizes = np.array([i for i in range(2, max_size)])
    # create the plot
    fig, ax = plt.subplots()
    ax.errorbar(sizes, golden_means, yerr=golden_interval, fmt ='-o', color='#5E81AC', capsize=10, ecolor='#2E3440', label='golden')
    ax.errorbar(sizes, standard_means, yerr=standard_interval, fmt ='-o', color='#BF616A', capsize=10, ecolor='#2E3440', label='standard')
    ax.set_title('Time vs subcircuit size')
    ax.set_xlabel('Subcircuit size (width)')
    ax.set_ylabel('Time (s)')
    plt.show()

# gen_random_circuit_specific_rotation("X", 5)
# one_cut_known_axis()

compare_golden_and_standard()
# compare_golden_and_standard(trials=100, max_size=3, shots=10)