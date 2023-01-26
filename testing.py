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


''' Create a random circuit where only 1 axis of the bloch
    sphere on the first qubit is rotated so only 2 measurements
    must be done later

    axis: the axis to rotate about
    subcirc_size: number of qubits in each subcircuit
'''
def gen_random_circuit_specific_rotation(axis, subcirc_size=2):
    # First half of our circuit (qubits 0 and 1)
    subcirc1 = QuantumCircuit(2)
    # Get the random value to rotate
    theta = random.uniform(0, 6.28)

    # Rotate just along the axis we want
    if axis == "X":
        subcirc1.rx(theta, [i for i in range(0, subcirc_size)])
    elif axis == "Y":
        subcirc1.ry(theta, [0])

    subcirc1.h(0)
    subcirc1.h(1)
        
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
    fullcirc.compose(subcirc2, qubits=[1,2], inplace=True)
    fullcirc.measure_all()

    # print(subcirc1)
    # print(subcirc2)
    # print(fullcirc)

    return fullcirc, subcirc1, subcirc2


''' For a given axis (X,Y,Z) create subcircuits and a full circuit
    to compare the reconstruction method with a run of the full
    circuit
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



def compare_golden_and_standard(trials=1000, max_size=5, shots=10000):
    axes = ["X", "Y"]
    golden_times = np.zeros([max_size-2, trials])
    standard_times = np.zeros([max_size-2, trials])

    sum_golden = 0
    sum_standard = 0

    for subcirc_size in range(2, max_size):
        ic(subcirc_size)
        for trial in range(trials):
            axis = random.choice(axes)
            circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis)

            start = time.time()
            pA, pB = run_subcirc_known_axis(subcirc1, subcirc2, axis, shots)
            end = time.time()
            elapsed = end-start
            golden_times[subcirc_size-2][trial] = elapsed
            sum_golden += elapsed

            start = time.time()
            pA, pB = run_subcirc(subcirc1, subcirc2, shots)
            end = time.time()
            elapsed = end-start
            standard_times[subcirc_size-2][trial] = elapsed
            sum_standard += elapsed

    ic(sum_golden)
    ic(sum_standard)

    






# gen_random_circuit_specific_rotation("X")
# one_cut_known_axis()

compare_golden_and_standard(trials=100, max_size=5, shots=1000)