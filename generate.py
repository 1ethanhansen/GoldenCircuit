import random
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit


''' Create a circuit that definitely does *not* have a golden cutting point
'''
def gen_random_circuit_not_golden(subcirc_size=2):
    # create some random gates for the upstream circuit
    # subcirc1 = random_circuit(subcirc_size, subcirc_size)
    subcirc1 = QuantumCircuit(subcirc_size)

    # make sure the last qubit in upstream circuit is not a golden cutting point
    theta1 = random.uniform(0.2, 1.4)
    theta2 = random.uniform(0.2, 1.4)
    # theta = 0.5

    subcirc1.rx(theta1, [i for i in range(0, subcirc_size)])
    subcirc1.ry(theta2, [i for i in range(0, subcirc_size)])

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
    theta = random.uniform(0.1, 1.5)

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


''' Create a completely random circuit split into upstream and
    downstream. No restrictions placed on rotation or what gates
    can be used
'''
def gen_completely_random_circuit(subcirc_size=2, subcirc_depth=2):
    # create some random gates for the upstream circuit
    subcirc1 = random_circuit(subcirc_size, subcirc_depth)

    # create the random downstream circuit
    subcirc2 = random_circuit(subcirc_size, subcirc_depth)

    # create the full circuit
    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[i for i in range(subcirc_size-1, subcirc_size*2-1)], inplace=True)
    fullcirc.measure_all()

    # print(subcirc1)
    # print(subcirc2)
    # print(fullcirc)

    return fullcirc, subcirc1, subcirc2

''' BEGIN SECTION WITH BENCHMARK CIRCUITS '''

def gen_GHZ_cut(subcirc_size=2):
    subcirc1 = QuantumCircuit(subcirc_size)
    subcirc1.h(0)
    for i in range(1, subcirc_size):
        subcirc1.cnot(i-1, i)

    subcirc2 = QuantumCircuit(subcirc_size)
    for j in range(1, subcirc_size):
        subcirc2.cnot(j-1, j)

    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[i for i in range(subcirc_size-1, subcirc_size*2-1)], inplace=True)
    fullcirc.measure_all()

    return fullcirc, subcirc1, subcirc2

def gen_bit_code_cut(subcirc_size=3):
    subcirc1 = QuantumCircuit(subcirc_size)
    subcirc1.h(0)
    for i in range(1, subcirc_size):
        subcirc1.cnot(i-1, i)

    subcirc2 = QuantumCircuit(subcirc_size)
    for j in range(1, subcirc_size):
        subcirc2.cnot(j-1, j)

    fullcirc = QuantumCircuit(subcirc_size*2 - 1)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[i for i in range(subcirc_size-1, subcirc_size*2-1)], inplace=True)
    fullcirc.measure_all()

    return fullcirc, subcirc1, subcirc2