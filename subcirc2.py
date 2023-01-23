import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile


def run_subcirc2(init):
    # create circuit
    circ = QuantumCircuit(3)
    # set up different initializations
    if init == '1':
        circ.x(0)
    if init == '+':
        circ.h(0)
    if init == '+i':
        circ.h(0)
        circ.s(0)

    circ.h([1,2])
    circ.t([1,2])
    circ.rx(1.57,2)
    circ.cz(0,2)
    circ.ry(1.57,2)
    circ.cz(0,1)
    circ.h([0,1,2])
    circ.measure_all()

    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)

    # Run and get counts
    result = simulator.run(circ).result()
    counts = result.get_counts(circ)
    # normalize count to get probability
    for keys in counts:
        counts[keys] = counts[keys]/1024
    return counts
