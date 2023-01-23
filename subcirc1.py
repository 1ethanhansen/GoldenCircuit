import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile


def run_subcirc1(meas):
    # create subcircuit
    circ = QuantumCircuit(3)
    circ.h(np.arange(0,3))
    circ.t(2)
    circ.cz(0,1)
    circ.cz(0,2)
    circ.rx(1.57,[0,1])
    circ.t([0,1])
    circ.h([0,1])

    # set up different measurements
    if meas == 'Y':
        circ.sdg(2)
        circ.h(2)
    if meas == 'X':
        circ.h(2)

    circ.measure_all()

    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)

    # Run and get counts
    result = simulator.run(circ).result()
    counts = result.get_counts(circ)
    # normalize result to get probability
    for keys in counts:
        counts[keys] = counts[keys]/1024
    return counts
