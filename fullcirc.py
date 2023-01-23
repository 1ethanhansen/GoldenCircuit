import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, transpile


def run_full_circuit():
    # Create circuit
    circ = QuantumCircuit(5)
    circ.h(np.arange(0,5))
    circ.t([2,3,4])
    circ.cz(0,1)
    circ.cz(0,2)
    circ.rx(1.57, 4)
    circ.rx(1.57,[0,1])
    circ.cz(2,4)
    circ.ry(1.57,4)
    circ.cz(2,3)
    circ.t([0,1])
    circ.h(np.arange(0,5))
    circ.measure_all()

    # Transpile for simulator
    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)

    # Run and get counts
    result = simulator.run(circ).result()
    counts = result.get_counts(circ)
    return counts
