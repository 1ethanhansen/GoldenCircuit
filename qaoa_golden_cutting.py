''' QAOA circuit code adapted from
    https://web.archive.org/web/20221114003819/https://qiskit.org/textbook/ch-applications/qaoa.html
'''

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from reconstruct import *
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from icecream import ic


''' For each axis (x,y,z) create subcircuits and a full circuit
    to compare the reconstruction method with a run of the full
    circuit
'''
def test_one_cut_known_axis(shots=10000):
    # Test only rotating one axis at a time
    axes = ["X", "Y", "Z"]
    # axes = ["Z"]

    for axis in axes:
        # Get the random circuit with a specific axis of rotation, then reconstruct
        circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis, True)
        pA, pB = run_subcirc_known_axis(subcirc1, subcirc2, axis, shots)
        print(f"pA: %s \n pB: %s" % (pA, pB))
        # pA, pB = run_subcirc(subcirc1, subcirc2, shots)
        exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
        # print(exact)

        # Run the full circuit for comparison with reconstruction
        simulator = Aer.get_backend('aer_simulator')
        circ_ = transpile(circ, simulator)
        counts = simulator.run(circ_, shots=shots).result().get_counts()
        # print(counts)

        # Plot both of the results to see visually if they are the same
        plot_histogram(exact, title=f"reconstructed for {axis}")
        plot_histogram(counts, title=f"full circ for {axis}")

    plt.show()


def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.
    
    Args:
        x: str
           solution bitstring
           
        G: networkx graph
        
    Returns:
        obj: float
             Objective
    """

    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1
            
    return obj


def compute_expectation(counts, G):
    """
    Computes expectation value based on measurement results
    
    Args:
        counts: dict
                key as bitstring, val as count
           
        G: networkx graph
        
    Returns:
        avg: float
             expectation value
    """
    
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        
        obj = maxcut_obj(bitstring[::-1], G)
        avg += obj * count
        sum_count += count
        
    return avg/sum_count


''' Create a QAOA circuit out of a balanced vertex separator graph
    so we can cut it at a golden cutting point early in the circuit.
'''
def create_qaoa_circ(G, theta):
    """
    Creates a parametrized qaoa circuit
    
    Args:  
        G: networkx graph
        theta: list
               unitary parameters
                     
    Returns:
        qc: qiskit circuit
    """
    
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    
    beta = theta[:p]
    gamma = theta[p:]
    
    # initial_state
    for i in range(1, nqubits):
        qc.h(i)
    
    for irep in range(0, p):
        
        # problem unitary
        for pair in list(G.edges()):
            qc.rzz(2 * gamma[irep], pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
            
    qc.measure_all()
        
    return qc

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(G, shots=512):
    """
    Runs parametrized circuit
    
    Args:
        G: networkx graph
        p: int,
           Number of repetitions of unitaries
    """
    
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots
    
    def execute_circ(theta):
        
        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, seed_simulator=10, 
                             nshots=shots).result().get_counts()
        
        return compute_expectation(counts, G)
    
    return execute_circ

G = nx.Graph()
G.add_nodes_from([*range(0,5)])
G.add_edges_from([(0,2), (1, 2), (3, 2), (4, 2)])
# G.add_nodes_from([0, 1, 2, 3])
# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
nx.draw(G, with_labels=True, alpha=0.8, node_size=500)
plt.savefig("balanced_graph.png")


# optimization
expectation = get_expectation(G, shots=2048)

res = minimize(expectation, 
                      [1.2, 1.5], 
                      method='COBYLA')
print(res)

# analyze results
backend = Aer.get_backend('aer_simulator')
backend.shots = 512

qc_res = create_qaoa_circ(G, res.x)

print(qc_res.decompose().decompose().decompose())

counts = backend.run(qc_res).result().get_counts()

ic(counts)

plot_histogram(counts)
plt.savefig("results.png")