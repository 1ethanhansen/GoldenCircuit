import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt



def gamma(beta,ahat,e):
    if beta == 0:
        return 2 * int(ahat == e) - 1
    elif beta == 1:
        return 2 * int(ahat == e) - 1
    elif beta == 2:
        return 2*int(ahat==e)


# run sub-circuits and return two rank-3 tensors
def run_subcirc(subcirc1, subcirc2,shots=10000):
    nA = subcirc1.width()
    nB = subcirc2.width()

    alpha = ['X','Y','Z']
    pA = np.zeros(shape=[2**(nA-1),2,3])
    pB = np.zeros(shape=[2**nB,2,3])

    for x in alpha:
        subcirc1_ = QuantumCircuit(nA).compose(subcirc1)
        beta = 2
        if x == 'X':
            beta = 0
            subcirc1_.h(nA-1)
        elif x == 'Y':
            beta = 1
            subcirc1_.sdg(nA-1)
            subcirc1_.h(nA-1)
        subcirc1_.measure_all()

        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(subcirc1_, simulator)
        result = simulator.run(circ,shots=shots).result()
        counts = result.get_counts(circ)

        for n in range(2**nA,2**(nA+1)):
            # ss = subcirc1.width()
            bstr = bin(n)
            string = bstr[3:len(bstr)]
            ahat = int(bstr[3]) # tensor index
            str_ind = int(bstr[4:len(bstr)][::-1],2) # tensor index

            if string not in counts:
                pA[str_ind, ahat, beta] = 0
            else:
                pA[str_ind, ahat, beta] = counts[string]/shots

    for x in alpha:
        for e in [0, 1]:
            init = QuantumCircuit(nB)
            beta = 2
            if e == 1:
                init.x(0)
            if x == 'X':
                beta = 0
                init.h(0)
            elif x == 'Y':
                beta = 1
                init.h(0)
                init.s(0)
            subcirc2_ = init.compose(subcirc2)
            subcirc2_.measure_all()

            circ = transpile(subcirc2_, simulator)
            result = simulator.run(circ,shots=shots).result()
            counts = result.get_counts(circ)

            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)][::-1]
                str_ind = int(string,2)

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shots
    return pA, pB


''' Given a subcirc1 which only rotates on a given axis,
    measure in the relevant axes and use that data to reconstruct
    the correct distribution
'''
def run_subcirc_known_axis(subcirc1, subcirc2, axis, shots=10000):
    # Get the number of qubits in each subcircuit
    nA = subcirc1.width()
    nB = subcirc2.width()

    # Get the bases we need to measure in
    # For example if we only rotate in the X axis, then
    # we should need to measure in the Y and Z axes
    alpha = ['X','Y','Z']
    alpha.remove(axis)

    pA = np.zeros(shape=[2**(nA-1),2,3])
    pB = np.zeros(shape=[2**nB,2,3])

    for x in alpha:
        subcirc1_ = QuantumCircuit(nA).compose(subcirc1)
        beta = 2
        if x == 'X':
            beta = 0
            subcirc1_.h(nA-1)
        elif x == 'Y':
            beta = 1
            subcirc1_.sdg(nA-1)
            subcirc1_.h(nA-1)
        subcirc1_.measure_all()

        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(subcirc1_, simulator)
        result = simulator.run(circ,shots=shots).result()
        counts = result.get_counts(circ)

        for n in range(2**nA,2**(nA+1)):
            # ss = subcirc1.width()
            bstr = bin(n)
            string = bstr[3:len(bstr)]
            ahat = int(bstr[3]) # tensor index
            str_ind = int(bstr[4:len(bstr)][::-1],2) # tensor index

            if string not in counts:
                pA[str_ind, ahat, beta] = 0
            else:
                pA[str_ind, ahat, beta] = counts[string]/shots

    for x in alpha:
        for e in [0, 1]:
            init = QuantumCircuit(nB)
            beta = 2
            if e == 1:
                init.x(0)
            if x == 'X':
                beta = 0
                init.h(0)
            elif x == 'Y':
                beta = 1
                init.h(0)
                init.s(0)
            subcirc2_ = init.compose(subcirc2)
            subcirc2_.measure_all()

            simulator = Aer.get_backend('aer_simulator')
            circ = transpile(subcirc2_, simulator)
            result = simulator.run(circ,shots=shots).result()
            counts = result.get_counts(circ)

            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)][::-1]
                str_ind = int(string,2)

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shots
    return pA, pB




def reconstruct_bstr(bstr,pA,pB,nA,nB):
    indB = int(bstr[3:3 + nB], 2)
    indA = int(bstr[3 + nB:len(bstr)], 2)

    p = [gamma(beta, ahat, e) * pA[indA, ahat, beta] * pB[indB, e, beta] for beta in [0, 1, 2] for ahat in [0, 1] for e in [0, 1]]
    p = np.sum(np.array(p)) / 2
    return p


def reconstruct_exact(pA,pB,nA,nB):
    p_rec = {}
    for n in range(2 ** (nA + nB - 1), 2 ** (nA + nB)):
        bstr = bin(n)
        string = bstr[3:len(bstr)]
        p = reconstruct_bstr(bstr, pA, pB, nA, nB)
        p_rec[string] = p

    for k in p_rec.keys():
        p_rec[k] = max(0, p_rec[k])
    return p_rec
