import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
import time
from icecream import ic



def gamma(beta,ahat,e):
    if beta == 0:
        return 2 * int(ahat == e) - 1
    elif beta == 1:
        return 2 * int(ahat == e) - 1
    elif beta == 2:
        return 2*int(ahat==e)


# run sub-circuits and return two rank-3 tensors
def run_subcirc(subcirc1, subcirc2, device, shots=10000):
    total_time = 0
    start_time = time.time()

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

        circ = transpile(subcirc1_, device)
        job = device.run(circ,shots=shots)
        ic("pA", job.job_id())
        counts = job.result().get_counts(circ)
        # Get the total time actually spent running the circuit and add to total
        if not device.configuration().simulator:
            total_time += job.result().time_taken

        for n in range(2**nA,2**(nA+1)):
            # ss = subcirc1.width()
            bstr = bin(n)
            string = bstr[3:len(bstr)]
            ahat = int(bstr[3]) # tensor index
            str_ind = int(bstr[4:len(bstr)],2) # tensor index

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

            circ = transpile(subcirc2_, device)
            job = device.run(circ,shots=shots)
            ic("pB", job.job_id())
            counts = job.result().get_counts(circ)
            # Get the total time actually spent running the circuit and add to total
            if not device.configuration().simulator:
                total_time += job.result().time_taken

            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)]
                str_ind = int(string,2)

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shots
    end_time = time.time()

    if device.configuration().simulator:
        total_time = end_time - start_time
    
    return pA, pB, total_time


''' function to create a batched job to send to IBMQ with all upstream and downstream
    subcircuits we want to run
'''
def run_subcirc_known_axis_batched(subcirc1, subcirc2, axis, device, shots=10000):
    total_time = 0
    start_time = time.time()
    # Get the number of qubits in each subcircuit
    nA = subcirc1.width()
    nB = subcirc2.width()

    # Get the bases we need to measure in
    # For example if we only rotate in the X axis, then
    # we should need to measure in the Y and Z axes
    alpha = ['X','Y','Z']
    alpha.remove(axis)

    # create the upstream (pA) subcircs:
    circs = []
    for x in alpha:
        subcirc1_ = subcirc1.copy()
        beta = 2
        if x == 'X':
            beta = 0
            subcirc1_.h(nA-1)
        elif x == 'Y':
            beta = 1
            subcirc1_.sdg(nA-1)
            subcirc1_.h(nA-1)
        subcirc1_.measure_all()

        circ = transpile(subcirc1_, device)
        circs.append(circ)

    # create the downstream (pB) subcircs:
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

            circ = transpile(subcirc2_, device)
            circs.append(circ)

    # Submit all the circuits to be run on IBMQ as one batched job
    job_manager = IBMQJobManager()
    job_set = job_manager.run(circs, backend=device, name='up_down-stream_subcircs')
    ic(job_set.job_set_id())
    results = job_set.results()

    pA = create_pA_from_batched_results(results, nA, alpha, shots)
    pB = create_pB_from_batched_results(results, nB, alpha, shots)

    # TODO: make this actually return the time!
    return pA, pB, 0


def create_pA_from_batched_results(results, nA, alpha, shots):
    # create pA 
    pA = np.zeros(shape=[2**(nA-1),2,3])
    for idx, op in enumerate(alpha):
        beta = 2
        if op == 'X':
            beta = 0
        elif op == 'Y':
            beta = 1
        counts = results.get_counts(idx)
        for n in range(2**nA,2**(nA+1)):
            # ss = subcirc1.width()
            bstr = bin(n)
            string = bstr[3:len(bstr)]
            ahat = int(bstr[3]) # tensor index
            str_ind = int(bstr[4:len(bstr)],2) # tensor index

            if string not in counts:
                pA[str_ind, ahat, beta] = 0
            else:
                pA[str_ind, ahat, beta] = counts[string]/shots

    return pA

def create_pB_from_batched_results(results, nB, alpha, shots):
    # create pB
    pB = np.zeros(shape=[2**nB,2,3])
    for idx, op in enumerate(alpha):
        for e in [0, 1]:
            beta = 2
            if op == 'X':
                beta = 0
            elif op == 'Y':
                beta = 1
            # because counts starts with `len(alpha)` pA entries, go from there
            count_index = len(alpha) + 2 * idx + e
            counts = results.get_counts(count_index)
            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)]
                str_ind = int(string,2)

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shots

    return pB

''' Given a subcirc1 which only rotates on a given axis,
    measure in the relevant axes and use that data to reconstruct
    the correct distribution

    return pA and pB tensors, along with the total time to run the circuits
'''
def run_subcirc_known_axis(subcirc1, subcirc2, axis, device, shots=10000):
    total_time = 0
    start_time = time.time()
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

        circ = transpile(subcirc1_, device)
        job = device.run(circ,shots=shots)
        ic("pA", job.job_id())
        counts = job.result().get_counts(circ)
        # Get the total time actually spent running the circuit and add to total
        if not device.configuration().simulator:
            total_time += job.result().time_taken

        for n in range(2**nA,2**(nA+1)):
            # ss = subcirc1.width()
            bstr = bin(n)
            string = bstr[3:len(bstr)]
            ahat = int(bstr[3]) # tensor index
            str_ind = int(bstr[4:len(bstr)],2) # tensor index

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

            circ = transpile(subcirc2_, device)
            job = device.run(circ,shots=shots)
            ic("pB", job.job_id())
            counts = job.result().get_counts(circ)
            # Get the total time actually spent running the circuit and add to total
            if not device.configuration().simulator:
                total_time += job.result().time_taken

            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)]
                str_ind = int(string,2)

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shots
    end_time = time.time()

    if device.configuration().simulator:
        total_time = end_time - start_time
    
    return pA, pB, total_time




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
