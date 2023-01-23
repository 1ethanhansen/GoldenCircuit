import matplotlib.pyplot as plt

from reconstruct import *
from qiskit.circuit.random import random_circuit
from qiskit.visualization import plot_histogram
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from main import *
import time
import random

from icecream import ic


def gen_random_circuit(nA,nB):
    A = random_circuit(nA, nA)
    B = random_circuit(nA, nB)
    C = random_circuit(nB-1,nA)
    D = random_circuit(nB,nB)
    E = random_circuit(1, nB)

    circ = QuantumCircuit(nA+nB).compose(A)
    circ.compose(E,qubits=[nA],inplace=True)
    circ.compose(C,qubits=list(range(nA+1,nA+nB)),inplace=True)
    circ.cnot(nA-1,nA)
    circ.barrier()
    circ.compose(B,qubits=list(range(0,nA)),inplace=True)
    circ.compose(D,qubits=list(range(nA,nA+nB)),inplace=True)
    circ.measure_all()

    subcirc1 = QuantumCircuit(nA+1).compose(A)
    subcirc1.compose(E,qubits=[nA],inplace=True)
    subcirc1.cnot(nA - 1, nA)
    subcirc1.compose(B, qubits=list(range(0, nA)), inplace=True)

    subcirc2 = QuantumCircuit(nB).compose(C,qubits=list(range(1,nB)))
    subcirc2.compose(D,inplace=True)

    print('full circ:\n', circ)
    
    print('subcirc1:\n', subcirc1)
    
    print('subcirc2:\n', subcirc2)

    return circ, subcirc1, subcirc2


''' Create a random circuit where only 1 axis of the bloch
    sphere on the first qubit is rotated so only 2 measurements
    must be done later

    This is hard-coded for a 3-qubit circuit split into two
    2-qubit circuits. Modifications can be done later for more
    general cases if needed.
'''
def gen_random_circuit_specific_rotation(axis, entangled=True):
    # First half of our circuit (qubits 0 and 1)
    subcirc1 = QuantumCircuit(2)
    # Get the random value to rotate
    theta = random.uniform(0, 6.28)

    # Rotate just along the axis we want
    # if axis == "X":
    #     subcirc1.rx(theta, [0])
    # elif axis == "Y":
    #     subcirc1.ry(theta, [0])
    # elif axis == "Z":
    #     subcirc1.h(0)
    #     subcirc1.rz(theta, [0])

    subcirc1.h(0)
    subcirc1.h(1)
        
    # Optionally add entangling gate between qubits 0 and 1
    # before the cut
    if entangled and axis == "Z":
        subcirc1.cz(0, 1)
    elif entangled and axis != "Z":
        subcirc1.cnot(0, 1)

    # subcirc1.rzz(3.14/2, 0, 1) 

    # Create a random second half of the circuit
    subcirc2 = random_circuit(2, 2)

    # create the full circuit
    fullcirc = QuantumCircuit(3)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[1,2], inplace=True)
    fullcirc.measure_all()

    print(subcirc1)
    print(subcirc2)
    print(fullcirc)

    return fullcirc, subcirc1, subcirc2


''' Create a random circuit where there are no gates before the
    cutting point. After the cutting point will be random gates.

    This is hard-coded for a 3-qubit circuit split into two
    2-qubit circuits. Modifications can be done later for more
    general cases if needed.
'''
def gen_random_circuit_no_gates():
    # First half of our circuit (qubits 0 and 1)
    subcirc1 = QuantumCircuit(2)

    # Create a random second half of the circuit
    subcirc2 = random_circuit(2, 2)

    # create the full circuit
    fullcirc = QuantumCircuit(3)
    fullcirc.compose(subcirc1, inplace=True)
    fullcirc.compose(subcirc2, qubits=[1,2], inplace=True)
    fullcirc.measure_all()

    print(subcirc1)
    print(subcirc2)
    print(fullcirc)

    return fullcirc, subcirc1, subcirc2


def gen_random_circuit_two(nA,nB, nC):
    A = random_circuit(nA,nA)
    B = random_circuit(nA,nB)
    C = random_circuit(1,nA)
    D = random_circuit(nB-1,nA)
    E = random_circuit(nB,nB)
    F = random_circuit(1,nB)
    G = random_circuit(nC-1,nB)
    H = random_circuit(nC,nC)
    I = random_circuit(nB,nC)

    circ = QuantumCircuit(nA+nB+nC).compose(A)
    circ.compose(C,qubits=[nA],inplace=True)
    circ.cnot(nA-1,nA)
    circ.compose(B,qubits=list(range(0,nA)),inplace=True)
    circ.compose(D,qubits=list(range(nA+1,nA+nB)),inplace=True)
    circ.compose(E,qubits=list(range(nA,nA+nB)),inplace=True)
    circ.compose(F,qubits=[nB],inplace=True)
    circ.cnot(nB-1,nB)
    circ.compose(I,qubits=list(range(nA, nA+nB)), inplace=True)
    circ.compose(G,qubits=list(range(nA+nB+1,nA+nB+nC)),inplace=True)
    circ.compose(H,qubits=list(range(nA+nB,nA+nB+nC)),inplace=True)
    circ.measure_all()

    subcirc1 = QuantumCircuit(nA+1).compose(A)
    subcirc1.compose(C,qubits=[nA],inplace=True)
    subcirc1.cnot(nA - 1, nA)
    subcirc1.compose(B, qubits=list(range(0, nA)), inplace=True)

    subcirc2 = QuantumCircuit(nB+1).compose(D,qubits=list(range(1,nB)))
    subcirc2.compose(E,inplace=True)
    subcirc2.compose(F,qubits=[nB], inplace=True)
    subcirc2.cnot(nB-1,nB)
    subcirc2.compose(I,qubits=list(range(0,nB)), inplace=True)

    subcirc3 = QuantumCircuit(nC).compose(G, qubits=list(range(1, nC)))
    subcirc3.compose(H, inplace=True)

    # print('full circ:', circ)
    #
    # print('subcirc1:', subcirc1)
    #
    # print('subcirc2:', subcirc2)
    #
    # print('subcirc3:', subcirc3)

    return circ, subcirc1, subcirc2, subcirc3


def totalVariationalDistance(p1, p2):
    # make sure  distributions are normalized
    sum = np.sum([p1[k] for k in p1.keys()])
    for k in p1.keys():
        p1[k] = p1[k]/sum

    sum = np.sum([p2[k] for k in p2.keys()])
    for k in p2.keys():
        p2[k] = p2[k] / sum

    sup = -10
    for k in p1.keys():
        if k in p2:
            sup = max(sup, abs(p1[k]-p2[k]))
        else:
            sup = max(sup, p1[k])

    for k in p2.keys():
        if k not in p1:
            sup = max(sup, p2[k])

    return sup


def averageVariationalDistance(p1,p2):
    # make sure  distributions are normalized
    sum = np.sum([p1[k] for k in p1.keys()])
    for k in p1.keys():
        p1[k] = p1[k] / sum

    sum = np.sum([p2[k] for k in p2.keys()])
    for k in p2.keys():
        p2[k] = p2[k] / sum

    dis = 0
    for k in p1.keys():
        if k in p2.keys():
            dis += abs(p1[k]-p2[k])*(p1[k]+p2[k])/2
        else:
            dis += p1[k]*p1[k]/2

    for k in p2.keys():
        if k not in p1.keys():
            dis += p2[k]*p2[k]/2

    return dis

def testOneCut(n,s):
    circ,subcirc1,subcirc2 = gen_random_circuit(n,n)
    pA, pB = run_subcirc(subcirc1, subcirc2)

    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(circ, simulator)
    result = simulator.run(circ,shots=10000).result()
    counts = result.get_counts(circ)

    # print(counts)

    t = time.time()
    exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
    exactTime = time.time() - t

    t = time.time()
    approx = reconstruct_approximate(pA,pB,subcirc1.width(),subcirc2.width(),s)
    approxTime = time.time() - t

    EvsA = averageVariationalDistance(exact, approx)
    EvsC = averageVariationalDistance(exact, counts)
    AvsC = averageVariationalDistance(approx, counts)

    return EvsA, EvsC, AvsC, exactTime, approxTime


''' For each axis (x,y,z) create subcircuits and a full circuit
    to compare the reconstruction method with a run of the full
    circuit
'''
def test_one_cut_known_axis(shots=10000):
    # Test only rotating one axis at a time
    # axes = ["X", "Y", "Z"]
    axes = ["Y"]

    for axis in axes:
        # Get the random circuit with a specific axis of rotation, then reconstruct
        circ,subcirc1,subcirc2 = gen_random_circuit_specific_rotation(axis, True)
        pA, pB = run_subcirc_known_axis(subcirc1, subcirc2, axis, shots)
        # pA, pB = run_subcirc(subcirc1, subcirc2, shots)
        print(f"pA: %s \n pB: %s" % (pA, pB))
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
    
''' Simple test function to attempt cutting after no gates - 
    that is, right at the start of the circuit
'''
def test_one_cut_no_gates(shots=10000):
    # Get the random circuit with a specific axis of rotation, then reconstruct
    circ,subcirc1,subcirc2 = gen_random_circuit_no_gates()
    # pA, pB = run_subcirc_known_axis(subcirc1, subcirc2, axis, shots)
    pA, pB = run_subcirc(subcirc1, subcirc2, shots)
    print(f"pA: %s \n pB: %s" % (pA, pB))
    exact = reconstruct_exact(pA,pB,subcirc1.width(),subcirc2.width())
    # print(exact)

    # Run the full circuit for comparison with reconstruction
    simulator = Aer.get_backend('aer_simulator')
    circ_ = transpile(circ, simulator)
    counts = simulator.run(circ_, shots=shots).result().get_counts()
    # print(counts)

    # Plot both of the results to see visually if they are the same
    plot_histogram(exact, title="reconstructed no gates")
    plot_histogram(counts, title="full no gates")

    plt.show()

def testOneCutMultiple(testRange, trial):
    EvsA = np.zeros(shape=[len(testRange),trial])
    EvsC = np.zeros(shape=[len(testRange),trial])
    AvsC = np.zeros(shape=[len(testRange),trial])

    exactTime = np.zeros(shape=[len(testRange),trial])
    approxTime = np.zeros(shape=[len(testRange),trial])

    for t in range(len(testRange)):
        for i in range(trial):
            EvsA[t,i], EvsC[t,i], AvsC[t,i], exactTime[t,i], approxTime[t,i] = testOneCut(testRange[t],testRange[t]*3000)

    combine = np.zeros(shape=[5,len(testRange),trial])
    combine[0, :, :] = EvsA
    combine[1, :, :] = EvsC
    combine[2, :, :] = AvsC
    combine[3, :, :] = exactTime
    combine[4, :, :] = approxTime
    np.save('OneCut_'+str(min(testRange))+'_'+str(max(testRange))+'.npy', combine)

    return EvsA, EvsC, AvsC, exactTime, approxTime


def testTwoCut(n,s):
    circ, subcirc1, subcirc2, subcirc3 = gen_random_circuit_two(n, n, n)
    pA, pB, pC = run_three_subcirc(subcirc1, subcirc2, subcirc3)



    t = time.time()
    exact = reconstruct_exact_seq(pA, pB, pC, subcirc1.width(), subcirc2.width(), subcirc3.width())
    exactTime = time.time() - t

    t = time.time()
    approx = reconstruct_approximate_two(pA, pB, pC, subcirc1.width(), subcirc2.width(), subcirc3.width(), s)
    approxTime = time.time() - t

    t = time.time()
    rand = reconstruct_approximate_two_randomized(pA, pB, pC, subcirc1.width(), subcirc2.width(), subcirc3.width(), s)
    randTime = time.time() - t

    EvsA = averageVariationalDistance(exact, approx)
    EvsR = averageVariationalDistance(exact, rand)

    return EvsA, EvsR, exactTime, approxTime, randTime


def testTwoCutMultiple(testRange, trial):
    EvsA = np.zeros(shape=[len(testRange), trial])
    EvsR = np.zeros(shape=[len(testRange), trial])

    exactTime = np.zeros(shape=[len(testRange), trial])
    approxTime = np.zeros(shape=[len(testRange), trial])
    randTime = np.zeros(shape=[len(testRange), trial])

    for t in range(len(testRange)):
        for i in range(trial):
            EvsA[t, i], EvsR[t, i], exactTime[t, i], approxTime[t, i], randTime[t, i] = testTwoCut(testRange[t],
                                                                                               testRange[t] * 3000)

    combine = np.zeros(shape=[5, len(testRange), trial])
    combine[0, :, :] = EvsA
    combine[1, :, :] = EvsR
    combine[2, :, :] = exactTime
    combine[3, :, :] = approxTime
    combine[4, :, :] = randTime
    np.save('TwoCut_' + str(min(testRange)) + '_' + str(max(testRange)) + '.npy', combine)

    return EvsA, EvsR, exactTime, approxTime, randTime


def testOneCutSample(sampleRange, trial, n):
    EvsA = np.zeros(shape=[len(sampleRange), trial])
    EvsC = np.zeros(shape=[len(sampleRange), trial])
    AvsC = np.zeros(shape=[len(sampleRange), trial])

    exactTime = np.zeros(shape=[len(sampleRange), trial])
    approxTime = np.zeros(shape=[len(sampleRange), trial])

    for t in range(len(sampleRange)):
        for i in range(trial):
            EvsA[t, i], EvsC[t, i], AvsC[t, i], exactTime[t, i], approxTime[t, i] = testOneCut(n, sampleRange[t])

    combine = np.zeros(shape=[5, len(sampleRange), trial])
    combine[0, :, :] = EvsA
    combine[1, :, :] = EvsC
    combine[2, :, :] = AvsC
    combine[3, :, :] = exactTime
    combine[4, :, :] = approxTime
    np.save('OneCutSam_' + str(min(sampleRange)) + '_' + str(max(sampleRange)) + '.npy', combine)

    return EvsA, EvsC, AvsC, exactTime, approxTime


# gen_random_circuit_specific_rotation("X")
# test_one_cut_known_axis()
# testOneCut(2,2)
test_one_cut_no_gates()

# # testRange = range(3,8)
# testRange = range(5,11)
# # print(testTwoCutMultiple(testRange,30))
# # print(testOneCutMultiple(testRange, 30))

# sampleRange = range(3000,18001,3000)
# # print(testOneCutSample(sampleRange, 30, 10))



# # combined = np.load('TwoCut_3_7.npy')
# combined = np.load('OneCut_5_10k(av).npy')
# # testRange = sampleRange
# ea = combined[0,:,:]
# ec = combined[1,:,:]
# ac = combined[2,:,:]
# et = combined[3,:,:]
# at = combined[4,:,:]

# # testRange = range(10,21,2)

# # plt.plot(testRange, np.median(ea,axis=1),label='Exact vs. Approximate')
# # plt.fill_between(testRange, np.quantile(ea,0.75,axis=1), np.quantile(ea,0.25,axis=1), alpha=0.3)
# # plt.plot(testRange, np.median(ec,axis=1),label='Exact vs. Full')
# # plt.fill_between(testRange, np.quantile(ec,0.75,axis=1), np.quantile(ec,0.25,axis=1), alpha=0.3)
# # plt.plot(testRange, np.median(ac,axis=1),label='Approximate vs. Full')
# # plt.fill_between(testRange, np.quantile(ac,0.75,axis=1), np.quantile(ac,0.25,axis=1), alpha=0.3)


# # plt.errorbar([x-0.04 for x in testRange], np.median(ea,axis=1),yerr=[ np.median(ea,axis=1)-np.quantile(ea,0.25,axis=1),np.quantile(ea,0.75,axis=1)- np.median(ea,axis=1)],label='Exact vs. Approximate',marker='o',markersize='10')
# plt.errorbar([x-0.02 for x in testRange], np.median(ec,axis=1),yerr=[ np.median(ec,axis=1)-np.quantile(ec,0.25,axis=1),np.quantile(ec,0.75,axis=1)- np.median(ec,axis=1)],label='Exact vs. Full',marker='o',markersize='10')
# plt.errorbar([x+0.02 for x in testRange], np.median(ac,axis=1),yerr=[ np.median(ac,axis=1)-np.quantile(ac,0.25,axis=1),np.quantile(ac,0.75,axis=1)- np.median(ac,axis=1)],label='Approximate vs. Full',marker='^',markersize='10')
# plt.yscale('log')
# plt.xlabel('Size of Subcircuits',fontsize=14)
# plt.ylabel('Average Distance',fontsize=14)
# plt.legend(fontsize=12)
# plt.grid()
# plt.show()

# # plt.plot(testRange, np.median(et,axis=1),label='Time for Exact',marker='o',markersize='10')
# # plt.fill_between(testRange, np.quantile(et,0.75,axis=1), np.quantile(et,0.25,axis=1), alpha=0.3)
# # plt.plot(testRange, np.median(at,axis=1),label='Time for Approximate',marker='^',markersize='10')
# # plt.fill_between(testRange, np.quantile(at,0.75,axis=1), np.quantile(at,0.25,axis=1), alpha=0.3)
# # plt.xlabel('Size of Subcircuits',fontsize=14)
# # plt.ylabel('Time (sec)',fontsize=14)
# # plt.legend(fontsize=12)
# # plt.grid()
# # plt.show()


# # two cut figure

# # plt.plot(testRange, np.median(ea,axis=1),label='Exact vs. Approximate')
# # plt.fill_between(testRange, np.quantile(ea,0.75,axis=1), np.quantile(ea,0.25,axis=1), alpha=0.3)
# # plt.plot(testRange, np.median(ec,axis=1),label='Exact vs. Approximate (randomized)')
# # plt.fill_between(testRange, np.quantile(ec,0.75,axis=1), np.quantile(ec,0.25,axis=1), alpha=0.3)

# # plt.errorbar([x-0.02 for x in testRange], np.median(ea,axis=1),yerr=[ np.median(ea,axis=1)-np.quantile(ea,0.25,axis=1),np.quantile(ea,0.75,axis=1)- np.median(ea,axis=1)],label='Exact vs. Approximate',marker='o',markersize='10')
# # plt.errorbar([x+0.02 for x in testRange], np.median(ec,axis=1),yerr=[ np.median(ec,axis=1)-np.quantile(ec,0.25,axis=1),np.quantile(ec,0.75,axis=1)- np.median(ec,axis=1)],label='Exact vs. Approximate (randomized)',marker='^',markersize='10')
# # plt.ylim([1e-4,1])
# # plt.yscale('log')
# # plt.xlabel('Size of Subcircuits',fontsize=14)
# # plt.ylabel('Average Distance',fontsize=14)
# # plt.legend(fontsize=12)
# # plt.grid()
# # plt.show()


# # plt.plot(testRange, np.median(ac,axis=1),label='Time for Exact',marker='o',markersize='10')
# # plt.fill_between(testRange, np.quantile(ac,0.75,axis=1), np.quantile(ac,0.25,axis=1), alpha=0.3)
# # plt.plot(testRange, np.median(et,axis=1),label='Time for Approximate',marker='^',markersize='10')
# # plt.fill_between(testRange, np.quantile(et,0.75,axis=1), np.quantile(et,0.25,axis=1), alpha=0.3)
# # plt.plot(testRange, np.median(at,axis=1),label='Time for Approximate (randomized)',marker='*',markersize='10')
# # plt.fill_between(testRange, np.quantile(at,0.75,axis=1), np.quantile(at,0.25,axis=1), alpha=0.3)
# # plt.legend(fontsize=12)
# # plt.xlabel('Size of Subcircuits',fontsize=14)
# # plt.ylabel('Time (sec)',fontsize=14)
# # plt.grid()
# # plt.show()





# #
# #
# #
# #
# # print(totatVariationalDistance(exact, approx))
# # print(totatVariationalDistance(exact, counts))
# # print(totatVariationalDistance(approx, counts))
# # print(averageVariationalDistance(exact, approx))
# # print(averageVariationalDistance(exact, counts))
# # print(averageVariationalDistance(approx, counts))




# # subcirc1 = create_subcirc1()
# # subcirc2 = create_subcirc2(1)
# # subcirc3 = create_subcirc3()
# # nA = subcirc1.width()
# # nB = subcirc2.width()
# # nC = subcirc3.width()
# # pA, pB, pC = run_three_subcirc(subcirc1, subcirc2, subcirc3)
# # exact = reconstruct_exact_seq(pA,pB,pC,nA,nB,nC)
# # approx = reconstruct_approximate_two(pA,pB,pC,nA,nB,nC,8000)
# #
# # circ = QuantumCircuit(7)
# # circ.h(np.arange(0,7))
# # circ.t([2,3,4])
# # circ.cz(0,1)
# # circ.cz(0,2)
# # circ.cz(5,6)
# # circ.rx(1.57, 4)
# # circ.rx(1.57,[0,1])
# # circ.rx(1.57,5)
# # circ.cz(2,4)
# # circ.cz(2,3)
# # circ.ry(1.57,4)
# # circ.h(4)
# # circ.ry(1.57,6)
# # circ.cz(4,5)
# # circ.cz(5,6)
# # circ.t([0,1])
# # circ.h(np.arange(0,7))
# # circ.measure_all()
# # simulator = Aer.get_backend('aer_simulator')
# # circ = transpile(circ, simulator)

# # Run and get counts
# # result = simulator.run(circ,shots=100000).result()
# # counts = result.get_counts(circ)
# #
# # print(totatVariationalDistance(exact, approx))
# # print(totatVariationalDistance(exact, counts))
# # print(totatVariationalDistance(approx, counts))
# # print(averageVariationalDistance(exact, approx))
# # print(averageVariationalDistance(exact, counts))
# # print(averageVariationalDistance(approx, counts))

