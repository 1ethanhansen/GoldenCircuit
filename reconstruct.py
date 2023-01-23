import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt


def create_subcirc1():
    subcirc1 = QuantumCircuit(3)
    subcirc1.h(np.arange(0,3))
    subcirc1.t(2)
    subcirc1.cz(0,1)
    subcirc1.cz(0,2)
    subcirc1.rx(1.57,[0,1])
    subcirc1.t([0,1])
    subcirc1.h([0,1])


    # subcirc1.h([0,1,2])

    # subcirc1 = QuantumCircuit(3)
    # subcirc1.h([0,1])
    # subcirc1.cnot(1,2)

    # subcirc1 = QuantumCircuit(4)
    # subcirc1.h([0,1,3])
    # subcirc1.cnot(1,2)
    return subcirc1


def create_subcirc2(k):
    if k == 1:
        subcirc2 = QuantumCircuit(3)
        subcirc2.h([1,2])
        subcirc2.t([1,2])
        subcirc2.rx(1.57,2)
        subcirc2.cz(0,2)
        subcirc2.cz(0,1)
        subcirc2.ry(1.57,2)
        subcirc2.h([0,1,2])

        # subcirc2.h([1,2])

    #     subcirc2.h([1,2])
    #     subcirc2.cnot(1,0)
    #     subcirc2.cnot(2,1)
    # if k == 2:
    #     subcirc2 = QuantumCircuit(2)
    #     subcirc2.h([1])
    #     subcirc2.cnot(1,0)
    return subcirc2


def create_subcirc3():
    # subcirc3 = QuantumCircuit(2)
    # subcirc3.h([1])
    # subcirc3.cnot(1,0)

    subcirc3 = QuantumCircuit(3)
    subcirc3.h([1,2])
    subcirc3.cz(1,2)
    subcirc3.rx(1.57,1)
    subcirc3.ry(1.57,2)
    subcirc3.cz(0,1)
    subcirc3.cz(1,2)
    subcirc3.h([0,1,2])

    # subcirc3.h([1,2])

    return subcirc3


def gamma(beta,ahat,e):
    if beta == 0:
        return 2 * int(ahat == e) - 1
    elif beta == 1:
        return 2 * int(ahat == e) - 1
    elif beta == 2:
        return 2*int(ahat==e)


# run sub-circuits and return two rank-3 tensors
def run_subcirc(subcirc1, subcirc2,shot=10000):
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
        result = simulator.run(circ,shots=shot).result()
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
                pA[str_ind, ahat, beta] = counts[string]/shot

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
            result = simulator.run(circ,shot=shot).result()
            counts = result.get_counts(circ)

            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)][::-1]
                str_ind = int(string,2)

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shot
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

    print(alpha)

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
            print(counts)

            for n in range(2**nB,2**(nB+1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)][::-1]
                str_ind = int(string,2)
                print(string)
                print(str_ind)
                print()

                if string not in counts:
                    pB[str_ind, e, beta] = 0
                else:
                    pB[str_ind, e, beta] = counts[string]/shots
    return pA, pB

def run_three_subcirc(subcirc1, subcirc2, subcirc3):
    nA = subcirc1.width()
    nB = subcirc2.width()
    nC = subcirc3.width()

    alpha = ['X', 'Y', 'Z']
    pA = np.zeros(shape=[2 ** (nA - 1), 2, 3])
    pB = np.zeros(shape=[2 ** (nB - 1), 2, 3, 2, 3])
    pC = np.zeros(shape=[2 ** nC, 2, 3])

    for x in alpha:
        subcirc1_ = QuantumCircuit(nA).compose(subcirc1)


        beta1 = 2
        if x == 'X':
            beta1 = 0
            subcirc1_.h(nA - 1)
        elif x == 'Y':
            beta1 = 1
            subcirc1_.sdg(nA - 1)
            subcirc1_.h(nA - 1)
        subcirc1_.measure_all()

        simulator = Aer.get_backend('aer_simulator')
        circ = transpile(subcirc1_, simulator)
        result = simulator.run(circ).result()
        counts = result.get_counts(circ)

        for n in range(2 ** nA, 2 ** (nA + 1)):
            # ss = subcirc1.width()
            bstr = bin(n)
            string = bstr[3:len(bstr)]
            ahat1 = int(bstr[3])  # tensor index
            str_ind = int(bstr[4:len(bstr)][::-1], 2)  # tensor index

            if string not in counts:
                pA[str_ind, ahat1, beta1] = 0
            else:
                pA[str_ind, ahat1, beta1] = counts[string] / 1024

    for x in alpha: # beta 1
        for e1 in [0, 1]: # e
            for y in alpha: # beta 2
                init = QuantumCircuit(nB)
                beta1 = 2
                if e1 == 1:
                    init.x(0)
                if x == 'X':
                    beta1 = 0
                    init.h(0)
                elif x == 'Y':
                    beta1 = 1
                    init.h(0)
                    init.s(0)
                subcirc2_ = init + subcirc2

                beta2 = 2
                if y == 'X':
                    beta2 = 0
                    subcirc2_.h(nB - 1)
                elif y == 'Y':
                    beta2 = 1
                    subcirc2_.sdg(nB - 1)
                    subcirc2_.h(nB - 1)
                subcirc2_.measure_all()

                simulator = Aer.get_backend('aer_simulator')
                circ = transpile(subcirc2_, simulator)
                result = simulator.run(circ).result()
                counts = result.get_counts(circ)

                for n in range(2 ** nB, 2 ** (nB + 1)):
                    bstr = bin(n)

                    string = bstr[3:len(bstr)][::-1]
                    ahat2 = int(bstr[3])  # tensor index
                    str_ind = int(bstr[4:len(bstr)][::-1], 2)  # tensor index

                    if string not in counts:
                        pB[str_ind, e1, beta1, ahat2, beta2] = 0
                    else:
                        pB[str_ind, e1, beta1, ahat2, beta2] = counts[string] / 1024

    for y in alpha:
        for e2 in [0, 1]:
            init = QuantumCircuit(nC)
            beta2 = 2
            if e2 == 1:
                init.x(0)
            if y == 'X':
                beta2 = 0
                init.h(0)
            elif y == 'Y':
                beta2 = 1
                init.h(0)
                init.s(0)
            subcirc3_ = init + subcirc3
            subcirc3_.measure_all()

            simulator = Aer.get_backend('aer_simulator')
            circ = transpile(subcirc3_, simulator)
            result = simulator.run(circ).result()
            counts = result.get_counts(circ)

            for n in range(2 ** nC, 2 ** (nC + 1)):
                bstr = bin(n)
                string = bstr[3:len(bstr)][::-1]
                str_ind = int(string, 2)

                if string not in counts:
                    pC[str_ind, e2, beta2] = 0
                else:
                    pC[str_ind, e2, beta2] = counts[string] / 1024
    return pA, pB, pC


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


def reconstruct_exact_seq(pA,pB,pC,nA,nB,nC):
    pint = np.zeros(shape=[2**(nB+nC-1),2,3])

    for e1 in [0,1]:
        for beta1 in [0,1,2]:
            for n in range(2**(nB+nC-1),2**(nB+nC)):
                bstr = bin(n)
                string = bstr[3:len(bstr)]
                ind = int(string,2)

                strC = bstr[3:3+nC]
                strB = bstr[3 + nC:3 + nC + nB - 1]

                indC = int(strC, 2)
                indB = int(strB, 2)

                p = [gamma(beta2,ahat2,e2) * pB[indB,e1,beta1,ahat2,beta2] * pC[indC,e2,beta2]
                     for beta2 in [0,1,2] for ahat2 in [0,1] for e2 in [0,1]]
                p = np.sum(np.array(p))/2
                pint[ind,e1,beta1] = p

    return reconstruct_exact(pA,pint,nA,nB+nC-1)



def reconstruct_approximate(pA,pB,nA,nB,num):
    burn = int(0.15*num)
    p_rec = {}

    x = bin(np.random.randint(2 ** (nA + nB - 1), 2 ** (nA + nB)))
    x = x[3:len(x)]
    for i in range(num+burn):
        n = np.random.randint(0,len(x))
        x_ = x
        if n == len(x)-1:
            x_ = x[0:n] + str((int(x_[n])+1)%2)
        else:
            x_ = x[0:n] + str((int(x_[n])+1)%2) + x[n+1:len(x_)]

        r = min(1,(reconstruct_bstr('xxx'+x_,pA,pB,nA,nB))/(reconstruct_bstr('xxx'+x,pA,pB,nA,nB)))
        # r = min(1,np.exp(((reconstruct_bstr('xxx'+x_,pA,pB,nA,nB)) - (reconstruct_bstr('xxx'+x,pA,pB,nA,nB)))*50))
        # if np.random.rand() < r:
        #     x = x_
        #     if i > burn:
        #         if x not in p_rec:
        #             p_rec[x] = 1
        #         else:
        #             p_rec[x] += 1
        # else:
        #     i -= 1

        if np.random.rand() < r:
            x = x_
        if i > burn:
            if x not in p_rec:
                p_rec[x] = 1
            else:
                p_rec[x] += 1

    return p_rec


def gen_string(length):
    x = bin(np.random.randint(2 ** (length), 2 ** (length+1)))
    x = x[3:len(x)]
    return x


def flip_string(x):
    n = np.random.randint(0, len(x))
    x_ = x
    if n == len(x) - 1:
        x_ = x[0:n] + str((int(x_[n]) + 1) % 2)
    else:
        x_ = x[0:n] + str((int(x_[n]) + 1) % 2) + x[n + 1:len(x_)]

    return x_


def reconstruct_approximate_two(pA,pB,pC,nA,nB,nC,num):
    burn = int(0.15*num)

    pint = np.zeros(shape=[2 ** (nB + nC - 1), 2, 3])
    xlist = []
    [xlist.append(gen_string(nB+nC-1)) for i in range(6)]
    count = [0, 0, 0, 0, 0, 0]
    p = 0
    while min(count) < num:
        c = 0
        for beta1 in [0,1,2]:
            for e1 in [0,1]:
                if count[c] < 2*num:
                    x = xlist.pop(0)
                    y = flip_string(x)

                    indC = int(x[0:nC], 2)
                    indB = int(x[nC:nC+nB-1], 2)
                    px = [gamma(beta2, ahat2, e2) * pB[indB, e1, beta1, ahat2, beta2] * pC[indC, e2, beta2]
                         for beta2 in [0, 1, 2] for ahat2 in [0, 1] for e2 in [0, 1]]
                    px = np.sum(np.array(px)) / 2

                    indy = int(y, 2)
                    indC = int(y[0:nC], 2)
                    indB = int(y[nC:nC+nB-1], 2)
                    py = [gamma(beta2, ahat2, e2) * pB[indB, e1, beta1, ahat2, beta2] * pC[indC, e2, beta2]
                          for beta2 in [0, 1, 2] for ahat2 in [0, 1] for e2 in [0, 1]]
                    py = np.sum(np.array(py)) / 2

                    r = min(1, (py)/(px))
                    if np.random.rand() < r:
                        xlist.append(y)
                    else:
                        xlist.append(x)
                    count[c] += 1
                    if count[c] > burn:
                        pint[indy, e1, beta1] += 1/(num-burn)

                c += 1


    p_rec = {}
    x = gen_string(nA+nB+nC-2)
    count = 0
    while count + burn < num:
        y = flip_string(x)

        indBC = int(x[0:nB + nC - 1], 2)
        indA = int(x[nB + nC - 1:len(y)], 2)
        px = [gamma(beta1, ahat1, e1) * pA[indA, ahat1, beta1] * pint[indBC, e1, beta1]
              for beta1 in [0, 1, 2] for ahat1 in [0, 1] for e1 in [0, 1]]
        px = np.sum(np.array(px)) / 2

        indBC = int(y[0:nB+nC-1], 2)
        indA = int(y[nB+nC-1:len(y)], 2)
        py = [gamma(beta1,ahat1,e1) * pA[indA,ahat1,beta1] * pint[indBC,e1,beta1]
             for beta1 in [0,1,2] for ahat1 in [0,1] for e1 in [0,1]]
        py = np.sum(np.array(py))/2

        r = min(1, (py)/(px))
        if np.random.rand() < r:
            x = y
        count += 1
        if count > burn:
            if x not in p_rec:
                p_rec[x] = 0
            else:
                p_rec[x] += 1
    return p_rec


def reconstruct_approximate_two_randomized(pA,pB,pC,nA,nB,nC,num):
    burn = int(0.15 * num)

    pint = np.zeros(shape=[2 ** (nB + nC - 1), 2, 3])
    x = gen_string(nB+nC-1)
    count = 0
    p = 0
    while count < num*3:
        c = 0
        beta1 = np.random.randint(0,3)
        e1 = np.random.randint(0,1)
        y = flip_string(x)

        indC = int(x[0:nC][::-1], 2)
        indB = int(x[nC:nC + nB - 1][::-1], 2)
        px = [gamma(beta2, ahat2, e2) * pB[indB, e1, beta1, ahat2, beta2] * pC[indC, e2, beta2]
              for beta2 in [0, 1, 2] for ahat2 in [0, 1] for e2 in [0, 1]]
        px = np.sum(np.array(px)) / 2

        indy = int(y, 2)
        indC = int(y[0:nC], 2)
        indB = int(y[nC:nC + nB - 1], 2)
        py = [gamma(beta2, ahat2, e2) * pB[indB, e1, beta1, ahat2, beta2] * pC[indC, e2, beta2]
              for beta2 in [0, 1, 2] for ahat2 in [0, 1] for e2 in [0, 1]]
        py = np.sum(np.array(py)) / 2

        r = min(1, (py) / (px))
        if np.random.rand() < r:
            x = y
        count += 1
        if count > burn:
            pint[indy, e1, beta1] += 1/(num-burn)


        p += 1
        # if p % 1000 == 0:
        #     print(count)


    p_rec = {}
    x = gen_string(nA + nB + nC - 2)
    count = 0
    while count < num:
        y = flip_string(x)

        indBC = int(x[0:nB + nC - 1][::-1], 2)
        indA = int(x[nB + nC - 1:len(y)][::-1], 2)
        px = [gamma(beta1, ahat1, e1) * pA[indA, ahat1, beta1] * pint[indBC, e1, beta1]
              for beta1 in [0, 1, 2] for ahat1 in [0, 1] for e1 in [0, 1]]
        px = np.sum(np.array(px)) / 2

        indBC = int(y[0:nB + nC - 1][::-1], 2)
        indA = int(y[nB + nC - 1:len(y)][::-1], 2)
        py = [gamma(beta1, ahat1, e1) * pA[indA, ahat1, beta1] * pint[indBC, e1, beta1]
              for beta1 in [0, 1, 2] for ahat1 in [0, 1] for e1 in [0, 1]]
        py = np.sum(np.array(py)) / 2

        r = min(1, (py) / (px))
        if np.random.rand() < r:
            x = y
        count += 1
        if count > burn:
            if x not in p_rec:
                p_rec[x] = 0
            else:
                p_rec[x] += 1
    return p_rec













