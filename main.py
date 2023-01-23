# import matplotlib.pyplot as plt
# import numpy as np
#
# from reconstruct import *
#
# subcirc1 = create_subcirc1()
# subcirc2 = create_subcirc2(1)
# subcirc3 = create_subcirc3()
# nA = subcirc1.width()
# nB = subcirc2.width()
# nC = subcirc3.width()
#
# qc = QuantumCircuit(5).compose(subcirc2)
# qc.compose(subcirc3,qubits=[2,3,4], inplace=True)
#
# print(subcirc1)
# print(subcirc2)
# # print(qc)
#
# num_sample = 4000
# # pA, pB = run_subcirc(subcirc1,subcirc2)
# # exact = reconstruct_exact(pA,pB,nA,nB)
# # approx = reconstruct_approximate(pA,pB,nA,nB,num_sample)
# # pA, pB = run_subcirc(subcirc1,qc)
# # exact = reconstruct_exact(pA,pB,nA,nB+nC-1)
# # approx = reconstruct_approximate(pA,pB,nA,nB+nC-1,num_sample)
# pA, pB, pC = run_three_subcirc(subcirc1, subcirc2, subcirc3)
# exact = reconstruct_exact_seq(pA,pB,pC,nA,nB,nC)
# approx = reconstruct_approximate_two(pA,pB,pC,nA,nB,nC,num_sample)
# # approx = reconstruct_approximate_two_randomized(pA,pB,pC,nA,nB,nC,8000)
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
#
# circ = QuantumCircuit(5)
# circ.h(np.arange(0,5))
# circ.t([2,3,4])
# circ.cz(0,1)
# circ.cz(0,2)
# circ.rx(1.57, 4)
# circ.rx(1.57,[0,1])
# circ.cz(2,4)
# circ.cz(2,3)
# circ.ry(1.57,4)
# circ.t([0,1])
# circ.h(np.arange(0,5))
#
# # circ = QuantumCircuit(7)
# # circ.h(np.arange(0,7))
#
# circ.measure_all()
# print(circ)
# #
# # circ.h([0,1,3,4])
# # circ.cnot(1,2)
# # circ.cnot(3,2)
# # circ.cnot(4,3)
# # circ.measure_all()
#
# # circ.h([1,3,7,9])
# # circ.cnot(1,3)
# # circ.cnot(3,4)
# # circ.cnot(6,7)
# # circ.cnot(9,8)
# # circ.measure_all()
# # print(circ)
# #
# # # Transpile for simulator
# simulator = Aer.get_backend('aer_simulator')
# circ = transpile(circ, simulator)
#
# # Run and get counts
# result = simulator.run(circ,shots=10000).result()
# counts = result.get_counts(circ)
# plot_histogram(counts,title='exact full')
#
# # p_rec = {}
# # for n in range(2**(nA+nB-1),2**(nA+nB)):
# #     bstr = bin(n)
# #     string = bstr[3:len(bstr)]
# #     p = reconstruct_bstr(bstr,pA,pB,nA,nB)
# #     p_rec[string] = p*10000
# #
# #
#
# #
# # sum = 0
# # for k in exact.keys():
# #     sum += exact[k]
# #
# #
# # sum = 0
# # for k in approx.keys():
# #     sum += approx[k]
# #
# # for k in approx.keys():
# #     approx[k] = approx[k]/sum
# #
# # for k in counts.keys():
# #     if counts[k]/num_sample > 0.02:
# #         if k in exact and k in approx:
# #             print(k, 'full', counts[k]/num_sample, ' exact: ', exact[k], ', approx: ', approx[k])
# #         else:
# #             if k not in exact:
# #                 print('exact missing')
# #             else:
# #                 print('approx missing')
# #
# plot_histogram(exact,title='exact recon')
# print(exact)
# plot_histogram(approx,title='approx')
# print(approx)
# plt.show()
#
#
#
#
#
