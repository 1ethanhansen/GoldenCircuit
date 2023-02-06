import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from icecream import ic

five_qubit_reconstructed = np.array([1.4074852085238574,
                                        0.017868887078699325,
                                        0.024213309596474844,
                                        0.006091082239396866,
                                        0.018155823945393917,
                                        0.5269854324922746,
                                        0.04826506825923811,
                                        0.030888714532464178,
                                        0.03580744717827404,
                                        1.085320314963532])
five_qubit_full = np.array([0.5517542628262189,
                            0.036953282633326506,
                            0.014257431624436339,
                            0.00595196698567,
                            0.009381460795516607,
                            1.1280944173293714,
                            0.038188494657883265,
                            0.01557064911715274,
                            0.04533825851015292,
                            0.5288732310281644])

seven_qubit_reconstructed = np.array([0.07526394884202078, 
                                        1.0829483048454138, 
                                        0.07958280792472788, 
                                        0.41526155600130654, 
                                        2.168616476483447, 
                                        0.23079945962832776, 
                                        0.12109898569985258, 
                                        1.0107215383364128,
                                        0.3663553976629582,
                                        0.33631449447336226])
seven_qubit_full = np.array([0.38561365905205575,
                                1.09490807941074,
                                0.10808728769875667,
                                1.4795683851309027,
                                1.3467772706328958,
                                0.22102615357080838,
                                0.17492552917486862,
                                16.208313364517377,
                                0.0450318546172423,
                                0.30973225630203866])

vals_to_plot = np.array([])
errs_to_plot = np.array([])
labels = ["5 qubit full circuit", "5 qubit reconstructed", "7 qubit full circuit", "7 qubit reconstructed"]

five_full_mean = five_qubit_full.mean()
five_reconstruct_mean = five_qubit_reconstructed.mean()
vals_to_plot = np.append(vals_to_plot, five_full_mean)
vals_to_plot = np.append(vals_to_plot, five_reconstruct_mean)
five_full_sem = five_full_mean / np.sqrt(len(five_qubit_full))
five_reconstruct_sem = five_reconstruct_mean / np.sqrt(len(five_qubit_reconstructed))

seven_full_mean = seven_qubit_full.mean()
seven_reconstruct_mean = seven_qubit_reconstructed.mean()
vals_to_plot = np.append(vals_to_plot, seven_full_mean)
vals_to_plot = np.append(vals_to_plot, seven_reconstruct_mean)
seven_full_sem = seven_full_mean / np.sqrt(len(seven_qubit_full))
seven_reconstruct_sem = seven_reconstruct_mean / np.sqrt(len(seven_qubit_reconstructed))

overall_interval = st.t.interval(confidence=0.95, df=len(five_qubit_full)-1, loc=five_full_mean, scale=five_full_sem)
plus_minus = (overall_interval[1] - overall_interval[0]) / 2
errs_to_plot = np.append(errs_to_plot, plus_minus)

overall_interval = st.t.interval(confidence=0.95, df=len(five_qubit_reconstructed)-1, loc=five_reconstruct_mean, scale=five_reconstruct_sem)
plus_minus = (overall_interval[1] - overall_interval[0]) / 2
errs_to_plot = np.append(errs_to_plot, plus_minus)

overall_interval = st.t.interval(confidence=0.95, df=len(seven_qubit_full)-1, loc=seven_full_mean, scale=seven_full_sem)
plus_minus = (overall_interval[1] - overall_interval[0]) / 2
errs_to_plot = np.append(errs_to_plot, plus_minus)

overall_interval = st.t.interval(confidence=0.95, df=len(seven_qubit_reconstructed)-1, loc=seven_reconstruct_mean, scale=seven_reconstruct_sem)
plus_minus = (overall_interval[1] - overall_interval[0]) / 2
errs_to_plot = np.append(errs_to_plot, plus_minus)

ic(vals_to_plot)
ic(errs_to_plot)


x_pos = np.arange(len(labels))
colors = ['#BF616A', '#FFD700', '#BF616A', '#FFD700']
fig, ax = plt.subplots()
ax.bar(x_pos, vals_to_plot, yerr=errs_to_plot, align='center', color=colors, ecolor='#2E3440', capsize=10)
ax.set_title('Comparison of fidelities')
ax.set_ylabel('Average weighted distance')
ax.set_xlabel('Run type and size of device')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)

plt.show()