import fullcirc, subcirc1, subcirc2
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt

# query counts from each measurement basis and initialization
# this actually stores the probability, so it's badly named
counts1I = subcirc1.run_subcirc1('I/Z')
counts1X = subcirc1.run_subcirc1('X')
counts1Y = subcirc1.run_subcirc1('Y')

count1 = [counts1I, counts1I, counts1X, counts1Y]

counts20 = subcirc2.run_subcirc2('0')
counts21 = subcirc2.run_subcirc2('1')
counts2p = subcirc2.run_subcirc2('+')
counts2pi = subcirc2.run_subcirc2('+i')

count2 = [counts20, counts21, counts2p, counts2pi]

# fix bit string and attribute shot
p_rec = {} # dictionary of probabilities reconstructed

for n in range(32,64): # I'm looping through 00000 ~ 11111 by taking the last bits
    bstr = bin(n)
    # let's take this as 01010
    string = bstr[3:8]
    # substring11 is 010, substring12 is 011
    # but the order is a bit weird just to fit how qiskit is coded
    substring11 = '0' + bstr[6:8]
    substring12 = '1' + bstr[6:8]
    # substring2 is 010
    substring2 = bstr[3:6]

    # just making sure we don't get an error from calling an empty key
    for c in count1:
        if substring11 not in c:
            c[substring11] = 0
        if substring12 not in c:
            c[substring12] = 0
    for c in count2:
        if substring2 not in c:
            c[substring2] = 0

    # find all the components needed as specified by CutQC paper
    p11 = counts1I[substring11] * 2
    p12 = counts1I[substring12] * 2
    p13 = counts1X[substring11] - counts1X[substring12]
    p14 = counts1Y[substring11] - counts1Y[substring12]

    p21 = counts20[substring2]
    p22 = counts21[substring2]
    p23 = 2 * counts2p[substring2] - p21 - p22
    p24 = 2 * counts2pi[substring2] - p21 - p22

    p_rec[string] = (p11 * p21 + p12 * p22 + p13 * p23 + p14 * p24)/2

# plot result
plot_histogram(p_rec,title='reconstructed')
print(p_rec)
full = fullcirc.run_full_circuit()
plot_histogram(full,title='full circuit')
print(full)
plt.show()
