import numpy as np
from icecream import ic

def totalVariationalDistance(p1, p2):
    # make sure distributions are normalized
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
    # make sure distributions are normalized
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


""" A new distance function where distances are
    weighted by the percentage error

    LateX for the formula is as follows:
    \[ dist(p_1, p_2) \equiv \sum_{k} \frac{|Pr(p_{1_k}) - Pr(p_{2_k})|^2}{Pr(p_{2_k})} \]

    p_2 is the expected distribution
"""
def weighted_distance(p1, p2):
    # make sure distributions are normalized
    # (likely unnecessary) but good practice
    sum = np.sum([p1[k] for k in p1.keys()])
    for k in p1.keys():
        p1[k] = p1[k]/sum

    sum = np.sum([p2[k] for k in p2.keys()])
    for k in p2.keys():
        p2[k] = p2[k] / sum

    # implementation of the function in the comment for this function
    total_distance = 0
    for k in p1.keys():
        if k in p2.keys():
            total_distance += ((p1[k] - p2[k])**2) / p2[k]
        else:
            # in the case of a 0 denominator, just add square of numerator
            total_distance += p1[k]**2
    for k in p2.keys():
        if k not in p1.keys():
            total_distance += abs(p2[k])

    return total_distance

def l2_norm_distance(p1, p2, expected_bits):
    # make sure distributions are normalized
    # (likely unnecessary but good practice)
    sum = np.sum([p1[k] for k in p1.keys()])
    for k in p1.keys():
        p1[k] = p1[k]/sum

    sum = np.sum([p2[k] for k in p2.keys()])
    for k in p2.keys():
        p2[k] = p2[k] / sum

    p1_vec = np.zeros([2**expected_bits])
    p2_vec = np.zeros([2**expected_bits])

    for dec_num_1 in range(2**expected_bits):
        # make sure we have all bitstrings in our counts
        bin_num = format(dec_num_1, '03b')
        if bin_num not in p1.keys():
            p1_vec[dec_num_1] = 0
        else:
            p1_vec[dec_num_1] = p1[bin_num]
    for dec_num_2 in range(2**expected_bits):
        # make sure we have all bitstrings in our counts
        bin_num = format(dec_num_2, '03b')
        if bin_num not in p2.keys():
            p2_vec[dec_num_2] = 0
        else:
            p2_vec[dec_num_2] = p2[bin_num]

    diff = p1_vec - p2_vec
    return np.linalg.norm(diff)

    