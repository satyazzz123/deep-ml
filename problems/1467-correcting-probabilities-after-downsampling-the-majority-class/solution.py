import numpy as np


def downsample_negatives(y, rate, rng):
    # TODO: keep all positives; keep each negative with probability rate (one rng.random(len(y)) draw)
    y=np.asarray(y)
    # idx=np.where(y==1)

    random_draws=rng.random(len(y))
    keep_mask=(y==1)| (random_draws<rate)
    return keep_mask




def corrected_probability(p_sampled, rate):
    # TODO: r p_s / (r p_s + 1 - p_s)
    return (rate*p_sampled)/(rate*p_sampled+1-p_sampled)


def logit_shift(rate):
    return np.log(rate)


def sampled_probability(p, rate):
    # TODO: p / (p + (1 - p) r)
    return (p)/(p+(1-p)*rate)


def calibration_gap(p, y):
    
    return np.mean(p)-np.mean(y)
