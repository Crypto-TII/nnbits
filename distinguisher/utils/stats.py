import matplotlib.pyplot as plt
import numpy as np

from distinguisher.utils import plotstyle
import matplotlib.ticker
from scipy.stats import chisquare, binom
from scipy import stats

def create_binomial_distribution(size, n, p=.5, convert_to_percentages=True):
    """
    Draw samples from a binomial distribution.

    0 1 1 0 1
    |_|_|_|_|
        n

    n: number of predicted bits, resp. the number of trials performed
    p: the probability of success in each trial
    size: ensemble members, resp. how many times are the n-bits predicted

    From: https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
    """
    s = np.random.binomial(n, p, size)
    if convert_to_percentages:
        s = s / n  # to obtain probabilities, we divide by n
    return s

def make_bins(N_TRIALS, P_EXP, N_BINS, my_observations=None):
    """
    The `stepsize` is determined according to the expected binomial distribution and the desired
    number of bins N_BINS.

    If my_observations are given, the number of bins may be adjusted, such that the calculated
    `stepsize` is preserved over the full range of observations (expected & my_observations).

    """
    # define bins according to expected probability distribution:
    xmin, xmax = binom.ppf(0.0001, N_TRIALS, P_EXP), binom.ppf(0.9999, N_TRIALS, P_EXP)
    stepsize = (xmax - xmin) / N_BINS
    bins = np.linspace(xmin, xmax, int((xmax - xmin) / stepsize))

    if my_observations is not None:
        xmin = min(xmin, min(my_observations))
        xmax = max(xmax, max(my_observations))
        bins = np.linspace(xmin, xmax, int((xmax - xmin) / stepsize))

    return bins


def get_expected_count_per_bin(bins, N_EXPERIMENTS, N_TRIALS, P_EXP):
    import math

    b_width = np.diff(bins).mean()

    expected = []

    for b in bins[:-1]:
        # collect all integers in the bin
        all_integers = list(range(math.ceil(b), math.floor(b + b_width) + 1))
        # calculate probability for each integer in the bin
        all_probs = [binom.pmf(x, N_TRIALS, P_EXP) for x in all_integers]
        # sum up all probabilities
        sum_probs = np.sum(all_probs)

        # the total expected count is the sum of the probabilities x N_EXPERIMENTS
        expected.append([b + b_width / 2, sum_probs * N_EXPERIMENTS])

    expected = np.array(expected)

    return expected


def create_figure(N_EXPERIMENTS=500, N_TRIALS=100000,
                  P_EXP=0.5,
                  P_OBS=0.5,
                  N_BINS=15,
                  ALPHA=.05, my_observations=None):
    #fig = plt.figure()

    print('=' * 70)
    print(f"""
    N_EXPERIMENTS={N_EXPERIMENTS} \t N_TRIALS={N_TRIALS:,.0f}
    P_EXP = {P_EXP} \t P_OBS = {P_OBS} 
    N_BINS = {N_BINS}
     """)

    # --- OBSERVATION ---#
    # assume our observation is the following binomial distribution
    if my_observations is None:
        observed_distribution = create_binomial_distribution(size=N_EXPERIMENTS, n=N_TRIALS, p=P_OBS,
                                                             convert_to_percentages=False)
    else:
        observed_distribution = my_observations

    # --- MODEL PART 1 ---#
    bins = make_bins(N_TRIALS, P_EXP, N_BINS, my_observations=observed_distribution)

    # for each bin calculate the expected count:
    expected = get_expected_count_per_bin(bins, N_EXPERIMENTS, N_TRIALS, P_EXP)
    # -------------------------------------------

    observed_hist = plt.hist(observed_distribution, ec='white', zorder=2, align='mid', bins=bins, label='observation')

    # --- MODEL PART 2---#
    # a small rescaling of the expected counts may be needed to ensure that the sum of the total counts match each other
    # (this is a prerequisite for the chi-square test)
    adjustment_factor = np.sum(observed_hist[0]) / np.sum(expected[:, 1])

    print(f'(needed adjustment factor between counts = {adjustment_factor:.4f})')
    expected[:, 1] = adjustment_factor * expected[:, 1]

    # ---- CHI SQUARE ----#
    # cast inputs to dtype np.float64 as recommended in this issue
    # https://github.com/scipy/scipy/issues/10159
    f_obs = np.array(observed_hist[0], dtype=np.float64)
    f_exp = np.array(expected[:, 1], dtype=np.float64)
    # ---
    # catch RuntimeWarning
    # RuntimeWarning: divide by zero encountered in true_divide
    # terms = (f_obs_float - f_exp)**2 / f_exp
    # This RuntimeWarning occurs if the expected counts in the observation region are zero.
    # In this case we set the chisquare_stat to np.nan and the p-value to 0.0
    divide_by_zero = np.argwhere((f_exp == 0) & (f_obs != 0))
    if len(divide_by_zero) > 0:
        chisquare_stat, chisquare_p = np.nan, 0.0
    else:
        chisquare_stat, chisquare_p = chisquare(f_obs, f_exp)
    #chisquare_stat, chisquare_p = chisquare(f_obs, f_exp)
    # adjust the markercolor
    if chisquare_p < ALPHA:
        mc = 'red'
    else:
        mc = 'green'

    # --- PLOT ---#
    plt.plot(expected[:, 0], expected[:, 1], 'o', ms='2', c=mc, label='expectation')
    plt.title(f'$\chi^2=${chisquare_stat:2.1g}  $p(\chi^2)=${chisquare_p:.4f}')
    plt.legend(fontsize='x-small')

    return plt.gcf()