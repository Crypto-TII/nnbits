import toml
import os
import numpy as np

from .filemanager import FileManager

def get_X(F):
    config = toml.load(F.filename_config())

    if config['PREDICT_LABEL']:
        X = np.empty((config['NEURAL_NETWORKS'], config['RESULTING N TOTAL BITS'] + 1))
    else:
        X = np.empty((config['NEURAL_NETWORKS'], config['RESULTING N TOTAL BITS']))
    X[:] = np.NaN

    network_id = 0

    filename = F.filename_selections()
    _selections = np.load(filename)
    selected_bits, not_selected_bits = _selections['selected_bits'], _selections['not_selected_bits']

    for filter_id in np.arange(config['NEURAL_NETWORKS']):
        filename = F.filename_accs(network_id)
        if os.path.isfile(filename):
            x = np.load(filename)
            X[network_id][selected_bits[filter_id]] = x

        network_id += 1

    return X

import warnings

def get_bit_accs(X):
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bit_accs = np.nanmean(X, axis=0)
    return bit_accs

def get_bit_stds(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bit_stds = np.nanstd(X, axis=0)
    return bit_stds

def get_bit_npred(X):
    bit_npred = np.sum(~np.isnan(X), axis=0)
    return bit_npred

def get_bit_ids_high_acc(bit_accs):
    #bit_accs = get_bit_accs(X)
    # if there is a nan value, set it to zero
    # (otherwise it will be handled as np.inf)
    _temp_bit_accs = bit_accs.copy()
    _temp_bit_accs[np.isnan(_temp_bit_accs)] = 0
    # the accuracies are sorted from low to high
    # --> reverse the argsort order with [::-1]
    sorted_ids = _temp_bit_accs.argsort()[::-1]
    return sorted_ids


def get_pvalue(n_trials, obs_mean):
    from scipy import stats

    successes = int(n_trials * obs_mean)
    pvalue = stats.binomtest(successes, n_trials, p=0.5).pvalue

    return pvalue


class BitAnalysis:
    def __init__(self, savepath):
        self.F = FileManager(savepath)
        config = toml.load(self.F.filename_config())
        self.n_trials = config['N_VAL']

    def update(self):
        self.X = get_X(self.F)
        self.bit_accs = get_bit_accs(self.X)
        self.bit_npred = get_bit_npred(self.X)
        self.bit_stds = get_bit_stds(self.X)
        self.bit_ids_high_acc = get_bit_ids_high_acc(self.bit_accs)

    def get_pvalue_of_bit_id(self, bit_id):
        obs_n = self.bit_npred[bit_id]
        obs_mean = self.bit_accs[bit_id]
        obs_std = self.bit_stds[bit_id]

        pvalue = get_pvalue(self.n_trials, obs_mean)
        return pvalue

    def get_pvalues_of_highest_acc(self, n_highest=1):
        pvalues = []

        for i in range(n_highest):
            bit_id = self.bit_ids_high_acc[i]
            pvalue = self.get_pvalue_of_bit_id(bit_id)
            pvalues.append(pvalue)

        return pvalues

def get_weak_and_strong_bits(X):
    bitbybit = np.nanmean(X, axis=0)
    distance_to_50 = np.abs(bitbybit - 0.5)
    sorted_bits = distance_to_50.argsort()
    weak_half = sorted_bits[len(bitbybit)//2:]
    strong_half = sorted_bits[:len(bitbybit)//2]
    return weak_half, strong_half