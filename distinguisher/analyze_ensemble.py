import numpy as np
import pandas as pd
import keras
from . import utils
import toml
from scipy.stats import binom_test

import os
# filter out tf info messages, for more info, see
#   https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def pValue(n, p, p0 = 0.5):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom_test.html
    """
    n_successes = int(p*n)
    return binom_test(x=n_successes, n=n, p=0.5, alternative='two-sided')

def get_analysis_df(folder, run=True):
    """Calculate the accuracies of each ensemble member for each bit ID

    :param folder: save_path in which to find a `pred` prediction folder
    :param run: if run=True the accuracies will be re-calculated.
        In this case the result will be stored in a file 'df_bit_level_analysis.pkl'.
        Afterwards run=False can be used to not re-calculate all accuracies, but load the pkl results.
    :return:
    """

    F = utils.files.FileManager(folder)

    if run:

        rows = []  # container to save the results.
        n_files = F.count_files(F.pred_path) # count files in the prediction path

        for index in range(n_files):

            # ---- load the predicted and actual target values:
            data = np.load(F.filename_of_predictions(index))
            y_pred = data['y_pred']
            y_true = data['y_true']

            # ---- get the actual bit indices from the config file:
            config = toml.load(F.filename_of_config(index))
            h_indices = config['bit_ids']['h_ids']

            # ---- evaluate the binary test accuracies for each single bit:
            for i in np.arange(y_true.shape[1]):

                y_true_i = y_true[:, i]
                y_pred_i = y_pred[:, i]

                bin_acc_test = keras.metrics.binary_accuracy(y_true_i[:].flatten(), y_pred_i[:].flatten())
                bin_acc_test = bin_acc_test.numpy()

                rows.append({'NNi': index,
                             'bitID': h_indices[i],
                             'A': bin_acc_test})

        df = pd.DataFrame(rows)
        df.to_pickle(F.filename_of_ensemble_analysis())

    else:
        df = pd.read_pickle(F.filename_of_ensemble_analysis())

    return df

def get_analysis_I(df):
    """
    # ---------------------------------------------------
    # Generate Analysis I
    # ---------------------------------------------------

    # for each neural network:
    #    extract the mean accuracy with which it can predict its bits.
    """
    rows = []

    for NNi in df.NNi.values:
        df_NNi = df[df.NNi == NNi]

        mean = df_NNi.A.mean()  # mean accuracy
        std = df_NNi.A.std()  # std of the accuracy

        rows.append({'NNi': NNi,
                     'A_mean': mean,
                     'A_std': std})

    df_I = pd.DataFrame(rows)
    df_I = df_I.sort_values(by='A_mean', ascending=False)

    return df_I

def get_analysis_II(df):
    """
    # ---------------------------------------------------
    # Generate Analysis II
    # ---------------------------------------------------

    # for each bit in the sequence:
    #    extract the mean accuracy with which it can be predicted.
    """
    rows = []

    for bitID in np.unique(df.bitID.values):
        df_bit = df[df.bitID == bitID]

        mean = df_bit.A.mean()  # mean accuracy
        std = df_bit.A.std()  # std of the accuracy
        n_NN = len(df_bit)  # number of neural networks predicting the bit

        rows.append({'bitID': bitID,
                     'A_mean': mean,
                     'A_std': std,
                     'n_NN': n_NN})

    df_II = pd.DataFrame(rows)
    df_II = df_II.sort_values(by='A_mean', ascending=False)

    return df_II

if __name__ == '__main__':

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description     = 'Analyze the ensemble.',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    # ---- add arguments to parser
    parser.add_argument(
        '--save_path',
        default = '_temp',
        type    = str,
        help    = 'The name of the folder in which to find the cfg files for the ensemble and save the h5 files for each model.')

    # ---------------------------------------------------
    args = parser.parse_args()
    # ---------------------------------------------------

    df = get_analysis_df(args.save_path, run=True)
