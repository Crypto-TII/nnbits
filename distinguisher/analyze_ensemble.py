import numpy as np
import pandas as pd
import keras
from . import utils
import toml
from scipy.stats import binomtest
from .utils.files import FileManager

import os

# filter out tf info messages,
#   for more info, see
#   https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def pValue(n, p, p0 = 0.5):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html#scipy.stats.binomtest

    n: Number of bits being predicted
    p: Observed binary accuracy of the prediction on a scale from 0...1

    Returns: The p-value of the hypothesis test.
    """
    n_successes = int(p*n)
    return binomtest(k=n_successes, n=n, p=0.5, alternative='two-sided').pvalue;

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

            if os.path.isfile(F.filename_of_predictions(index)):

                # ---- load the predicted and actual target values:
                data = np.load(F.filename_of_predictions(index))
                y_pred = data['y_pred']
                y_true = data['y_true']

                y_pred = y_pred.reshape(y_true.shape)

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

                    # ---------------------------------------------------
                    # Clean up backend
                    # ---------------------------------------------------
                    keras.backend.clear_session()

        df = pd.DataFrame(rows)
        df.to_pickle(F.filename_of_ensemble_analysis())

        # ---------------------------------------------------
        # Clean up backend
        # ---------------------------------------------------
        keras.backend.clear_session()

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

def get_pearson_corr(bx, cfg_indices, accuracies, save_path, n_bits=1024):
    """Returns a DataFrame with the Pearson correlations of each bit with the accuracy of bit bx.

    Idea:
    Construct a table in which the presence of each bit is indicated by 1 (bit was present) or 0 (bit was not present).
    Then correlate the presence of each bit with the obtained accuracy.
    In the following example the presence of bit 0 correlates with the obtained accuracy:

    | accuracy | bit 0 | bit 1 | bit 2 |
    ------------------------------------
    |   95%    |   1   |   0   |   0   |
    |    0%    |   0   |   1   |   1   |
    |   93%    |   1   |   1   |   0   |
    [TABLE 1]

    From [TABLE 1] the Pearson correlation of each bit's column with the accuracy column is calculated.
    The calculation uses the Panda-modules `corr` function and calculates

        pearson_r(bit_id) = corr(accuracy, bit_id presence).

    The resulting pearson_r for each bit is returned in [TABLE 2]:

    |  bit_id  | pearson_r |
    ------------------------
    |  bit 0   |    0.9    |
    |  bit 1   |    0.0    |
    |  bit 2   |    0.0    |
    [TABLE 2]

    Returns: DataFrame of the form of [TABLE 2].
    """

    # ----------------------------------------
    # Initialize a table similar to [TABLE 1]:
    # ----------------------------------------


    F = FileManager(save_path)

    pd.options.mode.chained_assignment = None

    # initialize a dataframe with all zeros
    df_bit = pd.DataFrame(0, index=np.arange(len(cfg_indices)),
                          columns=['cfg_index', 'accuracy'] + [f'B{_}' for _ in range(n_bits)])
    df_bit['cfg_index'] = cfg_indices
    df_bit['accuracy'] = accuracies

    # # for each possible bit initialize a column with 0s:
    # for new_column in np.arange(n_bits):
    #     df_bit[f'B{new_column}'] = [0] * len(df_bit)

    # for each cfg file, set the present bits to 1:
    for index, row in df_bit.iterrows():
        config = toml.load(F.filename_of_config(int(row.cfg_index)))
        lp_indices = config['bit_ids']['lp_ids']
        for lp in lp_indices:
            df_bit.loc[index, f'B{lp}'] = 1
    # ----------------------------------------

    # ----------------------------------------
    # Calculate the correlations similar to [TABLE 2]:
    # ----------------------------------------
    corrs = []
    for h in np.arange(n_bits):
        corr = df_bit['accuracy'].corr(df_bit[f'B{h}'])
        if h == bx:
            corr = 0.0
        corrs.append({'bit_id': h, 'pearson_r': corr})

    df_c = pd.DataFrame(corrs)
    df_c = df_c.sort_values(by='pearson_r', ascending=False)
    # ----------------------------------------

    return df_c

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

    # ---------------------------------------------------
    # Clean up backend
    # ---------------------------------------------------
    keras.backend.clear_session()
