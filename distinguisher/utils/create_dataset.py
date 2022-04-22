import numpy as np
import glob

# def get_bit_arrays(data_path, n_bits, index_start_file=0, index_stop_file=-1):
#
#     files = sorted(glob.glob(data_path+'*'))
#     files = np.array(files[index_start_file:index_stop_file])
#
#     all_arrays = []
#
#     for file in files:
#         with open(file, 'rb') as f:
#             b = f.read()
#             b = str(b).replace("b'input = 0b", '')
#             b = str(b).replace('b', '')
#             b = str(b).replace("'", '')
#
#             b = list(b)
#             b = np.array([float(element) for element in b])
#             if len(b) != n_bits:
#                 pass
#             else:
#                 all_arrays.append(b)
#
#     return np.asarray(all_arrays)


def create_back_and_forecasts(bit_arrays,
                              lp_indices: list,
                              h_from_backcast: str):
    """
    :param h_from_backcast:
        if 'remove' the horizon indices will be removed from the lookback period completely.
        if 'blackout' the horizon indices will still be present, but will all be set to 0.
    """

    ### From the given lp_indices for the lookback period LP create the indices of the horizon (forecast) H:
    all_indices = np.arange(len(bit_arrays[0]))
    all_indices = np.delete(all_indices, lp_indices)
    h_indices = all_indices

    ##################################################################
    # TODO change the following numpy slices to take
    if h_from_backcast == 'remove':
        backcasts = bit_arrays[:, lp_indices]

    elif h_from_backcast == 'blackout':
        bit_arrays_copy = bit_arrays.copy()
        bit_arrays_copy[:, h_indices] = 0
        backcasts = bit_arrays_copy

    forecasts = bit_arrays[:, h_indices]

    return backcasts, forecasts

def create_dataset(data_path,
                   train_ids,
                   val_ids,
                   test_ids,
                   lp_indices,
                   h_from_backcast='remove'):
    """

    :param data_path: the path to the *.npy file.
    :param train_ids:
    :param val_ids:
    :param test_ids:
    :param lp_indices:
    :param h_from_backcast: ('remove', 'blackout') see function create_back_and_forecasts()

    :return: Dictionary with keys 'x_train', 'y_train', 'x_val', 'y_val','x_test', 'y_test'
    """

    #if data_type == '0':
    #    bit_arrays = get_bit_arrays(data_path, index_start_file=index_start_file, index_stop_file=index_stop_file) # TO DO - inefficient: we currently load all data in the folder!
    #    bit_arrays = bit_arrays.astype(int)
    #if data_type == '1':
    bit_arrays = np.load(data_path)

    if 'npz' in data_path[-3:]:
        bit_arrays = bit_arrays['dataset']

    backcasts, forecasts = create_back_and_forecasts(bit_arrays, lp_indices, h_from_backcast)

    x_train, y_train    = backcasts[train_ids], forecasts[train_ids]
    x_val, y_val        = backcasts[val_ids], forecasts[val_ids]
    x_test, y_test      = backcasts[test_ids], forecasts[test_ids]

    dataset = {'x_train': x_train,
               'y_train': y_train,
               'x_val': x_val,
               'y_val': y_val,
               'x_test': x_test,
               'y_test': y_test}

    return dataset