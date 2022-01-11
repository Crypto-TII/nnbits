"""This is module defines the ensemble configuration files."""

__version__ = '0.1'
__author__ = 'TII (Technology Innovation Institute)'

import argparse
import numpy as np
import warnings
import random
import toml
from . import utils


def create_lp(n_bits: int,
              n_lp: int,
              category: str,
              lp0: int = None,
              target_bit: int = None):
    """
    Create the indices for the lookback period (LP).

    Given a sequence of n_bits
        bit(0), bit(1), ... bit(n_bits-1)
    some of those bit positions will be selected for the LP, e.g.
        bit(4), bit(10), ..., bit(n_bits-5).
    How the bit positions are selected is determined by the category:
        'sequential': bit(lp0), bit(lp0+1), ..., bit(lp0+n_lp)
        'random': randomly
        'target': a bit which is a target_bit can NOT appear in the LP and is excluded.

    :param n_bits: sequence length
    :param n_bits: sequence length
    :param n_lp: lookback period length
    :param category: 'sequential', 'random' or 'target' choice of bits for the look-back period LP
    :param lp0: starting point if category is 'sequential'
    :return: list of lookback period indices
    """
    lp_indices = None

    if category == 'sequential':
        lp_indices = np.arange(lp0, lp0 + n_lp) % n_bits
    elif category == 'random':
        lp_indices = np.sort(np.array(random.sample(range(n_bits), n_lp)))
    elif category == 'target':
        original_range = range(n_bits)
        new_range = np.delete(original_range, target_bit)
        new_range = new_range.tolist()
        lp_indices = np.sort(np.array(random.sample(new_range, n_lp)))
    else:
        warnings.WarningMessage('Invalid category.')

    return lp_indices

def sequence_ids(n_train, n_val, n_test):
    """
    Given the number of training, validation  and test sequences, (`n_train`, `n_val`, `n_test`)
    get the lists of the corresponding sequence ids.
    """
    all_file_indices = np.arange(n_train + n_val + n_test)
    train_ids = all_file_indices[:n_train].tolist()
    val_ids = all_file_indices[n_train:n_train + n_val].tolist()
    test_ids = all_file_indices[n_train + n_val:n_train + n_val + n_test].tolist()
    return train_ids, val_ids, test_ids

def write_cfg(index,
              category,
              lp0=0):
    """
    :param index: integer which identifies the ensemble member.
    :param category: 'sequential' or 'random' choice of bits for the look-back period LP
    :param LP0: start index for the look-back period (only relevant if the category is sequential)
    :return: None (writes configuration file)
    """

    # calculate the number of training, validation and test sequences:
    n_train = args.n_train_batches * args.batch_size
    n_val = args.n_val_batches * args.batch_size
    n_test = args.n_test_batches * args.batch_size

    # write down the lists of training, validation and test sequences:
    train_ids, val_ids, test_ids = sequence_ids(n_train, n_val, n_test)

    # ---- Define the indices of the lookback period LP and forecast region or "Horizon H"
    all_indices = np.arange(args.n_bits)
    lp_indices = create_lp(args.n_bits,
                           args.n_lp,
                           category,
                           lp0,
                           args.target_bit)
    all_indices = np.delete(all_indices, lp_indices)
    h_indices = all_indices
    # cast to list:
    lp_indices = lp_indices.tolist()
    h_indices = h_indices.tolist()

    # ---- Write the configuration into a toml file:
    train_info = {'epochs': args.epochs,
                       'batch_size': args.batch_size,
                       'data_path': args.data_path,
                       'n_bits': args.n_bits,
                       'n_train': n_train,
                       'n_val': n_val,
                       'n_test': n_test,
                       }

    file_ids = {'train_ids': train_ids,
                  'val_ids': val_ids,
                  'test_ids': test_ids}

    bit_ids = {'lp_ids': lp_indices,
                'h_ids': h_indices
                }

    configuration = {'train_info': train_info,
                     'bit_ids': bit_ids,
                     'file_ids': file_ids,
                     }

    config_filename = FileManager.filename_of_config(index)

    with open(config_filename, 'w') as configfile:
        toml.dump(configuration, configfile)


if __name__ == '__main__':

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(description='Generate *.cfg files for an NBEATS ensemble.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--save_path',
                        default='_temp',
                        type=str,
                        help='The name of the folder in which to save all results.')
    parser.add_argument('--data_path',
                        default="/opt/cryptanalysis_servers_shared_folder/",
                        type=str,
                        help='The name of the folder in which we find the data to be analyzed.')
    parser.add_argument('--data_type',
                        default='0',
                        type=str,
                        help="'0' for single files in the data_path, '1' for a single npy file")
    parser.add_argument('--batch_size',
                        default=1024,
                        type=int,
                        help='batch size for training, validation and testing.')
    parser.add_argument('--n_train_batches',
                        default=8,
                        type=int,
                        help='How many batches to use as training data.')
    parser.add_argument('--n_val_batches',
                        default=1,
                        type=int,
                        help='How many batches to use as validation data.')
    parser.add_argument('--n_test_batches',
                        default=8,
                        type=int,
                        help='How many batches to use as test data.')
    parser.add_argument('--n_ensemble',
                        default=100,
                        type=int,
                        help='How many NBEATS ensemble members are to be created (minimum 4)?')
    parser.add_argument('--target_bit',
                        default=-1,
                        type=int,
                        help='Should a particular bit be targeted by the ensemble? \n '
                             'The target_bit will always appear in the region to be predicted.')
    parser.add_argument('--n_bits',
                        default=1024,
                        type=int,
                        help='Number of bits in the sequence.')
    parser.add_argument('--n_lp',
                        default=768,
                        type=int,
                        help='Number of bits in the lookback period (input of the neural network).')
    parser.add_argument('--epochs',
                        default=35,
                        type=int,
                        help='For how many epochs to train.')
    # ---------------------------------------------------
    args = parser.parse_args()
    # ---------------------------------------------------
    if args.data_path == '':
        raise argparse.ArgumentTypeError(
            "Please provide a --data_path 'str' to continue (use -h for more information).")
    if args.n_ensemble < 4:
        raise argparse.ArgumentTypeError(
            "Please provide --n_ensemble 4 or larger to continue (use -h for more information).")
    # ---------------------------------------------------

    FileManager = utils.files.FileManager(args.save_path)
    FileManager.create_folders()

    # ---------------------------------------------------
    # Write the *.cfg files:
    # ---------------------------------------------------
    if args.target_bit == -1:
        # We write four *.cfg files which will cover all bits in a sequential manner.
        write_cfg(index=0, category='sequential', lp0=0 * args.n_bits // 4)
        write_cfg(index=1, category='sequential', lp0=1 * args.n_bits // 4)
        write_cfg(index=2, category='sequential', lp0=2 * args.n_bits // 4)
        write_cfg(index=3, category='sequential', lp0=3 * args.n_bits // 4)
        # The remaining *.cfg files will cover bit combinations in a random manner.
        for index in np.arange(4, args.n_ensemble):
            write_cfg(index=index, category='random')
    else:
        for index in np.arange(0, args.n_ensemble):
            write_cfg(index=index, category='target')
    # ---------------------------------------------------
