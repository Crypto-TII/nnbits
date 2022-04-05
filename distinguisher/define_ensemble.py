"""This is module defines the ensemble configuration files."""

__version__ = '0.1'
__author__ = 'TII (Technology Innovation Institute)'

import argparse
import numpy as np
import warnings
import random
import toml
import sys
from . import utils
from .utils.files import FileManager


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
    # all_file_indices = np.arange(n_train + n_val + n_test)
    # train_ids = all_file_indices[:n_train].tolist()
    # val_ids = all_file_indices[n_train:n_train + n_val].tolist()
    # test_ids = all_file_indices[n_train + n_val:n_train + n_val + n_test].tolist()
    train_ids = [0, n_train-1]
    val_ids = [n_train, n_train + n_val - 1]
    test_ids = [n_train + n_val, n_train + n_val + n_test - 1]
    return train_ids, val_ids, test_ids

def write_cfg(index,
              category,
              file_manager: FileManager,
              lp0=0, args=None):
    """
    :param index: integer which identifies the ensemble member.
    :param category: 'sequential' or 'random' choice of bits for the look-back period LP
    :param LP0: start index for the look-back period (only relevant if the category is sequential)
    :return: None (writes configuration file)
    """

    # calculate the number of training, validation and test sequences:
    n_train = args.n_train_batches * args.batch_size
    n_val = args.n_val_batches * args.batch_size_val
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
    train_info = {'model_id' : args.model_id,
                        'epochs': args.epochs,
                       'batch_size': args.batch_size,
                        'batch_size_val': args.batch_size_val,
                       'data_path': args.data_path,
                       'n_bits': args.n_bits,
                       'n_train': n_train,
                       'n_val': n_val,
                       'n_test': n_test,
                       }

    file_ids = {'train_ids': [train_ids[0], train_ids[-1]],
                  'val_ids': [val_ids[0], val_ids[-1]],
                  'test_ids': [test_ids[0], test_ids[-1]]}

    bit_ids = {'h_from_backcast': args.h_from_backcast,
                'lp_ids': lp_indices,
                'h_ids': h_indices
                }

    configuration = {'train_info': train_info,
                     'bit_ids': bit_ids,
                     'file_ids': file_ids,
                     }

    config_filename = file_manager.filename_of_config(index)

    with open(config_filename, 'w') as configfile:
        toml.dump(configuration, configfile)


def configure_argparse(parser):
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
    parser.add_argument('--batch_size_val',
                        default=-1,
                        type=int,
                        help='batch size for validation (if it should be different from the one used for training).')
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
    parser.add_argument('--target_bit_range',
                        default=[],
                        nargs='+',
                        help='A start and end point for a range of target bits? \n '
                             'For example --target_bit_range 0 64 for a range of [0,1,...,63] target bits.')
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
    parser.add_argument('--model_id',
                        default='nbeats_0',
                        type=str,
                        help='In models.py NBEATS with different parameters are defined and can chosen here.')
    parser.add_argument('--h_from_backcast',
                        default='remove',
                        type=str,
                        help="""What to do with the predicted bit ids (h) in the backcast region (input of the neural network)?
                            if 'remove' the horizon indices will be removed from the lookback period completely.
                            if 'blackout' the horizon indices will still be present, but will all be set to 0.""")
    parser.add_argument('--force_index',
                        default=-1,
                        type=int,
                        help="""If -1 the indices of the configuration files will be assigned automatically.
                            If force_index is set to another integer, the resulting configuration file(s) will be
                            assigned this index.
                             """)
    parser.add_argument('--config-from-file',
                        type=str,
                        help="""Load the configuration from file""")

def read_config(file_path):
    with open(file_path, 'r') as configfile:
        config  = toml.load(configfile)
        print(config)
        return config

def my_main(args):
    print(args)
    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(description='Generate *.cfg files for an NBEATS ensemble.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    configure_argparse(parser)
    args = parser.parse_args()

    if args.config_from_file:
        config = read_config(args.config_from_file)

    #sys.exit(0)
    #overwrite_config_from_arguments(config, parser)
    # ---------------------------------------------------
    # ---------------------------------------------------
    if args.target_bit_range:
        args.target_bit_range = [int(value) for value in args.target_bit_range]
        args.target_bit_range = np.arange(args.target_bit_range[0], args.target_bit_range[-1])
    #----------------------------------------------------
    # if no validation batch size is given, use the same batch size as for training:
    if args.batch_size_val == -1:
        args.batch_size_val = args.batch_size
    # ---------------------------------------------------
    if args.data_path == '':
        raise argparse.ArgumentTypeError(
            "Please provide a --data_path 'str' to continue (use -h for more information).")
    if args.force_index == -1:
        make_index = lambda x: x
    else:
        make_index = lambda x: args.force_index
    # ---------------------------------------------------

    file_manager = utils.files.FileManager(args.save_path)
    file_manager.create_folders()

    # ---------------------------------------------------
    # Write the *.cfg files:
    # ---------------------------------------------------
    if args.target_bit == -1 and len(args.target_bit_range)==0:
        for index in np.arange(0, min(4, args.n_ensemble)):
            # We write four *.cfg files which will cover all bits in a sequential manner.
            write_cfg(index=make_index(index), category='sequential', file_manager=file_manager, lp0=index * args.n_bits // 4, args=args)
        # The remaining *.cfg files will cover bit combinations in a random manner.
        for index in np.arange(4, args.n_ensemble):
            write_cfg(index=make_index(index), category='random', file_manager=file_manager, args=args)
    #--- args.target_bit_range
    elif len(args.target_bit_range):
        for index in args.target_bit_range:
            args.target_bit = index
            write_cfg(index=index, category='target', file_manager=file_manager, args=args)
            args.target_bit = -1
    #--- args.target_bit
    else:
        for index in np.arange(0, args.n_ensemble):
            write_cfg(index=make_index(index), category='target', file_manager=file_manager, args=args)
    # ---------------------------------------------------


if __name__ == '__main__':
    my_main(sys.argv[1:])