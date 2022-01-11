import numpy as np
from subprocess import run
from multiprocessing.dummy import Pool
from . import utils
import glob

import os

# filter out tf info messages, for more info, see
#   https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Train or Test the ensemble.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- add arguments to parser
    parser.add_argument('-mode',
                        type=str,
                        help="Mode: What to do with the ensemble? 'train' or 'test'.")

    parser.add_argument(
        '--save_path',
        default='_temp',
        type=str,
        help='The name of the folder in which to find the cfg files for the ensemble and save the model files.')

    parser.add_argument(
        '--processes',
        default=64,
        type=int,
        help="How many processes the CPU maximally launches simultaneously on the GPU \
                  (depends on the number of GPUs, ensemble members and number of training batches).")

    parser.add_argument(
        '--gpus',
        default=[0, 1, 2, 3],
        nargs='+',
        help='The GPU IDs to use during training.')

    parser.add_argument(
        '--overwrite',
        default=1,
        type=int,
        help="If '--overwrite 1' (True) existing model files will be overwritten. \
                   If '--overwrite 0' (False) existing model files will not be overwritten.")
    # ---------------------------------------------------
    args = parser.parse_args()
    # ---------------------------------------------------
    args.gpus = [int(value) for value in args.gpus]
    # ---------------------------------------------------

    F = utils.files.FileManager(args.save_path)
    cfg_files = glob.glob(F.cfg_path + '/*')

    # ---------------------------------------------------
    # Parallelized Ensemble Training
    # ---------------------------------------------------
    # ---------------------------------------------------
    def call_script(ensemble_args):
        """
        This script calls 'train_single.py' or 'test_single.py'.
        """

        cfg_id, gpu_id = ensemble_args  # unpack arguments

        if args.mode == 'train':
            print(f'training for cfg file {cfg_id} on gpu {gpu_id} was called...')
            script = 'distinguisher.train_single'
        elif args.mode == 'test':
            print(f'testing for cfg file {cfg_id} on gpu {gpu_id} was called...')
            script = 'distinguisher.test_single'
        else:
            script = None

        output = run(f"python -m {script} {cfg_id} {gpu_id} '{args.save_path}' {args.overwrite}",
                     capture_output=True,
                     shell=True)

        print(output.stdout.decode('ascii'))
        print(output.stderr.decode('ascii'))


    # ---------------------------------------------------

    # Pool for multiprocessing, the 'call_script' is called with different arguments:
    ensemble_args = []

    for index_id in np.arange(len(cfg_files)):
        cfg_id = str(index_id)
        gpu_id = args.gpus[index_id % len(args.gpus)]
        ensemble_args.append([cfg_id, gpu_id])

    pool = Pool(processes=args.processes)
    pool.map(call_script, ensemble_args)
    pool.close()
    pool.join()
    # ---------------------------------------------------
