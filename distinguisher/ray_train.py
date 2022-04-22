import numpy as np
import time
import pandas as pd
import ray
from ray.util import ActorPool
from . import utils
from . import training_tracker
import argparse
import os
from .models import ModelSelector
import toml
import glob
# filter out tf info messages, for more info, see
#   https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

ray.init();

def configure_argparse(parser):
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

#--- initialize one neural network per GPU
@ray.remote(num_gpus=1)
class RayNetwork(ModelSelector):
    def __init__(self, gpu:int):
        #--- GPU preparation
        import os
        gpu_str = f'{gpu}'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str

        # ---------------------------------------------------
        # Select Model to be trained
        # ---------------------------------------------------

        model_id = config['train_info']['model_id']
        # infer the dimension of the input and output layer from the training data
        time_steps, output_dim = len(config['bit_ids']['lp_ids']), len(config['bit_ids']['h_ids'])

        super().__init__(time_steps, output_dim, model_id,
                             filename_npzlog = None,
                              y_val_length = config['train_info']['batch_size_val'])

    def set_log(self, index):
        super().set_filename_npzlog(F.filename_of_validation_npz(index))

    def prepare_data(self, index):
        # ---------------------------------------------------
        # Create the Dataset
        # ---------------------------------------------------
        config = toml.load(F.filename_of_config(index))
        train_ids = np.arange(config['file_ids']['train_ids'][0], config['file_ids']['train_ids'][-1] + 1)
        val_ids = np.arange(config['file_ids']['val_ids'][0], config['file_ids']['val_ids'][-1] + 1)
        test_ids = np.arange(config['file_ids']['test_ids'][0], config['file_ids']['test_ids'][-1] + 1)

        dataset = utils.create_dataset.create_dataset(config['train_info']['data_path'],
                                                      train_ids,
                                                      val_ids,
                                                      test_ids[0:1],  # don't load test files during training
                                                      lp_indices=config['bit_ids']['lp_ids'],
                                                      h_from_backcast=config['bit_ids']['h_from_backcast'])
        super().passdataset(dataset)

    def get_weights(self):
        return super().model.get_weights()

if __name__ == '__main__':
    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Train or Test the ensemble.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    configure_argparse(parser)
    args = parser.parse_args()
    args.gpus = [int(value) for value in args.gpus]
    # ---------------------------------------------------
    # ---------------------------------------------------
    # Preparations
    # ---------------------------------------------------
    # Start a file manager
    F = utils.files.FileManager(args.save_path)
    # Read the config file
    config = toml.load(F.filename_of_config(0))
    # ---------------------------------------------------
    # ---------------------------------------------------
    # Ray Actor Pool
    # ---------------------------------------------------
    N_GPUS = 4

    # --- create a pool of actors
    Actors = [RayNetwork.remote(_) for _ in range(N_GPUS)]
    pool = ActorPool(Actors)

    @ray.remote
    def func(a, index):
        a.reset_weights.remote()
        a.prepare_data.remote(index)
        a.set_log.remote(index)
        a.train_model.remote(logfile=F.filename_of_training_progress(index))
        #a.model.evaluate(dataset['x_val'], dataset['y_val'], batch_size=config['train_info']['batch_size_val'])
        weights = ray.get(a.get_weights.remote())

    #----------------------------------------------------
    # start training tracker:
    from watchdog.observers import Observer

    event_handler = training_tracker.MyEventHandler(args.save_path, F.filename_of_log_ensemble_training())
    observer = Observer()
    observer.schedule(event_handler, path=args.save_path + '/hist/', recursive=False)
    observer.start()
    # ----------------------------------------------------
    # execute the pool
    cfg_files = glob.glob(F.cfg_path + '/*')
    tasks = pool.map(lambda a, v: func.remote(a, v), range(len(cfg_files)))
    print([v for v in tasks])
    #
    observer.stop()