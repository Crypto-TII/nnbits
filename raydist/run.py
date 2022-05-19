import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import toml
import argparse

import ray
from ray.util import ActorPool

from .filemanager import FileManager
from .trainingtracker import TrainingTracker
from .bitanalysis import get_X, get_weak_and_strong_bits
from .network import Network
from . import filters

def configure_argparse():
    parser = argparse.ArgumentParser(
        description='Run a distinguisher ensemble parallelized with Ray.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---------------------------------------------------

    parser.add_argument(
        '--savepath',
        default='_temp',
        type=str,
        help='The name of the folder in which to find the configuration file (as in example_config.cfg) and in which to save the analysis files.')
    _args = parser.parse_args()
    return _args

# ---------------------------------------------------
# Initialize a ray version of the network:
# ---------------------------------------------------
class RayNetwork(Network):
    def __init__(self,
                 obj_refs,
                 gpu: int,
                 model_id='gohr',
                 model_strength=1,
                 model_inputs = 64,
                 model_outputs = 1,
                 set_memory_growth=True,
                 data_strategy='remove',
                 epochs=10,
                 batchsize=4096):
        # --- GPU preparation
        import os

        self.gpu_str = f'{gpu}'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_str

        # --- Data preparation
        self.data_train = ray.get(obj_refs[0])
        self.data_val = ray.get(obj_refs[1])
        self.input_filters = ray.get(obj_refs[2])
        self.output_filters = ray.get(obj_refs[3])

        # --- Network preparation
        super().__init__(self.data_train,
                         self.data_val,
                         self.input_filters,
                         self.output_filters,
                         model_id,
                         model_strength,
                         model_inputs,
                         model_outputs,
                         set_memory_growth,
                         data_strategy,
                         epochs,
                         batchsize)
# ------------------------------

# ---------------------------------------------------
# Create function to be parallelized
# ---------------------------------------------------
@ray.remote
def parallelize(a, filemanager: FileManager, network_id, save_weights=False):
    a.create_model.remote()  # TODO: maybe this can be replaced by reset weights
    # a.reset_weights.remote()
    a.pass_filters.remote(network_id)
    a.pass_filters_test.remote(network_id)
    a.train.remote()
    a.save_history.remote(filemanager.filename_history(network_id))
    if save_weights:
        a.save_weights.remote(filemanager.filename_h5(network_id)) # TODO: maybe uncomment
    ray.get(a.test.remote(filemanager.filename_accs(network_id))) # testing
    return f'finalized id {network_id}'

def data_from_config(_config):
    data = np.load(_config['datapath'])
    _data_train = data.take(np.arange(0, _config['N_TRAIN']), axis=0)
    _data_val = data.take(np.arange(_config['N_TRAIN'], _config['N_TRAIN'] + _config['N_VAL']), axis=0)
    return _data_train, _data_val

def filters_from_config(_config, filename=None):
    n_output = _config['N_BITS'] - _config['N_INPUT_FILTER_ELEMENTS']
    # initialize the filters
    filter_constructor = getattr(filters, _config['FILTER_STRATEGY'])
    _input_filters, _output_filters = filter_constructor(_config['N_FILTERS'], _config['N_INPUT_FILTER_ELEMENTS'], n_output,
                                                         make_uniform=True,
                                                         list_of_target_bits=_config['TARGET_BITS'])

    if filename:
        np.savez(filename,
                 input_filters=_input_filters,
                 output_filters=_output_filters)

    return _input_filters, _output_filters


if __name__ == '__main__':

    EARLY_STOPPING_P_VALUE = 1e-10

    # Parse arguments from command line
    args = configure_argparse()

    # Set up the FileManager and read the configuration
    savepath = args.savepath
    F = FileManager(savepath)
    config = toml.load(F.filename_cfg())

    # Prepare data
    data_train, data_val = data_from_config(config)

    # Prepare filters
    if config['MAKE_NEW_FILTERS']:
        input_filters, output_filters = filters_from_config(config, filename=F.filename_filters())
    else:
        input_filters = np.load(F.filename_filters())['input_filters']
        output_filters = np.load(F.filename_filters())['output_filters']

    # CREATE THE RAY ACTOR POOL
    # --------------------------------
    # Initialize Ray
    ray.init()
    # Initialize a ray version of the network class with the correct number of
    # GPU and CPU resources:
    RayNetwork = ray.remote(num_gpus=config['NUM_GPUS'], num_cpus=config['NUM_CPUS'])(RayNetwork)

    # Store common and larger objects in the Ray Object Store:
    data_train_id = ray.put(data_train)
    data_val_id = ray.put(data_val)
    input_filters_id = ray.put(input_filters)
    output_filters_id = ray.put(output_filters)

    # enable GPU memory growth if there is more than one actor per GPU
    if config['N_ACTORS_PER_GPU'] > 1:
        SET_MEMORY_GROWTH = True
    else:
        SET_MEMORY_GROWTH = False

    # create a ray actor pool
    Actors = [RayNetwork.remote([data_train_id,
                                 data_val_id,
                                 input_filters_id,
                                 output_filters_id],
                                _ % config['N_GPUS'],
                                config['MODEL_ID'],
                                config['MODEL_STRENGTH'],
                                config['MODEL_INPUTS'],
                                config['MODEL_OUTPUTS'],
                                SET_MEMORY_GROWTH,
                                config['DATA_STRATEGY'],
                                config['N_EPOCHS'],
                                config['BATCHSIZE']) for _ in range(config['N_GPUS'] * config['N_ACTORS_PER_GPU'])]
    pool = ActorPool(Actors)

    # Launch a Training Tracker
    #--------------------------------
    # (the training tracker will print relevant current training information)
    from watchdog.observers import Observer

    event_handler = TrainingTracker(savepath, logfile=F.filename_run_log())
    observer = Observer()
    observer.schedule(event_handler, path=F.path_test_accuracies, recursive=False)
    observer.start()

    # Train the Actors on the Filters
    # --------------------------------
    tasks = pool.map(lambda actor, filter_id: parallelize.remote(actor, 
                                                                 F, 
                                                                 filter_id, 
                                                                 save_weights=config['SAVE_WEIGHTS']), range(config['N_FILTERS']))
    for t in tasks:
        # if early stopping is desired, exit the actor training early
        if config['EARLY_STOPPING'] and (event_handler.best_pval < EARLY_STOPPING_P_VALUE):
            terminate_actors = [actor.__ray_terminate__.remote() for actor in Actors]
            print('p-value is below limit ==> stop analysis.')
            break
        else:
            pass

    # Clean up Actors and Training Tracker
    kill_actors = [ray.kill(actor) for actor in Actors]
    observer.stop()