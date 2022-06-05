import numpy as np
import toml
import argparse
import pandas as pd

import ray
from ray.util import ActorPool

from .filemanager import FileManager
from .trainingtracker import TrainingTracker
from .network import Network
from . import selections


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
                 model_inputs=64,
                 model_outputs=1,
                 input_data_op='None',
                 predict_label=False,
                 epochs=10,
                 batchsize=4096,
                 verbose=False,
                 validation_batch_size=None):
        # --- GPU preparation
        import os

        self.gpu_str = f'{gpu}'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_str

        # --- Data preparation
        self.data_train = ray.get(obj_refs[0])
        self.data_val = ray.get(obj_refs[1])
        self.selected_bits = ray.get(obj_refs[2])
        self.not_selected_bits = ray.get(obj_refs[3])

        # --- Network preparation
        super().__init__(self.data_train,
                         self.data_val,
                         self.selected_bits,
                         self.not_selected_bits,
                         model_id,
                         model_strength,
                         model_inputs,
                         model_outputs,
                         input_data_op,
                         predict_label,
                         epochs,
                         batchsize,
                         verbose,
                         validation_batch_size)
# ------------------------------

# ---------------------------------------------------
# Create function to be parallelized
# ---------------------------------------------------
@ray.remote
def parallelize(a, filemanager: FileManager, network_id, save_weights=False):
    a.create_model.remote()  # TODO: maybe this can be replaced by reset weights
    # a.reset_weights.remote()
    a.pass_bit_selections.remote(network_id)
    a.pass_bit_selections_test.remote(network_id)
    a.train.remote()
    a.save_history.remote(filemanager.filename_history(network_id))
    if save_weights:
        a.save_weights.remote(filemanager.filename_h5(network_id))
    ray.get(a.test.remote(filemanager.filename_accs(network_id)))
    return f'finalized id {network_id}'

# ---------------------------------------------------
# Various configuration functions
# ---------------------------------------------------
def config_add_optional_defaults(_config):
    """
    If an argument is not provided in config and the argument is optional,
    this function sets the default values.
    """
    optional_defaults = {'PREDICT_LABEL': False,
                         'TARGET_BITS': [],
                         'N_RANDOM_BITS': 0,
                         'EARLY_STOPPING': True,
                         'EARLY_STOPPING_P_VALUE': 1e-10,
                         'CREATE_NEW_SELECTIONS': True,
                         'MODEL_STRENGTH': 1,
                         'VALIDATION_BATCH_SIZE': None,
                         'SAVE_WEIGHTS': False,
                         }

    # # check if a validation batch size has been given (if not, simply the whole validation data will be passed in one go)
    # VALIDATION_BATCH_SIZE = None
    # if 'VALIDATION_BATCH_SIZE' in config:
    #     VALIDATION_BATCH_SIZE = config['VALIDATION_BATCH_SIZE']

    for key in optional_defaults:
        if key in _config:
            pass
        else:
            _config[key] = optional_defaults[key]
    return _config

def data_from_config(_config):
    data = np.load(_config['DATAPATH'])
    _data_train = data.take(np.arange(0, _config['N_TRAIN']), axis=0)
    _data_val = data.take(np.arange(_config['N_TRAIN'], _config['N_TRAIN'] + _config['N_VAL']), axis=0)
    return _data_train, _data_val

def bit_selections_from_config(_config, filename=None):
    """
    Example:
        config = {
               'NEURAL_NETWORKS': 5,
                 'SELECT_BITS_STRATEGY': 'target',
                 'TARGET_BITS': [[1,2]]*5,
                 'N_RANDOM_BITS': 2,
                'PREDICT_LABEL': True,
        }

        config = inferred_config_settings(config, (10_000, 10))

        bit_selections_from_config(config)
    """
    # initialize the selections

    selection_constructor = getattr(selections, _config['SELECT_BITS_STRATEGY'])
    _selected_bits, _not_selected_bits = selection_constructor(_config['NEURAL_NETWORKS'],
                                                         _config['RESULTING N SELECTED BITS'],
                                                         _config['RESULTING N TOTAL BITS'],
                                                         make_uniform=True,
                                                         list_of_target_bits=_config['TARGET_BITS'])

    if filename:
        np.savez(filename,
                 selected_bits=_selected_bits,
                 not_selected_bits=_not_selected_bits)

    return _selected_bits, _not_selected_bits

def clean_target_bits_list(target_bits_list):
    """
    The TARGET_BITS can be passed as a list of integers or a list of lists
    'TARGET_BITS': [0, 1, 2] or 'TARGET_BITS': [[0,1], [1,2], [3,4]]

    Returns: 'TARGET_BITS' as a list of lists
    """
    if isinstance(target_bits_list[0], int):
        target_bits_list = [[_] for _ in target_bits_list]
    return target_bits_list

def inferred_config_settings(_config, data_train_shape, data_val_shape):
    """
    Example:
        config = {
                   'NEURAL_NETWORKS': 5,
                     'SELECT_BITS_STRATEGY': 'target',
                     'TARGET_BITS': [[1,2]]*5,
                     'N_RANDOM_BITS': 2,
                    'PREDICT_LABEL': True,
        }

        config = inferred_config_settings(config, (10_000, 10))
    """

    _config['RESULTING BITS IN DATA-ROW'] = data_train_shape[1]

    # Infer the number of selected bits:
    if _config['SELECT_BITS_STRATEGY'] == 'random':
        _config['RESULTING N SELECTED BITS'] = _config['N_RANDOM_BITS']
    elif _config['SELECT_BITS_STRATEGY'] == 'target':
        _config['RESULTING N SELECTED BITS'] = len(_config['TARGET_BITS'][0])
    elif _config['SELECT_BITS_STRATEGY'] == 'None':
        _config['RESULTING N SELECTED BITS'] = 0

        # Infer the number of total bits:
    #   (if there is no label it is equivalent to the 'RESULTING BITS IN DATA-ROW')
    _config['RESULTING N TOTAL BITS'] = _config['RESULTING BITS IN DATA-ROW']
    #   (if there is, however, a label in the data row, subtract 1)
    if _config['PREDICT_LABEL']:
        _config['RESULTING N TOTAL BITS'] -= 1

    # Infer the number of input neurons:
    _config['RESULTING NN INPUT NEURONS'] = _config['RESULTING N TOTAL BITS']
    if _config['SELECT_BITS_STRATEGY'] == 'zero':
        _config['RESULTING NN INPUT NEURONS'] -= _config['RESULTING N SELECTED BITS']

    # Infer the number of output neurons:
    if _config['PREDICT_LABEL']:
        _config['RESULTING NN OUTPUT NEURONS'] = 1
    else:
        _config['RESULTING NN OUTPUT NEURONS'] = _config['RESULTING N SELECTED BITS']

    # Infer the validation batch size:
    if _config['VALIDATION_BATCH_SIZE'] is None:
        _config['VALIDATION_BATCH_SIZE'] = data_val_shape[0]

    # potential config clean up steps
    if _config['SELECT_BITS_STRATEGY'] == 'target':
        _config['TARGET_BITS'] = clean_target_bits_list(_config['TARGET_BITS'])

    return _config

if __name__ == '__main__':

    # Parse arguments from command line
    args = configure_argparse()

    # Set up the FileManager and read the configuration
    savepath = args.savepath
    F = FileManager(savepath)
    config = toml.load(F.filename_config())
    config = config_add_optional_defaults(config)

    # Prepare data
    data_train, data_val = data_from_config(config)

    # Infer config settings
    config = inferred_config_settings(config, data_train.shape, data_val.shape)

    # Print configuration
    print("="*70)
    df = pd.DataFrame.from_dict(config, orient='index', columns=['value'])
    print(df.to_markdown())
    print('\n')

    with open(F.filename_config(), 'w') as configfile:
        toml.dump(config, configfile)

    # Prepare bit selections
    if config['CREATE_NEW_SELECTIONS']:
        selected_bits, not_selected_bits = bit_selections_from_config(config, filename=F.filename_selections())
    else:
        selected_bits = np.load(F.filename_selections())['selected_bits']
        not_selected_bits = np.load(F.filename_selections())['not_selected_bits']

    if config['NEURAL_NETWORKS'] > 1:
        # CREATE THE RAY ACTOR POOL
        # --------------------------------
        # Initialize Ray
        ray.init()
        # Initialize a ray version of the network class with the correct number of
        # GPU and CPU resources:
        RayNetwork = ray.remote(num_gpus=config['GPU_PER_ACTOR'], num_cpus=config['CPU_PER_ACTOR'])(RayNetwork)

        # Store common and larger objects in the Ray Object Store:
        data_train_id = ray.put(data_train)
        data_val_id = ray.put(data_val)
        selected_bits_id = ray.put(selected_bits)
        not_selected_bits_id = ray.put(not_selected_bits)

        # # enable GPU memory growth if there is more than one actor per GPU
        # if config['N_ACTORS_PER_GPU'] > 1:
        #     SET_MEMORY_GROWTH = True
        # else:
        #     SET_MEMORY_GROWTH = False

        # create a ray actor pool
        Actors = [RayNetwork.remote([data_train_id,
                                     data_val_id,
                                     selected_bits_id,
                                     not_selected_bits_id],
                                    _ % config['N_GPUS'],
                                    config['NEURAL_NETWORK_MODEL'],
                                    config['MODEL_STRENGTH'],
                                    config['RESULTING NN INPUT NEURONS'],
                                    config['RESULTING NN OUTPUT NEURONS'],
                                    config['INPUT_DATA_OP'],
                                    config['PREDICT_LABEL'],
                                    config['N_EPOCHS'],
                                    config['BATCHSIZE'],
                                    verbose=False,
                                    validation_batch_size=config['VALIDATION_BATCH_SIZE'],
                                    ) for _ in range(config['N_GPUS'] * config['N_ACTORS_PER_GPU'])]
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
                                                                     save_weights=config['SAVE_WEIGHTS']), range(config['NEURAL_NETWORKS']))
        for t in tasks:
            # if early stopping is desired, exit the actor training early
            if config['EARLY_STOPPING'] and (event_handler.best_pval < config['EARLY_STOPPING_P_VALUE']):
                terminate_actors = [actor.__ray_terminate__.remote() for actor in Actors]
                print('p-value is below limit ==> stop analysis.')
                break
            else:
                pass

        # Clean up Actors and Training Tracker
        kill_actors = [ray.kill(actor) for actor in Actors]
        observer.stop()

    elif config['NEURAL_NETWORKS'] == 1:
        # TRAIN THE NETWORK WITHOUT A RAY ACTOR POOL
        # --------------------------------
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        network = Network(data_train,
                          data_val,
                          selected_bits,
                          not_selected_bits,
                          config['NEURAL_NETWORK_MODEL'],
                          config['MODEL_STRENGTH'],
                          config['RESULTING NN INPUT NEURONS'],
                          config['RESULTING NN OUTPUT NEURONS'],
                          config['INPUT_DATA_OP'],
                          config['PREDICT_LABEL'],
                          config['N_EPOCHS'],
                          config['BATCHSIZE'],
                          verbose=True,
                          validation_batch_size=config['VALIDATION_BATCH_SIZE'],
                          )

        def parallelize(a, filemanager: FileManager, network_id, save_weights=False):
            a.create_model()
            a.pass_bit_selections(network_id)
            a.pass_bit_selections_test(network_id)
            a.train()
            a.save_history(filemanager.filename_history(network_id))
            if save_weights:
                a.save_weights.remote(filemanager.filename_h5(network_id))  # TODO: maybe uncomment
            #a.test(filemanager.filename_accs(network_id))  # testing
            return f'finalized id {network_id}'

        parallelize(network, F, 0)

