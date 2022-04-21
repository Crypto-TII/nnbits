import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import toml

import ray
from ray.util import ActorPool

from .filemanager import FileManager
from .filtermanager import FilterManager
from .trainingtracker import TrainingTracker
from .bitanalysis import get_X, get_weak_and_strong_bits
from .network import Network

if __name__ == '__main__':

    ray.init()

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description='Run a distinguisher ensemble parallelized with Ray.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--savepath',
        default='_temp',
        type=str,
        help='The name of the folder in which to find the configuration file (as in example_config.cfg) and in which to save the analysis files.')
    args = parser.parse_args()

    # ---------------------------------------------------
    # Set up
    # ---------------------------------------------------
    savepath = args.savepath
    F = FileManager(savepath)
    config = toml.load(F.filename_cfg())

    # ---------------------------------------------------
    # Initialize a ray version of the network:
    # ---------------------------------------------------
    @ray.remote(num_gpus=config['NUM_GPUS'], num_cpus=config['NUM_CPUS'])
    class RayNetwork(Network):
        def __init__(self,
                     obj_refs,
                     gpu: int,
                     model_id='nbeats',
                     model_strength=1,
                     set_memory_growth=True,
                     data_strategy='remove',
                     epochs=10):
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
                             set_memory_growth,
                             data_strategy,
                             epochs)
    # ------------------------------

    # ---------------------------------------------------
    # Create function to be parallelized
    # ---------------------------------------------------
    @ray.remote
    def func(a, F, network_id, filter_id):
        a.create_model.remote()  # TODO: maybe this can be replaced by reset weights
        # a.reset_weights.remote()
        a.pass_filters.remote(filter_id)
        a.train.remote()
        a.save_history.remote(F.filename_history(network_id))
        # a.save_weights.remote(F.filename_h5(network_id)) # TODO: maybe uncomment

        # --- testing
        a.pass_filters_test.remote(filter_id)
        ray.get(a.test.remote(F.filename_accs(network_id)))
        return f'finalized id {network_id}'


    # --- Training Tracker
    from watchdog.observers import Observer

    event_handler = TrainingTracker(savepath, logfile=F.filename_run_log())
    observer = Observer()
    observer.schedule(event_handler, path=F.path_test_accuracies, recursive=False)
    observer.start()

    # --- prepare data
    data = np.load(config['datapath'])

    data_train = data.take(np.arange(0, config['N_TRAIN']), axis=0)
    data_val = data.take(np.arange(config['N_TRAIN'], config['N_TRAIN'] + config['N_VAL']), axis=0)

    data_train_id = ray.put(data_train)
    data_val_id = ray.put(data_val)
    # ------------------------------

    # enable GPU memory growth
    if config['N_ACTORS_PER_GPU'] > 1:
        set_memory_growth = True
    else:
        set_memory_growth = False

    run_id = 0
    f0 = 0

    # prepare filters
    filtermanager = FilterManager(config['N_BITS'])

    # for each filter / input pair
    for N_FILTER, N_INPUT, FILTER_STRATEGY in zip(config['N_FILTERS'], config['N_INPUTS'], config['FILTER_STRATEGY']):

        N_OUTPUT = config['N_BITS'] - N_INPUT

        # evaluate bits
        if run_id > 0:
            X = get_X(F, run_id)
            weak_half, strong_half = get_weak_and_strong_bits(X)
        #         #--- make plot
        #         plt.figure()
        #         plt.scatter(np.arange(config['N_BITS']), np.nanmean(X, axis=0))
        #         plt.scatter(np.arange(config['N_BITS'])[weak_half], np.nanmean(X[:, weak_half], axis=0))
        #         plt.show()
        #         #----
        else:
            weak_half, strong_half = None, None
        # -----------------------------------------

        # initialize the filters
        if FILTER_STRATEGY == 'random':
            input_filters, output_filters = filtermanager.create_random_filters(N_FILTER, N_INPUT, make_uniform=True)
        elif FILTER_STRATEGY == 'weak':
            input_filters, output_filters = filtermanager.create_weakbit_filters(N_FILTER, N_INPUT,
                                                                                 weak_bit_ids=weak_half,
                                                                                 strong_bit_ids=strong_half)
        elif FILTER_STRATEGY == 'target':
            input_filters, output_filters = filtermanager.create_target_filters(N_FILTER, N_INPUT,
                                                                                config['TARGET_BITS'][run_id])
        elif FILTER_STRATEGY == 'gohr_with_target':
            input_filters, output_filters = filtermanager.create_gohr_with_target_filters(N_FILTER, N_INPUT,
                                                                                          config['TARGET_BITS'][run_id])

        np.savez(F.filename_filters(run=run_id),
                 input_filters=input_filters,
                 output_filters=output_filters)

        input_filters_id = ray.put(input_filters)
        output_filters_id = ray.put(output_filters)

        # ------------------------------
        # create a ray actor pool
        Actors = [RayNetwork.remote([data_train_id,
                                     data_val_id,
                                     input_filters_id,
                                     output_filters_id],
                                    _ % config['N_GPUS'],
                                    config['MODEL_ID'],
                                    config['MODEL_STRENGTH'],
                                    set_memory_growth,
                                    config['DATA_STRATEGY'],
                                    config['N_EPOCHS']) for _ in range(config['N_GPUS'] * config['N_ACTORS_PER_GPU'])]
        pool = ActorPool(Actors)

        # train the actors on the filters
        tasks = pool.map(lambda actor, filter_id: func.remote(actor, F, filter_id + f0, filter_id), range(N_FILTER))
        results = [v for v in tasks]
        # display([v for v in tasks])
        # clean up actors
        kill_actors = [ray.kill(actor) for actor in Actors]

        # ------------------------------
        f0 += N_FILTER
        run_id += 1

        if event_handler.best_pval < 1e-10:
            print('p-value is below limit ==> stop analysis.')
            break

    # ------------------------------
    observer.stop()