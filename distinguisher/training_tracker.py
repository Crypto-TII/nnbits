import sys
save_path = str(sys.argv[1])

from . import utils


import datetime
import glob
import pandas as pd
import numpy as np
import toml
import time

from watchdog.events import FileSystemEventHandler

MIN_LOG_INTERVAL = 15

class MyEventHandler(FileSystemEventHandler):
    """
    This subclass of the watchdog FileSystemEventHandler is used to track the training progress.
    It will look for new files created in the `save_path`.
    Once a file is created, it will generate a report with help of the `evaluate_training_progress` function.
    """

    def __init__(self, save_path, logfile=None):
        super().__init__()
        self.save_path = save_path
        self.string = ''
        self.logfile = logfile

        self.F = utils.files.FileManager(save_path)

        self.log_time = time.time()

        create_translator(self.save_path)

        print(f"| {'time':^25} | {'n started':^15} | {'n finished':^15} | {'best val acc (%) (n)':^20} | {'best bit acc (%) (n)':^20} |")
        print("-" * 111)

        if self.logfile:
            with open(self.logfile, 'a') as myfile:
                myfile.write(','.join(['time','n_ensemble','n_training','n_finished','max_val_acc','max_n','max_bit_acc','max_bit_acc_bit'"\n"]))

            self.make_report()

    def make_report(self):

        string, listinfo = evaluate_training_progress(self.save_path)

        now_time = time.time()
        time_delta = now_time - self.log_time

        # print('==========')
        # print(string)
        # print(self.string)
        # print(time_delta)
        # print('==========')

        if (string != self.string): # and (time_delta > MIN_LOG_INTERVAL):

            self.string = string

            #--- get the ensemble vote and safe it:
            X = ensemble_vote(self.F)
            bit_accuracies = get_bit_accuracies(X)
            strftime = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
            np.savez(f'{self.F.hist_path}/X_{strftime}.npz', bit_accuracies=bit_accuracies)

            #--- find the maximum single bit accuracy:
            try:
                weakest_bit = np.nanargmax(bit_accuracies)
                weakest_bit_acc = bit_accuracies[weakest_bit]
            except ValueError: # if all bit_accuracies are np.nan the np.nanargmax will throw a value error. catch it:
                weakest_bit = None
                weakest_bit_acc = 0.5

            #--- append and print the log info:
            str_best_bit_acc = f'{weakest_bit_acc * 100:.10f} ({weakest_bit})'
            string += f'{str_best_bit_acc:^20} |'
            listinfo += [f'{weakest_bit_acc * 100:.10f}', f'{weakest_bit}'"\n"]

            print(string)

            if self.logfile:
                with open(self.logfile, 'a') as myfile:
                    myfile.write(','.join(listinfo))

            self.log_time = time.time()

    def on_created(self, event):
        self.make_report()

def get_bit_accuracies(X):
    """
    X = ensemble_vote(F)
    """
    import warnings
    # I expect to see RuntimeWarnings in this block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bit_accuracies = np.nanmean(X, axis=0)
    return bit_accuracies

def ensemble_vote(F):
    #-----------------------------------------
    # create the translator file
    #-----------------------------------------
    # create an empty array of the correct dimensions
    n_ensemble = F.count_files('/cfg/')
    config = toml.load(F.filename_of_config(0))
    n_bits  = config['train_info']['n_bits']
    X = np.empty((n_ensemble, n_bits))
    X[:] = np.NaN

    # load the results of all npz files:
    data = np.load(F.filename_of_npz_translator())
    translator = data['translator']

    n_active_ensemble = 0

    for cfg_index in range(n_ensemble):
        bit_indices = translator[cfg_index]
        try:
            data = np.load(F.filename_of_validation_npz(cfg_index))
            X[cfg_index][bit_indices] = data['X']
            n_active_ensemble += 1
        except:
            pass
    #-----------------------------------------
    return X

def create_translator(save_path):
    # -----------------------------------------
    # create the translator file
    # -----------------------------------------
    # The translator file contains a numpy array with all h_indices.
    # Row `x` of the translator array contains the h_indices of ensemble member `x`.

    F = utils.files.FileManager(save_path)

    # First create an empty translator array of the right shape:
    n_ensemble = F.count_files('/cfg/')
    config = toml.load(F.filename_of_config(0))
    translator = np.ones((n_ensemble, len(config['bit_ids']['h_ids'])), dtype=int)
    print(translator.shape)
    # Fill the translator array with the `h_indices` of each ensemble member:
    for cfg_index in range(n_ensemble):
        config = toml.load(F.filename_of_config(cfg_index))
        h_bits = config['bit_ids']['h_ids']
        translator[cfg_index] = h_bits

    # Save the translator array:
    filename_translator = F.filename_of_npz_translator()
    np.savez(filename_translator, translator=translator)
    # -----------------------------------------

def evaluate_training_progress(save_path):
    """
    This function is used to quantify the training progress of the ensemble in `save_path` and gives report on it in form of a string.
    """

    F = utils.files.FileManager(save_path)

    n_ensemble = F.count_files('/cfg/')

    str_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # -------
    # NUMBER OF MODELS THAT STARTED TRAINING
    # -------
    # --- Create stores ---#
    # storage for ensemble member `n` with highest validation accuracy `acc`:
    max_n_acc = [None, 0.5]
    max_bit_acc = [None, 0.5]
    weakest_bit_acc = 0.5
    weakest_bit = None
    # storage for number of ensemble members `n_training` which have started training:
    n_training = 0

    # --- Evaluate training progress ---#
    # for each ensemble index `n`, check if a training progress file already exists.
    # if it does, extract the maximum validation accuracy and if it is the highest one, store it.
    for n in range(n_ensemble):
        try:
            df = pd.read_csv(F.filename_of_training_progress(n), sep=';')
            #--- find maximum validation binary accuracy:
            this_value = df.val_binary_accuracy.max()
            if this_value > max_n_acc[1]:
                max_n_acc = [n, this_value]
            #--- counter up
            n_training += 1
        except:
            pass

    # -------
    # NUMBER OF MODELS THAT FINISHED TRAINING
    # -------
    n_finished = len(glob.glob(F.filename_of_model('*')))

    # -------
    # OUTPUT STRING
    # -------
    # string = f"""{time} \t {n_training}/{n_ensemble}\t ensemble members have started training
    #             \t {n_finished}/{n_ensemble}\t ensemble members have finalized training
    #             \t {max_n_acc[1] * 100:.10f}%\t best validation accuracy of ensemble member {max_n_acc[0]}
    #             \t {max_bit_acc[1] * 100:.10f}%\t best single bit validation accuracy of ensemble member {max_bit_acc[0]}"""
    str_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    str_started = f'{n_training}/{n_ensemble}'
    str_finished = f'{n_finished}/{n_ensemble}'
    str_best_val_acc = f'{max_n_acc[1] * 100:.10f} ({max_n_acc[0]})'
    string = f"| {str_time:^25} | {str_started:^15} | {str_finished:^15} | {str_best_val_acc:^20} | "

    percent_finished = 100 * n_finished / n_ensemble

    list_info = [str_time, str(n_ensemble), str(n_training), str(n_finished), f'{max_n_acc[1] * 100:.10f}', f'{max_n_acc[0]}']

    return string, list_info


# percent_finished = 0.0
# N_SECONDS = 5
# print(f'The training progress will be evaluated every {N_SECONDS}s...')
#
# while percent_finished != 100.0:
#     print(evaluate_training_progress(F))
#     if percent_finished == 100:
#         break
#     else:
#         time.sleep(N_SECONDS)

# quit()