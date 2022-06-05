from watchdog.events import FileSystemEventHandler
import time
import datetime
import glob
import pandas as pd
import toml
import numpy as np

from .filemanager import FileManager
from .bitanalysis import BitAnalysis

class TrainingTracker(FileSystemEventHandler):
    """
    This subclass of the watchdog FileSystemEventHandler is used to track the training progress.
    It will look for new files created in the `save_path`.
    Once a file is created, it will generate a report with help of the `evaluate_training_progress` function.
    """

    def __init__(self, savepath, logfile=None):
        super().__init__()

        self.string = ''
        self.logfile = logfile

        self.F = FileManager(savepath)
        config = toml.load(self.F.filename_config())
        self.n_bits = config['RESULTING N TOTAL BITS']
        self.n_ensemble = np.sum(config['NEURAL_NETWORKS'])

        self.bitanalysis = BitAnalysis(savepath)

        self.log_time = time.time()

        # --- print string
        self.string = f"""|| {'time':^25} | {'NN finished':^15} | {'pred. bits':^10} || {'best bit':^10} | {'acc (%)':^10} | {'n pred':^10} | {'p value':^10} ||"""
        print(self.string)
        self.string = "=" * len(self.string)
        print(self.string)

        self.df_dict = []

        self.make_report()

    def evaluate(self):

        # '2022-04-21_10h26m29s'
        self.now = datetime.datetime.now()
        self.strf_time = self.now.strftime('%Y-%m-%d_%Hh%Mm%Ss')

        self.bitanalysis.update()

        # n finished 36/100
        self.n_finished = len(glob.glob(self.F.filename_accs('*')))

        # unpredicted bits
        self.n_pred = len(np.where(self.bitanalysis.bit_npred != 0)[0])

        # best bit acc
        try:
            self.best_bit = self.bitanalysis.bit_ids_high_acc[0]
            self.best_acc = self.bitanalysis.bit_accs[self.best_bit]
            self.best_npred = self.bitanalysis.bit_npred[self.best_bit]
            self.best_pval = self.bitanalysis.get_pvalue_of_bit_id(self.best_bit)
        except ValueError:
            self.best_bit = np.nan
            self.best_acc = np.nan
            self.best_npred = np.nan
            self.best_pval = np.nan

    def make_report(self):

        self.evaluate()

        strf_finished = f'{self.n_finished}/{self.n_ensemble}'
        strf_npred = f'{self.n_pred}/{self.n_bits}'

        strf_best_acc = f'{self.best_acc * 100:.3f}'
        strf_best_p = f'{self.best_pval:.2g}'

        #         if self.logfile:
        #             with open(self.logfile, 'a') as myfile:
        #                 myfile.write(','.join(listinfo))

        string = f"""|| {self.strf_time:^25} | {strf_finished:^15} | {strf_npred:^10} || {self.best_bit:^10} | {strf_best_acc:^10} | {self.best_npred :^10} | {strf_best_p:^10} ||"""

        if string != self.string:
            print(string)
            self.string = string

        self.df_dict.append({'time': self.now,
                             'NN finished': self.n_finished,
                             'NN ensemble': self.n_ensemble,
                             'pred. bits': self.n_pred,
                             'bits': self.n_bits,
                             'best bit': self.best_bit,
                             'acc (%)': self.best_acc * 100,
                             'n pred': self.best_npred,
                             'p value': self.best_pval})

        if self.logfile:
            df = pd.DataFrame(self.df_dict)
            df.to_pickle(self.logfile)

    def on_created(self, event):
        self.make_report()