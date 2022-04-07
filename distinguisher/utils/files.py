import os
import glob
import shutil

class FileManager:
    """
    Manages filenames and folders.
    """
    def __init__(self, save_path):

        self.save_path = save_path
        self.cfg_path = f'{save_path}/cfg'
        self.hist_path = f'{save_path}/hist'
        self.h5_path = f'{save_path}/h5'
        self.pred_path = f'{save_path}/pred'

    def create_folders(self):
        """ creates folders """
        _my_paths = [self.save_path
                    , self.cfg_path
                    , self.hist_path
                    , self.h5_path
                    , self.pred_path]

        # Create the directories defined above:
        for path in _my_paths:

            try:
                os.makedirs(path)
            except FileExistsError:
                pass

    def filename_of_log_ensemble_training(self):
        return f'{self.save_path}/log_ensemble_training.csv'

    def filename_of_BitwiseStatTest_df(self):
        """ stores filename """
        return f'{self.save_path}/BitwiseStatTest_df.pkl'

    def filename_of_config(self, index):
        """ stores filename """
        return f'{self.cfg_path}/cfg_{index}.cfg'

    def filename_of_training_history(self, index):
        """ stores filename """
        return f'{self.hist_path}/training_history_{index}.pkl'

    def filename_of_validation_history(self, index):
        """ stores filename """
        return f'{self.hist_path}/validation_history_{index}.pkl'

    def filename_of_training_info(self, index):
        """ stores filename """
        return f'{self.hist_path}/training_info_{index}.csv'

    def filename_of_training_progress(self, index):
        """ stores filename """
        return f'{self.hist_path}/training_progress_{index}.csv'

    def filename_of_validation_npz(self, index):
        """ stores filename """
        return f'{self.hist_path}/validation_{index}.npz'

    def filename_of_npz_translator(self):
        """ load the translator vian
        >>> T = np.load(F.filename_of_npz_translator())['translator'] """
        return f'{self.hist_path}/translator.npz'

    def filename_of_model(self, index):
        """ stores filename """
        return f'{self.h5_path}/model_{index}.h5'

    def filename_of_predictions(self, index):
        """ stores filename """
        return f'{self.pred_path}/pred_{index}.npz'

    def filename_of_ensemble_analysis(self):
        """ stores filename """
        return f'{self.save_path}/df_ensemble_analysis.pkl'

    def count_files(self, folder):
        if self.save_path in folder:
            path = f'{folder}/*'
        else:
            path = f'{self.save_path}/{folder}/*'
        files = glob.glob(path)
        return len(files)

    def copy_over_cfg(self, src_save_path, only_indices=[]):

        self.create_folders()

        if only_indices:
            for index in only_indices:
                shutil.copy(f'{src_save_path}/cfg/cfg_{index}.cfg', self.cfg_path)

        else:
            shutil.rmtree(self.cfg_path)
            shutil.copytree(f'{src_save_path}/cfg', self.cfg_path)

        n_cfg = self.count_files(self.cfg_path)
        print(f'Finished copying. cfg contains {n_cfg} files.')