import os
import glob

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

    def filename_of_config(self, index):
        """ stores filename """
        return f'{self.cfg_path}/cfg_{index}.cfg'

    def filename_of_training_history(self, index):
        """ stores filename """
        return f'{self.hist_path}/training_history_{index}.pkl'

    def filename_of_validation_history(self, index):
        """ stores filename """
        return f'{self.hist_path}/validation_history_{index}.pkl'

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
        files = glob.glob(folder+'/*')
        return len(files)