import os


class FileManager:
    """ This class simply stores filenames for convenience."""

    def __init__(self, savepath):

        self.savepath = savepath

        # create sub-directories if not available:
        make_paths = ['h5', 'hist', 'bit_selections', 'test_accuracies', 'test_accuracies_bit_by_bit']

        for path in make_paths:
            setattr(self, f'path_{path}', f'{self.savepath}/{path}')

            try:
                os.makedirs(f'{self.savepath}/{path}')
            except FileExistsError:
                pass

    def filename_run_log(self):
        return f'{self.savepath}/run_log.pkl'

    def filename_config(self):
        return f'{self.savepath}/config.cfg'

    def filename_h5(self, network_id):
        return f'{self.savepath}/h5/{network_id}.h5'

    def filename_history(self, network_id):
        return f'{self.savepath}/hist/{network_id}.pkl'

    def filename_selections(self, run=0):
        return f'{self.savepath}/bit_selections/selection{run}.npz'

    def filename_accs(self, network_id):
        return f'{self.savepath}/test_accuracies/{network_id}.npy'
    
    def filename_bitbybitaccs(self, network_id):
        return f'{self.savepath}/test_accuracies_bit_by_bit/{network_id}.npy'