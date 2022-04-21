import numpy as np

class FilterManager(object):

    def __init__(self, N_BITS):
        """
        Example for Speck 32:
            N_BITS = 1024
            N_FILTERS = [24, 24, 24, 24]
            N_INPUTS = [784, 841, 900, 961]

        The filter manager will create 4 rounds of filters in this example.
        In each round 24 filters are created.
        On the input of each filter, N_INPUTS[run_id] of the N_BITS will be chosen.

        run_id = 0: random choice, however with assuring that each bit is predicted the approximate same number of times
        run_id > 0: interesting bits of the previous run are identified and
                    passed to the FilterManager as weak_bit_ids and strong_bit_ids.
        """

        self.N_BITS = N_BITS

    def create_random_filters(self, n_filters, n_input_bits, make_uniform=True):
        """make_uniform: ensure that all indices in N_BITS occur approximately the same number of times."""

        n_output_bits = self.N_BITS - n_input_bits

        input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
        output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

        p = np.ones((self.N_BITS))
        p = p / self.N_BITS

        for i in np.arange(n_filters):

            all_bit_ids = np.arange(self.N_BITS)

            input_filter = np.random.choice(all_bit_ids, size=n_input_bits, replace=False, p=p)

            input_filters[i] = np.sort(input_filter)
            output_filters[i] = np.delete(all_bit_ids, input_filters[i])

            if make_uniform:
                counts = np.zeros(self.N_BITS, dtype=np.float32)
                e, c = np.unique(input_filters[0:i + 1, :].flatten(), return_counts=True)
                counts[e] = c
                counts = np.abs(max(counts) - 0.01 - counts)
                p = counts / np.sum(counts)  # normalize counts to 1 to obtain probability

        input_filters = input_filters.astype(int)
        output_filters = output_filters.astype(int)

        return input_filters, output_filters

    def create_target_filters(self, n_filters, n_input_bits, list_of_target_bits):

        # list_of_target_bits = list_of_target_bits.tolist()

        n_output_bits = self.N_BITS - n_input_bits

        input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
        output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

        all_bit_ids = np.arange(self.N_BITS)

        # 1 bit is chosen by the `list_of_target_bits`. Choose the rest randomly.
        n_random_choice = n_output_bits - 1

        for i in np.arange(n_filters):
            if n_random_choice > 0:
                random_choice = np.random.choice(all_bit_ids, size=n_random_choice, replace=False)
            else:
                random_choice = []
            input_filter = np.delete(all_bit_ids, list(random_choice) + [list_of_target_bits[i]])
            input_filters[i] = np.sort(input_filter)
            output_filters[i] = np.delete(all_bit_ids, input_filters[i])

        input_filters = input_filters.astype(int)
        output_filters = output_filters.astype(int)

        return input_filters, output_filters

    def create_gohr_with_target_filters(self, n_filters, n_input_bits, list_of_target_bits):

        n_output_bits = 1

        input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
        output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

        all_bit_ids = np.arange(n_input_bits)

        for i in np.arange(n_filters):
            # input_filter = np.delete(all_bit_ids, [list_of_target_bits[i]])
            input_filters[i] = np.array(list_of_target_bits[i])
            output_filters[i] = np.array([n_input_bits])  # the last bit is the one to be at the output

        input_filters = input_filters.astype(int)
        output_filters = output_filters.astype(int)

        return input_filters, output_filters

    def create_weakbit_filters(self, n_filters, n_input_bits, weak_bit_ids, strong_bit_ids):

        n_output_bits = self.N_BITS - n_input_bits

        input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
        output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

        N_WEAK = int(n_output_bits * 1 / 2 * 6 / 4)
        N_STRONG = n_output_bits - N_WEAK

        for i in np.arange(n_filters):
            all_bit_ids = np.arange(self.N_BITS)

            # p = np.arange(1,len(weak_bit_ids)+1)
            # p = p / np.sum(p)

            output_filter = list(np.random.choice(weak_bit_ids, size=(N_WEAK), replace=False)) + \
                            list(np.random.choice(strong_bit_ids, size=(N_STRONG), replace=False))
            output_filter = np.array(output_filter)
            output_filter = np.sort(output_filter)
            output_filters[i] = output_filter
            input_filters[i] = np.delete(all_bit_ids, output_filters[i])

        input_filters = input_filters.astype(int)
        output_filters = output_filters.astype(int)

        return input_filters, output_filters