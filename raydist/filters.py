import numpy as np

def random(n_filters, n_input_bits, n_output_bits, make_uniform=True, **kwargs):
    """make_uniform: ensure that all indices in N_BITS occur approximately the same number of times."""

    n_bits = n_input_bits + n_output_bits

    input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
    output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

    p = np.ones((n_bits))
    p = p / n_bits

    for i in np.arange(n_filters):

        all_bit_ids = np.arange(n_bits)

        input_filter = np.random.choice(all_bit_ids, size=n_input_bits, replace=False, p=p)

        input_filters[i] = np.sort(input_filter)
        output_filters[i] = np.delete(all_bit_ids, input_filters[i])

        if make_uniform:
            counts = np.zeros(n_bits, dtype=np.float32)
            e, c = np.unique(input_filters[0:i + 1, :].flatten(), return_counts=True)
            counts[e] = c
            counts = np.abs(max(counts) - 0.01 - counts)
            p = counts / np.sum(counts)  # normalize counts to 1 to obtain probability

    input_filters = input_filters.astype(int)
    output_filters = output_filters.astype(int)

    return input_filters, output_filters

def target(n_filters, n_input_bits, n_output_bits, list_of_target_bits=None, **kwargs):
    """ A single bit may be chosen as target for each of the n_filters.
    The target bit will appear in the output filter with 100% probability.
    If there is more than one output bit, the rest of the bits will be chosen randomly.
    """

    # list_of_target_bits = list_of_target_bits.tolist()

    n_bits = n_input_bits + n_output_bits

    input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
    output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

    all_bit_ids = np.arange(n_bits)

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

def gohr_with_target(n_filters, n_input_bits, n_output_bits, list_of_target_bits=None, **kwargs):
    """
    Here, the input filter contains one bit to be zero-ed in the `list_of_target_bits`

    0 1 2 3 ... 63 | 64
    |____________|    |
        bit ids     label
        ^
    set to zero
    """

    assert n_output_bits == 1, 'For this filter type the number of output bits has to be 1.'
    assert list_of_target_bits is not None, 'Please provide a list of target bits.'

    input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
    output_filters = np.zeros((n_filters, n_output_bits), dtype=int)

    for i in np.arange(n_filters):
        input_filters[i] = np.array(list_of_target_bits[i])
        output_filters[i] = np.array([n_input_bits])  # the last bit is the one to be at the output

    input_filters = input_filters.astype(int)
    output_filters = output_filters.astype(int)

    return input_filters, output_filters

# def weakbit(self, n_filters, n_input_bits, weak_bit_ids, strong_bit_ids):
#
#     n_output_bits = self.N_BITS - n_input_bits
#
#     input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
#     output_filters = np.zeros((n_filters, n_output_bits), dtype=int)
#
#     N_WEAK = int(n_output_bits * 1 / 2 * 6 / 4)
#     N_STRONG = n_output_bits - N_WEAK
#
#     for i in np.arange(n_filters):
#         all_bit_ids = np.arange(self.N_BITS)
#
#         # p = np.arange(1,len(weak_bit_ids)+1)
#         # p = p / np.sum(p)
#
#         output_filter = list(np.random.choice(weak_bit_ids, size=(N_WEAK), replace=False)) + \
#                         list(np.random.choice(strong_bit_ids, size=(N_STRONG), replace=False))
#         output_filter = np.array(output_filter)
#         output_filter = np.sort(output_filter)
#         output_filters[i] = output_filter
#         input_filters[i] = np.delete(all_bit_ids, output_filters[i])
#
#     input_filters = input_filters.astype(int)
#     output_filters = output_filters.astype(int)
#
#     return input_filters, output_filters