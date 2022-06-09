import numpy as np

def random(n_selections, n_selected_bits, n_total, make_uniform=True, **kwargs):
    """make_uniform: ensure that all indices in N_BITS occur approximately the same number of times.

    Example:
        selected, not_selected = selections.random(3, 2, 10)
        selected, not_selected

        (array([[6, 9],
        [2, 4],
        [5, 8]]),
         array([[0, 1, 2, 3, 4, 5, 7, 8],
                [0, 1, 3, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 6, 7, 9]]))

    """

    selected_bits = np.zeros((n_selections, n_selected_bits), dtype=int)
    not_selected_bits = np.zeros((n_selections, n_total-n_selected_bits), dtype=int)

    p = np.ones((n_total))
    p = p / n_total

    for i in np.arange(n_selections):

        all_bit_ids = np.arange(n_total)

        selected_bits[i] = np.random.choice(all_bit_ids, size=n_selected_bits, replace=False, p=p)
        selected_bits[i] = np.sort(selected_bits[i])
        not_selected_bits[i] = np.delete(all_bit_ids, selected_bits[i])

        if make_uniform:
            counts = np.zeros(n_total, dtype=np.float32)
            e, c = np.unique(selected_bits[0:i + 1, :].flatten(), return_counts=True)
            counts[e] = c
            counts = np.abs(max(counts) - 0.01 - counts)
            p = counts / np.sum(counts)  # normalize counts to 1 to obtain probability

    selected_bits = selected_bits.astype(int)
    not_selected_bits = not_selected_bits.astype(int)

    return selected_bits, not_selected_bits

def target(n_selections, n_selected_bits, n_total, list_of_target_bits=None, **kwargs):
    """ A single bit may be chosen as target for each of the n_filters.
    The target bit will appear on the output side with 100% probability.

    Example:
        selected, not_selected = selections.target(5, 1, 10, list_of_target_bits = [0,1,2,3,4,5,6,7,8,9])

        (array([[0],
        [1],
        [2],
        [3],
        [4]]),
         array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 5, 6, 7, 8, 9]]))
    """

    # list_of_target_bits = list_of_target_bits.tolist()

    #n_bits = n_selected_bits + n_not_selected_bits

    selected_bits = np.zeros((n_selections, n_selected_bits), dtype=int)
    not_selected_bits = np.zeros((n_selections, n_total-n_selected_bits), dtype=int)

    all_bit_ids = np.arange(n_total)

    # 1 bit is chosen by the `list_of_target_bits`. Choose the rest randomly.
    # n_random_choice = n_output_bits - 1

    for i in np.arange(n_selections):
        # if n_random_choice > 0:
        #     random_choice = np.random.choice(all_bit_ids, size=n_random_choice, replace=False)
        # else:
        #     random_choice = []
        # input_filter = np.delete(all_bit_ids, list(random_choice) + [list_of_target_bits[i]])
        target_bits = [list_of_target_bits[i]] if isinstance(list_of_target_bits[i], int) else list_of_target_bits[i]
        selected_bits[i] = target_bits # np.sort(input_filter)
        not_selected = np.delete(all_bit_ids, target_bits)
        not_selected_bits[i] = np.sort(not_selected)# np.delete(all_bit_ids, selected_bits[i])

    selected_bits = selected_bits.astype(int)
    not_selected_bits = not_selected_bits.astype(int)

    return selected_bits, not_selected_bits

# def gohr_with_target(n_filters, n_input_bits, n_output_bits, list_of_target_bits=None, **kwargs):
#     """
#     Here, the input filter contains one bit to be zero-ed in the `list_of_target_bits`
#
#     0 1 2 3 ... 63 | 64
#     |____________|    |
#         bit ids     label
#         ^
#     set to zero
#     """
#
#     assert n_output_bits == 1, 'For this filter type the number of output bits has to be 1.'
#     assert list_of_target_bits is not None, 'Please provide a list of target bits.'
#
#     input_filters = np.zeros((n_filters, n_input_bits), dtype=int)
#     output_filters = np.zeros((n_filters, n_output_bits), dtype=int)
#
#     for i in np.arange(n_filters):
#         input_filters[i] = np.array(list_of_target_bits[i])
#         output_filters[i] = np.array([n_input_bits])  # the last bit is the one to be at the output
#
#     input_filters = input_filters.astype(int)
#     output_filters = output_filters.astype(int)
#
#     return input_filters, output_filters