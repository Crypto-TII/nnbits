import numpy as np

from nnbits.filemanager import FileManager
from nnbits.bitanalysis import get_X
import warnings

# the file manager knows where to find all files in the working directory:
F = FileManager('demo_speck32_round7') 

# get the accuracies for all networks on all single bits:
X = get_X(F)

# the bit accuracies are calculated by taking the mean over all neural networks which predict the particular bit 
with warnings.catch_warnings(): # we expect the following operation to throw a warning
    warnings.simplefilter("ignore", category=RuntimeWarning)
    bit_accuracies = np.nanmean(X, axis=0)
# find the best bit
best_bit_id = np.nanargmax(bit_accuracies)

#### Create a figure ######
import matplotlib.pyplot as plt

# here we visualize the obtained `bit_accuracies`:
plt.figure(figsize=(10, 3), dpi=150)
plt.plot(bit_accuracies, 'o', markersize=3, linestyle = 'None', label="mean validation accuracy of each bit")
# here we visualize the best bit
plt.plot(best_bit_id, bit_accuracies[best_bit_id], marker='x', c='C0', linestyle = 'None', label=f"best bit {int(best_bit_id)} with {float(bit_accuracies[best_bit_id])*100:.1f}% accuracy")

plt.legend(loc='upper left')
plt.xlabel('bit id')
plt.ylabel('validation accuracy')
plt.gcf().savefig('./result.png', bbox_inches='tight')