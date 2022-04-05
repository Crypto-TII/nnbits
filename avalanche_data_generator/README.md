This is the avalanche dataset generator for 
* speck32/64
* speck64/128
* speck96/144
* speck128/128
* aes128/128

To use the dataset generator, please run the generateData.py under the corresponding directory as followed example:
```
$ cd speck_32_64
$ python3 generateData.py
```
Running `generateData.py` will import the speck64/128 code, `speck_k128_p64_o64_r27.py`, and generate the dataset. 
The dataset would be saved in the compressed numpy format with filename `dataset_$cipher__round_$round_$amount_samples.npz`.
There are `$amount` samples of round `$round` generated in this file. The execution time would be recorded in file `execution_time.txt`

In `genearateData.py`, you can set up the parameter::
* `number_of_samples` indicates how many samples to generate, for example, here are total 10**6 samples need to be generated.

To use the dataset, please load the file:
```
import numpy as np
loaded = np.load(filename)
dataset = loaded['dataset']
```
