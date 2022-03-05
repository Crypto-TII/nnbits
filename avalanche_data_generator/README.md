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
Running `generateData.py` will import the speck64/128 code, which is `speck_k128_p64_o64_r27.py` and generate the dataset. 
The dataset would be saved in the numpy format with filename `data_$amount_rounds_$round_samples_$idx.npy`.
There are `$amount` samples of round `$round` in this file. It is the `$idx`-th file of this round. The `$round` is started from 0. 

In `genearateData.py`, there are several parameters you can change:
* `number_of_samples` indicates how many samples to genearate, for example, here are total 10**6 samples need to be generated.
* `number_of_rounds` indicates that how many loops to generate the samples, for example, here we use 50 loops to generate samples, so total there would be 50 files for each round, and in each files it contains 10**6/50=20000 samples in one file.
