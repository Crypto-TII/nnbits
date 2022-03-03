This is the avalanche dataset generator for 
* speck32/64
* speck64/128
* speck96/144
* speck128/128
* aes128/128

To use the dataset generator, please run the generateData.py under the correspoding directory
```$ python3 generateData.py```

The dataset would be saved in the numpy format. 

For example, under the speck32/64 directory, when you run `generateData.py`, it will import the speck64/128 code, which is `speck_k128_p64_o64_r27.py`, and output the dataset with filename `data_$sizeofdataset_rounds_$round_samples_$samplenumber.npy`.
There are `$sizeofdataset` samples in this file, which are all generated from the ourput of round `$round`, and this is the  means the dataset is the `$samplenumber`th sample file of this round.

In `genearateData.py`, there is several parameter you can change:
`number_of_samples` indicates how many samples to genearate, for example, here are total 10**6 samples need to be generated.
`number_of_rounds` indicates that how many loops to generate the samples, for example, here we use 50 loops to generate samples, so total there would be 50 files for each round, and in each files it contains 10**6/50=20000 samples in one file.
