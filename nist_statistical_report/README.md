This is the report generator of the nist statistical tools for 
* speck32/64
* speck64/128
* speck96/144
* speck128/128
* aes128/128

First, please install the NIST STS library following the webpage instruction:
https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final

After installing the library, there should be an `asset` execution file at the root directory.

To generate the NIST statistical report with our dataset, please, put the `generate_NIST_statistical_report.py` file under the root directory of the nist statistical libry and execute:
```
$ python3 generate_NIST_statistical_report.py
```
Running `generate_NIST_statistical_report.py` will import the dataset generated before and call the `asset` execution file to run the statistical test. 

In `generate_NIST_statistical_report.py`, you can set up the parameters::
* `dataset_directory_path`: the path of the dataset
* `cipher_name`: name of the cipher
* `block_size`: block size of the cipher
* `round_start`: the round to start the statistical test (included, start from 0)
* `round_end`: the round to end the statistical test (excluded)
* `seq_num`: number of sequences for the statistical test
* `seq_len`: length of one sequence for the statistical test
* `data_segments`: how many segments in the dataset (default 50, depends on the loops when generates the dataset)
