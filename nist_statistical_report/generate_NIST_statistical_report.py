# Please run this script under the statistical test tool directory

import sys
import os
import time
import pathlib
from datetime import timedelta
import shutil
import numpy as np
from math import ceil

############### CONFIGURATION ####################
dataset_directory_path = '/opt/cryptanalysis_servers_shared_folder/NBEATSCipherDistinguisher/D5/speck_32_64/'
cipher_name = "speck_32_64"
block_size = 32
round_start = 0
round_end = 22
seq_num = 300
seq_len = 1048576
data_segments = 50
############# END CONFIGURATION ##################


sample_size = block_size*block_size
data_amount_in_one_seq = ceil(seq_len/sample_size)

statistics_report_folder = 'nist_statistics_report_'+cipher_name+'_'+str(seq_num)+'seq_'+str(seq_len)+'bits/'
try:
    os.mkdir(statistics_report_folder)
except OSError as e:
    print("The statistic report folder had existed already. Please state a new folder")
    sys.exit(0)

rounds = [i for i in range(round_start, round_end)]
for r in rounds:
    dataset_filename = 'nist_input.txt'
    statistics_report_folder_round = statistics_report_folder+'round_'+str(r)+'/'

    # inistialize the directory enviorment
    try:
        shutil.rmtree('experiments/')
    except OSError as e:
        print("experiments had been removed.")
	
    # prepare data
    merge_data_time = time.time()
    count = 0
    max_count = (seq_num+1)*(data_amount_in_one_seq+1)
    data = ""
    for s in range(data_segments):
        numpy_filename = dataset_directory_path + "dataset_"+cipher_name+"__round_"+str(r)+"_20000_samples_"+str(s)+".npz"
        array = np.load(numpy_filename)	
        array = array['dataset']
        for sample in array:
            for i in sample:
                data += str(i)
            count += 1
            if count == max_count:
                break
        if count == max_count:
            break
    f_out = open(dataset_filename, "w")
    f_out.write(data)
    f_out.close()
    merge_data_time = time.time() - merge_data_time

    # run statistics 
    nist_statistics_report_time = time.time()

    for dname in [ "AlgorithmTesting", "BBS", "CCG", "G-SHA1", "LCG", "MODEXP", "MS", "QCG1", "QCG2", "XOR" ]:
        path_prefix = "experiments/" + dname + "/"
        pathlib.Path(path_prefix + "Frequency").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "BlockFrequency").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "Runs").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "LongestRun").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "Rank").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "FFT").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "NonOverlappingTemplate").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "OverlappingTemplate").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "Universal").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "LinearComplexity").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "Serial").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "ApproximateEntropy").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "CumulativeSums").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "RandomExcursions").mkdir(parents=True, exist_ok=False, mode=0o777)
        pathlib.Path(path_prefix + "RandomExcursionsVariant").mkdir(parents=True, exist_ok=False, mode=0o777)
    output_code = os.system("./assess " + \
                            str(data_amount_in_one_seq*sample_size) + " " + \
                            dataset_filename + " " + \
                            str(seq_num) + " " + \
                            '0' + " " + \
                            "experiments/" + " " + \
                            15 * '1')
    os.rename('experiments/', statistics_report_folder_round)
    nist_statistics_report_time = time.time() - nist_statistics_report_time

    f_out = open(statistics_report_folder+"execution_time.txt", "a")
    f_out.write("round " + str(r) + ": merge data " + str(timedelta(seconds=merge_data_time)) + ", compute nist statistics report " + str(timedelta(seconds=nist_statistics_report_time)) + "\n")
    f_out.close()
