#### CONFIGURATION ######

number_of_samples = 10**6

#########################

import aes_block_cipher_k128_p128_o128_r10 as data_generator
import numpy as np
import time
from datetime import timedelta

loops = 50
for i in range(50):
    t = time.time()
    dataset=data_generator.generate_avalanche_dataset(int(number_of_samples/50))
    for r in range(len(dataset)):
        np.savez_compressed("dataset_aes_128_128_"+"_round_"+str(r)+'_'+str(number_of_samples)+"_samples_"+str(i),dataset=dataset[r])
    t = time.time() - t
    f_out = open("execution_time.txt", "a")
    f_out.write(str(timedelta(seconds=t))+"\n")
    f_out.close()
