#### CONFIGURATION ######

number_of_samples = 10**6

#########################

import speck_k128_p128_o128_r32 as data_generator
import numpy as np
import time
from datetime import timedelta

t = time.time()
dataset=data_generator.generate_avalanche_dataset(number_of_samples)
for r in range(len(dataset)):
    np.save("dataset_speck_128_128_"+"_rounds_"+str(r)+'_'+str(number_of_samples)+"_samples.npy",dataset[r])
t = time.time() - t
f_out = open("execution_time.txt", "a")
f_out.write(str(timedelta(seconds=t))+"\n")
f_out.close()
