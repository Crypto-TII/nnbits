#### CONFIGURATION ######

number_of_samples = 10**6

#########################

import speck_k64_p32_o32_r22 as data_generator
import numpy as np
import time
from datetime import timedelta

loops = 50
for i in range(loops):
    t = time.time()
    dataset=data_generator.generate_avalanche_dataset(int(number_of_samples/loops))
    for r in range(len(dataset)):
        np.savez_compressed("dataset_speck_32_64_"+"_round_"+str(r)+'_'+str(number_of_samples)+"_samples_"+str(i),dataset=dataset[r])
    t = time.time() - t
    f_out = open("execution_time.txt", "a")
    f_out.write(str(timedelta(seconds=t))+"\n")
    f_out.close()
