#### CONFIGURATION ######

number_of_samples = 10**6
number_of_rounds = 50

#########################

import speck_k144_p96_o96_r29 as data_generator
import numpy as np
import time
from datetime import timedelta

number_of_samples_per_round = int(number_of_samples / number_of_rounds)

for i in range(number_of_rounds):
    t = time.time()
    dataset=data_generator.generate_avalanche_dataset(number_of_samples_per_round)
    for r in range(len(dataset)):
        np.save("data_"+str(number_of_samples_per_round)+"_rounds_"+str(r)+"_samples_"+str(i)+".npy",dataset[r])
    t = time.time() - t
    f_out = open("execution_time.txt", "a")
    f_out.write(str(i)+": "+str(timedelta(seconds=t))+"\n")
    f_out.close()
