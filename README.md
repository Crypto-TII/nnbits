
# NNBits: Bit Profiling with a Deep Learning Ensemble Based Distinguisher

We introduce a deep learning ensemble (NNBits) as a tool for bit-profiling and evaluation of cryptographic (pseudo) random bit sequences. 

_(This work has been submitted to FSE'22 and is currently under review)_


# NNBits: Bit Profiling with a Deep Learning Ensemble Based Distinguisher

We introduce a deep learning ensemble (NNBits) as a tool for bit-profiling and evaluation of cryptographic (pseudo) random bit sequences. 

_(This work has been submitted to FSE'22 and is currently under review)_

- [Demos](#demos)
  * [Jupyter Notebook](#jupyter-notebook)
  * [Command line interface](#command-line-interface)
- [Background information](#background-information)
  * [Methodology](#methodology)
  * [Repository structure](#repository-structure)
  * [Training output](#training-output)
  * [Working directory structure](#working-directory-structure)
  * [How to set GPU parameters](#how-to-set-gpu-parameters)
  * [How to prepare the dataset](#how-to-prepare-the-dataset)
  * [How to add a new model](#how-to-add-a-new-model)
- [Citation](#citation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


# Demos 

## Jupyter Notebook

A [demo notebook](demo.ipynb) is included in this repository. Please clone the repository and install the requirements
```bash
git clone https://github.com/Crypto-TII/nnbits
cd nnbits
pip install -r requirements.txt
```
In `conda` you can install Jupyter Lab via `conda install -c conda-forge jupyterlab` and launch Jupyter Lab via `jupyter lab`.

<!-- To open the demo in Google Colab (allows to execute Python code directly in the browser and provides one GPU and one TPU for free) please click here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Crypto-TII/nbeats_statistical_test/blob/main/demo.ipynb) -->

## Command line interface

### Clone repository and create folders
```bash
# clone repository
git clone https://github.com/Crypto-TII/nnbits
# change working directory
cd nnbits/
# install requirements
pip install -r requirements.txt
# create dataset directory
mkdir 'speck_32_64'
# create working directory for analysis results
mkdir 'demo_speck32_roundid6'
```
### Create a configuration file
Create a configuration file `config.cfg` file in the working directory (e.g. via `vim 'demo_speck32_roundid6/config.cfg'`) with the following content. 
The configuration file contains all details for the ensemble training, such as the path for the data to be analyzed (`DATAPATH`) or the number of `NEURAL_NETWORKS`. 
```python
DATAPATH = "speck_32_64/round6_sequences300k.npy"
# ensemble settings
NEURAL_NETWORK_MODEL = "gohr_generalized"
NEURAL_NETWORKS = 100
SELECT_BITS_STRATEGY = "random"
INPUT_DATA_OP = "zero"
N_RANDOM_BITS = 63
# hardware settings <------------ adjust according to your GPU hardware (check with nvidia-smi)
N_GPUS = 4 			# how many GPUs do you have available?
N_ACTORS_PER_GPU = 4		# divide the GPU memory by ~3800 MiB for training a generalized Gohr's network on the avalanche dataset of Speck32/64
GPU_PER_ACTOR = 0.25		# <= 1/N_ACTORS_PER_GPU
CPU_PER_ACTOR = 5		# depends on your CPU cores << N_CPU_CORES / N_ACTORS
# training settings
N_EPOCHS = 5
N_TRAIN = 145000
N_VAL = 145000
N_TEST = 0
BATCHSIZE = 5000
```

### Create the dataset
(Python)
```python 
#### Create the dataset ######
number_of_samples = 300_000

from avalanche_data_generator.speck_32_64 import speck_k64_p32_o32_r22 as data_generator
import numpy as np

dataset = data_generator.generate_avalanche_dataset(int(number_of_samples))

np.save(f"speck_32_64/round6_sequences300k.npy", dataset[6])
```
### Run the NNBits analysis
(bash)
```bash
python -m nnbits.run --savepath 'demo_speck32_roundid6'
```

### Analyze the outcome 
(Python)
```python 
import numpy as np

from nnbits.filemanager import FileManager
from nnbits.bitanalysis import get_X

# the file manager knows where to find all files in the working directory:
F = FileManager('demo_speck32_roundid6') 

# get the accuracies for all networks on all single bits:
X = get_X(F)

# the bit accuracies are calculated by taking the mean over all neural networks which predict the particular bit 
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
plt.gcf().savefig('result.png', bbox_inches='tight')
```
![image](https://user-images.githubusercontent.com/73515327/191898706-0b18f16a-e04d-4711-8222-8fdb0a3e7ff0.png)

# Background information

## Methodology 

An ensemble of deep neural networks is trained and tested on a `*.npy` file which contains sequences of potential random data.  

1. Each ensemble member is a neural network with a unique bit `selection`: The respective bit selection will define some bits of the sequence as inputs, and the remaining bits as outputs of the neural network. The input bits will be set to zero at the input of the neural network. The neural network will be trained to predict the output bits. The number of selections, and therefore ensemble members is defined in the `*.cfg` configuration file. 
2. Each ensemble member is trained on the training data as defined in the `*.cfg` file. 
3. Each ensemble member is tested on the test data as defined in the `*.cfg` file. 

## Repository structure 
This repository contains the following files:
```
nnbits
|   |   README.md               <-  the file which generates the current view
|   |_
|_  demo.ipynb                  <-  demo notebook
|_  nnbits
    |_  run.py                  <-  run the ensemble distinguisher (see `demo.ipynb`) 
    |_  selections.py           <-  generates bit selections, see [Methodology](#methodology) 
    |_  metric.py               <-  defines a bit-by-bit accuracy as metric
    |_  network.py              <-  handles routines for the deep-learning models in folder `models`
    |_  models                  <-  contains the following deep learning models
        |_  gohr_generalized.py     <-  a generalized version of Gohr's network
        |_  resnet50.py             <-  ResNet50 implementation 
        |_  vgg16.py                <-  VGG-16 implementation
        |_  nbeats.py               <-  N-BEATS network 
    |_  trainingtracker.py      <-  keeps track of the ensemble training progress
    |_  filemanager.py          <-  keeps track of filenames
```

## Training output 

The output gives the following information:
```
====================================
speck_32_64/round0_sequences300k.npy
||           time            |   NN finished   | pred. bits ||  best bit  |  acc (%)   |   n pred   |  p value   ||
===================================================================================================================
||   2022-05-19_12h33m59s    |      0/100      |   0/1024   ||    nan     |    nan     |    nan     |    nan     ||
||   2022-05-19_12h34m41s    |      1/100      |  63/1024   ||    143     |  100.000   |     1      |     0      ||
||   2022-05-19_12h34m41s    |      3/100      |  122/1024  ||    237     |  100.000   |     1      |     0      ||
...
||   2022-05-19_12h34m42s    |     16/100      |  762/1024  ||    511     |  100.000   |     1      |     0      ||
p-value is below limit ==> stop analysis.
```
Topmost is the `*.npy` file which has been analyzed by the ensemble. 
The tabular output gives the following information in real-time during the training of the ensemble:
* The `time` column gives a timestamp for the row and the rest of the row indicates the ensemble training status. 
* `NN finished` is the neural networks which have already finalized their training. 
*  `pred. bits` indicates how many bits of the total unit length were already present at the output of the `NN finished`. For example the avalanche unit of Speck 32 has a length of `1024` bits and in the last timesteps `762/1024` of those bits had been predicted by one of the neural networks.
* `best bit` the bit which can be predicted with the highest accuracy:
    * `acc` mean test accuracy of the `best bit` 
    * `n pred` how many neural networks have already predicted `best bit`
    * `p value` what is the p-value for the observation of `acc` 

## Working directory structure
Running these commands will create a folder located in path `save_path` with the following structure
```
save_path
    cfg                 <- *.cfg ensemble configuration file
    |_  h5              <- *.h5 neural network model files which contain the weights of each neural network
    |_  hist            <- *.pkl files which contain the training history of each ensemble member 
    |_  pred            <- *.npy files with the predictions of each ensemble member (generated by running test_ensemble.py)
```

## How to set GPU parameters

If you execute the code on a new machine or on a new dataset or with a new model, the parameters which are likely to change are the ones relating to how many actors work in parallel on each GPU

```python
# How many GPUs are available?
N_GPUS = 4 
# Each GPU will host a certain number of actors. 
# Note: This number will depend on the model and data complexity. 
# We have used N_ACTORS_PER_GPU = 3 for Speck 32 and N_ACTORS_PER_GPU=2 for Speck 128.
N_ACTORS_PER_GPU = 6 
# GPU fraction per actor (typically 0.33 for Speck 32 and 0.5 for Speck 128 for GPUs with 40GB memory per GPU):
NUM_GPUS=0.15
# CPUs per actor (typically 8 for Speck 32 and 28 for Speck 128 if you have 128 CPU cores available):
NUM_CPUS=5
```

You can find useful information about GPU usage by running `watch -n 0.5 nvidia-smi` while running the code. 
The snapshot below shows that the memory of `GPU 0` is almost full (`39354MiB / 40536MiB`). This means `N_ACTORS_PER_GPU` has to be reduced. 
The GPU fraction used by each actor (`NUM_GPUS`) has to be modified accordingly.
```
Sun May 22 10:41:14 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 495.29.05    Driver Version: 495.29.05    CUDA Version: 11.5     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:01:00.0 Off |                    0 |
| N/A   38C    P0    53W / 275W |  39354MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

## How to prepare the dataset
The `{data_path}` contains a single `*.npy` file with X sequences of length 1024 bits for SPECK 32/64, for example:
```python 
>>> filename = '/home/anna/NBEATSD4/data_5rounds_1000000_samples.npy'
>>> data = np.load(filename)
>>> print(data.shape)
(1000000, 1024) # 1'000'000 rows with n_bits=1024 in each row.
>>> print(data[0])
array([0, 0, 0, ..., 1, 0, 1], dtype=uint8)
```

Often machine learning data is saved in the format of `X.npy`, `Y.npy` `X_val.npy`, `Y_val.npy`. The following routine produces a dataset of the expected format for `NNBits`: 
```python 
#load training and validation data
X = np.load('X.npy')
Y = np.load('Y.npy')
X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')
#combine the data: concatenate Y as a column to X
train   = np.c_[X, Y]
val     = np.c_[X_val, Y_val]
#combine the data: concatenate rows
final   = np.r_[train, val]
#save final
np.save('nnbits_dataset.npy', final)
```

## How to add a new model
1. Add your TensorFlow model `my_model.py` to the folder `models/`. 
2. Add your TensorFlow model to the initialization file `models/__init__.py` by adding a line 
   ```python 
   from .my_model import create_model_routine as my_model_id
   ```
3. Call NNBits and set the configuration parameter `'NEURAL_NETWORK_MODEL': 'my_model_id'`

# Citation
If you use this code in your work, please cite the following [paper]()
```
@inproceedings{

}
```


# Demos 

## Command line interface

### Clone repository and create folders
```bash
# clone repository
git clone https://github.com/Crypto-TII/nnbits
# change working directory
cd nnbits/
# install requirements
pip install -r requirements.txt
# create dataset directory
mkdir 'speck_32_64'
# create working directory for analysis results
mkdir 'demo_speck32_roundid6'
```
### Create a configuration file
Create a configuration file `config.cfg` file in the working directory (e.g. via `vim 'demo_speck32_roundid6/config.cfg'`) with the following content. 
The configuration file contains all details for the ensemble training, such as the path for the data to be analyzed (`DATAPATH`) or the number of `NEURAL_NETWORKS`. 
```python
DATAPATH = "speck_32_64/round6_sequences300k.npy"
# ensemble settings
NEURAL_NETWORK_MODEL = "gohr_generalized"
NEURAL_NETWORKS = 100
SELECT_BITS_STRATEGY = "random"
INPUT_DATA_OP = "zero"
N_RANDOM_BITS = 63
# hardware settings
N_GPUS = 1
N_ACTORS_PER_GPU = 1
GPU_PER_ACTOR = 1
CPU_PER_ACTOR = 1
# training settings
N_EPOCHS = 5
N_TRAIN = 145000
N_VAL = 145000
N_TEST = 0
BATCHSIZE = 5000
```

### Create the dataset
(Python)
```python 
#### Create the dataset ######
number_of_samples = 300_000

from avalanche_data_generator.speck_32_64 import speck_k64_p32_o32_r22 as data_generator
import numpy as np

dataset = data_generator.generate_avalanche_dataset(int(number_of_samples))

np.save(f"speck_32_64/round6_sequences300k.npy", dataset[6])
```
### Run the NNBits analysis
(bash)
```bash
python -m nnbits.run --savepath 'demo_speck32_roundid6'
```

### Analyze the outcome 
(Python)
```python 
import numpy as np

from nnbits.filemanager import FileManager
from nnbits.bitanalysis import get_X

# the file manager knows where to find all files in the working directory:
F = FileManager('demo_speck32_roundid6') 

# get the accuracies for all networks on all single bits:
X = get_X(F)

# the bit accuracies are calculated by taking the mean over all neural networks which predict the particular bit 
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
plt.gcf().savefig('result.png', bbox_inches='tight')
```
![image](https://user-images.githubusercontent.com/73515327/191898706-0b18f16a-e04d-4711-8222-8fdb0a3e7ff0.png)


## Jupyter Notebook

A [demo notebook](demo.ipynb) is included in this repository. Please clone the repository and install the requirements
```bash
git clone https://github.com/Crypto-TII/nnbits
cd nnbits
pip install -r requirements.txt
```
Launch the demo notebook.

<!-- To open the demo in Google Colab (allows to execute Python code directly in the browser and provides one GPU and one TPU for free) please click here

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Crypto-TII/nbeats_statistical_test/blob/main/demo.ipynb) -->

# Background information

## Methodology 

An ensemble of deep neural networks is trained and tested on a `*.npy` file which contains sequences of potential random data.  

1. Each ensemble member is a neural network with a unique bit `selection`: The respective bit selection will define some bits of the sequence as inputs, and the remaining bits as outputs of the neural network. The input bits will be set to zero at the input of the neural network. The neural network will be trained to predict the output bits. The number of selections, and therefore ensemble members is defined in the `*.cfg` configuration file. 
2. Each ensemble member is trained on the training data as defined in the `*.cfg` file. 
3. Each ensemble member is tested on the test data as defined in the `*.cfg` file. 

## Repository structure 
This repository contains the following files:
```
nnbits
|   |   README.md               <-  the file which generates the current view
|   |_
|_  demo.ipynb                  <-  demo notebook
|_  nnbits
    |_  run.py                  <-  run the ensemble distinguisher (see `demo.ipynb`) 
    |_  selections.py           <-  generates bit selections, see [Methodology](#methodology) 
    |_  metric.py               <-  defines a bit-by-bit accuracy as metric
    |_  network.py              <-  handles routines for the deep-learning models in folder `models`
    |_  models                  <-  contains the following deep learning models
        |_  gohr_generalized.py     <-  a generalized version of Gohr's network
        |_  resnet50.py             <-  ResNet50 implementation 
        |_  vgg16.py                <-  VGG-16 implementation
        |_  nbeats.py               <-  N-BEATS network 
    |_  trainingtracker.py      <-  keeps track of the ensemble training progress
    |_  filemanager.py          <-  keeps track of filenames
```

## Training output 

The output gives the following information:
```
====================================
speck_32_64/round0_sequences300k.npy
||           time            |   NN finished   | pred. bits ||  best bit  |  acc (%)   |   n pred   |  p value   ||
===================================================================================================================
||   2022-05-19_12h33m59s    |      0/100      |   0/1024   ||    nan     |    nan     |    nan     |    nan     ||
||   2022-05-19_12h34m41s    |      1/100      |  63/1024   ||    143     |  100.000   |     1      |     0      ||
||   2022-05-19_12h34m41s    |      3/100      |  122/1024  ||    237     |  100.000   |     1      |     0      ||
...
||   2022-05-19_12h34m42s    |     16/100      |  762/1024  ||    511     |  100.000   |     1      |     0      ||
p-value is below limit ==> stop analysis.
```
Topmost is the `*.npy` file which has been analyzed by the ensemble. 
The tabular output gives the following information in real-time during the training of the ensemble:
* The `time` column gives a timestamp for the row and the rest of the row indicates the ensemble training status. 
* `NN finished` is the neural networks which have already finalized their training. 
*  `pred. bits` indicates how many bits of the total unit length were already present at the output of the `NN finished`. For example the avalanche unit of Speck 32 has a length of `1024` bits and in the last timesteps `762/1024` of those bits had been predicted by one of the neural networks.
* `best bit` the bit which can be predicted with the highest accuracy:
    * `acc` mean test accuracy of the `best bit` 
    * `n pred` how many neural networks have already predicted `best bit`
    * `p value` what is the p-value for the observation of `acc` 

## Working directory structure
Running these commands will create a folder located in path `save_path` with the following structure
```
save_path
    cfg                 <- *.cfg ensemble configuration file
    |_  h5              <- *.h5 neural network model files which contain the weights of each neural network
    |_  hist            <- *.pkl files which contain the training history of each ensemble member 
    |_  pred            <- *.npy files with the predictions of each ensemble member (generated by running test_ensemble.py)
```

## How to set GPU parameters

If you execute the code on a new machine or on a new dataset or with a new model, the parameters which are likely to change are the ones relating to how many actors work in parallel on each GPU

```python
# How many GPUs are available?
N_GPUS = 4 
# Each GPU will host a certain number of actors. 
# Note: This number will depend on the model and data complexity. 
# We have used N_ACTORS_PER_GPU = 3 for Speck 32 and N_ACTORS_PER_GPU=2 for Speck 128.
N_ACTORS_PER_GPU = 6 
# GPU fraction per actor (typically 0.33 for Speck 32 and 0.5 for Speck 128 for GPUs with 40GB memory per GPU):
NUM_GPUS=0.15
# CPUs per actor (typically 8 for Speck 32 and 28 for Speck 128 if you have 128 CPU cores available):
NUM_CPUS=5
```

You can find useful information about GPU usage by running `watch -n 0.5 nvidia-smi` while running the code. 
The snapshot below shows that the memory of `GPU 0` is almost full (`39354MiB / 40536MiB`). This means `N_ACTORS_PER_GPU` has to be reduced. 
The GPU fraction used by each actor (`NUM_GPUS`) has to be modified accordingly.
```
Sun May 22 10:41:14 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 495.29.05    Driver Version: 495.29.05    CUDA Version: 11.5     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:01:00.0 Off |                    0 |
| N/A   38C    P0    53W / 275W |  39354MiB / 40536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
```

## How to prepare the dataset
The `{data_path}` contains a single `*.npy` file with X sequences of length 1024 bits for SPECK 32/64, for example:
```python 
>>> filename = '/home/anna/NBEATSD4/data_5rounds_1000000_samples.npy'
>>> data = np.load(filename)
>>> print(data.shape)
(1000000, 1024) # 1'000'000 rows with n_bits=1024 in each row.
>>> print(data[0])
array([0, 0, 0, ..., 1, 0, 1], dtype=uint8)
```

Often machine learning data is saved in the format of `X.npy`, `Y.npy` `X_val.npy`, `Y_val.npy`. The following routine produces a dataset of the expected format for `NNBits`: 
```python 
#load training and validation data
X = np.load('X.npy')
Y = np.load('Y.npy')
X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')
#combine the data: concatenate Y as a column to X
train   = np.c_[X, Y]
val     = np.c_[X_val, Y_val]
#combine the data: concatenate rows
final   = np.r_[train, val]
#save final
np.save('nnbits_dataset.npy', final)
```

## How to add a new model
1. Add your TensorFlow model `my_model.py` to the folder `models/`. 
2. Add your TensorFlow model to the initialization file `models/__init__.py` by adding a line 
   ```python 
   from .my_model import create_model_routine as my_model_id
   ```
3. Call NNBits and set the configuration parameter `'NEURAL_NETWORK_MODEL': 'my_model_id'`

# Citation
If you use this code in your work, please cite the following [paper]()
```
@inproceedings{

}
```
