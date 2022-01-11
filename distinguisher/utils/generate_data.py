import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import urandom
import sys
np.set_printoptions(threshold=sys.maxsize)


def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
    key = (0x1918,0x1110,0x0908,0x0100)
    pt = (0x6574, 0x694c)
    ks = expand_key(key, 22)
    ct = encrypt(pt, ks)
    if (ct == (0xa868, 0x42f2)):
        print("Testvector verified.")
        return(True);
    else:
        print("Testvector not verified.")
        return(False);

#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
    X = np.zeros((32*2 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
    for i in range(32*2* WORD_SIZE()):
        index = i // (WORD_SIZE()*2);
        offset = 2*WORD_SIZE() - (i % (2*WORD_SIZE())) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return(X);


#baseline training data generator
def make_train_data(n, nr):
    """
    Description
    -----------
    The generation of the avalanche dataset is described for example in [1, section 2.1].
    The following code generates a plaintext avalanche dataset.
    Short description, adapted from [1]:
    Given random input plaintexts and a key of all zeros, derived blocks are generated in the following manner:
    Each derived block is based on the XOR of the ciphertext formed using the fixed key and the perturbed random plaintext
    with the i-th bit changed, for 1<=i<=32.
    The derived blocks are all concatenated to one sequence.

    [1]
    Soto, J. (1999). Randomness testing of the advanced encryption standard candidate algorithms.
    In US Department of Commerce, Technology Administration, National Institute of Standards and Technology.
    http://www.nist.gov/customcf/get_pdf.cfm?pub_id=151193

    :param n:
    :param nr:
    :return:
    """
    # key is zero
    keys = np.zeros(4*n,dtype=np.uint16).reshape(4,-1);

    # left and right input words are randomly chosen
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);

    # expand key to number of rounds
    ks = expand_key(keys, nr);

    delta32=[0 for i in range(0,32)]

    for j in range(0,32):

        # shift 1 by j positions
        diff32=1<<j

        # obtain the left part of diff32 by rotating to the right by 16 bits
        # obtain the right part of diff32 by AND with 16 `1`s
        diff=(diff32>>16, diff32 & 0xffff)

        # choose the outputs according to the XOR with the perturbation
        plain1l = plain0l ^ diff[0];
        plain1r = plain0r ^ diff[1];

        # encrypt to number of rounds
        ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
        ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);

        # XOR of left round input and output, resp. right round input and output
        deltaL = ctdata0l^ctdata1l
        deltaR = ctdata0r^ctdata1r

        # to obtain a 32 bit word, we concatenate the left and right Delta
        delta32[j] = (np.uint32(deltaL)<<16)^deltaR

    X=convert_to_binary(delta32)
    np.save(datapath+"data_"+str(nr)+"rounds_"+str(n)+"_samples.npy",X)
    #np.save("/home/anna/NBEATSD4/data_"+str(nr)+"rounds_"+str(n)+"_samples.npy",X)
    return(X);

if __name__ == 'main':

    datapath = '/opt/cryptanalysis_servers_shared_folder/NBEATSCipherDistinguisher/D1/'

    for rounds in range(1, 22+1):
        make_train_data(10**6, rounds)