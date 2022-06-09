import numpy as np
from pickle import dump
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
from os import urandom

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

def convert_to_binary(arr):
    X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
    for i in range(len(arr) * WORD_SIZE()):
        index = i // WORD_SIZE();
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
        X[i] = (arr[index] >> offset) & 1;
    X = X.transpose();
    return(X);

def make_train_data(n, nr, diff=(0x0040,0)):
    Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
    plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
    num_rand_samples = np.sum(Y==0);
    plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
    ks = expand_key(keys, nr);
    ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
    ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
    X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
    return(X,Y);

def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5, reg_param=0.0001,        final_activation='sigmoid'):
    #Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size * 2,));
    rs = Reshape((2 * num_blocks, word_size))(inp);
    perm = Permute((2,1))(rs);
    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm);
    conv0 = BatchNormalization()(conv0);
    conv0 = Activation('relu')(conv0);
    #add residual blocks
    shortcut = conv0;
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
        conv1 = BatchNormalization()(conv1);
        conv1 = Activation('relu')(conv1);
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
        conv2 = BatchNormalization()(conv2);
        conv2 = Activation('relu')(conv2);
        shortcut = Add()([shortcut, conv2]);
    #add prediction head
    flat1 = Flatten()(shortcut);
    dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(flat1);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
    model = Model(inputs=inp, outputs=out);
    return(model);

bs = 5000;

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr);
    return(res);

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True);
    return(res);