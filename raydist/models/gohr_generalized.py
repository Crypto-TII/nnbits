def create_gohr_generalized_model(input_neurons=32, output_neurons=10, model_strength=1, set_memory_growth=True):
    """
    CHANGE ME
    """
    # --- prepare GPU
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    from keras.models import Model
    from keras.layers import Dense, Conv1D, Conv2D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, \
        Activation
    from keras.layers import Concatenate, MaxPooling2D
    from keras.regularizers import l2
    import numpy as np

    img_sqrt = int(np.sqrt(input_neurons))

    # num_blocks=2
    num_filters = 32 * 4 #* model_strength
    d1 = 64 #* model_strength
    d2 = 64 #* model_strength
    # word_size=16
    ks = 3
    depth = model_strength
    reg_param = 10 ** -5
    final_activation = 'sigmoid'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, amsgrad=True)  # 'Adam'
    loss = 'mse'  # 'Huber'

    # loss = 'mse'

    # put input in square shape instead of word-like structure
    inp = Input(shape=(input_neurons,));
    # rs = Reshape((2 * num_blocks, word_size))(inp);
    rs = Reshape((img_sqrt, img_sqrt))(inp)  # changed rs = Reshape((img_sqrt, img_sqrt, 1))(inp)
    rs = Permute((2, 1))(rs)
    # ---- search correlations in one direction
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(rs);  # changed
    conv0 = BatchNormalization()(conv0);
    conv0 = Activation('relu')(conv0);
    # add residual blocks
    shortcut = conv0;
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
        conv1 = BatchNormalization()(conv1);
        conv1 = Activation('relu')(conv1);
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1);
        conv2 = BatchNormalization()(conv2);
        conv2 = Activation('relu')(conv2);
        shortcut = Add()([shortcut, conv2]);

    # ---- search correlations in the other direction
    #     perm = Permute((2,1))(rs);
    #     conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm); # changed
    #     conv0 = BatchNormalization()(conv0);
    #     conv0 = Activation('relu')(conv0);
    #     shortcut1 = conv0;
    #     for i in range(depth):
    #         conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut1);
    #         conv1 = BatchNormalization()(conv1);
    #         conv1 = Activation('relu')(conv1);
    #         conv2 = Conv1D(num_filters, kernel_size=ks, padding='same',kernel_regularizer=l2(reg_param))(conv1);
    #         conv2 = BatchNormalization()(conv2);
    #         conv2 = Activation('relu')(conv2);
    #         shortcut1 = Add()([shortcut1, conv2]);

    # ---- bring them together
    # shortcut = Concatenate()([shortcut, shortcut1])#Add()([shortcut, shortcut1])
    # add prediction head
    flat1 = Flatten()(shortcut);
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1);
    dense1 = BatchNormalization()(dense1);
    dense1 = Activation('relu')(dense1);
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
    dense2 = BatchNormalization()(dense2);
    dense2 = Activation('relu')(dense2);
    out = Dense(output_neurons, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2);
    model = Model(inputs=inp, outputs=out);
    # -------------#-------------#-------------#-------------#

    model.compile(loss=loss,
                  optimizer=optimizer,
                  run_eagerly=False)  # , metrics=['acc'])

    #     from gohr import cyclic_lr
    #     from keras.callbacks import LearningRateScheduler
    #     lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001));

    model.callbacks = []  # [lr]
    return model