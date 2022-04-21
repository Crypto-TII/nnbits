def create_mlp_model(input_neurons=64, output_neurons=1, model_strength=1, set_memory_growth=True):
    # --- prepare GPU
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    from keras.models import Model
    from keras.layers import Dense, Input

    inp = Input(shape=(input_neurons,))
    x = inp
    for i in range(12):
        x = Dense(64, activation="relu")(x)
    out = Dense(output_neurons, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer='adam', loss='mse',
                  run_eagerly=False
                  # added by authors to Gohr's script for fairness, as run_eagerly=True will slow down the run
                  );

    from gohr.gohr import cyclic_lr
    from keras.callbacks import LearningRateScheduler
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));

    model.callbacks = [lr]

    return model