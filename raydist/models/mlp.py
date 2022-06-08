def create_mlp_model(input_neurons=64, output_neurons=1, model_strength=1, set_memory_growth=True):
    """
    In [1] the best MLP model for Speck32 is found in the second row of table 3.
    The MLP has 6 layers with neurons=(128, 256, 112, 256, 128, 64).

    [1] Baksi, A., Breier, J., Dasu, V. A., & Hou, X. (2014). Machine Learning Attacks on Speck. 4–9.
    https://www.esat.kuleuven.be/cosic/events/silc2020/wp-content/uploads/sites/4/2021/09/Submission10.pdf
    """
    # ---------------------------------------------------
    # Prepare GPU
    # ---------------------------------------------------
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # ---------------------------------------------------
    # Imports
    # ---------------------------------------------------
    from keras.models import Model
    from keras.layers import Dense, Input

    # ---------------------------------------------------
    # Model definition
    # ---------------------------------------------------
    inp = Input(shape=(input_neurons,))
    x = inp

    for i in range(model_strength):
        for neurons in [128, 256, 112, 256, 128, 64]:
            x = Dense(neurons, activation="relu")(x)

    out = Dense(output_neurons, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    # ---------------------------------------------------
    # Model compilation
    # ---------------------------------------------------
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    loss = 'mse'

    model.compile(optimizer=optimizer,
                  loss=loss,
                  run_eagerly=False,
                  metrics = ['acc']
                  )

    # ---------------------------------------------------
    # Model callbacks
    # ---------------------------------------------------
    model.callbacks = []

    return model