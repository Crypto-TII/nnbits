def create_resnet50_model(input_neurons=32, output_neurons=10, model_strength=1, set_memory_growth=True):

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
    import numpy as np
    from keras import Model
    from keras.layers import Input, Reshape, Concatenate, Dense
    from keras.applications.resnet import ResNet50

    # ---------------------------------------------------
    # Model parameters
    # ---------------------------------------------------
    img_size_target = int(np.sqrt(input_neurons))

    # ---------------------------------------------------
    # Model definition
    # ---------------------------------------------------
    input_0 = Input(shape=(input_neurons,))
    img_input = Reshape((img_size_target, img_size_target, 1))(input_0)
    img_conc = Concatenate()([img_input, img_input, img_input])

    model = ResNet50(weights=None, input_tensor=img_conc)

    # substitute the Dense last layer with 1000 outputs, with one of a suitable number of output neurons
    x = model.layers[-2].output
    x = Dense(output_neurons, activation='sigmoid')(x)
    model = Model(inputs=model.input, outputs=x)

    # ---------------------------------------------------
    # Model compilation
    # ---------------------------------------------------
    optimizer = 'Adam'
    loss = 'binary_crossentropy'

    model.compile(loss=loss,
                  optimizer=optimizer,
                  run_eagerly=False)

    # ---------------------------------------------------
    # Model callbacks
    # ---------------------------------------------------
    model.callbacks = []

    return model