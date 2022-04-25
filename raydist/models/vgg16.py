def create_vgg16_model(input_neurons=32, output_neurons=10, model_strength=1, set_memory_growth=True):
    """
    CHANGE ME
    """
    # Prepare GPU
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # Define model
    import numpy as np
    from keras import Model
    from keras.layers import Input, Reshape, Concatenate, Dense
    from keras.applications.vgg16 import VGG16

    # add a reshaping to the input of VGG16
    img_size_target = int(np.sqrt(input_neurons))

    input_0 = Input(shape=(input_neurons,))
    img_input = Reshape((img_size_target, img_size_target, 1))(input_0)
    img_conc = Concatenate()([img_input, img_input, img_input])

    model = VGG16(weights=None, input_tensor=img_conc)

    # substitute the Dense last layer with 1000 outputs, with one of a suitable number of output neurons
    x = model.layers[-2].output
    x = Dense(output_neurons, activation='sigmoid')(x)
    model = Model(inputs=model.input, outputs=x)

    # Compile model
    optimizer = 'Adam' #tf.keras.optimizers.Adam(learning_rate=0.002, amsgrad=True)  # 'Adam'
    loss = 'binary_crossentropy'

    model.compile(loss=loss,
                  optimizer=optimizer,
                  run_eagerly=False)

    model.callbacks = []

    return model