def set_random_seeds(seed_value=0):
    import numpy as np
    import tensorflow as tf
    import random

    # Set numpy pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # Set tensorflow pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # Set random package seed value
    random.seed(seed_value)

def create_toy_resnet_model(input_neurons=32, output_neurons=10, model_strength=1, set_memory_growth=True):
    # --- prepare GPU
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    import numpy as np
    from keras import Model
    from keras.layers import Input, Reshape, Concatenate, Dense, GlobalAveragePooling2D, Flatten, Dropout
    from keras.applications.resnet import ResNet50

    set_random_seeds(42)

    # reshape input
    img_size_target = int(np.sqrt(input_neurons))

    input_0 = Input(shape=(input_neurons))
    img_input_0 = Reshape((img_size_target, img_size_target,1))(input_0)

    img_conc = Concatenate()([img_input_0, img_input_0, img_input_0]) #img_input_0 #Concatenate()([img_input_0, img_input_2, img_input_3])#

    model = ResNet50(weights=None, input_tensor=img_conc)

    """ HOW MANY LAYERS OF ResNet50 TO KEEP?
    ##########################################
    >>> model = resnet50(8**2, 1)

    >>> for i, layer in enumerate(model.layers):
    >>>    print(i, '\t', layer.output_shape, '\t', layer.name)

    ===> after the first 20 layers the output is only 1 neuron in size. Therefore, we keep only the first 20 layers.
    """
    x = model.layers[20].output

    # Add a prediction head
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Flatten(name="flatten")(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(output_neurons, activation='sigmoid')(x)

    model = Model(inputs=model.input, outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, amsgrad=True)
    loss = 'mse'

    model.compile(loss=loss,
                  optimizer=optimizer,
                  run_eagerly=False,
                  metrics=['acc'])

    model.callbacks = []

    return model