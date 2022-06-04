def create_nbeats_model(input_neurons=32, output_neurons=10, model_strength=1,
                        set_memory_growth=True
                        ):
    # --- prepare GPU
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # --- prepare model
    from tensorflow import keras
    from keras import layers
    from nbeats_keras.model import NBeatsNet as NBeatsKeras

    optimizer = 'Adam'
    loss = 'Huber'  # keras.losses.BinaryCrossentropy(from_logits=True)
    stack_types = ['generic'] * 6
    thetas_dim = [int(8 * model_strength)] * 6
    nb_blocks_per_stack = 4
    hidden_layer_units = int(64 * model_strength)
    share_weights_in_stack = False

    model = NBeatsKeras(backcast_length=input_neurons,
                        forecast_length=output_neurons,
                        stack_types=stack_types,
                        nb_blocks_per_stack=nb_blocks_per_stack,
                        thetas_dim=thetas_dim,
                        share_weights_in_stack=share_weights_in_stack,
                        hidden_layer_units=hidden_layer_units)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  run_eagerly=False)
    # metrics=['accuracy'])

    model.callbacks = []

    return model