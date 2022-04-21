from ..gohr_original.gohr import make_resnet, cyclic_lr

def create_gohrs_model(input_neurons=32, output_neurons=10, model_strength=1, set_memory_growth=True):
    assert input_neurons == 64, "Gohr's model is only defined for input_neurons = 64"
    assert output_neurons == 1, "Gohr's model is only defined for output_neurons = 1"
    assert type(model_strength) == int, 'model_strength has to be an integer'

    # --- prepare GPU
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if set_memory_growth:
        tf.config.experimental.set_memory_growth(gpus[0], True)


    from keras.callbacks import LearningRateScheduler

    model = make_resnet(depth=model_strength, reg_param=10 ** -5);
    model.compile(optimizer='adam', loss='mse',
                  run_eagerly=False
                  # added by authors to Gohr's script for fairness, as run_eagerly=True will slow down the run
                  );

    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001));

    model.callbacks = [lr]

    return model