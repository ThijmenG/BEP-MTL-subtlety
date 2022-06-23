## imports
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv3D, GlobalAveragePooling3D, MaxPooling3D
import os
from LCR_Model_support import get_submodules_from_kwargs, load_model_weights

def VGG16(
        lr,
        VERBOSE,
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(128,128,64,1),
        pooling=None,
        classes=1,
        stride_size=2,
        init_filters=64,
        max_filters=512,
        repetitions=(2, 2, 3, 3, 3),
        **kwargs
):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    backend, tensorflow.keras.layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    tensorflow.keras.mixed_precision.set_global_policy('mixed_float16')

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # if stride_size is scalar make it tuple of length 5 with elements tuple of size 3
    # (stride for each dimension for more flexibility)
    if type(stride_size) not in (tuple, list):
        stride_size = [
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
        ]
    else:
        stride_size = list(stride_size)

    if len(stride_size) < 3:
        print('Error: stride_size length must be 3 or more')
        return None

    if len(stride_size) != len(repetitions):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    for i in range(len(stride_size)):
        if type(stride_size[i]) not in (tuple, list):
            stride_size[i] = (stride_size[i], stride_size[i], stride_size[i])

    if input_tensor is None:
        img_input = Input(shape=input_shape, dtype='float16')
    else:
        if not tensorflow.keras.backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape, dtype='float16')
        else:
            img_input = input_tensor

    x = img_input
    for stage, rep in enumerate(repetitions):
        for i in range(rep):
            x = Conv3D(
                init_filters,
                (3, 3, 3),
                activation='relu',
                padding='same',
                name='block{}_conv{}'.format(stage + 1, i + 1)
            )(x)

        x = MaxPooling3D(
            stride_size[stage],
            strides=stride_size[stage],
            name='block{}_pool'.format(stage + 1)
        )(x)

        init_filters *= 2
        if init_filters > max_filters:
            init_filters = max_filters

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions', dtype =  'float32')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # Load weights.
    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, 'vgg16', weights, classes, include_top, **kwargs)

    conv_base = model

    # add a densely connected classifier on top of conv base
    model = tensorflow.keras.models.Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu', dtype = 'float32'))
    model.add(Dense(1, activation='sigmoid', dtype= 'float32'))

    opt = tensorflow.keras.optimizers.RMSprop(learning_rate=lr)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    if VERBOSE:
        model.summary()
    return model

def VGG16_multi(
        lr,
        VERBOSE,
        batch_size,
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=(128,128,64,1),
        pooling=None,
        classes=1,
        stride_size=2,
        init_filters=64,
        max_filters=512,
        repetitions=(2, 2, 3, 3, 3),
        **kwargs
):
    """Instantiates the VGG16 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    backend, tensorflow.keras.layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # if stride_size is scalar make it tuple of length 5 with elements tuple of size 3
    # (stride for each dimension for more flexibility)
    if type(stride_size) not in (tuple, list):
        stride_size = [
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
        ]
    else:
        stride_size = list(stride_size)

    if len(stride_size) < 3:
        print('Error: stride_size length must be 3 or more')
        return None

    if len(stride_size) != len(repetitions):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    for i in range(len(stride_size)):
        if type(stride_size[i]) not in (tuple, list):
            stride_size[i] = (stride_size[i], stride_size[i], stride_size[i])

    if input_tensor is None:
        img_input = Input(shape=input_shape, dtype = 'float16')
    else:
        if not tensorflow.keras.backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape, dtype = 'float16')
        else:
            img_input = input_tensor

    x = img_input
    for stage, rep in enumerate(repetitions):
        for i in range(rep):
            x = Conv3D(
                init_filters,
                (3, 3, 3),
                activation='relu',
                padding='same',
                name='block{}_conv{}'.format(stage + 1, i + 1)
            )(x)

        x = MaxPooling3D(
            stride_size[stage],
            strides=stride_size[stage],
            name='block{}_pool'.format(stage + 1)
        )(x)

        init_filters *= 2
        if init_filters > max_filters:
            init_filters = max_filters

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions', dtype = 'float32')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling3D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # Load weights.
    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, 'vgg16', weights, classes, include_top, **kwargs)
    conv_base = model

    # add a densely connected classifier on top of conv base
    x = Flatten()(conv_base.output)
    x = Dense(256, activation='relu')(x)

    out_class = Dense(1, activation='sigmoid', name='out_class', dtype = 'float32')(x)
    out_anno = Dense(1, activation='linear', name='out_anno', dtype = 'float32')(x)

    model = Model(conv_base.input, outputs=[out_class, out_anno])

    opt = tensorflow.keras.optimizers.RMSprop(learning_rate=lr)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    model.compile(optimizer=opt,
        loss={'out_class': 'binary_crossentropy', 'out_anno': 'mse'},
        loss_weights={'out_class': 0.5, 'out_anno': 0.5},
        metrics={'out_class': 'accuracy', 'out_anno': 'accuracy'}) #waarom niet ook hier accuracy?

    if VERBOSE:
        model.summary()

    return model
