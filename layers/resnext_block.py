from tflearn.layers.normalization import batch_normalization
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, grouped_conv_2d, avg_pool_2d

def resnext_block(incoming, nb_blocks, out_channels, cardinality,
                  downsample=False, downsample_strides=2, activation='relu',
                  batch_norm=True, weights_init='variance_scaling',
                  regularizer='L2', weight_decay=0.0001, bias=True,
                  bias_init='zeros', trainable=True, restore=True,
                  reuse=False, scope=None, name="ResNeXtBlock"):
    resnext = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Bottleneck width related to cardinality for perplexity conservation
    # compare to ResNet (see paper, Table 2).
    card_values = [1, 2, 4, 8, 32]
    bottleneck_values = [64, 40, 24, 14, 4]
    bottleneck_size = bottleneck_values[card_values.index(cardinality)]
    # Group width for reference
    group_width = [64, 80, 96, 112, 128]

    assert cardinality in card_values, "cardinality must be in [1, 2, 4, 8, 32]"

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        for i in range(nb_blocks):

            identity = resnext
            if not downsample:
                downsample_strides = 1

            resnext = conv_2d(resnext, bottleneck_size, 1,
                              downsample_strides, 'valid',
                              'linear', bias, weights_init,
                              bias_init, regularizer, weight_decay,
                              trainable, restore)

            if batch_norm:
                resnext = batch_normalization(resnext, trainable=trainable)
            resnext = tflearn.activation(resnext, activation)

            resnext = grouped_conv_2d(resnext, cardinality, 3, 1, 'same',
                                      'linear', False, weights_init,
                                      bias_init, regularizer, weight_decay,
                                      trainable, restore)
            if batch_norm:
                resnext = batch_normalization(resnext, trainable=trainable)
            resnext = tflearn.activation(resnext, activation)

            resnext = conv_2d(resnext, out_channels, 1, 1, 'valid',
                              activation, bias, weights_init,
                              bias_init, regularizer, weight_decay,
                              trainable, restore)

            if batch_norm:
                resnext = batch_normalization(resnext, trainable=trainable)

            # Downsampling
            if downsample_strides > 1:
                identity = avg_pool_2d(identity, 1, downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels) // 2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            resnext = resnext + identity
            resnext = tflearn.activation(resnext, activation)

        return resnext