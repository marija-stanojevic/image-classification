from tflearn.layers.normalization import batch_normalization
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d

def residual_block(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock"):

    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                #I think conv_2d should go before not after batch normalization and activation
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)

            # Downsampling
            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)

            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels

            resnet = resnet + identity

    return resnet
