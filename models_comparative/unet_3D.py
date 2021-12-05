# import sys
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '/home/camilo/Programacion/master_thesis')

import tensorflow as tf
from tensorflow.keras import Model, Input, layers
# from model.config import *

class Conv3D_Unet(layers.Layer):
    def __init__(self, filters, kernels, strides, padding='same', act_fnc='relu', pool_size=(2,2,2), dropout_lyr=False, dropout_rate=0.5, factor=2, **kwarks):
        super(Conv3D_Unet, self).__init__(**kwarks)

        # Attributes
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.pool_size = pool_size
        self.max_pool_lyr = None
        self.dropout_lyr = None
        self.dropout_rate = dropout_rate

        if (act_fnc == 'relu'):
            act_fnc = tf.nn.relu

        self.activation = act_fnc

        # Layers

        # Convolutional layers
        self.conv_lyr_1 = layers.Conv3D(
            filters=self.filters,
            kernel_size=self.kernels,
            strides=self.strides,
            kernel_initializer='he_normal',
            padding=self.padding,
        )

        self.conv_lyr_2 = layers.Conv3D(
            filters=self.filters*factor,
            kernel_size=self.kernels,
            strides=self.strides,
            kernel_initializer='he_normal',
            padding=self.padding,
        )
        
        # Activation functions
        self.act_lyr_1 = layers.Activation(self.activation)
        self.act_lyr_2 = layers.Activation(self.activation)
        
        # Batchs norms
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()

        if (dropout_lyr):
            self.dropout_lyr = layers.Dropout(self.dropout_rate)

    def call(self, inputs):
        x = self.conv_lyr_1(inputs)
        x = self.batch_norm_1(x)
        x = self.act_lyr_1(x)
        x = self.conv_lyr_2(x)
        x = self.batch_norm_2(x)
        x = self.act_lyr_2(x)

        if (self.dropout_lyr):
            x = self.dropout_lyr(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # Attributes
            'filters': self.filters,
            'kernels' : self.kernels,
            'strides' : self.strides,
            'padding' : self.padding,
            'pool_size' : self.pool_size,
            # Layers

        })
        return config

class Up_conv3D_unet(layers.Layer):
    def __init__(self, filters, up_size=2, kernels=3, strides=1, act_fnc='relu', **kwarks):
        super(Up_conv3D_unet, self).__init__(**kwarks)

        self.filters = filters
        self.up_size = up_size
        self.kernels = kernels
        self.strides = strides

        if (act_fnc == 'relu'):
            act_fnc = tf.nn.relu

        self.activation = act_fnc

        # Layers
        self.up_conv_1 = layers.UpSampling3D(
            size=self.up_size
        )

        self.conv_block = Conv3D_Unet(
            filters=self.filters,
            kernels=self.kernels,
            strides=self.strides,
            dropout_lyr=False,
            factor=1,
        )

    def call(self, inputs, skip_connection):
        x = self.up_conv_1(inputs)
        # print("\n\n", skip_connection.shape, " ", x.shape)
        x = layers.Concatenate()([x, skip_connection])
        # print("Concat: ", x.shape)
        x = self.conv_block(x)
        # print("after conv: ", x.shape)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # Attributes
            'filters': self.filters,
            'kernels' : self.kernels,
            'strides' : self.strides,
            'up_size' : self.up_size,
            # Layers

        })
        return config

def build_unet3D_model(config):
    """
        Returns Unet model

        https://arxiv.org/pdf/1606.06650.pdf

        Params:
            - Config: ml_collection that retrieves all the needed configuration
            for 3D unet
        
        Returns:
            - Model: Model
    """

    encoder_filters = [
        {'filters': 8, 'dropout': False, 'pool':True},
        {'filters': 16, 'dropout': False, 'pool':True},
        {'filters': 32, 'dropout': True, 'pool':True},
        {'filters': 64, 'dropout': True, 'pool':False},
    ]

    # 3D input image shape (64,64,64, 1)
    inputs = Input(shape=config.image_size)
    
    x = inputs
    encoder_layers = []
    for encoder_filter in encoder_filters:
        # Conv3D + ReLu + Conv3D + ReLu + MaxPooling 
        x = Conv3D_Unet(
            filters=encoder_filter['filters'],
            kernels=3,
            strides=1,
            dropout_lyr=encoder_filter['dropout'],
        )(x)

        if (encoder_filter['pool']):
            encoder_layers.append(x)
            x = layers.MaxPooling3D(pool_size=2)(x)

    for idx in range(len(encoder_layers)-1, -1, -1):
        # Upsampĺing + Conv3D + ReLu + Conv3D + ReLu + MaxPooling
        x = Up_conv3D_unet(
            filters=encoder_filters[idx+1]['filters']
        )(x, encoder_layers[idx])

    # Segmentation head
    x = layers.Conv3D(
        filters=config.n_classes, 
        kernel_size=1, 
        strides=1, 
        padding='same',
        activation='softmax'
    )(x)

    return Model(inputs, x)
    
# def main():
#     config = get_config_local_path()#get_config_test()
#     model = unet3D_model(config)
#     model.summary()

#     tf.keras.utils.plot_model(
#         model,
#         to_file='unet_model.png',
#         show_shapes=True,
#         show_dtype=True,
#         show_layer_names=True,
#         rankdir="TB",
#         expand_nested=False,
#         dpi=96,
#     )

# if __name__ == "__main__":
#     main()