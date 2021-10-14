import tensorflow as tf
from tensorflow.keras import layers

class ConvolutionalBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        # self.padding = padding
        # self.dropout_rate = dropout_rate
        self.activation = 'relu'
        
        # Layers
        self.conv_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )

        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation(self.activation)

    def call(self, inputs):
        x = self.conv_a(inputs)
        x = self.bn_a(x)
        return self.activation_fnc(x)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'activation' : self.activation,
            # layers
            'conv_a' : self.conv_a,
            'bn_a' : self.bn_a,
            'activation_fnc' : self.activation_fnc,
        })
        return config

class MLPBlock(layers.Layer):
    def __init__(self, hidden_units, dropout_rate, activation=None, **kwarks):
        super(MLPBlock, self).__init__(**kwarks)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        if not activation:
            activation = tf.nn.gelu

        self.activation = activation

        # creating layers
        self.layers = []

        for units in self.hidden_units:
            self.layers.append(layers.Dense(units, activation=self.activation))
            self.layers.append(layers.Dropout(self.dropout_rate))

    def call(self, inputs):

        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_units' : self.hidden_units,
            'dropout_rate' : self.dropout_rate,
            'activation' : self.activation,
            'layers' : self.layers,
        })
        return config

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwarks):
        super(Patches, self).__init__(**kwarks)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        
        # patches = tf.image.extract_patches(
        #     images=images,
        #     sizes=[1, self.patch_size, self.patch_size, 1],
        #     strides=[1, self.patch_size, self.patch_size, 1],
        #     rates=[1, 1, 1, 1],
        #     padding="VALID",
        # )

        patches = tf.extract_volume_patches(
            input=images,
            ksizes=[1, self.patch_size, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, self.patch_size, 1],
            padding='VALID',
        )

        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwarks):
        super(PatchEncoder, self).__init__(**kwarks)
        self.num_patches = num_patches
        
        # Layers
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches,
            # layers
            'projection' : self.projection,
            'position_embedding' : self.position_embedding,
        })
        return config

class ConvProjection(layers.Layer):
    def __init__(self, reshape_dim, projection_dim, num_patches, **kwargs):
        super(ConvProjection, self).__init__(**kwargs)
        self.reshape_dim = reshape_dim
        self.projection_dim = projection_dim
        self.num_patches = num_patches

        # Layers
        self.reshape_lyr = layers.Reshape(
            target_shape=(
                self.reshape_dim,
                self.projection_dim
            ),
        )

        # conv layer encoder
        self.conv_layer_encoded = PatchEncoder(
            num_patches=self.num_patches,
            projection_dim=self.projection_dim,
        )

    def call(self, conv_input):
        output = self.reshape_lyr(conv_input)
        output = self.conv_layer_encoded(output)
        return output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'reshape_dim' : self.reshape_dim,
            'projection_dim' : self.projection_dim,
            'num_patches' : self.num_patches,
            # layers
            'reshape_lyr' : self.reshape_lyr,
            'conv_layer_encoded' : self.conv_layer_encoded
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, projection_dim, dropout_rate, normalization_rate, transformer_units, **kwarks):
        super(TransformerBlock, self).__init__(**kwarks)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.normalization_rate = normalization_rate
        self.transformer_units = transformer_units

        # Layers
        self.ln_a = layers.LayerNormalization(epsilon=self.normalization_rate)
        self.attention_layer_a = layers.MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim = self.projection_dim,
            dropout = self.dropout_rate,
        )
        self.add_a = layers.Add()

        self.ln_b = layers.LayerNormalization(epsilon=self.normalization_rate)
        self.mlp_block_b = MLPBlock(
            hidden_units=self.transformer_units, 
            dropout_rate=self.dropout_rate
        )

        self.softmax_b = layers.Activation(activation='softmax')


        self.add_b = layers.Add()

    def call(self, encoded_patches):
        x1 = self.ln_a(encoded_patches)
        attention_layer = self.attention_layer_a(x1, x1)
        # print(attention_layer.shape)
        # attention_layer = self.softmax_b(attention_layer)
        
        x2 = self.add_a([attention_layer, encoded_patches])
        x3 = self.ln_b(x2)
        x3 = self.mlp_block_b(x3)
        return self.add_b([x3, x2])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads' : self.num_heads,
            'projection_dim' : self.projection_dim,
            'dropout_rate' : self.dropout_rate,
            'normalization_rate' : self.normalization_rate,
            'transformer_units' : self.transformer_units,
            # layers
            'ln_a' : self.ln_a,
            'attention_layer_a' : self.attention_layer_a,
            'add_a' : self.add_a,
            'ln_b' : self.ln_b,
            'mlp_block_b' : self.mlp_block_b,
            'add_b' : self.add_b,
        })
        return config

class DecoderBlockCup(layers.Layer):

    def __init__(self, target_shape, filters, normalization_rate, pool_size=(2, 2, 1), kernel_size=3, activation=None, upsample=True, **kwarks):
        super(DecoderBlockCup, self).__init__(**kwarks)
        self.normalization_rate = normalization_rate
        self.target_shape = target_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.upsample = upsample

        if not activation:
            activation = tf.nn.relu

        self.activation = activation

        # Layers
        self.ln_a = layers.LayerNormalization(epsilon=self.normalization_rate, name="decoder_block_cup_ln_a")
        self.reshape_a = layers.Reshape(
            target_shape=self.target_shape, 
            name="decoder_block_cup_reshape_1"
        )
        # self.conv_a = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size*2, strides=1, padding='same')
        self.conv_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=self.kernel_size*2, 
            strides=1, 
            padding='same'
        )
        # self.max_pool_a = layers.MaxPooling3D(pool_size=self.pool_size)
        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation('relu')
        self.upsample_a = layers.UpSampling3D(
            size=(2, 2, 2)
        )

    def call(self, encoder_output):
        # x = self.ln_a(encoder_output)
        x = self.reshape_a(encoder_output)
        x = self.conv_a(x)
        x = self.bn_a(x)
        x = self.activation_fnc(x)
        if (self.upsample):
            x = self.upsample_a(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'normalization_rate' : self.normalization_rate,
            'target_shape' : self.target_shape,
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'pool_size' : self.pool_size,
            # layers
            'ln_a' : self.ln_a,
            'reshape_a' : self.reshape_a,
        })
        return config

class DecoderTransposeBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=(2, 2, 2), activation='relu', **kwargs):
        super(DecoderTransposeBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation

        # Layers
        self.conv_tr = layers.Conv3DTranspose(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
        )

        self.bn = layers.BatchNormalization()
        self.activation_layer = layers.Activation(self.activation)

    def call(self, inputs):
        x = self.conv_tr(inputs)
        x = self.bn(x)
        x = self.activation_layer(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides,
            'activation': self.activation,
            # layers
            'conv_tr' : self.conv_tr,
            'bn' : self.bn,
            'activation_layer' : self.activation_layer,
        })
        return config

class DecoderUpsampleBlock(layers.Layer):
    
    def __init__(self, filters, kernel_size=3, strides=(1, 1, 1), pool_size=(2, 2, 1), activation='relu', **kwarks):
        super(DecoderUpsampleBlock, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.activation = activation

        # Layers
        self.upsample_a = layers.UpSampling3D(
            size=(2, 2, 2)
        )

        self.conv_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )

        # self.max_pool_a = layers.MaxPooling2D(pool_size=self.pool_size)
        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation(self.activation)
        
    def call(self, decoder_input):
        x = self.conv_a(decoder_input)
        x = self.activation_fnc(x)
        x = self.bn_a(x)
        x = self.upsample_a(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides,
            'pool_size' : self.pool_size,
            'activation': self.activation,
            # layers
            'conv_a' : self.conv_a,
            'bn_a' : self.bn_a,
            'upsample_a' : self.upsample_a,
            'activation_fnc' : self.activation_fnc,
        })
        return config

class DecoderSegmentationHead(layers.Layer):

    def __init__(self, filters, activation='softmax', kernel_size=1, strides=(1, 1, 1), **kwarks):
        super(DecoderSegmentationHead, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation

        # Layers
        self.conv_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )

        self.activation_layer = layers.Activation(self.activation)
    
    def call(self, decoder_upsample_block):
        x = self.conv_a(decoder_upsample_block)
        x = self.activation_layer(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides,
            'activation': self.activation,
            # layers
            'conv_a' : self.conv_a,
            'activation_layer' : self.activation_layer,
        })
        return config

class ConnectionComponents(layers.Layer):
    def __init__(self, filters, kernel_size, **kwarks):
        super(ConnectionComponents, self).__init__(**kwarks)

        self.filters = filters
        self.kernel_size = kernel_size

        self.conv_1_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size), 
            strides=1, 
            padding='same'
        )

        self.conv_1_b = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            strides=1,
            padding='same'
        )

        self.activation_layer = layers.Activation('relu')
        self.activation_layer_2 = layers.Activation('relu')

        self.add_layer = layers.Add()
        self.bn_1_b = layers.BatchNormalization()
        self.bn_1_a = layers.BatchNormalization()
        self.bn_out = layers.BatchNormalization()

    def call(self, input):
        shortcut = input
        path_1 = self.conv_1_b(shortcut)
        path_1 = self.bn_1_b(path_1)
        
        # conv 3x3
        path_2 = self.conv_1_a(input)
        path_2 = self.bn_1_a(path_2)
        path_2 = self.activation_layer(path_2)

        # add layer
        out = self.add_layer([path_1, path_2])
        out = self.activation_layer_2(out)
        out = self.bn_out(out)

        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            # layers
            'conv_1_a' : self.conv_1_a,
            'bn_1_a' : self.bn_1_a,
            'conv_1_b' : self.conv_1_b,
            'bn_1_b' : self.bn_1_b,
            'add_layer': self.add_layer,
            'activation_layer': self.activation_layer,
            'activation_layer_2': self.activation_layer_2,
            'bn_out': self.bn_out,
        })
        return config

class EncoderDecoderConnections(layers.Layer):
    
    def __init__(self, filters, kernel_size, upsample=True, **kwarks):
        super(EncoderDecoderConnections, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample = upsample

        # self.concatenate = layers.Concatenate()
        self.upsample_lyr = layers.UpSampling3D(
            size=(2, 2, 2)
        )

        self.con_comp_1 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size
        )

        self.con_comp_2 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size
        )

        self.con_comp_3 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size
        )

        self.con_comp_4 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size
        )


    def call(self, encoder_input):
        # coding res path
        out = self.con_comp_1(encoder_input)
        out = self.con_comp_2(out)
        out = self.con_comp_3(out)
        out = self.con_comp_4(out)
        
        if (self.upsample):
            out = self.upsample_lyr(out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'kernel_size' : self.kernel_size,
            'upsample' : self.upsample,
            # layers
            'upsample_lyr': self.upsample_lyr,
            'con_comp_1': self.con_comp_1,
            'con_comp_2': self.con_comp_2,
            'con_comp_3': self.con_comp_3,
            'con_comp_4': self.con_comp_4,
        })
        return config

class DecoderDense(layers.Layer):
    def __init__(self, normalization_rate, **kwarks):
        super(DecoderDense, self).__init__(**kwarks)
        self.normalization_rate = normalization_rate

        # Layers
        self.ln_a = layers.LayerNormalization(epsilon=self.normalization_rate, name="decoder_block_cup_ln_a")
        self.flatten_a = layers.Flatten()
        self.dropout_a = layers.Dropout(0.5)
        self.reshape_a = layers.Reshape(target_shape=(256,256,1))

    def call(self, inputs):
        x = self.ln_a(inputs)
        x = self.flatten_a(x)
        x = self.dropout_a(x)
        return self.reshape_a(x)
        
    