import tensorflow as tf
from tensorflow.keras import layers

class ConvolutionalBlock(layers.Layer):

    def __init__(self, filters, kernel_size, padding, dropout_rate, activation, **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape):
        self.w = tf.random_normal_initializer(mean=0.0, stddev=1e-4)

        if (self.bias):
            self.b = tf.constant_initializer(0.0)
        else:
            self.b = None

        self.conv_a = layers.Conv2D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=1, 
            padding='same',
            kernel_initializer=self.w,
            use_bias=True,
            bias_initializer=self.b
        )

        self.max_pool_a = layers.MaxPool2D(pool_size=(2,2))
        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation('relu')

    def call(self, inputs):
        x = self.conv_a(inputs)
        x = self.max_pool_a(x)
        x = self.bn_a(x)
        return self.activation_fnc(x)

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

class DecoderBlockCup(layers.Layer):

    def __init__(self, target_shape, filters, normalization_rate, pool_size=(2, 1), kernel_size=3, activation=None, **kwarks):
        super(DecoderBlockCup, self).__init__(**kwarks)
        self.normalization_rate = normalization_rate
        self.target_shape = target_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        if not activation:
            activation = tf.nn.relu

        self.activation = activation

        # Layers
        self.ln_a = layers.LayerNormalization(epsilon=self.normalization_rate, name="decoder_block_cup_ln_a")
        self.reshape_a = layers.Reshape(target_shape=self.target_shape, name="decoder_block_cup_reshape_1")
        # self.conv_a = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size*2, strides=1, padding='same')
        self.conv_a = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size*2, strides=1, padding='same')
        self.max_pool_a = layers.MaxPooling2D(pool_size=self.pool_size)
        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation('relu')
        self.upsample_a = layers.UpSampling2D(
            size=(2,2), interpolation='bilinear'
        )

    def call(self, encoder_output):
        x = self.ln_a(encoder_output)
        x = self.reshape_a(x)
        
        # x = self.conv_a(x)
        # x = self.max_pool_a(x)
        # x = self.bn_a(x)
        # return self.activation_fnc(x)
        # x = self.conv_a(x)    
        # x = self.max_pool_a(x)
        # x = self.bn_a(x)
        # x = self.activation_fnc(x)
        # x = self.upsample_a(x)
        return x

class DecoderUpsampleBlock(layers.Layer):
    
    def __init__(self, filters, kernel_size=3, strides=(1, 1), pool_size=(2, 1), **kwarks):
        super(DecoderUpsampleBlock, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size

        # Layers
        self.upsample_a = layers.UpSampling2D(
            size=(16,16), interpolation='bilinear'
        )

        self.conv_a = layers.Conv2D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )

        self.max_pool_a = layers.MaxPooling2D(pool_size=self.pool_size)
        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation('relu')
        
    def call(self, decoder_input):
        # x = self.upsample_a(decoder_input)
        # x = self.upsample_a(x)

        x = self.conv_a(decoder_input)
        # x = self.conv_a(x)

        # x = self.conv_transpose_a(decoder_input)    
        # x = self.max_pool_a(x)
        # x = self.bn_a(x)
        # x = self.activation_fnc(x)
        x = self.upsample_a(x)
        return x

class DecoderSegmentationHead(layers.Layer):

    def __init__(self, filters=1, kernel_size=3, strides=1, target_shape=(256, 256, 16), **kwarks):
        super(DecoderSegmentationHead, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.target_shape = target_shape

        # Layers
        self.reshape_a = layers.Reshape(target_shape=(self.target_shape))
        self.conv_a = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='same',
        )
    
    def call(self, decoder_upsample_block):
        x = self.reshape_a(decoder_upsample_block)
        return self.conv_a(x)

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
        
    