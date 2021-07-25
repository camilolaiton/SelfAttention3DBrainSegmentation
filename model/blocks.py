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

    def call(self, inputs):
        x = layers.Conv2D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=1, 
            padding='same'
        )(inputs)
        x = layers.MaxPool2D(pool_size=(2,2))(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation('relu')(x)

class MLPBlock(layers.Layer):
    def __init__(self, hidden_units, dropout_rate, activation=None, **kwarks):
        super(MLPBlock, self).__init__(**kwarks)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        if not activation:
            activation = tf.nn.gelu

        self.activation = activation

    def call(self, inputs):
        for units in self.hidden_units:
            inputs = layers.Dense(units, activation=self.activation)(inputs)
            inputs = layers.Dropout(self.dropout_rate)(inputs)

        return inputs

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
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

    def call(self, encoded_patches):
        x1 = layers.LayerNormalization(epsilon=self.normalization_rate)(encoded_patches)
        attention_layer = layers.MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim = self.projection_dim,
            dropout = self.dropout_rate,
        )(x1, x1)
        
        x2 = layers.Add()([attention_layer, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=self.normalization_rate)(x2)
        x3 = MLPBlock(hidden_units=self.transformer_units, dropout_rate=self.dropout_rate)(x3)
        return layers.Add()([x3, x2])

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
        

    def call(self, encoder_output):
        x = layers.LayerNormalization(epsilon=self.normalization_rate, name="ln_1")(encoder_output)
        x = layers.Reshape(target_shape=self.target_shape, name="reshape_1")(x)
        
        x = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding='same')(x)
        x = layers.MaxPooling2D(pool_size=self.pool_size)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation('relu')(x)

class DecoderUpsampleBlock(layers.Layer):
    
    def __init__(self, filters, kernel_size=3, strides=(1, 2), pool_size=(2, 1), **kwarks):
        super(DecoderUpsampleBlock, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size

    def call(self, decoder_input):
        x = layers.Conv2DTranspose(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            padding='same'
        )(decoder_input)    
        x = layers.MaxPooling2D(pool_size=self.pool_size)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation('relu')(x)