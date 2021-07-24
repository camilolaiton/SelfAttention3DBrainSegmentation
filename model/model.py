import tensorflow as tf
from tensorflow.keras import Model, Input, layers, losses
from config import get_initial_config

class ConvolutionalBlock(layers.Layer):

    def __init__(self, idx, filters, kernel_size, padding, dropout_rate, activation, **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)
        self.__idx = idx
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__padding = padding
        self.__dropout_rate = dropout_rate
        self.__activation = activation

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
        # x3 = mlp(x3, hidden_units=self.transformer_units, dropout_rate=self.dropout_rate)
        # print(f"X2 shape: {x2.shape} X3 shape: {x3.shape}")

        return layers.Add()([x3, x2])

class DecoderBlockCup(layers.Layer):

    def __init__(self, **kwarks):
        super(DecoderBlockCup, self).__init__(**kwarks)

    def call(self):
        x = layers.LayerNormalization(epsilon=config.transformer.normalization_rate, name="ln_1")(encoded_patches)
        x = layers.Reshape(target_shape=(config.transformer.projection_dim, config.image_height//16, config.image_width//16), name="reshape_1")(x)
        
        x = layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(pool_size=(2, 1))(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation('relu')(x)

def build_model(config):
    mlp_head_units = [2048, 1024]

    inputs = Input(shape=config.image_size)
    patches = Patches(config.transformer.patch_size)(inputs)
    encoded_patches = PatchEncoder(config.transformer.num_patches, config.transformer.projection_dim)(patches)

    print(f"enconded patches size {encoded_patches.shape}")
    for idx in range(config.transformer.layers):
        # print(f"Starting {idx} iteration")
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads, 
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name="transformer_block_"+str(idx)
        )(encoded_patches)

    x = layers.LayerNormalization(epsilon=config.transformer.normalization_rate, name="ln_1")(encoded_patches)
    x = layers.Reshape(target_shape=(config.transformer.projection_dim, config.image_height//16, config.image_width//16), name="reshape_1")(x)
    
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(1, 2), padding='same')(x)    
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(1, 2), padding='same')(x)    
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(1, 2), padding='same')(x)    
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=(1, 2), padding='same')(x)    
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.UpSampling3D(size=(1, 2, 2))(x)
    
    # x = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(x)
    # x = layers.MaxPooling2D(pool_size=(2, 1))(x)
    # x = layers.Conv2DTranspose(filters=1, kernel_size=1, strides=2, activation='relu')(x)
    # x = layers.Flatten(name="flatten_1")(x)
    # x = layers.LayerNormalization(epsilon=config.transformer.normalization_rate)(x)
    # x = layers.Dense(24576)(x)
    # representation = layers.Reshape(target_shape=(config.transformer.projection_dim, config.image_height/tf.cast(tf.constant(16), tf.int32), config.image_width/tf.cast(tf.constant(16), tf.int32)), name="reshape_1")(representation)
    # representation = layers.Dropout(0.5)(representation)
    # features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # logits = layers.Dense(100)(features)

    return Model(inputs=inputs, outputs=x)

if __name__ == "__main__":
    config = get_initial_config()
    model = build_model(config)
    model.summary()