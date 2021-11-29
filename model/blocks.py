import tensorflow as tf
from tensorflow.python.keras.backend import dropout
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from .attention_block_modified import multihead_attention_3d

class RandomInvert(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, global_seed=12, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.global_seed = global_seed
        
    def call(self, inputs, training=True):
        tf.random.set_seed(self.global_seed)
        if tf.random.uniform([]) < self.prob:
            return tf.cast(255.0 - inputs, dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
        
    def get_config(self):
        config = {
            'prob': self.prob,
        }
        base_config = super(RandomInvert, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
# Custom Layer 2 
class RandomEqualize(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, global_seed=12, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
        self.global_seed = global_seed
        
    def call(self, inputs, training=True):
        tf.random.set_seed(self.global_seed)
        if tf.random.uniform([]) < self.prob:
            return tf.cast(tfa.image.equalize(inputs), dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
        
    def get_config(self):
        config = {
            'prob': self.prob,
        }
        base_config = super(RandomEqualize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape
        
# Custom Layer 3
class RandomCutout(tf.keras.layers.Layer):
    def __init__(self, prob=0.5, mask_size=(20, 20), replace=0, global_seed=12, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob 
        self.replace = replace
        self.mask_size = mask_size
        self.global_seed = global_seed
        
    def call(self, inputs, training=True):
        tf.random.set_seed(self.global_seed)
        if tf.random.uniform([]) < self.prob:
            inputs = tfa.image.random_cutout(inputs,
                                           mask_size=self.mask_size,
                                           constant_values=self.replace)  
            return tf.cast(inputs, dtype=tf.float32)
        else: 
            return tf.cast(inputs, dtype=tf.float32)
        
    def get_config(self):
        config = {
            'prob': self.prob,
            'replace': self.replace,
            'mask_size': self.mask_size
        }
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Keras3DAugmentation(layers.Layer):
    def __init__(self, global_seed, input_width, input_height, input_depth, input_channel=1, modeling='3D', **kwargs):
        super(Keras3DAugmentation, self).__init__(**kwargs)
        # Built-in Layers
        # **IMPORTANT**: Same seed to get same augmented output on each modality's slices. 
        self.global_seed = global_seed
        # rng = tf.random.Generator.from_seed(self.global_seed)
        # set_seed = rng.make_seeds()[0]

        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.input_channel = input_channel
        self.modeling = modeling

        self.random_flip = preprocessing.RandomFlip("horizontal", seed=self.global_seed)
        self.random_rotate = preprocessing.RandomRotation(factor=0.01, seed=self.global_seed)
        self.random_translation = preprocessing.RandomTranslation(height_factor=0.0, 
                                                                  width_factor=0.1, 
                                                                  seed=self.global_seed)
        self.random_contrast = preprocessing.RandomContrast(factor=0.6, seed=self.global_seed)
        self.random_crop = preprocessing.RandomCrop(int(input_height*0.90), int(input_width*0.90), 
                                                    seed=self.global_seed)
        self.resize_crop   = preprocessing.Resizing(input_height, input_width)
        self.random_height = preprocessing.RandomHeight(factor=(0.1, 0.1), seed=self.global_seed)
        self.random_width  = preprocessing.RandomWidth(factor=(0.1, 0.1), seed=self.global_seed)
        self.random_rotate = preprocessing.RandomRotation(factor=(-0.1, 0.1), fill_mode='wrap',
                                                          seed=self.global_seed)

        # CustomLayers
        # self.random_equalize = RandomEqualize(prob=0.6)
        # self.random_invert   = RandomInvert(prob=0.1)
        # self.random_cutout   = RandomCutout(prob=0.8, replace=0,
        #                                     mask_size=(int(input_height * 0.1), 
        #                                                int(input_width * 0.1)))
        
    def call(self, inputs):
        # Split the inputs wrt to input_channel 
        # For example: 224, 224, 10, 4 will be splitted by 4 (input_channel)
        # Output: [224, 224, 10, 1] * 4
        # splitted_modalities = tf.split(tf.cast(inputs, tf.float32), self.input_channel, axis=-1)
        
        # if self.modeling == '3D':
        #     # Removing the last axis, no needed for now. 
        #     splitted_modalities = [tf.squeeze(i, axis=-1) for i in splitted_modalities] 
       
        # Will contain the augmented outputs
        # flair = []
        # t1w = []
        # t1wce = []
        # t2w = []
        
        # Iterate over the the each modality 
        # for j, each_modality in enumerate(splitted_modalities):
        # print("bef: ", inputs.shape)
        inputs = tf.squeeze(inputs, axis=-1)
        # print("aft: ", inputs.shape)

        x = self.random_flip(inputs)
        # print("x: ", x.shape)
        x = self.random_rotate(x)
        x = self.random_translation(x)
        # x = self.random_cutout(x)
        x = self.random_contrast(x)
        # x = self.random_height(x)
        # x = self.random_width(x)
        x = self.random_rotate(x)
        # x = self.random_invert(x)
        # x = self.random_crop(x)
        # x = self.resize_crop(x)
            
            # if j == 0:
            #     flair.append(tf.expand_dims(x, axis=-1))
            # elif j == 1:
            #     t1w.append(tf.expand_dims(x, axis=-1))
            # elif j == 2:
            #     t1wce.append(tf.expand_dims(x, axis=-1))
            # elif j == 3:
            #     t2w.append(tf.expand_dims(x, axis=-1))

        if self.modeling == '3D':
            # image = tf.stack([flair, t1w, t1wce, t2w], axis=-1)
            # print("end: ", x.shape)
            # image = tf.reshape(x, [-1, self.input_height, self.input_width, 
                                    #    self.input_depth])
            image = tf.expand_dims(x, axis=-1)
            # print(image.shape)
            return image
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'global_seed': self.global_seed,
            'input_width' : self.input_width,
            'input_height' : self.input_height,
            'input_depth' : self.input_depth,
            'input_channel' : self.input_channel,
            # layers
            'modeling' : self.modeling
        })
        return config
        
class ConvolutionalBlock(layers.Layer):

    def __init__(self, filters, kernel_size, strides, activation='relu', padding='same', **kwargs):
        super(ConvolutionalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        # self.padding = padding
        # self.dropout_rate = dropout_rate

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu

        self.activation = activation
        
        # Layers
        self.conv_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides,
            kernel_initializer='he_normal',
            padding=padding
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
    def __init__(self, hidden_units, dropout_rate, activation='relu', **kwarks):
        super(MLPBlock, self).__init__(**kwarks)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu
        

        self.activation = activation

        # creating layers
        self.layers = []

        for units in self.hidden_units:
            self.layers.append(layers.Dense(units, activation=self.activation, kernel_initializer='he_normal'))
            self.layers.append(layers.SpatialDropout3D(self.dropout_rate))#layers.Dropout(self.dropout_rate))

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

        # batch, 64,64,64
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

def reshape_range(tensor, i, j, shape):
	"""Reshapes a tensor between dimensions i and j."""

	target_shape = tf.concat(
			[tf.shape(tensor)[:i], shape, tf.shape(tensor)[j:]],
			axis=0)

	return tf.reshape(tensor, target_shape)

def flatten_3d(x):
    x_shape = tf.shape(x)
    # print("XSHAPE: ", x_shape)
    x = reshape_range(x, 2, 5, [tf.reduce_prod(x_shape[2:5])])
    return x

def combine_last_two_dimensions(x):
	"""Reshape x so that the last two dimension become one.
	Args:
		x: a Tensor with shape [..., a, b]
	Returns:
		a Tensor with shape [..., a*b]
	"""

	old_shape = x.get_shape().dims
	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]

	ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
	ret.set_shape(new_shape)

	return ret

class msa_3d(layers.Layer):
    def __init__(self, key_filters, value_filters, output_filters, num_heads, dropout_rate=0.5, **kwarks):
        super(msa_3d, self).__init__(**kwarks)
        self.num_heads = num_heads
        self.key_filters = key_filters
        self.value_filters = value_filters
        self.output_filters = output_filters
        self.padding = 'same'
        self.key_filters_per_head = self.key_filters // self.num_heads
        self.dropout_rate = dropout_rate
        # Layers
        # attention_layer = multihead_attention_3d(x1, 512, 512, 64, 4, False, layer_type='SAME')
        self.q_conv = layers.Conv3D(
            filters=self.key_filters, 
            kernel_size=1, 
            strides=1,
            kernel_initializer='he_normal',
            use_bias=True,
            padding=self.padding
        )

        self.k_conv = layers.Conv3D(
            filters=self.key_filters, 
            kernel_size=1, 
            strides=1,
            kernel_initializer='he_normal',
            use_bias=True,
            padding=self.padding
        )

        self.v_conv = layers.Conv3D(
            filters=self.value_filters, 
            kernel_size=1, 
            strides=1,
            kernel_initializer='he_normal',
            use_bias=True,
            padding=self.padding
        )

    def call(self, inputs):
        q, k, v = self.compute_qkv_3d(inputs)
        
        q = tf.transpose(self.split_last_dimension(q, self.num_heads), [0, 4, 1, 2, 3, 5])
        k = tf.transpose(self.split_last_dimension(k, self.num_heads), [0, 4, 1, 2, 3, 5])
        v = tf.transpose(self.split_last_dimension(v, self.num_heads), [0, 4, 1, 2, 3, 5])
        
        q = q * self.key_filters_per_head**-0.5
        
        # print(q, "\n", k, "\n", v)
        output = self.global_attention_3d(q, k, v)
        # print(output)
        output = combine_last_two_dimensions(tf.transpose(output, [0, 2, 3, 4, 1, 5]))
        output = layers.Conv3D(
            filters=self.output_filters, 
            kernel_size=1, 
            strides=1,
            kernel_initializer='he_normal',
            use_bias=True,
            padding=self.padding
        )(output)

        return output

    def compute_qkv_3d(self, inputs):
        q = self.q_conv(inputs)
        k = self.k_conv(inputs)
        v = self.v_conv(inputs)
        return q, k, v

    def global_attention_3d(self, q, k, v, name=None):
        new_shape = tf.concat([tf.shape(q)[0:-1], [v.shape[-1]]], 0)
        
        q_new = flatten_3d(q)
        k_new = flatten_3d(k)
        v_new = flatten_3d(v)

        # attention
        # print(q_new, "\n", k_new)
        logits = tf.matmul(q_new, k_new, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights = tf.keras.layers.Dropout(weights, self.dropout_rate)(weights)
        output = tf.matmul(weights, v_new)
        output = tf.reshape(output, new_shape)

        return output

    def split_last_dimension(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
            x: a Tensor with shape [..., m]
            n: an integer.
        Returns:
            a Tensor with shape [..., n, m/n]
        """

        old_shape = x.get_shape().dims
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        
        return ret

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads' : self.num_heads,
            'key_filters' : self.key_filters,
            'value_filters' : self.value_filters,
            'output_filters' : self.output_filters,
            'key_filters_per_head' : self.key_filters_per_head,
            'dropout_rate' : self.dropout_rate,
            # layers
            'q_conv' : self.q_conv,
            'k_conv' : self.k_conv,
            'v_conv' : self.v_conv,
        })
        return config

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, projection_dim, dropout_rate, normalization_rate, transformer_units, activation='relu', **kwarks):
        super(TransformerBlock, self).__init__(**kwarks)
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.normalization_rate = normalization_rate
        self.transformer_units = transformer_units

        # Layers
        self.ln_a = layers.LayerNormalization(epsilon=self.normalization_rate)
        # self.attention_layer_a = layers.MultiHeadAttention(
        #     num_heads = self.num_heads,
        #     key_dim = self.projection_dim,
        #     dropout = self.dropout_rate,
        #     kernel_initializer='he_normal',
        # )

        self.attention_layer_a = msa_3d(512, 512, 64, 4, 0.2)

        self.add_a = layers.Add()

        self.ln_b = layers.LayerNormalization(epsilon=self.normalization_rate)
        self.mlp_block_b = MLPBlock(
            hidden_units=self.transformer_units, 
            dropout_rate=self.dropout_rate,
            activation='elu'
        )

        self.add_b = layers.Add()

    def call(self, encoded_patches):
        x1 = self.ln_a(encoded_patches)
        
        # attention_layer = multihead_attention_3d(x1, 512, 512, 64, 4, False, layer_type='SAME')
        attention_layer = self.attention_layer_a(x1)
        # attention_layer = self.attention_layer_a(x1, x1)
        
        x2 = self.add_a([attention_layer, encoded_patches])
        x3 = self.ln_b(x2)
        x3 = self.mlp_block_b(x3)
        x3 = self.add_b([x3, x2])

        # attention_layer = self.attention_layer_a(encoded_patches, encoded_patches)
        # x1 = self.add_a([attention_layer, encoded_patches])
        # x2 = self.ln_a(x1)

        # x3 = self.mlp_block_b(x2)
        # x3 = self.add_b([x3, x2])
        # x3 = self.ln_b(x3)
        return x3

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

    def __init__(self, target_shape, filters, normalization_rate, pool_size=(2, 2, 1), kernel_size=3, activation='relu', upsample=True, **kwarks):
        super(DecoderBlockCup, self).__init__(**kwarks)
        self.normalization_rate = normalization_rate
        self.target_shape = target_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.upsample = upsample

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
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
            kernel_size=self.kernel_size,#*2, 
            strides=1, 
            kernel_initializer='he_normal',
            padding='same'
        )
        # self.max_pool_a = layers.MaxPooling3D(pool_size=self.pool_size)
        self.bn_a = layers.BatchNormalization()
        self.activation_fnc = layers.Activation(self.activation)
        self.upsample_a = layers.UpSampling3D(
            size=(2, 2, 2)
        )

    def call(self, encoder_output):
        # encoder_output = self.ln_a(encoder_output)
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

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu

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

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu
            
        self.activation = activation

        # Layers
        self.upsample_a = layers.UpSampling3D(
            size=(2, 2, 2)
        )

        self.conv_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            strides=self.strides, 
            kernel_initializer='he_normal',
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
            kernel_initializer='he_normal',
            padding='same'
        )

        self.activation_layer = layers.Activation(self.activation, dtype='float32')
    
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
    def __init__(self, filters, kernel_size, activation='relu', **kwarks):
        super(ConnectionComponents, self).__init__(**kwarks)

        self.filters = filters
        self.kernel_size = kernel_size

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu

        self.activation = activation

        self.conv_1_a = layers.Conv3D(
            filters=self.filters, 
            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size), 
            strides=1, 
            kernel_initializer='he_normal',
            padding='same'
        )

        self.conv_1_b = layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            strides=1,
            kernel_initializer='he_normal',
            padding='same'
        )

        self.activation_layer = layers.Activation(self.activation)
        self.activation_layer_2 = layers.Activation(self.activation)

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
    
    def __init__(self, filters, kernel_size, upsample=True, activation='relu', **kwarks):
        super(EncoderDecoderConnections, self).__init__(**kwarks)
        self.filters = filters
        self.kernel_size = kernel_size
        self.upsample = upsample

        if (activation == 'leaky_relu'):
            activation = tf.nn.leaky_relu
        elif (activation == 'elu'):
            activation = tf.nn.elu
        else:
            activation = tf.nn.relu

        self.activation = activation

        # self.concatenate = layers.Concatenate()
        self.upsample_lyr = layers.UpSampling3D(
            size=(2, 2, 2)
        )

        self.con_comp_1 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size,
            activation=activation,
        )

        self.con_comp_2 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size,
            activation=activation,
        )

        self.con_comp_3 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size,
            activation=activation,
        )

        self.con_comp_4 = ConnectionComponents(
            filters=self.filters, 
            kernel_size=self.kernel_size,
            activation=activation,
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
        
    
