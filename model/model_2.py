import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from .blocks import *
from .config import *

def model_local_path(config, inputs):
    # [First path]

    conv_layers = inputs
    conv_blocks = []

    for filters in config.enc_filters:
        
        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            activation=config.act_func,
            name=f"conv_block_{filters}_stride1_0"
        )(conv_layers)
        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            activation=config.act_func,
            name=f"conv_block_{filters}_stride1_1"
        )(conv_layers)

        conv_blocks.append(conv_layers)

        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=2,
            activation=config.act_func,
            name=f"down_conv_block_{filters}"
        )(conv_layers)

    conv_proj = ConvProjection(
        config.conv_projection,#config.transformer.projection_dim,
        config.transformer.projection_dim,
        num_patches=config.transformer.num_patches, #512
        name='conv_projection'
    )(conv_layers)

    transformer_layers_path_1 = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        conv_proj = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units,
            activation='relu',
            name=f"transformer_block_{idx}"
        )(conv_proj)
        transformer_layers_path_1.append(conv_proj)

    dec = config.transformer.patch_size
    decoder_block_cup = DecoderBlockCup(
        target_shape=(
            config.image_height//dec, 
            config.image_width//dec, 
            config.image_depth//dec, 
            config.transformer.projection_dim
        ),
        filters=config.transformer.projection_dim,
        kernel_size=7,
        normalization_rate=config.transformer.normalization_rate,
        upsample=False,
        name=f'decoder_cup_{0}'
    )(conv_proj)

    deconv_layers = decoder_block_cup

    deconv_layers = ConvolutionalBlock(
        filters=config.dec_filters[0],
        kernel_size=3,
        strides=1,
        activation=config.act_func,
        name=f"deconv_block_{config.dec_filters[0]}_stride1_0"
    )(deconv_layers)

    deconv_layers = ConvolutionalBlock(
        filters=config.dec_filters[0],
        kernel_size=3,
        strides=1,
        activation=config.act_func,
        name=f"deconv_block_{config.dec_filters[0]}_stride1_1"
    )(deconv_layers)

    if (config.decoder_conv_localpath):
        deconv_layers = DecoderTransposeBlock(
            filters=config.dec_filters[0],
            activation=config.act_func,
            name=f"transpose_{config.dec_filters[0]}"
        )(deconv_layers)
    else:
        deconv_layers = DecoderUpsampleBlock(
            filters=config.dec_filters[0], 
            kernel_size=3,
            activation=config.act_func,
            name=f"upsample_{config.dec_filters[0]}"
        )(deconv_layers)

    if (config.skip_connections):

        skip_conn_1 = EncoderDecoderConnections(
            filters=config.dec_filters[0],
            kernel_size=3,
            upsample=False,
            activation=config.act_func,
            name=f"skip_connection_{config.dec_filters[0]}"
        )(conv_blocks[-1])

        # print("conv block: ", conv_blocks[-1-i].shape, " ", skip_conn_1.shape)
        # print("deconv block: ", deconv_layers.shape)
        deconv_layers = layers.Concatenate()([skip_conn_1, deconv_layers])

    i = 1

    for filters in config.dec_filters[1:]:
        shape = deconv_layers.shape[-1]

        deconv_layers = ConvolutionalBlock(
            filters=shape/2,
            kernel_size=3,
            strides=1,
            activation=config.act_func,
            name=f"deconv_block_{filters}_stride1_0"
        )(deconv_layers)

        deconv_layers = ConvolutionalBlock(
            filters=shape/4,
            kernel_size=3,
            strides=1,
            activation=config.act_func,
            name=f"deconv_block_{filters}_stride1_1"
        )(deconv_layers)

        if (config.decoder_conv_localpath):
            deconv_layers = DecoderTransposeBlock(
                filters=filters,
                activation=config.act_func,
                name=f"transpose_{filters}"
            )(deconv_layers)
        else:
            deconv_layers = DecoderUpsampleBlock(
                filters=filters, 
                kernel_size=3,
                activation=config.act_func,
                name=f"upsample_{filters}"
            )(deconv_layers)

        if (config.skip_connections):

            skip_conn_1 = EncoderDecoderConnections(
                filters=filters,
                kernel_size=3,
                upsample=False,
                activation=config.act_func,
                name=f"skip_connection_{filters}"
            )(conv_blocks[-1-i])

            # print("conv block: ", conv_blocks[-1-i].shape, " ", skip_conn_1.shape)
            # print("deconv block: ", deconv_layers.shape)
            i += 1
            deconv_layers = layers.Concatenate()([skip_conn_1, deconv_layers])

    return deconv_layers

def model_local_path_2(config, inputs):
    # [First path]

    conv_layers = inputs
    conv_blocks = []

    for filters in config.enc_filters:
        
        # conv_layers = ConvolutionalBlock(
        #     filters=filters,
        #     kernel_size=3,
        #     strides=1,
        #     activation=config.act_func,
        #     name=f"conv_block_{filters}_stride1_0"
        # )(conv_layers)
        # conv_layers = ConvolutionalBlock(
        #     filters=filters,
        #     kernel_size=3,
        #     strides=1,
        #     activation=config.act_func,
        #     name=f"conv_block_{filters}_stride1_1"
        # )(conv_layers)

        conv_blocks.append(conv_layers)

        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=2,
            activation=config.act_func,
            name=f"down_conv_block_{filters}"
        )(conv_layers)

    conv_proj = ConvProjection(
        config.conv_projection,#config.transformer.projection_dim,
        config.transformer.projection_dim,
        num_patches=config.transformer.num_patches, #512
        name='conv_projection'
    )(conv_layers)

    transformer_layers_path_1 = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        conv_proj = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units,
            activation='relu',
            name=f"transformer_block_{idx}"
        )(conv_proj)
        transformer_layers_path_1.append(conv_proj)

    dec = config.transformer.patch_size
    decoder_block_cup = DecoderBlockCup(
        target_shape=(
            config.image_height//dec, 
            config.image_width//dec, 
            config.image_depth//dec, 
            config.transformer.projection_dim
        ),
        filters=config.transformer.projection_dim,
        kernel_size=7,
        normalization_rate=config.transformer.normalization_rate,
        upsample=False,
        name=f'decoder_cup_{0}'
    )(conv_proj)

    deconv_layers = decoder_block_cup

    # deconv_layers = ConvolutionalBlock(
    #     filters=config.dec_filters[0],
    #     kernel_size=3,
    #     strides=1,
    #     activation=config.act_func,
    #     name=f"deconv_block_{config.dec_filters[0]}_stride1_0"
    # )(deconv_layers)

    # deconv_layers = ConvolutionalBlock(
    #     filters=config.dec_filters[0],
    #     kernel_size=3,
    #     strides=1,
    #     activation=config.act_func,
    #     name=f"deconv_block_{config.dec_filters[0]}_stride1_1"
    # )(deconv_layers)

    if (config.decoder_conv_localpath):
        deconv_layers = DecoderTransposeBlock(
            filters=config.dec_filters[0],
            activation=config.act_func,
            name=f"transpose_{config.dec_filters[0]}"
        )(deconv_layers)
    else:
        deconv_layers = DecoderUpsampleBlock(
            filters=config.dec_filters[0], 
            kernel_size=3,
            activation=config.act_func,
            name=f"upsample_{config.dec_filters[0]}"
        )(deconv_layers)

    if (config.skip_connections):

        skip_conn_1 = EncoderDecoderConnections(
            filters=config.dec_filters[0],
            kernel_size=3,
            upsample=False,
            activation=config.act_func,
            name=f"skip_connection_{config.dec_filters[0]}"
        )(conv_blocks[-1])

        # print("conv block: ", conv_blocks[-1-i].shape, " ", skip_conn_1.shape)
        # print("deconv block: ", deconv_layers.shape)
        deconv_layers = layers.Concatenate()([skip_conn_1, deconv_layers])

    i = 1

    for filters in config.dec_filters[1:]:
        shape = deconv_layers.shape[-1]

        # deconv_layers = ConvolutionalBlock(
        #     filters=shape/2,
        #     kernel_size=3,
        #     strides=1,
        #     activation=config.act_func,
        #     name=f"deconv_block_{filters}_stride1_0"
        # )(deconv_layers)

        # deconv_layers = ConvolutionalBlock(
        #     filters=shape/4,
        #     kernel_size=3,
        #     strides=1,
        #     activation=config.act_func,
        #     name=f"deconv_block_{filters}_stride1_1"
        # )(deconv_layers)

        if (config.decoder_conv_localpath):
            deconv_layers = DecoderTransposeBlock(
                filters=filters,
                activation=config.act_func,
                name=f"transpose_{filters}"
            )(deconv_layers)
        else:
            deconv_layers = DecoderUpsampleBlock(
                filters=filters, 
                kernel_size=3,
                activation=config.act_func,
                name=f"upsample_{filters}"
            )(deconv_layers)

        if (config.skip_connections):

            skip_conn_1 = EncoderDecoderConnections(
                filters=filters,
                kernel_size=3,
                upsample=False,
                activation=config.act_func,
                name=f"skip_connection_{filters}"
            )(conv_blocks[-1-i])

            # print("conv block: ", conv_blocks[-1-i].shape, " ", skip_conn_1.shape)
            # print("deconv block: ", deconv_layers.shape)
            i += 1
            deconv_layers = layers.Concatenate()([skip_conn_1, deconv_layers])

    return deconv_layers

def model_global_path(config, inputs):
    # Split the volume into multiple volumes of 16x16x16
    patches = Patches(
        patch_size=config.transformer.patch_size, 
        name='patches_0'
    )(inputs)

    # Adding positional encoding to patches
    encoded_patches = PatchEncoder(
        num_patches=config.transformer.num_patches,
        projection_dim=config.transformer.projection_dim,
        name='encoded_patches_0',
    )(patches)

    transformer_layers_path_2 = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_path_2{idx}"
        )(encoded_patches)
        transformer_layers_path_2.append(encoded_patches)
    
    # Reshape and initiating decoder
    # (64, 128) => (4, 4, 4, 128) => (8, 8, 8, 128)
    dec = config.transformer.patch_size
    decoder_block_cup = DecoderBlockCup(
        target_shape=(
            config.image_height//dec, 
            config.image_width//dec, 
            config.image_depth//dec, 
            config.transformer.projection_dim
        ),
        filters=config.transformer.projection_dim,
        normalization_rate=config.transformer.normalization_rate,
        activation=config.act_func,
        name=f'decoder_cup_path_2_0'
    )(encoded_patches)

    # Upsample layer : (8, 8, 8, 64) => (16, 16, 16, 32)
    decoder_up_block_0 = None
    if (config.decoder_conv_globalpath):
        decoder_up_block_0 = DecoderTransposeBlock(
            filters=32,
            activation=config.act_func,
            name=f"decoder_trans_0"
        )(decoder_block_cup)
    else:
        decoder_up_block_0 = DecoderUpsampleBlock(
            filters=32, 
            kernel_size=3,
            activation=config.act_func,
            name=f"decoder_upsample_0"
        )(decoder_block_cup)

    # Skip connection with transformer layer
    if (config.skip_connections):
        skip_conn_0 = DecoderBlockCup(
            target_shape=(
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim
            ),
            filters=config.transformer.projection_dim,
            normalization_rate=None,
            activation=config.act_func,
            name='reshaping_trans_skip_0'
        )(transformer_layers_path_2[-2])

        skip_conn_0 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            activation=config.act_func,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (16, 16, 16, 32) => (32, 32, 32, 16)
    decoder_up_block_1 = None
    if (config.decoder_conv_globalpath):
        decoder_up_block_1 = DecoderTransposeBlock(
            filters=16,
            activation=config.act_func,
            name=f"decoder_trans_1"
        )(decoder_up_block_0)
    else:
        decoder_up_block_1 = DecoderUpsampleBlock(
            filters=16, 
            kernel_size=3,
            activation=config.act_func,
            name=f"decoder_upsample_1"
        )(decoder_up_block_0)

    # Skip connection with transformer layer
    if (config.skip_connections):
        skip_conn_1 = DecoderBlockCup(
            target_shape=(
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim
            ),
            filters=config.transformer.projection_dim,
            normalization_rate=None,
            activation=config.act_func,
            name='reshaping_trans_skip_1'
        )(transformer_layers_path_2[-3])

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            activation=config.act_func,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            activation=config.act_func,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    # Upsample layer : (32, 32, 32, 16) => (64, 64, 64, 8)
    decoder_up_block_2 = None
    if (config.decoder_conv_globalpath):
        decoder_up_block_2 = DecoderTransposeBlock(
            filters=8,
            activation=config.act_func,
            name=f"decoder_trans_2"
        )(decoder_up_block_1)
    else:
        decoder_up_block_2 = DecoderUpsampleBlock(
            filters=8, 
            kernel_size=3,
            activation=config.act_func,
            name="decoder_upsample_2"
        )(decoder_up_block_1)

    # Skip connection with transformer layer
    if (config.skip_connections):
        skip_conn_2 = DecoderBlockCup(
            target_shape=(
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim
            ),
            filters=config.transformer.projection_dim,
            normalization_rate=None,
            activation=config.act_func,
            name='reshaping_trans_skip_2'
        )(transformer_layers_path_2[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            activation=config.act_func,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            activation=config.act_func,
            name="skip_connection_2_1"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=8,
            kernel_size=3,
            activation=config.act_func,
            name="skip_connection_2_2"
        )(skip_conn_2)

        decoder_up_block_2 = layers.Add()([decoder_up_block_2, skip_conn_2])

    return decoder_up_block_2

def build_model(config):
    # Here I have an input size of 64x64x64x1
    inputs = Input(shape=config.image_size)
    
    # Data augmentation
    data_aug = None
    if (config.data_augmentation):
        data_aug = Keras3DAugmentation(
            12, 
            config.image_width, 
            config.image_height, 
            config.image_channels, 
            name='data_aug'
        )(inputs)
    else:
        data_aug = inputs

    local_path = model_local_path_2(config, data_aug)

    # global_path = model_global_path(config, inputs)

    # head = layers.Add()([local_path, global_path])

    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(local_path)#(head)

    return Model(inputs, segmentation_head)

def main():
    config = get_config_local_path()#get_config_test()
    model = build_model(config)
    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file='test_model.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

if __name__ == "__main__":
    main()