import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.callbacks import ModelCheckpoint
from .blocks import *
from .config import *
from .losses import dice_coef_3cat, IoU_coef
from tensorflow.keras import models
import segmentation_models as sm
import numpy as np

# import tensorflow_addons as tfa
def build_model_patchsize_16(config):
    # Here I have an input size of 256x256x256x1
    inputs = Input(shape=config.image_size)

    # Split the volume into multiple volumes of 16x16x16
    patches = Patches(
        patch_size=config.transformer.patch_size, 
        name='patches_0'
    )(inputs)

    # Adding positional encoding to patches
    # ( (256**3)x(16**3) = 4096)
    encoded_patches = PatchEncoder(
        num_patches=config.transformer.num_patches,
        projection_dim=config.transformer.projection_dim,
        name='encoded_patches_0',
    )(patches)

    transformer_layers = []
    # Successive transformer layers
    # Layers of (4096, 192)
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(encoded_patches)
        transformer_layers.append(encoded_patches)

    # Reshape and initiating decoder
    # (4096, 192) => (16, 16, 16, 192) => (32, 32, 32, 64)
    dec = 16
    decoder_block_cup = DecoderBlockCup(
        target_shape=(
            config.image_height//dec, 
            config.image_width//dec, 
            config.image_depth//dec, 
            config.transformer.projection_dim
        ),
        filters=64,
        normalization_rate=config.transformer.normalization_rate,
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    # Upsample layer : (32, 32, 32, 64) => (64, 64, 64, 32)
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
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
            filters=64,
            normalization_rate=None,
            name='reshaping_trans_skip_0'
        )(transformer_layers[-3])

        skip_conn_0 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (64, 64, 64, 32) => (128, 128, 128, 16)
    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=16, 
        kernel_size=3,
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
            filters=64,
            normalization_rate=None,
            name='reshaping_trans_skip_1'
        )(transformer_layers[-5])

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    # Upsample layer : (128, 128, 128, 16) => (256, 256, 256, 8)
    decoder_up_block_2 = DecoderUpsampleBlock(
        filters=8, 
        kernel_size=3,
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
            filters=64,
            normalization_rate=None,
            name='reshaping_trans_skip_2'
        )(transformer_layers[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=8,
            kernel_size=3,
            name="skip_connection_2_2"
        )(skip_conn_2)

        decoder_up_block_2 = layers.Add()([decoder_up_block_2, skip_conn_2])

    # Segmentation Head : (256, 256, 256, 8) => (256, 256, 256, 4)
    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(decoder_up_block_2)

    return Model(inputs=inputs, outputs=segmentation_head)

def build_model_patchsize_64(config):
    # Here I have an input size of 256x256x256x1
    inputs = Input(shape=config.image_size)

    # Split the volume into multiple volumes of 16x16x16
    patches = Patches(
        patch_size=config.transformer.patch_size, 
        name='patches_0'
    )(inputs)

    # Adding positional encoding to patches
    # ( (256**3)x(16**3) = 4096)
    encoded_patches = PatchEncoder(
        num_patches=config.transformer.num_patches,
        projection_dim=config.transformer.projection_dim,
        name='encoded_patches_0',
    )(patches)

    transformer_layers = []
    # Successive transformer layers
    # Layers of (4096, 192)
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(encoded_patches)
        transformer_layers.append(encoded_patches)

    # Reshape and initiating decoder
    # (4096, 192) => (16, 16, 16, 192) => (32, 32, 32, 64)
    dec = 16
    decoder_block_cup = DecoderBlockCup(
        target_shape=(
            config.image_height//dec, 
            config.image_width//dec, 
            config.image_depth//dec, 
            config.transformer.projection_dim
        ),
        filters=64,
        normalization_rate=config.transformer.normalization_rate,
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    # Upsample layer : (32, 32, 32, 64) => (64, 64, 64, 32)
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
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
            filters=64,
            normalization_rate=None,
            name='reshaping_trans_skip_0'
        )(transformer_layers[-3])

        skip_conn_0 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (64, 64, 64, 32) => (128, 128, 128, 16)
    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=16, 
        kernel_size=3,
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
            filters=64,
            normalization_rate=None,
            name='reshaping_trans_skip_1'
        )(transformer_layers[-5])

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    # Upsample layer : (128, 128, 128, 16) => (256, 256, 256, 8)
    decoder_up_block_2 = DecoderUpsampleBlock(
        filters=8, 
        kernel_size=3,
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
            filters=64,
            normalization_rate=None,
            name='reshaping_trans_skip_2'
        )(transformer_layers[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=8,
            kernel_size=3,
            name="skip_connection_2_2"
        )(skip_conn_2)

        decoder_up_block_2 = layers.Add()([decoder_up_block_2, skip_conn_2])

    # Segmentation Head : (256, 256, 256, 8) => (256, 256, 256, 4)
    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(decoder_up_block_2)

    return Model(inputs=inputs, outputs=segmentation_head)

def build_model_patchsize_32(config):
    # Here I have an input size of 256x256x256x1
    inputs = Input(shape=config.image_size)

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

    transformer_layers = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(encoded_patches)
        transformer_layers.append(encoded_patches)

    # Reshape and initiating decoder
    # (512, 128) => (16, 16, 16, 128)
    dec = config.transformer.patch_size
    decoder_block_cup = DecoderBlockCup(
        target_shape=(
            config.image_height//dec, 
            config.image_width//dec, 
            config.image_depth//dec, 
            config.transformer.projection_dim
        ),
        filters=128,
        normalization_rate=config.transformer.normalization_rate,
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    # Upsample layer : (16, 16, 16, 128) => (32, 32, 32, 64)
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=64, 
        kernel_size=3,
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
            filters=128,
            normalization_rate=None,
            name='reshaping_trans_skip_0'
        )(transformer_layers[-1])

        skip_conn_0 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (32, 32, 32, 64) => (64, 64, 64, 32)
    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
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
            filters=128,
            normalization_rate=None,
            name='reshaping_trans_skip_1'
        )(transformer_layers[-2])

        skip_conn_1 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    # Upsample layer : (64, 64, 64, 32) => (128, 128, 128, 16)
    decoder_up_block_2 = DecoderUpsampleBlock(
        filters=16, 
        kernel_size=3,
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
            filters=128,
            normalization_rate=None,
            name='reshaping_trans_skip_2'
        )(transformer_layers[-3])

        skip_conn_2 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_2_2"
        )(skip_conn_2)

        decoder_up_block_2 = layers.Add()([decoder_up_block_2, skip_conn_2])
    
    # Upsample layer : (128, 128, 128, 16) => (256, 256, 256, 8)
    decoder_up_block_3 = DecoderUpsampleBlock(
        filters=8, 
        kernel_size=3,
        name="decoder_upsample_3"
    )(decoder_up_block_2)

    # Skip connection with transformer layer
    if (config.skip_connections):
        skip_conn_3 = DecoderBlockCup(
            target_shape=(
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim
            ),
            filters=128,
            normalization_rate=None,
            name='reshaping_trans_skip_3'
        )(transformer_layers[-4])

        skip_conn_3 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_3_0"
        )(skip_conn_3)

        skip_conn_3 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_3_1"
        )(skip_conn_3)

        skip_conn_3 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_3_2"
        )(skip_conn_3)

        skip_conn_3 = EncoderDecoderConnections(
            filters=8,
            kernel_size=3,
            name="skip_connection_3_3"
        )(skip_conn_3)

        decoder_up_block_3 = layers.Add()([decoder_up_block_3, skip_conn_3])

    # Segmentation Head : (256, 256, 256, 8) => (256, 256, 256, 4)
    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(decoder_up_block_3)

    return Model(inputs=inputs, outputs=segmentation_head)

def build_model_patchified_patchsize16(config):
    # Here I have an input size of 64x64x64x1
    inputs = Input(shape=config.image_size)

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

    transformer_layers = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(encoded_patches)
        transformer_layers.append(encoded_patches)
    
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
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    # Upsample layer : (8, 8, 8, 128) => (16, 16, 16, 64)
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=64, 
        kernel_size=3,
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
            name='reshaping_trans_skip_0'
        )(transformer_layers[-3])

        skip_conn_0 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (16, 16, 16, 64) => (32, 32, 32, 32)
    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
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
            name='reshaping_trans_skip_1'
        )(transformer_layers[-5])

        skip_conn_1 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    # Upsample layer : (32, 32, 32, 32) => (64, 64, 64, 16)
    decoder_up_block_2 = DecoderUpsampleBlock(
        filters=16, 
        kernel_size=3,
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
            name='reshaping_trans_skip_2'
        )(transformer_layers[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_2_2"
        )(skip_conn_2)

        decoder_up_block_2 = layers.Add()([decoder_up_block_2, skip_conn_2])

    # Segmentation Head : (64, 64, 64, 16) => (64, 64, 64, 4)
    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(decoder_up_block_2)

    return Model(inputs=inputs, outputs=segmentation_head)

def build_model_patchified_patchsize8(config):
    # Here I have an input size of 64x64x64x1
    inputs = Input(shape=config.image_size)

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

    transformer_layers = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(encoded_patches)
        transformer_layers.append(encoded_patches)
    
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
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    # Upsample layer : (8, 8, 8, 128) => (16, 16, 16, 64)
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=64, 
        kernel_size=3,
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
            name='reshaping_trans_skip_0'
        )(transformer_layers[-3])

        skip_conn_0 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (16, 16, 16, 64) => (32, 32, 32, 32)
    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
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
            name='reshaping_trans_skip_1'
        )(transformer_layers[-5])

        skip_conn_1 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    conv_block_1 = ConvolutionalBlock(
        filters=16,
        kernel_size=(1, 1, 1),
        strides=(1, 1, 1),
        name='conv_block_1'
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
            name='reshaping_trans_skip_2'
        )(transformer_layers[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        conv_block_1 = layers.Add()([conv_block_1, skip_conn_2])

    # Segmentation Head : (64, 64, 64, 16) => (64, 64, 64, 4)
    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(conv_block_1)

    return Model(inputs=inputs, outputs=segmentation_head)

def test_model(config):
    # Here I have an input size of 64x64x64x1
    inputs = Input(shape=config.image_size)

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

    transformer_layers = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        encoded_patches = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(encoded_patches)
        transformer_layers.append(encoded_patches)

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
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    conv_tr_1 = DecoderTransposeBlock(
        filters=64,
        kernel_size=3,
        name='conv_tr_1'
    )(decoder_block_cup)

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
            name='reshaping_trans_skip_1'
        )(transformer_layers[-2])

        skip_conn_1 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        conv_tr_1 = layers.Add()([conv_tr_1, skip_conn_1])

    conv_tr_2 = DecoderTransposeBlock(
        filters=32,
        kernel_size=3,
        name='conv_tr_2'
    )(conv_tr_1)

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
            name='reshaping_trans_skip_2'
        )(transformer_layers[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=64,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        conv_tr_2 = layers.Add()([conv_tr_2, skip_conn_2])

    conv_tr_3 = DecoderTransposeBlock(
        filters=16,
        kernel_size=3,
        name='conv_tr_3'
    )(conv_tr_2)

    conv_tr_3 = layers.Dropout(config.dropout)(conv_tr_3)

    # Segmentation Head : (64, 64, 64, 16) => (64, 64, 64, 4)
    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(conv_tr_3)

    return Model(inputs, segmentation_head)

def test_model_2(config):
    # Here I have an input size of 64x64x64x1
    inputs = Input(shape=config.image_size)

    conv_layers = inputs
    conv_blocks = []

    for filters in [8, 16, 32, 64]:

        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"conv_block_{filters}_stride1_0"
        )(conv_layers)
        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"conv_block_{filters}_stride1_1"
        )(conv_layers)

        conv_blocks.append(conv_layers)

        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=2,
            name=f"down_conv_block_{filters}"
        )(conv_layers)

    conv_proj = ConvProjection(
        config.transformer.projection_dim,
        config.transformer.projection_dim,
        num_patches=config.transformer.num_patches,
        name='conv_projection'
    )(conv_layers)

    transformer_layers = []
    # Successive transformer layers
    for idx in range(config.transformer.layers):
        conv_proj = TransformerBlock(
            num_heads=config.transformer.num_heads,
            projection_dim=config.transformer.projection_dim, 
            dropout_rate=config.transformer.dropout_rate, 
            normalization_rate=config.transformer.normalization_rate, 
            transformer_units=config.transformer.units, 
            name=f"transformer_block_{idx}"
        )(conv_proj)
        transformer_layers.append(conv_proj)

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
        name=f'decoder_cup_{0}'
    )(conv_proj)

    deconv_layers = decoder_block_cup

    i = 0
    for filters in [64, 32, 16, 8]:
        deconv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"deconv_block_{filters}_stride1_0"
        )(deconv_layers)

        deconv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"deconv_block_{filters}_stride1_1"
        )(deconv_layers)

        # deconv_layers = DecoderTransposeBlock(
        #     filters=filters,
        #     name=f"up_transpose_{filters}"
        # )(deconv_layers)

        deconv_layers = DecoderUpsampleBlock(
            filters=filters, 
            kernel_size=3,
            name=f"up_transpose_{filters}"
        )(deconv_layers)

        if (config.skip_connections):

            skip_conn_1 = EncoderDecoderConnections(
                filters=filters,
                kernel_size=3,
                upsample=False,
                name=f"skip_connection_{filters}"
            )(conv_blocks[-1-i])

            # print("conv block: ", conv_blocks[-1-i].shape, " ", skip_conn_1.shape)
            # print("deconv block: ", deconv_layers.shape)
            i += 1
            deconv_layers = layers.Add()([skip_conn_1, deconv_layers])

    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(deconv_layers)

    return Model(inputs, segmentation_head)

def test_model_3(config):
    # Here I have an input size of 64x64x64x1
    inputs = Input(shape=config.image_size)

    # [First path]
    conv_layers = inputs
    conv_blocks = []

    for filters in [8, 16, 32, 64]:

        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"conv_block_{filters}_stride1_0"
        )(conv_layers)
        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"conv_block_{filters}_stride1_1"
        )(conv_layers)

        conv_blocks.append(conv_layers)

        conv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=2,
            name=f"down_conv_block_{filters}"
        )(conv_layers)

    conv_proj = ConvProjection(
        config.transformer.projection_dim,
        config.transformer.projection_dim,
        num_patches=config.transformer.num_patches,
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
        normalization_rate=config.transformer.normalization_rate,
        upsample=False,
        name=f'decoder_cup_{0}'
    )(conv_proj)

    deconv_layers = decoder_block_cup

    i = 0
    for filters in [64, 32, 16, 8]:
        deconv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"deconv_block_{filters}_stride1_0"
        )(deconv_layers)

        deconv_layers = ConvolutionalBlock(
            filters=filters,
            kernel_size=3,
            strides=1,
            name=f"deconv_block_{filters}_stride1_1"
        )(deconv_layers)

        # deconv_layers = DecoderTransposeBlock(
        #     filters=filters,
        #     name=f"up_transpose_{filters}"
        # )(deconv_layers)

        deconv_layers = DecoderUpsampleBlock(
            filters=filters, 
            kernel_size=3,
            name=f"up_transpose_{filters}"
        )(deconv_layers)

        if (config.skip_connections):

            skip_conn_1 = EncoderDecoderConnections(
                filters=filters,
                kernel_size=3,
                upsample=False,
                name=f"skip_connection_{filters}"
            )(conv_blocks[-1-i])

            # print("conv block: ", conv_blocks[-1-i].shape, " ", skip_conn_1.shape)
            # print("deconv block: ", deconv_layers.shape)
            i += 1
            deconv_layers = layers.Add()([skip_conn_1, deconv_layers])

    # [second path]

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
        name=f'decoder_cup_path_2_0'
    )(encoded_patches)

    # Upsample layer : (8, 8, 8, 64) => (16, 16, 16, 32)
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
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
            name='reshaping_trans_skip_0'
        )(transformer_layers_path_2[-2])

        skip_conn_0 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_0_0"
        )(skip_conn_0)

        decoder_up_block_0 = layers.Add()([decoder_up_block_0, skip_conn_0])

    # Upsample layer : (16, 16, 16, 32) => (32, 32, 32, 16)
    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=16, 
        kernel_size=3,
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
            name='reshaping_trans_skip_1'
        )(transformer_layers_path_2[-3])

        skip_conn_1 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_1_0"
        )(skip_conn_1)

        skip_conn_1 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_1_1"
        )(skip_conn_1)

        decoder_up_block_1 = layers.Add()([decoder_up_block_1, skip_conn_1])

    # Upsample layer : (32, 32, 32, 16) => (64, 64, 64, 8)
    decoder_up_block_2 = DecoderUpsampleBlock(
        filters=8, 
        kernel_size=3,
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
            name='reshaping_trans_skip_2'
        )(transformer_layers_path_2[0])

        skip_conn_2 = EncoderDecoderConnections(
            filters=32,
            kernel_size=3,
            name="skip_connection_2_0"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=16,
            kernel_size=3,
            name="skip_connection_2_1"
        )(skip_conn_2)

        skip_conn_2 = EncoderDecoderConnections(
            filters=8,
            kernel_size=3,
            name="skip_connection_2_2"
        )(skip_conn_2)

        decoder_up_block_2 = layers.Add()([decoder_up_block_2, skip_conn_2])

    deconv_layers = layers.Add()([deconv_layers, decoder_up_block_2])

    segmentation_head = DecoderSegmentationHead(
        filters=config.n_classes, 
        kernel_size=1,
        name="segmentation_head"
    )(deconv_layers)

    return Model(inputs, segmentation_head)

def plot_model(model, input_shape=(None, 256, 256, 1)):
    all_layers = []
    for layer in model.layers:
        try: 
            all_layers.extend(layer.layers)
        except AttributeError as e:
            print(e, " ", layer.name)

    model_plot = tf.keras.models.Sequential(all_layers)
    model_plot.build(input_shape)
    print(list(all_layers))
    tf.keras.utils.plot_model(
        model_plot,
        to_file="model/model_architecture_2.png",
        show_shapes=True,
        show_dtype=True,
        # show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )

if __name__ == "__main__":
    config_num = 1
    architecture_image_name = f"model/model_architecture_skip_connection_{config_num}.png"
    config = None
    model = None

    if (config_num == 1):
        config = get_config_32()
        model = build_model_patchsize_32(config)
        # model = build_model_test(config)


    # elif (config_num == 2):
    #     config = get_config_2()
    #     model = build_model_2(config)

    # else:
    #     config = get_config_3()
    #     model = build_model_3(config)

    print(f"[+] Building model {config.config_name} with config {config}")    
    model.summary()

    if not (config.skip_connections):
        architecture_image_name = architecture_image_name.replace("_skip_connection", "")
    
    tf.keras.utils.plot_model(
        model,
        to_file=architecture_image_name,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )

    # plot_model(model)

    # wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
    # dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
    # focal_loss = sm.losses.CategoricalFocalLoss()
    # loss = dice_loss + (1 * focal_loss)

    # optimizer = tf.optimizers.SGD(
    #     learning_rate=config.learning_rate, 
    #     momentum=config.momentum,
    #     name='optimizer_SGD_0'
    # )

    # model.compile(
    #     optimizer=optimizer,
    #     loss=loss,
    #     metrics=[
    #         # tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    #         'accuracy',
    #         dice_coef_3cat,
    #         sm.metrics.IOUScore(threshold=0.5),
    #         # IoU_coef
    #     ],
    # )

    # model.save('model/model.h5')