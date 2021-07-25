import tensorflow as tf
from tensorflow.keras import Model, Input
from blocks import *
from config import get_initial_config

def build_model(config):
    inputs = Input(shape=config.image_size)
    patches = Patches(config.transformer.patch_size, name="patches_0")(inputs)
    encoded_patches = PatchEncoder(
        num_patches=config.transformer.num_patches, 
        projection_dim=config.transformer.projection_dim,
        name='encoded_patches_0',    
    )(patches)

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

    x = None
    filters=16
    for idx in range(5):
        if not idx:
            x = DecoderBlockCup(
                target_shape=(config.transformer.projection_dim, config.image_height//16, config.image_width//16),
                filters=filters,
                normalization_rate=config.transformer.normalization_rate,
                name=f'decoder_cup_{idx}'
            )(encoded_patches)
        else:
            filters *= 2
            x = DecoderUpsampleBlock(
                filters=filters,
                name=f'decoder_upsample_{idx-1}'
            )(x)

    x = DecoderSegmentationHead(
        name='decoder_segmentation_end_0'
    )(x)

    return Model(inputs=inputs, outputs=x)

if __name__ == "__main__":
    config = get_initial_config()
    model = build_model(config)
    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file="model/model_architecture.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    # model.save('model/model.h5')