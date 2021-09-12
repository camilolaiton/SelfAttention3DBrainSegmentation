import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.callbacks import ModelCheckpoint
from blocks import *
from config import *
from metrics import dice_coef, IoU_coef
from tensorflow.keras import models
import segmentation_models as sm
import numpy as np

# import tensorflow_addons as tfa
def build_model_test(config):
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
    dec = 32
    decoder_block_cup = DecoderBlockCup(
        target_shape=(config.image_height//dec, config.image_width//dec, config.image_depth//dec, config.transformer.projection_dim//2),
        filters=64,
        normalization_rate=config.transformer.normalization_rate,
        name=f'decoder_cup_{0}'
    )(encoded_patches)

    # Decoder upsample blocks & skip connections
    
    decoder_up_block_0 = DecoderUpsampleBlock(
        filters=32, 
        kernel_size=3,
        name=f"decoder_upsample_0"
    )(decoder_block_cup)

    skip_conn_0 = EncoderDecoderConnections(
        filters=32,
        kernel_size=3,
        name="skip_connection_0"
    )(
        transformer_layers[-1], 
        decoder_up_block_0,
        {
            "length_res_block": 4,
            "target_shape": (
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim//2
            ),
            "length_reshape": 0,
            "filters_reshape": []
        }
    )
    skip_conn_0 = layers.Add()([decoder_up_block_0, skip_conn_0])


    decoder_up_block_1 = DecoderUpsampleBlock(
        filters=16, 
        kernel_size=3,
        name=f"decoder_upsample_1"
    )(skip_conn_0)

    skip_conn_1 = EncoderDecoderConnections(
        filters=16,
        kernel_size=3,
        name="skip_connection_1"
    )(
        transformer_layers[-4], 
        decoder_up_block_1, 
        {
            "length_res_block": 4,
            "target_shape": (
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim//2
            ),
            "length_reshape": 1,
            "filters_reshape": [32]
        }
    )
    skip_conn_1 = layers.Add()([decoder_up_block_1, skip_conn_1])


    decoder_up_block_2 = DecoderUpsampleBlock(
        filters=8, 
        kernel_size=3,
        name=f"decoder_upsample_2"
    )(skip_conn_1)

    skip_conn_2 = EncoderDecoderConnections(
        filters=8,
        kernel_size=3,
        name="skip_connection_2"
    )(
        transformer_layers[-6], 
        decoder_up_block_2, 
        {
            "length_res_block": 4,
            "target_shape": (
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim//2
            ),
            "filters_reshape": [32, 16]
        }
    )
    skip_conn_2 = layers.Add()([decoder_up_block_2, skip_conn_2])

    
    decoder_up_block_3 = DecoderUpsampleBlock(
        filters=1, 
        kernel_size=3,
        name=f"decoder_upsample_3"
    )(skip_conn_2)

    skip_conn_3 = EncoderDecoderConnections(
        filters=1,
        kernel_size=3,
        name="skip_connection_3"
    )(
        transformer_layers[0], 
        decoder_up_block_3, 
        {
            "length_res_block": 4,
            "target_shape": (
                config.image_height//dec, 
                config.image_width//dec, 
                config.image_depth//dec, 
                config.transformer.projection_dim//2
            ),
            "filters_reshape": [32, 16, 8]
        }
    )
    skip_conn_3 = layers.Add()([decoder_up_block_3, skip_conn_3])

    return Model(inputs=inputs, outputs=skip_conn_3)

def build_model(config):
    inputs = Input(shape=config.image_size)
    patches = Patches(config.transformer.patch_size, name="patches_0")(inputs)
    encoded_patches = PatchEncoder(
        num_patches=config.transformer.num_patches, 
        projection_dim=config.transformer.projection_dim,
        name='encoded_patches_0',    
    )(patches)

    # print(f"enconded patches size {encoded_patches.shape}")

    transformer_blocks = []
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
        transformer_blocks.append(encoded_patches)
    
    filters = [3, 128, 64, 32, 1]
    residual_block = None
    for idx in range(len(filters)):
      if not idx:
        x = DecoderBlockCup(
            target_shape=(config.image_height//64, config.image_width//64, config.image_depth//64, config.transformer.projection_dim),
            filters=filters[idx],
            normalization_rate=config.transformer.normalization_rate,
            name=f'decoder_cup_{idx}'
        )(encoded_patches)
      else:
          x = DecoderUpsampleBlock(filters=filters[idx], kernel_size=3)(x)

          if (config.residual_blocks):
            num = 0
            if (idx == 1):
              num = -3
            elif (idx == 2):
              num = int(len(transformer_blocks)/2)

            if idx == 1:
              residual_block = DecoderBlockCup(
                  target_shape=(config.image_height//16, config.image_width//16, config.transformer.projection_dim),
                  filters=filters[idx],
                  normalization_rate=config.transformer.normalization_rate,
                  name=f'decoder_cup_{idx}'
              )(transformer_blocks[-3])
              residual_block = DecoderUpsampleBlock(filters=filters[idx], kernel_size=3)(residual_block)
              x = layers.Add()([x, residual_block])


            if (idx == 2):
              residual_block = DecoderBlockCup(
                  target_shape=(config.image_height//8, config.image_width//8, config.transformer.projection_dim),
                  filters=filters[idx-1],
                  normalization_rate=config.transformer.normalization_rate,
                  name=f'decoder_cup_{idx}'
              )(transformer_blocks[-5])
              residual_block = DecoderUpsampleBlock(filters=filters[idx-1], kernel_size=3)(residual_block)
              residual_block = DecoderUpsampleBlock(filters=filters[idx], kernel_size=3)(residual_block)

              x = layers.Add()([x, residual_block])
            
            if (idx == 3):
              residual_block = DecoderBlockCup(
                  target_shape=(config.image_height//8, config.image_width//8, config.transformer.projection_dim),
                  filters=filters[idx-1],
                  normalization_rate=config.transformer.normalization_rate,
                  name=f'decoder_cup_{idx}'
              )(transformer_blocks[-7])
              residual_block = DecoderUpsampleBlock(filters=filters[idx-2], kernel_size=3)(residual_block)
              residual_block = DecoderUpsampleBlock(filters=filters[idx-1], kernel_size=3)(residual_block)
              residual_block = DecoderUpsampleBlock(filters=filters[idx], kernel_size=3)(residual_block)

              x = layers.Add()([x, residual_block])

    # x = DecoderUpsampleBlock(filters=32, kernel_size=3)(x)
    # x = DecoderUpsampleBlock(filters=3, kernel_size=3)(x)

    return Model(inputs=inputs, outputs=x)
    

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
        config = get_config_1()
        # model = build_model(config)
        model = build_model_test(config)


    # elif (config_num == 2):
    #     config = get_config_2()
    #     model = build_model_2(config)

    # else:
    #     config = get_config_3()
    #     model = build_model_3(config)

    print(f"[+] Building model {config_num} with config {config}")    
    model.summary()

    if not (config.residual_blocks):
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

    wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    loss = dice_loss + (1 * focal_loss)

    optimizer = tf.optimizers.SGD(
        learning_rate=config.learning_rate, 
        momentum=config.momentum,
        name='optimizer_SGD_0'
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            # tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            dice_coef,
            sm.metrics.IOUScore(threshold=0.5),
            # IoU_coef
        ],
    )

    # model.save('model/model.h5')