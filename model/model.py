import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from blocks import *
from config import *
from metrics import dice_coef, IoU_coef
import tensorflow_addons as tfa

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
    
    x = DecoderBlockCup(
        target_shape=(config.image_height//16, config.image_width//16, config.transformer.projection_dim),
        filters=3,
        normalization_rate=config.transformer.normalization_rate,
        name=f'decoder_cup_{idx}'
    )(encoded_patches)

    x = DecoderUpsampleBlock(filters=2, kernel_size=1)(x)

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
        model = build_model(config)

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

    optimizer = tf.optimizers.SGD(
        learning_rate=config.learning_rate, 
        momentum=config.momentum,
        name='optimizer_SGD_0'
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            dice_coef,
            IoU_coef
        ],
    )

    # model.save('model/model.h5')