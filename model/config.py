import ml_collections

def get_config_1():
    """
        Returns the transformer configuration 1
    """

    config = ml_collections.ConfigDict()
    config.config_name = "architecture_1"
    config.dataset_path = 'dataset_3D/'
    config.learning_rate = 0.001
    config.weight_decay = 1e-4
    config.momentum = 0.9
    config.batch_size = 2
    config.num_epochs = 20
    config.image_height = 256
    config.image_width = 256
    config.image_depth = 256
    config.image_channels = 1
    config.image_size = (config.image_height, config.image_width, config.image_depth, config.image_channels)
    config.skip_connections = True
    config.data_augmentation = False
    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.patch_size = 16
    config.transformer.num_patches = (config.image_size[0] // config.transformer.patch_size)**3
    config.transformer.projection_dim = 192
    config.transformer.units = [
        config.transformer.projection_dim * 3, # (3) 1536 --  (4) 2048
        config.transformer.projection_dim, # 512
        # config.transformer.projection_dim, # 64
    ]
    config.transformer.layers = 8
    config.transformer.num_heads = 5
    config.transformer.dropout_rate = 0.1
    config.transformer.normalization_rate = 1e-6

    config.n_classes = 4
    config.activation = 'softmax'

    return config