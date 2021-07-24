import ml_collections

def get_initial_config():
    """
        Returns the transformer initial configuration
    """

    config = ml_collections.ConfigDict()
    config.dataset_path = 'dataset/'
    config.learning_rate = 0.01
    config.weight_decay = 1e-4
    config.batch_size = 256
    config.num_epochs = 10
    config.image_height = 256
    config.image_width = 256
    config.image_channels = 1
    config.image_size = (config.image_height, config.image_width, config.image_channels)
    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.patch_size = 16
    config.transformer.num_patches = (config.image_size[0] // config.transformer.patch_size)**2
    config.transformer.projection_dim = 512
    config.transformer.units = [
        config.transformer.projection_dim * 3, # (3) 1536 --  (4) 2048
        config.transformer.projection_dim, # 512
        # config.transformer.projection_dim, # 64
    ]
    config.transformer.layers = 12
    config.transformer.num_heads = 12
    config.transformer.dropout_rate = 0.1
    config.transformer.normalization_rate = 1e-6

    config.n_classes = 1
    config.activation = 'softmax'

    return config