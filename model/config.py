import ml_collections

def get_initial_config():
    """
        Returns the transformer initial configuration
    """

    config = ml_collections.ConfigDict()
    config.learning_rate = 0.001
    config.batch_size = 256
    config.num_epochs = 10
    config.image_size = (256, 256, 1)
    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.patch_size = 16
    config.transformer.num_patches = (config.image_size[0] // config.transformer.patch_size)**2
    config.transformer.projection_dim = 64
    config.transformer.units = [
        config.transformer.projection_dim * 2,
        config.transformer.projection_dim,
    ]
    config.transformer.layers = 12
    config.transformer.num_heads = 4
    config.transformer.dropout_rate = 0.1
    config.transformer.normalization_rate = 1e-6

    config.n_classes = 1
    config.activation = 'softmax'

    return config