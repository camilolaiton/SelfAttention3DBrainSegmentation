"""
    Made by:
        - Camilo Laiton
        Universidad Nacional de Colombia, Colombia
        2021-1
        GitHub: https://github.com/camilolaiton/

        This file belongs to the private repository "master_thesis" where
        I save all the files that are related to my thesis which is called
        "Método para la segmentación de imágenes de resonancia magnética 
        cerebrales usando una arquitectura de red neuronal basada en modelos
        de atención".
"""

# import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import elasticdeform
from utils import utils
# from matplotlib import pyplot
from model.config import *
from model.model import *
from model.model_2 import build_model
from model.losses import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from volumentations import *
import pickle
import glob
import segmentation_models as sm
from augmend import Augmend, Elastic, FlipRot90
import argparse
from tensorflow.keras import mixed_precision
sm.set_framework('tf.keras')

# import tensorflow_addons as tfa

# all_files_loc = "datapsycho/imglake/population/train/image_files/"
# all_files = os.listdir(all_files_loc)

# image_label_map = {
#         "image_file_{}.npy".format(i+1): "label_file_{}.npy".format(i+1)
#         for i in range(int(len(all_files)/2))}
# partition = [item for item in all_files if "image_file" in item]

# class DataGenerator(keras.utils.Sequence):

#     def __init__(self, file_list):
#         """Constructor can be expanded,
#            with batch size, dimentation etc.
#         """
#         self.file_list = file_list
#         self.on_epoch_end()

#     def __len__(self):
#       'Take all batches in each iteration'
#       return int(len(self.file_list))

#     def __getitem__(self, index):
#       'Get next batch'
#       # Generate indexes of the batch
#       indexes = self.indexes[index:(index+1)]

#       # single file
#       file_list_temp = [self.file_list[k] for k in indexes]

#       # Set of X_train and y_train
#       X, y = self.__data_generation(file_list_temp)

#       return X, y

#     def on_epoch_end(self):
#       'Updates indexes after each epoch'
#       self.indexes = np.arange(len(self.file_list))

#     def __data_generation(self, file_list_temp):
#       'Generates data containing batch_size samples'
#       data_loc = "datapsycho/imglake/population/train/image_files/"
#       # Generate data
#       for ID in file_list_temp:
#           x_file_path = os.path.join(data_loc, ID)
#           y_file_path = os.path.join(data_loc, image_label_map.get(ID))

#           # Store sample
#           X = np.load(x_file_path)

#           # Store class
#           y = np.load(y_file_path)

#       return X, y

def eslastic_deform_datagen_individual(img):
    # def el_deform(img):
    img_deformed = elasticdeform.deform_grid(np.reshape(img, (256, 256)), displacement=np.random.randn(2,3,3)*3)
    return np.expand_dims(img_deformed, axis=2)

    # return el_deform

def elastic_deform_data_gen(img, msk):
    img = np.reshape(img, (256, 256))
    msk = np.reshape(msk, (256, 256))

    displacement = np.random.randn(2, 3, 3) * 9
    img_deformed = elasticdeform.deform_grid(img, displacement=displacement)
    msk_deformed = elasticdeform.deform_grid(msk, displacement=displacement)
    # img_deformed, msk_deformed = elasticdeform.deform_random_grid([img, msk], sigma=7, points=3)
    return np.expand_dims(img_deformed, axis=2), np.expand_dims(msk_deformed, axis=2)

def create_train_dataset(config:dict):
    data_gen_args = dict(
        # rescale=1./255,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=90,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # zoom_range=0.2,
        # preprocessing_function=eslastic_deform_datagen_individual#(displacement=config['ELASTIC_DEFORM_DISPLACEMENT'])
    )

    if (config["DATA_AUGMENTATION"]):
        data_gen_args['preprocessing_function'] = eslastic_deform_datagen_individual

    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + 'orig', 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
        shuffle=False,
    )

    msk_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + config['LABEL'], 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
        shuffle=False,
    )

    return zip(img_generator, msk_generator)

def create_validation_dataset(config:dict):
    data_gen_args = dict(
        # rescale=1./255,
    )

    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + 'orig', 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
        shuffle=False,
    )

    msk_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + config['LABEL'], 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
        shuffle=False,
    )

    return zip(img_generator, msk_generator)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Slice', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    
    arr = display_list[i]
    res = arr.nonzero()
    print(arr[res])

    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='bone')
    plt.axis('off')
  plt.show()

def show_dataset(datagen, config, num=1):
    for i in range(0, num):
        image,mask = next(datagen)
        print(image[0].shape, " ", mask.shape)

        display([image[0], mask[0]])
        # utils.elastic_deform_2(image[0], mask[0])
        # img, msk = elastic_deform_data_gen(image[0], mask[0])
        # image[0] = img
        # mask[0] = msk

def testing_datagens(config):
    img = tf.keras.preprocessing.image.load_img('/home/camilo/Programacion/master_thesis/data/HLN-12/HLN-12-1/slices/axial/HLN-12-1_161.png', grayscale=True)
    msk = tf.keras.preprocessing.image.load_img('/home/camilo/Programacion/master_thesis/data/HLN-12/HLN-12-1/segSlices/left-cerebellum-white-matter/axial/HLN-12-1_161.png', grayscale=True)
    
    msk_data = tf.keras.preprocessing.image.img_to_array(msk)
    data = tf.keras.preprocessing.image.img_to_array(img)
    # expand dimension to one sample
    msk_samples = np.expand_dims(msk_data, 0)
    samples = np.expand_dims(data, 0)
    # create image data augmentation generator
    data_gen_args = dict(
        # rescale=1./255,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=90,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # zoom_range=0.2,
        preprocessing_function=eslastic_deform_datagen_individual#(displacement=config['ELASTIC_DEFORM_DISPLACEMENT'])
    )

    datagen = ImageDataGenerator(**data_gen_args)
    msk_datagen = ImageDataGenerator(**data_gen_args)
    
    # prepare iterator
    it = datagen.flow(samples, batch_size=1, seed=12)
    it_msk = msk_datagen.flow(msk_samples, batch_size=1, seed=12)
    utils.helperPlottingOverlay(img, msk)
    # generate samples and plot
    for i in range(9):
        # define subplot
        # pyplot.subplot(330 + 1 + i)
        # generate batch of images
        batch = it.next()
        batch_msk = it_msk.next()
        # convert to unsigned integers for viewing
        image = batch[0]
        mask = batch_msk[0]

        # plot raw pixel data
        utils.helperPlottingOverlay(image, mask)
        # pyplot.imshow(image)
    # show the figure
    # pyplot.show()

def load_files_py(img_path, msk_path):
    img = np.load(img_path).astype(np.float32)
    msk = np.load(msk_path).astype(np.float32)
    return img, msk

def load_files(img_path, msk_path):
    return tf.numpy_function(
        load_files_py,
        inp=[img_path, msk_path],
        Tout=[tf.float32, tf.float32]
    )

def get_augmentation():
    return Compose([
        # Rotate((-5, 5), (0, 0), (0, 0), p=0.5),
        # RandomCropFromBorders(crop_value=0.1, p=0.3),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize(patch_size, interpolation=1, always_apply=True, p=1.0),
        # Flip(0, p=0.5),
        # Flip(1, p=0.5),
        # RandomRotate90((0, 1), p=0.6),
        # GaussianNoise(var_limit=(0, 5), p=0.5),
        # RandomGamma(gamma_limit=(0.5, 1.5), p=0.7),
    ], p=1.0)

def augmentor_py(img, msk):
    aug = get_augmentation()#(64,64,64))
    data = {'image': img, 'msk': msk}
    aug_data = aug(**data)
    img = aug_data['image']
    msk = aug_data['msk']
    return tf.cast(img, tf.float32), tf.cast(msk, tf.float32)
    # return np.ndarray.astype(img, np.float32), np.ndarray.astype(msk, np.float32)

def augmentation(img, msk):
    img = img.astype(np.float32)
    msk = msk.astype(np.float32)
    total_img = []
    total_msk = []
    img = np.squeeze(img)
    msk = np.argmax(msk, axis=4)

    aug = Augmend()
    
    # aug.add([
    #     FlipRot90(axis=(0, 1, 2)),
    #     FlipRot90(axis=(0, 1, 2)),
    # ], probability=0.9)

    aug.add([
        Elastic(axis=(0, 1, 2), amount=5, order=1, use_gpu=False),
        Elastic(axis=(0, 1, 2), amount=5, order=0, use_gpu=False),
    ], probability=0.9)

    for i in range(img.shape[0]):
        img_res, msk_res = aug([img[i, :, :, :], msk[i, :, :, :]])
        # print(img_res.shape, " ", msk_res.shape)
        total_img.append(img_res)#.astype(np.float32))
        total_msk.append(msk_res)#.astype(np.float32))
    return np.expand_dims(total_img, axis=-1).astype(np.float16), to_categorical(total_msk).astype(np.float16) 

def augmentor(img, msk):
    aug_img = tf.numpy_function(
        augmentation,#augmentor_py,
        inp=[img, msk],
        Tout=[tf.float16, tf.float16]
    )
    #aug_img.set_shape((64, 64, 64, 1))
    return aug_img

def main():

    # Selecting cuda device

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    tf.keras.backend.clear_session()

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    
    SEED = 12
    mb_limit = 9500
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 10GB of memory on the GPU
        try:
            # Setting visible devices
            tf.config.set_visible_devices(gpus, 'GPU')

            # Setting memory growth
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_memory_growth(gpus[1], True)

            # Setting max memory
            # tf.config.experimental.set_per_process_memory_fraction(0.80)
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mb_limit)])

            tf.config.experimental.set_virtual_device_configuration(gpus[1], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mb_limit)])

            # tf.config.experimental.set_per_process_memory_growth(True)

        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--retrain', metavar='retr', type=int,
                        help='Retrain architecture', default=0)

    parser.add_argument('--folder_name', metavar='folder', type=str,
                        help='Insert the folder for insights')

    parser.add_argument('--lr_epoch_start', metavar='lr_decrease', type=int,
                        help='Start epoch lr decrease', default=10)
    args = vars(parser.parse_args())

    retrain = args['retrain']
    training_folder = 'trainings/' + args['folder_name']

    model_path = f"{training_folder}/model_trained_architecture.hdf5"
    # model_path = f"{training_folder}/checkpoints_4/model_trained_09_0.68.hdf5"

    utils.create_folder(f"{training_folder}/checkpoints")

    # creating model
    # config = get_config_patchified()
    config = get_config_local_path()#get_config_test()
    model = None

    # Mirrored strategy for parallel training
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Setting up weights 
    weights = utils.read_test_to_list(config.dataset_path + 'weights.txt')

    if (weights == False):
        end_path = '/train/masks'
        image_files = [file for file in glob.glob(config.dataset_path + end_path + '/*') if file.endswith('.npy')]
        weights, label_to_frequency_dict = utils.median_frequency_balancing(image_files=image_files, num_classes=config.n_classes)
        if (weights == False):
            print("Please check the path")
            exit()
        utils.write_list_to_txt(weights, config.dataset_path + 'weights.txt')
        print("Weights calculated: ", weights)
    else:
        weights = [float(weight) for weight in weights]
        # weights = [0.0, 1, 2.7, 3]
        # weights = [0.0, 2.3499980585022096, 6.680915101433645, 7.439929426050408]
        print("Weights read! ", weights)

    # Setting up neural network loss
    #loss = tversky_loss()#
    
    with mirrored_strategy.scope():
    # model = build_model_patchified_patchsize16(config)
    # model = build_model_patchified_patchsize16(config)
        model = build_model(config)#test_model_3(config)
    
    if (retrain):
        model.load_weights(model_path)

    loss = None
    if config.loss_fnc == 'tversky':
        loss = tversky_loss
        print('Using tversky loss...')
    elif config.loss_fnc == 'crossentropy':
        loss = 'categorical_crossentropy'
        print('Using categorical crossentropy loss...')
    elif config.loss_fnc == 'dice_focal_loss':
        loss = dice_focal_loss(weights)
        print('Using dice focal loss...')
    elif config.loss_fnc == 'weighted_crossentropy':
        loss = weighted_categorical_crossentropy(weights)
        print("Using weighted crossentropy")
    elif config.loss_fnc == 'gen_dice':
        loss = generalized_dice_loss(weights)
    elif config.loss_fnc == 'focal_tversky':
        loss = focal_tversky
    elif config.loss_fnc == 'dice_categorical':
        loss = dice_categorical(weights)
    else:
        print("No loss function")
        exit()
    
    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    #     return lr
    

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     config.learning_rate,
    #     decay_steps=5,
    #     decay_rate=0.01,
    #     staircase=True
    # )

    optimizer = None

    if (config.optimizer == 'SGD'):
        optimizer = tf.optimizers.SGD(
            learning_rate=config.learning_rate, 
            momentum=config.momentum,
            nesterov=True,
            name='optimizer_SGD_0'
        )
    elif (config.optimizer == 'adamax'):
        optimizer = tf.keras.optimizers.Adamax(
            learning_rate=config.learning_rate, 
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-07,
            name="Adamax",
        )
    elif (config.optimizer == 'adam'):
        optimizer = tf.optimizers.Adam(
            learning_rate=config.learning_rate,
            beta_1=0.9, 
            beta_2=0.999, 
            epsilon=1e-07,
            amsgrad=False, 
            name='optimizer_Adam'
        )

    # lr_metric = get_lr_metric(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            # 'accuracy',
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),
        ],
    )
    
    print(f"[+] Building model with config {config}")
    model.summary()
    
    tf.keras.utils.plot_model(
        model,
        to_file=f"{training_folder}/trained_architecture.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,
    )
    
    # Setting up variables for data generators
    # TRAIN_IMGS_DIR = config.dataset_path + 'train/images/'
    # TRAIN_MSKS_DIR = config.dataset_path + 'train/masks/'

    # TEST_IMGS_DIR = config.dataset_path + 'test/images/'
    # TEST_MSKS_DIR = config.dataset_path + 'test/masks/'

    # train_imgs_lst = os.listdir(TRAIN_IMGS_DIR)
    # train_msks_lst = os.listdir(TRAIN_MSKS_DIR)

    # test_imgs_lst = os.listdir(TEST_IMGS_DIR)
    # test_msks_lst = os.listdir(TEST_MSKS_DIR)

    image_list_train = sorted(glob.glob(
        config.dataset_path + 'train/images/*'))
    mask_list_train = sorted(glob.glob(
        config.dataset_path + 'train/masks/*'))
    print(config.dataset_path, " ", len(image_list_train), " ", len(mask_list_train))
    image_list_test = sorted(glob.glob(
        config.dataset_path + 'test/images/*'))
    mask_list_test = sorted(glob.glob(
        config.dataset_path + 'test/masks/*'))

    # Getting image data generators
    # train_datagen = utils.mri_generator(
    #     TRAIN_IMGS_DIR,
    #     train_imgs_lst,
    #     TRAIN_MSKS_DIR,
    #     train_msks_lst,
    #     config.batch_size
    # )

    # reading for training
    # half = int(len(image_list_train)*0.5)
    # train_imgs = utils.read_files_from_directory(image_list_train, half)
    # train_msks = utils.read_files_from_directory(mask_list_train, half)

    # Reading for validation
    # test_imgs = utils.read_files_from_directory(image_list_test)
    # test_msks = utils.read_files_from_directory(mask_list_test)

    train_datagen = tf.data.Dataset.from_tensor_slices(
        (image_list_train, mask_list_train)
    )

    # val_datagen = utils.mri_generator(
    #     TEST_IMGS_DIR,
    #     test_imgs_lst,
    #     TEST_MSKS_DIR,
    #     test_msks_lst,
    #     config.batch_size
    # )
    # weights = [0, 0.2, 0.4, 0.4]
    val_datagen = tf.data.Dataset.from_tensor_slices(
        # (test_imgs, test_msks)
        (image_list_test, mask_list_test)
    )

    dataset = {
        "train" : train_datagen,
        "val" : val_datagen
    }

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset['train'] = dataset['train'].shuffle(buffer_size=config.batch_size, seed=SEED)
    dataset['train'] = dataset['train'].map(load_files).map(augmentor, num_parallel_calls=AUTOTUNE)
    if (config.unbatch):
        dataset['train'] = dataset['train'].unbatch()
    dataset['train'] = dataset['train'].repeat()
    # dataset['train'] = dataset['train'].shuffle(config.batch_size, reshuffle_each_iteration=True)
    dataset['train'] = dataset['train'].batch(config.batch_size)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
    dataset['train'] = dataset['train'].with_options(options)

    dataset['val'] = dataset['val'].map(load_files)
    if (config.unbatch):
        dataset['val'] = dataset['val'].unbatch()
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(config.batch_size)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)
    dataset['val'] = dataset['val'].with_options(options)
    
    # Setting up callbacks
    monitor = 'val_iou_score'
    mode = 'max'

    # Early stopping
    early_stop = EarlyStopping(
        monitor=monitor, 
        mode=mode, 
        verbose=1,
        patience=20
    )

    # Model Checkpoing
    model_check = ModelCheckpoint(
        f"{training_folder}/model_trained_architecture.hdf5", 
        save_best_only=True,
        save_weights_only=True, 
        monitor=monitor, 
        mode=mode
    )

    model_check_2 = ModelCheckpoint(
        training_folder + "/checkpoints/model_trained_{epoch:02d}_{val_iou_score:.2f}_{val_f1-score:.2f}.hdf5", 
        save_best_only=False,
        save_weights_only=True, 
        monitor=monitor, 
        mode=mode,
        period=5
    )

    tb = TensorBoard(
        log_dir=f"{training_folder}/logs_tr_2", 
        profile_batch=(4, 8),
        write_graph=True, 
        update_freq='epoch'
    )

    pltau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=2, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0
    )

    def scheduler(epoch, lr):
        if epoch < args['lr_epoch_start']:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    factor = 1

    if (config.unbatch):
        factor = 64

    steps_per_epoch = (len(image_list_train)*factor)//config.batch_size
    val_steps_per_epoch = (len(image_list_test)*factor)//config.batch_size

    utils.write_dict_to_txt(
        config, 
        f"{training_folder}/trained_architecture_config.txt"
    )
    # class_weights = {
    #     0: weights[0],
    #     1: weights[1],
    #     2: weights[2],
    #     3: weights[3],
    # }

    history = model.fit(dataset['train'],
        steps_per_epoch=steps_per_epoch,
        epochs=config.num_epochs,
        # batch_size=config.batch_size,
        verbose=1,
        validation_data=dataset['val'],
        validation_steps=val_steps_per_epoch,
        callbacks=[early_stop, model_check, model_check_2, tb, pltau, lr_callback],
        # class_weight=class_weights
    )

    with open(f"{training_folder}/history.obj", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

if __name__ == "__main__":
    main()
