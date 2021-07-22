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

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import elasticdeform
import random
from utils import utils
from matplotlib import pyplot
# import tensorflow_addons as tfa

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
        rescale=1./255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        # rotation_range=90,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # zoom_range=0.2,
        preprocessing_function=eslastic_deform_datagen_individual#(displacement=config['ELASTIC_DEFORM_DISPLACEMENT'])
    )

    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + 'orig', 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    msk_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'train/' + config['VIEW_TRAINIG'] + config['LABEL'], 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    return zip(img_generator, msk_generator)

def create_validation_dataset(config:dict):
    data_gen_args = dict(
        rescale=1./255,
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
    )

    msk_generator = datagen.flow_from_directory(
        config['DATASET_PATH'] + 'test/' + config['VIEW_TRAINIG'] + config['LABEL'], 
        target_size=config['IMAGE_SIZE'],
        class_mode=None,
        color_mode='grayscale',
        batch_size=config['BATCH_SIZE'],
        seed=12,
        follow_links=True,
    )

    return zip(img_generator, msk_generator)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Slice', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
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

def main():
    BATCH_SIZE = 64
    IMAGE_SIZE = (256, 256)
    DATASET_PATH = 'dataset/'
    VIEW_TRAINIG = 'axial/'
    LABEL = 'left-cerebellum-white-matter'
    ELASTIC_DEFORM_DISPLACEMENT = np.array([
        [
            [  8.05204059,  13.24886814, -12.10855166],
            [  2.4468842,   -0.30459105,  -3.61833851],
            [ -9.41539836,   9.4857399,   10.59295401,]
        ],
        [
            [-11.69137525,   2.7400518,   -9.11486785],
            [  3.71851547,  -1.41123399,  -3.32899794],
            [  2.30374345,   0.41622831,  -1.77370728],
        ]
    ])#np.random.randn(2, 3, 3) * 7
    print("ELDIS: ", ELASTIC_DEFORM_DISPLACEMENT)
    """
        [[[  8.05204059  13.24886814 -12.10855166]
        [  2.4468842   -0.30459105  -3.61833851]
        [ -9.41539836   9.4857399   10.59295401]]

        [[-11.69137525   2.7400518   -9.11486785]
        [  3.71851547  -1.41123399  -3.32899794]
        [  2.30374345   0.41622831  -1.77370728]]]


        [[[ -4.87267665  -5.8617854   19.74015017]
        [  8.24165151   8.96995615  -8.77903898]
        [  4.33526268   8.96590878   6.46129555]]

        [[  2.03945636 -12.10953444  -2.45797345]
        [ -1.77627598   0.83095988   4.03630251]
        [ 11.17687835   7.9816394    1.49157364]]]



        [[[ -5.29220521  -1.03176988   8.57819707]
        [  5.4735756   -1.8623361    4.15340418]
        [  5.78356656   0.02207705  -4.696546  ]]

        [[-14.51778563   1.89981693  -3.19687128]
        [ -4.12490684  -8.58324571  -3.49991011]
        [  2.63476564  -1.13689161  -1.54028576]]]
    """

    config = {
        'DATASET_PATH': DATASET_PATH,
        'VIEW_TRAINIG': VIEW_TRAINIG,
        'LABEL': LABEL,
        'IMAGE_SIZE': IMAGE_SIZE,
        'BATCH_SIZE': BATCH_SIZE,
        'ELASTIC_DEFORM_DISPLACEMENT': ELASTIC_DEFORM_DISPLACEMENT,
    }
    
    train_gen = create_train_dataset(config=config)
    val_gen = create_validation_dataset(config=config)
    
    # show_dataset(datagen=train_gen, config=config, num=150)
    testing_datagens(config)

if __name__ == "__main__":
    main()