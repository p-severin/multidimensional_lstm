import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.util import view_as_windows
from sklearn.externals._pilutil import imresize
from tensorflow.python.keras.utils import to_categorical

from sys import platform

if platform == 'linux':
    directory_voc_dataset = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'
else:
    directory_voc_dataset = '/Users/patrykseweryn/PycharmProjects/datasets/voc_dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'


class Dataset:
    extensions = dict(image='.jpg',
                      segmentation='.png',
                      annotations='.txt')

    def __init__(self, directory_images, subsets, image_shape=(96, 96)):
        self.main_directory = partial(os.path.join, directory_images)
        self.directories = dict(image=self.main_directory('JPEGImages'),
                                segmentation=self.main_directory(
                                    'SegmentationClass'),
                                annotations=self.main_directory(
                                    'ImageSets/Segmentation'))
        self.subsets = subsets
        self.rgb_images = dict(train=[],
                               val=[])
        self.image_shape = image_shape
        self.segmentation_shape = (self.image_shape[0] // 3, self.image_shape[1] // 3)
        self.data = dict()

        for subset in self.subsets:
            self.data[subset] = self.get_image_numbers(subset)

    def get_image_numbers(self, subset):

        if subset not in ['train', 'trainval', 'val']:
            raise Exception('No such data subset exists.')

        train_file = os.path.join(self.directories['annotations'],
                                  subset + self.extensions['annotations'])
        csv_data = pd.read_csv(train_file, header=None)
        return csv_data.values.reshape((-1))

    def image_generator(self, subset, how_many_images):
        images = []
        y = []
        if how_many_images == -1:
            number_of_elements = len(self.data[subset])
        else:
            number_of_elements = how_many_images

        for file in self.data[subset][:number_of_elements]:
            image = self.open_image(file, 'image')
            segmentation = self.open_image(file, 'segmentation')
            segmentation[segmentation == 255] = 0
            image = imresize(image,
                             self.image_shape,
                             interp='bilinear')

            image_v = np.flip(image, axis=0)
            image_h = np.flip(image, axis=1)
            image_vh = np.flip(image_v, axis=1)
            segmentation = imresize(segmentation,
                                    size=self.segmentation_shape,
                                    interp='nearest')
            # plt.subplot(221)
            # plt.imshow(image)
            # plt.subplot(222)
            # plt.imshow(image_v)
            # plt.subplot(223)
            # plt.imshow(image_h)
            # plt.subplot(224)
            # plt.imshow(image_vh)
            # plt.show()
            # plt.imshow(segmentation, vmin=0, vmax=20)
            # plt.show()
            # images.append(image)
            images.append((image, image_v, image_h, image_vh))
            y.append(segmentation)
        return images, y

    def create_patches(self, subset, how_many_images):
        images = []
        y = []
        if how_many_images == -1:
            number_of_elements = len(self.data[subset])
        else:
            number_of_elements = how_many_images

        for file in self.data[subset][:number_of_elements]:
            image = self.open_image(file, 'image')
            segmentation = self.open_image(file, 'segmentation')
            segmentation[segmentation == 255] = 0
            image = imresize(image,
                             self.image_shape,
                             interp='bilinear')
            segmentation = imresize(segmentation,
                                    size=self.segmentation_shape,
                                    interp='nearest')
            patches_rgb, patches_segmentation = self.extract_patches((image, segmentation))
            patches_segmentation = self.__one_hot_encode_y(patches_segmentation)
            image_v = np.flip(patches_rgb, axis=0)
            image_h = np.flip(patches_rgb, axis=1)
            image_vh = np.flip(patches_rgb, axis=1)
            images.append((patches_rgb, image_v, image_h, image_vh))
            y.append(patches_segmentation)
        images = np.array(images, dtype=np.float32)
        y = np.array(y)
        images *= 1./255
        return images, y

    @staticmethod
    def __one_hot_encode_y(y_dataset):
        one_hot_encoded = to_categorical(y_dataset, num_classes=21)
        y_dataset = one_hot_encoded
        return y_dataset

    def open_image(self, file_name, image_type):
        image_path = os.path.join(self.directories[image_type],
                                  file_name + self.extensions[image_type])
        image = Image.open(image_path)
        image = np.array(image)
        return image

    def plot_image_and_its_segmentation(self, image, segmentation_image):
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(segmentation_image)
        plt.show()


    def extract_patches(self, pair_of_images):
        rgb_image = pair_of_images[0]
        segmentation = pair_of_images[1]
        patch_shape = (3, 3, 3)
        segmentation_shape = (1, 1)
        step = 3
        segmentation_step = 1
        patches_rgb = view_as_windows(rgb_image, patch_shape, step=step)
        patches_rgb = np.reshape(patches_rgb, (patches_rgb.shape[0], patches_rgb.shape[1], np.prod(patch_shape)))
        patches_segmentation = view_as_windows(segmentation, segmentation_shape,
                                               step=segmentation_step)
        patches_segmentation = np.reshape(patches_segmentation,
                                          (patches_segmentation.shape[0], patches_segmentation.shape[1], np.prod(segmentation_shape)))
        return patches_rgb, patches_segmentation





def create_dataset():
    dataset = Dataset(directory_voc_dataset, subsets=['train', 'val'])
    X, y = dataset.image_generator('train', how_many_images=100)
    print('unique: {}'.format(np.unique(y)))

    X = np.array(X, dtype=np.float32)
    x_min = np.min(X)
    x_max = np.max(X)
    X = (X - x_min) / (x_max - x_min)
    print(X.shape)
    print(np.min(X), np.max(X))

    y = Dataset.__one_hot_encode_y(y)
    y = np.array(y, dtype=np.uint8)
    print(y.shape)
    return X, y


if __name__ == '__main__':
    dataset = Dataset(directory_voc_dataset, subsets=['train', 'val'])
    X, y = dataset.create_patches('train', how_many_images=100)
    print(X.shape)
    print(y.shape)
