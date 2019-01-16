import os
from functools import partial

from skimage.transform import resize
from sklearn.externals._pilutil import imresize
from sklearn.feature_extraction import image
from skimage.util import view_as_windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

directory_voc_dataset = '/Users/patrykseweryn/PycharmProjects/datasets/voc_dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'


class Dataset:
    extensions = dict(image='.jpg',
                      segmentation='.png',
                      annotations='.txt')

    def __init__(self, directory_images, subsets):
        self.main_directory = partial(os.path.join, directory_images)
        self.directories = dict(image=self.main_directory('JPEGImages'),
                                segmentation=self.main_directory(
                                    'SegmentationClass'),
                                annotations=self.main_directory(
                                    'ImageSets/Segmentation'))
        self.subsets = subsets
        self.rgb_images = dict(train=[],
                               val=[])
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

    def image_generator(self, subset):
        for file in self.data[subset]:
            image = self.open_image(file, 'image')
            segmentation = self.open_image(file, 'segmentation')
            if 15 in np.unique(segmentation):
                # self.plot_image_and_its_segmentation(image, segmentation)
                print(np.unique(segmentation))
                yield image, segmentation
            else:
                continue

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


if __name__ == '__main__':
    dataset = Dataset(directory_voc_dataset, subsets=['train', 'val'])
    image_generator = dataset.image_generator('train')
    for pair_of_images in image_generator:
        rgb_image = pair_of_images[0]
        segmentation = pair_of_images[1]

        image_shape = (270, 270)
        rgb_image = imresize(rgb_image, image_shape, interp='bilinear')
        segmentation = imresize(segmentation, size=image_shape, interp='nearest')
        print(np.unique(segmentation))



        patch_shape = (15, 15, 3)
        segmentation_shape = (15, 15)
        step = 15

        patches_rgb = view_as_windows(rgb_image, patch_shape, step=step)
        patches_rgb = np.reshape(patches_rgb, (-1, *patch_shape))

        patches_segmentation = view_as_windows(segmentation, segmentation_shape, step=step)
        patches_segmentation = np.reshape(patches_segmentation, (-1, *segmentation_shape))

        print(patches_rgb.shape)
        print(patches_segmentation.shape)

        plt.subplot(211)
        plt.imshow(rgb_image)
        plt.subplot(212)
        plt.imshow(segmentation)
        plt.show()

        original_shape = rgb_image.shape
        print('original shape: {}'.format(original_shape))

        # patches_rgb = view_as_windows(rgb_image, patch_shape, step=3)
        # patches_rgb = np.reshape(patches_rgb, (-1, *patch_shape))
        # print(patches_rgb.shape)
        for patch in patches_rgb:
            flipped = np.fliplr(patch)
            flipped_ver = np.flipud(patch)
            plt.subplot(311)
            plt.imshow(patch)
            plt.subplot(312)
            plt.imshow(flipped)
            plt.subplot(313)
            plt.imshow(flipped_ver)
            plt.show()



        # blocks_of_image = np.reshape(image, (-1, 3, 3, 3))
        # print(blocks_of_image.shape)

        # height, width, channels = rgb_image.shape
        # stride = 3
        #
        # print(rgb_image.shape)
        #
        # floored_height = (height) // 3 * 3
        # floored_width = (width) // 3 * 3
        #
        # print(type(floored_height))
        #
        # patches = []
        # for i in range(0, floored_height, stride):
        #     for j in range(0, floored_width, stride):
        #         patches.append(rgb_image[i: i+stride, j: j+stride, :])
        #
        # patches = np.array(patches)
        # print(patches.shape)
        #
        # original = np.zeros((floored_height, floored_width, channels), np.uint8)
        # print(original.shape)
        #
        # for i in range(0, floored_height, stride):
        #     for j in range(0, floored_width, stride):
        #         original[i: i+stride, j: j+stride, :] = patches[int(i / 3) * int(floored_width / 3) + int(j / 3)]
        # plt.imshow(original)
        # plt.show()
        #
        # plt.imshow(rgb_image)
        # plt.show()