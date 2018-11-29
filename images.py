import os
from functools import partial

import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imresize

directory_voc_dataset = r'C:\Users\patry\Documents\datasets\voc_dataset\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'


class Dataset:
    extensions = dict(image='.jpg',
                      segmentation='.png',
                      annotations='.txt')
    width, height, channels = 256, 256, 3

    def __init__(self, directory_images, subsets=['train']):
        self.main_directory = partial(os.path.join, directory_images)
        self.directories = dict(image=self.main_directory('JPEGImages'),
                                segmentation=self.main_directory('SegmentationClass'),
                                annotations=self.main_directory('ImageSets\Segmentation'))
        self.subsets = subsets
        self.rgb_images = dict(train=[],
                               val=[])

        for subset in self.subsets:
            self.data = self.get_image_numbers(subset)
            print(self.data)

    def get_image_numbers(self, subset):

        if subset not in ['train', 'trainval', 'val']:
            raise Exception('No such data subset exists.')

        train_file = os.path.join(self.directories['annotations'], subset + self.extensions['annotations'])
        csv_data = pd.read_csv(train_file, header=None)
        return csv_data.values.reshape((-1))

    def get_batch_of_images(self, size=4):
        print(self.data.shape)
        batch_of_files = np.random.choice(self.data, size=size, replace=False)
        batch_x = np.zeros((size, self.width, self.height, self.channels))
        batch_y = np.zeros((size, self.width, self.height))
        for i, file in enumerate(batch_of_files):
            image = self.open_image(file, 'image')
            segmentation = self.open_image(file, 'segmentation')
            batch_x[i] = image
            batch_y[i] = segmentation
            self.plot_image_and_its_segmentation(image, segmentation)
        return batch_x, batch_y

    def open_image(self, file_name, image_type):
        assert isinstance(file_name, str) and isinstance(image_type, str)
        image_path = os.path.join(self.directories[image_type], file_name + self.extensions[image_type])
        image = Image.open(image_path)
        image = np.array(image)
        image = imresize(image, (self.width, self.height), interp='nearest')
        return image

    def plot_image_and_its_segmentation(self, image, segmentation_image):
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(segmentation_image)
        plt.show()


if __name__ == '__main__':
    dataset = Dataset(directory_voc_dataset, subsets=['train', 'val'])
    images = dataset.get_batch_of_images()
    print(images)
