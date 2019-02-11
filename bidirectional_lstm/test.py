from keras.engine import Model
from keras.models import load_model
import numpy as np
from keras.optimizers import Adam

from bidirectional_lstm.custom_loss_function import class_weighted_pixelwise_crossentropy
from bidirectional_lstm.data_generator import DataGenerator
from bidirectional_lstm.models import build_model
from utils.images import Dataset
import matplotlib.pyplot as plt

validation_data_dir = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'

batch_size = 1


def validate(model: Model, batch_size):
    """Train the model.
    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
    """



    dataset = Dataset(validation_data_dir, subsets=['val'], image_shape=(90, 90))
    # X, y = dataset.image_generator('val', how_many_images=100)
    X, y = dataset.create_patches('val', how_many_images=100)
    print(X.shape)

    for i in range(100):
        # X = np.expand_dims(X, axis=0)
        y_pred = model.predict([np.expand_dims(X[i, 0], 0),
                                np.expand_dims(X[i, 1], 0),
                                np.expand_dims(X[i, 2], 0),
                                np.expand_dims(X[i, 3], 0)], batch_size=batch_size, verbose=1)
        y_pred = np.argmax(y_pred, axis=3)

        y_true = np.argmax(y[i], axis=2)

        plt.subplot(121)
        plt.imshow(y_pred[0], vmin=0, vmax=20)
        plt.subplot(122)
        plt.imshow(y_true, vmin=0, vmax=20)
        plt.show()
        print(y_pred.shape)


if __name__ == '__main__':
    model = build_model(30, 30, 27)
    model.summary()
    optimizer = Adam(lr=10e-4)
    model.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    # model.load_weights('/home/pseweryn/Projects/multidimensional_lstm/repository/models/weights.02-3.5613.hdf5')
    model.load_weights('/home/pseweryn/Projects/multidimensional_lstm/repository/models/weights.05-3.4559.hdf5')
    validate(model, batch_size=8)
