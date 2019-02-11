import os

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from bidirectional_lstm.custom_loss_function import class_weighted_pixelwise_crossentropy
from bidirectional_lstm.models import build_model
from bidirectional_lstm.data_generator import DataGenerator

if __name__ == '__main__':
    directory_voc_dataset = '/home/pseweryn/Repositories/VOCdevkit/VOC2012'

    training_generator = DataGenerator(directory_voc_dataset, 'train', batch_size=4, shuffle=True, n_channels=27, dim=(30, 30))
    validation_generator = DataGenerator(directory_voc_dataset, 'val', batch_size=4, shuffle=False, n_channels=27, dim=(30, 30))

    model = build_model(30, 30, 27)
    model.summary()
    optimizer = Adam(lr=10e-4)
    model.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    callbacks = [ModelCheckpoint(os.path.join('/home/pseweryn/Projects/multidimensional_lstm/repository/models/', 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'),
                                 save_best_only=True)]

    model.fit_generator(training_generator, epochs=5, verbose=1, callbacks=callbacks,
                        validation_data=validation_generator,
                        class_weight='auto')
