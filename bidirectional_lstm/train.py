import os

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from bidirectional_lstm.models import build_model
from bidirectional_lstm.data_generator import DataGenerator

if __name__ == '__main__':
    directory_voc_dataset = '/Users/patrykseweryn/PycharmProjects/datasets/voc_dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'

    training_generator = DataGenerator(directory_voc_dataset, 'train', batch_size=8, shuffle=True)
    validation_generator = DataGenerator(directory_voc_dataset, 'val', batch_size=8, shuffle=False)

    model = build_model()
    optimizer = Adam(lr=10e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [ModelCheckpoint(os.path.join('./saved_models', 'weights.{epoch:02d}-{val_loss:.4f}.hdf5'),
                                 save_best_only=True)]

    model.fit_generator(training_generator, epochs=5, verbose=1, callbacks=callbacks,
                        validation_data=validation_generator)
