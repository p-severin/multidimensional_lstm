from keras.engine import Model
from keras.models import load_model
import numpy as np

from bidirectional_lstm.data_generator import DataGenerator

validation_data_dir = '/home/pseweryn/Projects/repositories/VOCdevkit/VOC2012'
path_to_models = './saved_models/'

model_type = 'lstm'
batch_size = 16
num_classes = 355
epochs = 5


def validate(model: Model, batch_size):
    """Train the model.
    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
    """

    validation_generator = DataGenerator(validation_data_dir, subset='val', batch_size=batch_size)


    # loss, acc = model.evaluate_generator(
    #     validation_generator,
    #     steps=STEP_SIZE_VAL)
    y_pred = model.predict_generator(generator=validation_generator,
                                     verbose=1)

    # print('loss: {}, acc: {}'.format(loss, acc))
    y_pred = np.argmax(y_pred, axis=3)
    print(y_pred.shape)
    # print(y_pred.shape)
    # matrix = confusion_matrix(y_true, y_pred)
    # print(matrix)
    # f1_score_value = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
    # print('f1_score: {}'.format(f1_score_value))


if __name__ == '__main__':
    model = load_model('./saved_models/weights.01-1.2619.hdf5')
    validate(model, batch_size=8)
