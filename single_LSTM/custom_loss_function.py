from keras import backend as K

weights = [1] + 20 * [3]

def class_weighted_pixelwise_crossentropy(target, output):
    output = K.clip(output, 10e-8, 1. - 10e-8)
    return -K.sum(target * weights * K.log(output), axis=3)
