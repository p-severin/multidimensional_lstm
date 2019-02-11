from keras import backend as K

weights = [1] + 20 * [3]

def custom_objective(y_true, y_pred):

    punishment_rate = 3
    cost = K.abs(y_pred - y_true)
    condition = K.equal(y_true, 0)
    error = K.switch(condition, punishment_rate * cost, cost)
    return error

def class_weighted_pixelwise_crossentropy(target, output):
    output = K.clip(output, 10e-8, 1.-10e-8)
    return -K.sum(target * weights * K.log(output), axis=3)