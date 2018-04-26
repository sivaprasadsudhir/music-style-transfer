import numpy as np

from keras import backend as K

def regularized_mse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    y_pred_2d = K.reshape(y_pred, (K.shape(y_pred)[0], 2, 257, 251, 2))
    combine_samples = K.sum(y_pred_2d, axis=0)
    combine_coeff = K.sum(combine_samples, axis=-1)
    combine_instr = K.sum(combine_coeff, axis=0)
    combine_freq = K.sum(combine_instr, axis=0)
    regularization = combine_freq[0] # get the time 0 total

    return mse + 100 * regularization
