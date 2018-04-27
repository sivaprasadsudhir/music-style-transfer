import numpy as np

from keras import backend as K

edge_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
edge_kernel = K.variable(edge_kernel)
# edge_kernel = K.reshape(edge_kernel, (1, 3, 3, 1))

thresh_lower = 4e-1
thresh_upper = 9e-1

def regularized_mse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    y_pred_2d = K.reshape(y_pred, (K.shape(y_pred)[0], 2, 257, 251, 2))
    y_pred_thresh_lower = K.cast(K.greater(y_pred_2d, thresh_lower), K.floatx())
    y_pred_thresh_upper = K.cast(K.less(y_pred_2d, thresh_upper), K.floatx())
    y_pred_in_range = K.cast(K.greater(y_pred_thresh_lower + y_pred_thresh_upper, 1.1), K.floatx())

    y_pred_in_range = K.sum(y_pred_in_range, axis=-1) # sum complex coefficients
    y_pred_in_range = K.sum(y_pred_in_range, axis=-1) # sum time dimension
    y_pred_in_range = K.sum(y_pred_in_range, axis=-1) # sum frequency dimension

    # now y_pred_in_range should have shape (batch_size, 2)  -- (2 is because of shared autoencoder)
    # print(K.int_shape(y_pred_in_range))
    # y_pred_in_range = K.print_tensor(y_pred_in_range, message='thresh')
    regularization = K.mean(y_pred_in_range)

    return mse + 1e-6 * regularization

def norm(t):
    s = K.square(t)
    norm_squared = K.sum(s, axis=-1)
    return K.reshape(K.sqrt(norm_squared), (K.shape(norm_squared)[0], 251, 1))

def edge_regularized_mse(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)

    # imagine spectograms of two instruments being stacked on top of each other
    y_true_2d = K.reshape(y_true, (K.shape(y_true)[0] * 514, 251, 2))
    y_pred_2d = K.reshape(y_pred, (K.shape(y_pred)[0] * 514, 251, 2))
    y_true_2d = norm(y_true_2d)
    y_pred_2d = norm(y_pred_2d)

    print(K.int_shape(y_true_2d))
    y_true_edge = K.conv2d(y_true_2d, kernel=edge_kernel, data_format="channels_last")
    y_pred_edge = K.conv2d(y_pred, kernel=edge_kernel, data_format="channels_last")
    #
    # edge_mse = K.mean(K.square(y_pred_edge - y_true_edge), axis=-1)
    # edge_mse = K.print_tensor(edge_mse, message='edge_mse')
    return mse
