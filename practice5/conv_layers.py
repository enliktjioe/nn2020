from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Get the dimensions of the input data x
    N, C, H, W = x.shape

    # Get the dimensions of the filter weights
    F, _, HH, WW = w.shape

    # Get the stride and pad parameters
    stride = conv_param['stride']
    padding = conv_param['pad']

    # Compute the output volume size of the hidden layer and the weights
    out_H = 1 + ((H + 2 * padding - HH) / stride)
    out_W = 1 + ((H + 2 * padding - WW) / stride)

    # Initialize the output container
    out = np.zeros((N, F, int(out_H), int(out_W)))

    # Pad the input data starting at position (1,1) and using zero constant values
    pad_input = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    for i in range(N):
        for j in range(C):
            pad_input[i, j] = np.pad(x[i, j], (1, 1), 'constant', constant_values=(0, 0))

    # Process the loop through the layer and then apply the filter
    for i in range(N):
        for j in range(int(out_H)):
            for k in range(int(out_W)):
                for l in range(F):
                    # Get the input values to be convoluted
                    x_now = pad_input[i, :, j * stride:j * stride + HH, k * stride:k * stride + WW]
                    # Get the indexed filter
                    kernel = w[l]
                    # Convolve the input data using the indexed filter
                    out[i, l, j, k] = np.sum(x_now * kernel)
                # When convolution is done, add the bias
                out[i, :, j, k] = out[i, :, j, k] + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

