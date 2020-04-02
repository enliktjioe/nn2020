from builtins import object
import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # Set the value of input dimension
        C,H,W = input_dim

        # Initialize value of all three layers
        # Using gaussian random numbers with mean 0
        # Using scale counted by the weight scale hyper parameters and correct dim
        # The weight dimensions in 2nd layer must be divided by 2 because of maxpool size 2x2
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C,
                                                               filter_size,
                                                               filter_size))
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*(int(H/2))*(int(W/2)),hidden_dim))
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))

        # Initialize all three biases value with zero values and appropriate dimension
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # Do the fast Convolution Forward Pass
        convolution_out, convolution_cache = conv_forward_im2col(X, W1, b1, conv_param)
        # Do the Relu Forward Activation
        relu_out1, relu_out1_cache = relu_forward(convolution_out)
        # Do the maxpool forward fast
        maxpool_out, maxpool_cache = max_pool_forward_fast(relu_out1, pool_param)
        # Do the Relu Activation for the final layer
        affine_relu_out, affine_relu_cache = affine_relu_forward(maxpool_out, W2, b2)
        # Do the final affine forward pass to get the the scores
        scores, scores_cache = affine_forward(affine_relu_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # Evaluate the value of the softmax loss and get derivative of scores
        loss, derivative_scores = softmax_loss(scores, y)

        # Apply Regularization to the loss
        loss += 0.5 * (self.reg * (
                    np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2)))

        # Do the Affine_backward module to compute backward pass of the layer
        layer2_dx, layer2_dw, layer2_db = affine_backward(derivative_scores, scores_cache)

        # Save the value of the gradients of weights & biases
        grads['W3'] = layer2_dw + self.reg * self.params['W3']
        grads['b3'] = layer2_db

        # Use the affine_relu_backward module to compute backward pass of layer 2
        layer1_dx, layer1_dw, layer1_db = affine_relu_backward(layer2_dx, affine_relu_cache)

        # Store the gradients of Weights & biases for layer 2
        grads['W2'] = layer1_dw + self.reg * self.params['W2']
        grads['b2'] = layer1_db

        # Do the fast maxpool backward pass
        maxpool_dx = max_pool_backward_fast(layer1_dx, maxpool_cache)

        # Do the relu backward pass of convolution layer
        relu_dx = relu_backward(maxpool_dx, relu_out1_cache)

        # Do the the backward pass for the convolution later
        convolution_dx, convolution_dw, convolution_db = conv_backward_im2col(relu_dx, convolution_cache)

        # Save the gradients of weights & biases for the layer 1
        grads['W1'] = convolution_dw + self.reg * self.params['W1']
        grads['b1'] = convolution_db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
