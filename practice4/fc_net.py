from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        # See also: http://cs231n.github.io/neural-networks-2/#init                #
        ############################################################################
        # Initialize all the weights inside the self.params dictionary, with numpy
        # random numbers generator and multiply it by the weight_scale.
        # Dimensions will use the network layers dimensions as following:
        # 1st Layer Dimension = input x hidden
        # 2nd Layer Dimension = hidden x number of classes(output)
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)

        # Initialize the biases with zero values in correspondence with the right
        # vector dimensions
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # 1. Initialize all the values for Weights and Biases
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        # 2. Use forward pass with Relu activation for the hidden layer
        h, cachel1 = affine_relu_forward(X, W1, b1)

        # 3. Use forward pass for the output of hidden layer
        scores, cachel2 = affine_forward(h, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        dh, grads["W2"], grads["b2"] = affine_backward(dscores, cachel2)

        dx, grads["W1"], grads["b1"] = affine_relu_backward(dh, cachel1)

        grads["W2"] += W2 * self.reg
        grads["W1"] += W1 * self.reg
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # 1. Get the random value of weights for the first layer
        W1_l1 = np.random.normal(0, weight_scale, input_dim * hidden_dims[0])

        # 2. Store the value of the first layer weights in the params dictionary
        self.params["W1"] = W1_l1.reshape((input_dim, hidden_dims[0]))

        # 3. Initialize the bias value of the first layer with zero value
        self.params['b1'] = np.zeros((hidden_dims[0]))

        # 4. Do the loop throughout the initialization for intermediary layers
        for i in xrange(1, self.num_layers - 1):
            # 4.1 Initialize the value of indexed weights with random numbers
            # with mean 0 and defined weight scale.
            # Reshaped to maintain dimensionality
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale,
                                                             hidden_dims[i - 1] * hidden_dims[i]).reshape(
                (hidden_dims[i - 1], hidden_dims[i]))

            # 4.2 Initialize the indexed biases in the dictionary with zeros
            self.params['b' + str(i + 1)] = np.zeros((hidden_dims[i]))

        # 5. Initialize the final layer weights with random numbers with mean 0 and defined weight scale.
        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale,
                                                                   hidden_dims[-1] * num_classes).reshape(
            (hidden_dims[-1], num_classes))

        # 6. Initialize the final layer biases in the dictionary with zeros
        self.params['b' + str(self.num_layers)] = np.zeros((num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # 1. Create a variable to store the cache data from every layer
        cache_list = []

        # 2. Create a copy of the input data(X)
        input_data = X.copy()

        # 3. If dropout in use
        if self.use_dropout:
            for i in xrange(1, self.num_layers):
                # Do Affine Forward Pass
                a, fc_cache = affine_forward(input_data, self.params['W' + str(i)], self.params['b' + str(i)])
                # Do Relu activation for Forward Pass
                r, relu_cache = relu_forward(a)
                # Relu activation's output used to perform dropout forward pass
                dropout_output, dropout_cache = dropout_forward(r, self.dropout_param)
                # Store the cached values
                cache = (fc_cache, relu_cache, dropout_cache)
                # Store all the cached values into the cache_list
                cache_list.append(cache)
                # Set value of input data as the current output for next layer
                input_data = dropout_output

            # Perfrom the affine forward pass for the last layer
            scores, dr_cache_ln = affine_forward(input_data, self.params['W' + str(self.num_layers)],
                                                 self.params['b' + str(self.num_layers)])
            # Store current cache to Cache list
            cache_list.append(dr_cache_ln)

        # 4. If dropout not in use, run normal mode
        else:
            for i in xrange(1, self.num_layers):
                # Do the forward pass with Relu activation
                layer_output, layer_cache = affine_relu_forward(input_data, self.params['W' + str(i)],
                                                                self.params['b' + str(i)])
                # Store current cached values into the cache list
                cache_list.append(layer_cache)
                # Set input data as current output for next layer
                input_data = layer_output

            # Do the Affine Forward Pass into the last layer
            scores, cache_ln = affine_forward(input_data, self.params['W' + str(self.num_layers)],
                                              self.params['b' + str(self.num_layers)])
            # Store the value of current cache to Cache list
            cache_list.append(cache_ln)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # 1. Using softmax_loss() to get the loss and dout value
        loss, dout = softmax_loss(scores, y)

        # 2. Initialize variable to store derivatives of weights (dw) and bias (db)
        list_dw = []
        list_db = []

        # 3. Extract and delete the last layer entry in the Cache List
        cache = cache_list.pop()

        # 4. Do the affine backward pass on cached data for the last layer
        dx, dw, db = affine_backward(dout, cache)

        # 5. Store the derivative of weights and biases at index 0
        list_dw.insert(0, dw)
        list_db.insert(0, db)
        dout = dx

        # 6. If dropout is in used
        if self.use_dropout:
            # Loop through the cached entries for all the intermediary layers
            for i in xrange(len(list_cache)):
                cache = cahcelist_cache.pop()
                # Get data by extracting it from the cache file
                fc_cache, relu_cache, dropout_cache = cache
                # Do the dropout backward pass
                dd = dropout_backward(dout, dropout_cache)
                # Do the Relu activation for backward pass
                dr = relu_backward(dd, relu_cache)
                # Do the Perfrom Affine backward Pass
                dx, dw, db = affine_backward(dr, fc_cache)
                # Update list of derivatives of weights and biases
                list_dw.insert(0, dw)
                list_db.insert(0, db)
                # Set derivative of output as derivative of x
                dout = dx

            para_loss = 0

            # Loop through the values in list of derivatives of weights
            for i in xrange(len(list_dw)):
                # Apply regularization to the weights
                W = self.params['W' + str(i + 1)]
                list_dw[i] += self.reg * W
                # Use para_loss variable to store the iterative penalty terms for the regularization
                para_loss += np.sum(W ** 2)
            # Regularize the loss
            loss += 0.5 * self.reg * para_loss

            # Loop through and update the grads dictionary entries for derivatives of weights and biases
            for i in xrange(len(list_dw)):
                grads['W' + str(i + 1)] = list_dw[i]
                grads['b' + str(i + 1)] = list_db[i]

            # If dropout is not specified, run normal mode
        else:
            # Loop through the cached entries for all the intermediary layers
            for i in xrange(len(list_cache)):
                # Extract and remove the last entry in the cache list
                cache = list_cache.pop()
                # Perform Backward pass with Relu activation
                dx, dw, db = affine_relu_backward(dout, cache)
                # Update list of derivatives of weights and biases
                list_dw.insert(0, dw)
                list_db.insert(0, db)
                # Set derivative of output as derivative of x
                dout = dx
            para_loss = 0

            # Loop through the values in list of derivatives of weights
            for i in xrange(len(list_dw)):
                # Apply regularization to the weights
                W = self.params['W' + str(i + 1)]
                list_dw[i] += self.reg * W
                # Use para_loss variable to store the iterative penalty terms for the regularization
                para_loss += np.sum(W ** 2)
            # Regularize the loss
            loss += 0.5 * self.reg * para_loss

            # Loop through and update the grads dictionary entries for derivatives of weights and biases
            for i in xrange(len(list_dw)):
                grads['W' + str(i + 1)] = list_dw[i]
                grads['b' + str(i + 1)] = list_db[i]

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
