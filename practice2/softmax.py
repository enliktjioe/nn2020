import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax (cross-entropy) loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    You might or might not want to transform it into one-hot form (not obligatory)
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  # In this naive implementation we have a for loop over the N samples
  for i, x in enumerate(X):
    #############################################################################
    # TODO: Compute the cross-entropy loss using an explicit loop and store the #
    # sum of losses in "loss".                                                  #
    # If you are not careful in implementing softmax, it is easy to run into    #
    # numeric instability, because exp(a) is huge if a is large.                #
    #############################################################################
    # TODO: should use explicit loops here
    z = X[i].dot(W)

    z -= np.max(z)

    p = np.exp(z) / (np.sum(np.exp(z)))

    loss += -np.log(p[y[i]])

    #############################################################################
    # TODO: Compute the gradient using explicit loops and store the sum over    #
    # samples in dW.                                                            #
    #############################################################################
    for k in range(W.shape[1]):
      dW[:, k] += (p[k] - (k == y[i])) * X[i]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
  # now we turn the sum into an average by dividing with N
  loss /= X.shape[0]
  dW /= X.shape[0]

  # Add regularization to the loss and gradients.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax (cross-entropy) loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the cross-entropy loss and its gradient using no loops.     #
  # Store the loss in loss and the gradient in dW.                            #
  # Make sure you take the average.                                           #
  # If you are not careful with softmax, you migh run into numeric instability#
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  # Add regularization to the loss and gradients.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW

