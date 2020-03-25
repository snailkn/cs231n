from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n_train, n_classes, n_feature = X.shape[0], W.shape[1], X.shape[1]
    for i in range(n_train):
        pred = X[i].dot(W)
        temp1 = np.exp(pred).sum()
        for j in range(n_classes):
            temp2 = np.exp(pred[j]) / temp1
            if j == y[i]:
                temp2 -= 1
            dW[:, j] += temp2 * X[i]
        loss += np.log(temp1) - pred[y[i]]
        
    loss /= n_train
    dW /= n_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n_train, n_classes = X.shape[0], W.shape[1]
    pred = X.dot(W)
    temp = np.sum(np.exp(pred), axis=1)
    loss = np.mean(np.log(temp) - pred[range(n_train), y]) + 0.5 * reg * np.sum(W * W)
    
    y_matrix = np.zeros((n_train, n_classes))
    y_matrix[range(n_train), y] = 1
    dW = X.T.dot(np.exp(pred) / temp.reshape((n_train, 1)).dot(np.ones((1, n_classes))) - y_matrix) / n_train + reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
