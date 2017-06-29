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
  for i in xrange(X.shape[0]):
    scores=np.exp(X[i].dot(W))
    total=np.sum(scores)
    scores/=total
    loss-=np.log(scores[y[i]])
    for j in xrange(W.shape[1]):
      if j==y[i]:
        dW[:,y[i]]+=X[i]*(scores[y[i]]-1)
      else:
        dW[:,j]+=X[i]*(scores[j])
  dW/=X.shape[0]
  dW+=(2*reg)*(W)
  loss/=X.shape[0]
  loss+=(reg)*np.sum(W*W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  temp1=np.exp(X.dot(W))
  temp2=np.sum(temp1,axis=1).reshape(X.shape[0],1)
  temp2=np.tile(temp2,(1,W.shape[1]))
  temp1/=temp2
  temp4=np.zeros((X.shape[0],W.shape[1]))
  temp4[range(X.shape[0]),list(y)]=1
  dW+=(X.T).dot(temp1)-(X.T).dot(temp4)
  dW/=X.shape[0]
  dW+=(2*reg)*(W)
  temp3=temp1[range(X.shape[0]),list(y)]
  temp3=-np.log(temp3)
  loss=np.sum(temp3)/X.shape[0]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

