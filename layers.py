import numpy as np


def affine_forward(x, theta, theta0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (m, d_1, ..., d_k) and contains a minibatch of m
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension d = d_1 * ... * d_k, and
  then transform it to an output vector of dimension h.

  Inputs:
  - x: A numpy array containing input data, of shape (m, d_1, ..., d_k)
  - theta: A numpy array of weights, of shape (d, h)
  - theta0: A numpy array of biases, of shape (h,)
  
  Returns a tuple of:
  - out: output, of shape (m, h)
  - cache: (x, theta, theta0)
  """
  out = None
  m = x.shape[0]
  d = np.product(x.shape[1:])
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # 2 lines of code expected
  # X = np.reshape(x, (m, d))
  x_reshape = x.reshape(x.shape[0], theta.shape[0])
#   out =  x_reshape @ theta + np.repeat(np.atleast_2d(theta0), m, axis=0)
  out =  x_reshape @ theta + theta0
  #out = x.reshape(x.shape[0], theta.shape[0]).dot(theta) + theta0

  # out = theta.T @ X.T + np.repeat(np.atleast_2d(theta0), m, axis=0).T
  # out = out.T
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (m, h)
  - cache: Tuple of:
    - x: Input data, of shape (m, d_1, ... d_k)
    - theta: Weights, of shape (d,h)
    - theta0: biases, of shape (h,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (m, d1, ..., d_k)
  - dtheta: Gradient with respect to theta, of shape (d, h)
  - dtheta0: Gradient with respect to theta0, of shape (h,)
  """
  x, theta, theta0 = cache
  dx, dtheta, dtheta0 = None, None, None
  m = x.shape[0]
  d = np.product(x.shape[1:])
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # Hint: do not forget to reshape x into (m,d) form
  # 4-5 lines of code expected
    
  x_reshape = x.reshape(x.shape[0], theta.shape[0])
  dx = (dout @ theta.T).reshape(x.shape)
  dtheta = x_reshape.T @ dout
  dtheta0 = np.sum(dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # 1 line of code expected
  out = np.where(x > 0, x, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # 1 line of code expected. Hint: use np.where
  dx = np.where(cache > 0, dout, 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    # 2 lines of code expected
    drop = 1 - p
    mask = (np.random.rand(*x.shape) < drop) / drop 
    out = x * mask
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    # 1 line of code expected
    
    drop = 1 - p
    out = x * (np.random.rand(*x.shape) < drop) 
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, theta, theta0, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of m data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (m, C, H, W)
  - theta: Filter weights of shape (F, C, HH, WW)
  - theta0: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (m, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, theta, theta0, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = theta.shape
  H_pr = int(1+(H+2*pad - HH) / stride)
  W_pr = int(1+(W+2*pad - WW) / stride)
  out  = np.zeros((N, F, H_pr, W_pr))
  x_new = np.pad(x, ((0,), (0,), (pad,), (pad, )), 'constant', constant_values = (0,0))

  for n in range(N):
    for f in range(F):
      for i in range(H_pr):
        for j in range(W_pr):
          for k in range(HH):
            for l in range(WW):
              for c in range(C):
                out[n,f,i,j] += x_new[n,c,stride*i+k,stride*j+l] * theta[f,c,k,l]   
                out[n,f,i,j] += theta0[f] 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, theta, theta0, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, theta, theta0, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dtheta: Gradient with respect to theta
  - dtheta0: Gradient with respect to theta0
  """
  dx, dtheta, dtheta0 = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, theta, theta0, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']
  N, C, H, W = x.shape
  F, C, HH, WW = theta.shape
  H_pr = int(1+(H+2*pad - HH) / stride)
  W_pr = int(1+(W+2*pad - WW) / stride)

  x_new = np.pad(x, ((0,), (0,), (pad,), (pad, )), 'constant', constant_values = (0,0))
  dx = np.zeros(x_new.shape)
  dtheta = np.zeros(theta.shape)
  dtheta0 = np.zeros(theta0.shape)

  dtheta0 = np.sum(dout, axis= (0,2,3))
  for n in range(N):
    for f in range(F):
      for i in range(H_pr):
        for j in range(W_pr):
          for k in range(HH):
            for l in range(WW):
              for c in range(C):
                dtheta[f,c,k,l] += dout[n,f,i,j] * x_new[n,c,stride*i+k,stride*j+l]
                dx[n,c,stride*i+k,stride*j+l] += dout[n,f,i,j] * theta[f,c,k,l]  
  if pad > 0:
    dx = dx[:,:,pad:-pad, pad:-pad]
  
                
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dtheta, dtheta0


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (m, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  p_height = pool_param["pool_height"]
  p_width = pool_param["pool_width"]
  stride = pool_param['stride']
  N, C, H, W = x.shape
  H_pr = int(1 + (H-p_height)/stride)
  W_pr = int(1 + (W-p_width)/stride)
  out = np.zeros((N,C,H_pr, W_pr))


  for n in range(N):
    for i in range(H_pr):
      for j in range(W_pr):
        for c in range(C):
          out[n,c,i,j] = np.max(x[n,c,stride*i:stride*i+p_height, stride*j:stride*j+p_width])
         


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x,pool_param = cache
  p_height = pool_param["pool_height"]
  p_width = pool_param["pool_width"]
  stride = pool_param['stride']
  N, C, H, W = x.shape
  H_pr = int(1 + (H-p_height)/stride)
  W_pr = int(1 + (W-p_width)/stride)
  dx = np.zeros(x.shape)

  for n in range(N):
    for i in range(H_pr):
      for j in range(W_pr):
        for c in range(C):
          pool = x[n,c,stride*i:stride*i+p_height, stride*j:stride*j+p_width]
          idx_keep = (pool == np.max(pool))
          dx[n,c,stride*i:stride*i+p_height, stride*j:stride*j+p_width] = idx_keep * dout[n,c,i,j]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx



def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  m = x.shape[0]
  correct_class_scores = x[np.arange(m), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(m), y] = 0
  loss = np.sum(margins) / m
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(m), y] -= num_pos
  dx /= m
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (m, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (m,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  m = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(m), y])) / m
  dx = probs.copy()
  dx[np.arange(m), y] -= 1
  dx /= m
  return loss, dx
