from builtins import range
import numpy as np

def affine_forward(x, w, b):
    N = x.shape[0]
    X = x.reshape(N, -1)
    out = np.dot(X, w) + b
    cache = (x, w, b)

    return out, cache

def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    X = x.reshape(N, -1)
    dx = np.dot(dout, w.T).reshape(*x.shape)
    dw = np.dot(X.T, dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x

    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0

    return dx

def softmax_loss(x, y):
    N = x.shape[0]
    shifted_x = x - np.max(x, axis=1, keepdims=True)
    softmax = np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=1).reshape(-1,1)
    loss = np.sum(-np.log(softmax[range(N), y]))
    loss /= N

    dx = softmax.copy()
    dx[range(N), y] -= 1
    dx /= N

    return loss, dx