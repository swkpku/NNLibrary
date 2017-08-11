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
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx