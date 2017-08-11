from builtins import range
from builtins import object
import numpy as np

from NNLibrary.layers import *

class TwoLayerNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        a1_out, a1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        r1_out, r1_cache = relu_forward(a1_out)
        a2_out, a2_cache = affine_forward(r1_out, self.params['W2'], self.params['b2'])
        scores = a2_out

        if y is None:
            return scores

        grads = {}

        loss, dscores = softmax_loss(a2_out, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        dx2, dw2, db2 = affine_backward(dscores, a2_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        dx2 = relu_backward(dx2, r1_cache)
        dx1, dw1, db1 = affine_backward(dx2, a1_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads