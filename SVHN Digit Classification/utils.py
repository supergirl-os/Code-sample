# ==================================================================
# Course: Understanding Deep Neural Networks
# Teacher: Zhang Yi
# Student: Wang Yaxuan
# ID:   2019141440341
#
# Ten-category classification problem on SVHN dataset
# ====================================================================
import numpy as np
import matplotlib.pyplot as plt


# define the activation function
# f = lambda s: 1 / (1 + np.exp(-s))
def f(z):
    return 1 / (1 + np.exp(-z))


# define the derivative of activation function
df = lambda s: f(s) * (1 - f(s))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)  # overflow
    return np.exp(x) / np.sum(np.exp(x))


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# lr schedule
def lr_schedule(epoch):
    learning_rate = 0.5
    if 3 < epoch < 6:
        learning_rate = 0.8
    elif 5 < epoch < 9:
        learning_rate = 0.5
    elif epoch > 8:
        learning_rate = 0.3
    return learning_rate


# Define Cost Function
def cost(a, y):
    J = 1 / 2 * np.sum((a - y) ** 2)
    return J


def cross_entropy_error(y, t):
    # For one dim data, y,t need to be row vector
    # ensure batch_size=1
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # t one-hot label
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # 
    return -np.sum(t * np.log(y + 1e-7).T) / batch_size


# Define Evaluation Index
def accuracy(a, y):
    mini_batch = a.shape[1]
    idx_a = np.argmax(a, axis=0)
    idx_y = np.argmax(y, axis=0)
    acc = sum(idx_a == idx_y) / mini_batch
    return acc


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (data num, channel, height, length)
    filter_h : filter height
    filter_w : filter length
    stride :
    pad :

    Returns
    -------
    col : 2-dim
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 4-dim e.g.(10, 1, 28, 28)
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def display(array):
    x = range(0, 20, 1)
    plt.title('Loss')
    plt.plot(x, array, color='blue', marker='o', label='train loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

