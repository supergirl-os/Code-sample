import numpy as np
from utils import im2col, col2im, f, cross_entropy_error
from collections import OrderedDict
import utils as u
from module import *
import pickle
from utils import *


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # output of softmax
        self.t = None  # label

    def forward(self, x, t):
        self.t = t
        self.y = u.softmax(x)
        self.loss = u.cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[1]
        dx = (self.y - self.t.T) / batch_size
        return dx


class ConvNet:
    # conv - relu - pool - linear - relu - linear - softmax
    # input_dim：input data channels, height, length
    # conv_param：hyperparameters of convolutional layer
    # hidden_size：number of neurons in hidden layer
    # output_size：number of neurons in the last fully connected layer
    # weight_init_std：standard deviation of weights
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']  # hyperparameters of convolutional layer（dict）
        filter_size = conv_param['filter_size']  # convolution kernel size
        filter_pad = conv_param['pad']  # 
        filter_stride = conv_param['stride']  # 
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        # weights initialize
        # parameter initialization of the convolution layer
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # parameter initialization of the two Linear layers
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size,
                                            hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        # generate layer
        self.layers = OrderedDict()
        # add layers in OrderedDict 
        # name as 'Conv1'、'Relu1'、'Pool1'、'Linear1'、'Relu2'、'Affine2'
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def backward(self, x, y):
        # forward
        self.loss(x, y)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        # Save the gradients of the weight parameters calculated during the learning process to the grads dictionary
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]


class Trainer:
    def __init__(self, batch_size, epochs, network, optimizer='SGD', optimizer_param={'lr': 0.01}):
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network
        optimizer_class_dict = {'sgd': SGD,  'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.current_iter = 0
        self.current_epoch = 0
        self.train_loss = []
        self.train_loss_all = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

    def train(self, x_train, train_labels, x_test, test_labels):
        train_size = x_train.shape[0]
        batch_len = math.ceil(train_size / self.batch_size)
        for epoch in range(self.epochs):
            index = np.random.permutation(train_size)  # for divide the training set into random batch
            for k in range(batch_len):
                start_index = k * self.batch_size
                end_index = min((k + 1) * self.batch_size, train_size)
                batch_indices = index[start_index:end_index]
                x_batch = x_train[batch_indices]  # initialize the first layer
                y_batch = train_labels[:, batch_indices]  # get the according labels
                grads = self.network.backward(x_batch, y_batch)
                self.optimizer.update(self.network.params, grads)
                loss = self.network.loss(x_batch, y_batch)/self.batch_size
                self.train_loss_all.append(loss)
                # print("train_loss = ", loss)
            self.train_loss.append(self.train_loss_all[-1])     # train_loss of each Epoch
            # test network
            x_test_sample = x_test[0:500]
            y_test_sample = test_labels[:,0:500]
            test_size = x_test[0]
            grads = self.network.backward(x_test_sample, y_test_sample)
            self.optimizer.update(self.network.params, grads)
            test_loss = self.network.loss(x_test_sample, y_test_sample) / test_size
            self.test_loss.append(test_loss)
            print("train loss: ", self.train_loss[-1], " test loss: ",self.test_loss[-1])
        # Draw graph to show train_loss
        # u.display(self.train_loss, "Epoch")
        # u.display(self.train_loss_all, "TotalBatchSize")



