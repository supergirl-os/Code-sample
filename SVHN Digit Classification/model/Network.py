# Simple BP algorithm, feedforward neural network
import numpy as np
import utils as u
import math
import matplotlib.pyplot as plt
import pickle
from module import LayerOfWeight


class Network:
    def __init__(self, L, alpha, epochs, batch_size,if_scheduler):
        self.L = L  # NN layers
        self.alpha = alpha  # learning rate
        self.epochs = epochs  # training epoch
        self.batch_size = batch_size  # sample of each mini batch
        self.if_scheduler = if_scheduler    # lr schedule
        self.layer = []  # structure
        self.a = {}  # neurons
        self.z = []  # temp z
        self.delta = {}  # save delta
        self.J = []  # cost of each mini batch
        self.acc = []  # accuracy of each mini batch
        self.show_acc=[]
        self.show_J = []

    def designer(self):
        # Network Architecture Design
        if self.L - 1 == 2:
            self.layer = [
                LayerOfWeight(256, self.alpha),
                LayerOfWeight(10, self.alpha)
            ]
        elif self.L - 1 == 5:
            self.layer = [
                LayerOfWeight(512, self.alpha),
                LayerOfWeight(256, self.alpha),
                LayerOfWeight(128, self.alpha),
                LayerOfWeight(64, self.alpha),
                LayerOfWeight(10, self.alpha)
            ]
        else:
            self.layer = [  # L = 8
                LayerOfWeight(2048, self.alpha),
                LayerOfWeight(1024, self.alpha),
                LayerOfWeight(512, self.alpha),
                LayerOfWeight(256, self.alpha),
                LayerOfWeight(128, self.alpha),
                LayerOfWeight(64, self.alpha),
                LayerOfWeight(10, self.alpha)
            ]

    def fc(self, a):
        # forward 
        self.z.append(a)
        for layer in self.layer:
            a, z = layer.fc(a)  # call that layer
            self.z.append(z)
        return a  # return the output of the last layer

    def bc(self, delta):
        # backward
        for i, layer in enumerate(self.layer[::-1]):
            delta = layer.bc(self.z[len(self.z) - 2 - i], delta)

    def update(self):
        for layer in self.layer:
            layer.update()

    def train(self, x_train, train_labels, x_test, test_labels):
        # Step 6: Train the Network
        train_size = 73257  # number of train_set
        test_size = 26032
        batch_len = math.ceil(train_size / self.batch_size)  # batch of each epoch

        for epoch in range(self.epochs):
            if self.if_scheduler:
                self.alpha = u.lr_schedule(epoch)    # lr schedule on
            index = np.random.permutation(train_size)  # for divide the training set into random batch
            for k in range(batch_len):
                start_index = k * self.batch_size
                end_index = min((k + 1) * self.batch_size, train_size)
                batch_indices = index[start_index:end_index]
                self.a[1] = x_train[:, batch_indices]  # initialize the first layer
                y = train_labels[:, batch_indices]  # get the according labels
                # forward computation
                self.a[self.L] = self.fc(self.a[1])
                self.delta[self.L] = (self.a[self.L] - y) * (self.a[self.L] * (1 - self.a[self.L]))
                # backward computation
                self.bc(self.delta[self.L])
                # update weights
                self.update()
                self.J.append(u.cost(self.a[self.L], y) / self.batch_size)
                self.acc.append(u.accuracy(self.a[self.L], y))
            # Test the Network
            self.a[1] = x_test
            y = test_labels
            # forward computation
            self.a[self.L] = self.fc(self.a[1])
            self.show_acc.append(self.acc[-1])
            self.show_J.append(self.J[-1])
            print("---------------Each epoch-----------------")
            print(epoch,
                  "training loss:{:.5f} \t test loss:{:.5f}".format(self.J[-1], u.cost(self.a[self.L], y) / test_size))
            print(epoch,
                  "training acc:{:.2f} \t test acc:{:.2f}".format(self.acc[-1] * 100,
                                                                  (u.accuracy(self.a[self.L], y)) * 100))
            # print(epoch, "training loss:", self.J[-1], 'test loss:', u.cross_entropy_error(self.a[self.L],
            # y) / test_size)

    def show(self):
        # show the cost line
        x = range(0, self.epochs, 1)
        plt.title('Loss')
        plt.plot(x, self.show_J, color='red', marker='o', label='train loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        plt.title('Accuracy')
        plt.plot(x, self.show_acc, color='red', marker='o', label='train acc')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.show()

    def save_model(self):
        # Store the Network Parameters && save model
        w = []
        for layer in self.layer:
            w.append(layer.w)
        with open("model.pkl", 'wb') as f:
            pickle.dump(w, f)
            print("Saved successfully!")
