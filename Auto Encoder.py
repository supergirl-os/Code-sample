import cv2
import os, math, pickle, time
import os.path
from matplotlib import pyplot as plt
from bc import bc
from fc import fc
import numpy as np

path_1 = "random_animal/"
path_2 = "tiger/"
path_3 = "lion/"
IMAGE_SIZE = 50         
dst_path_ = ["blind_data/", "train_data/", "test_data/"]
def normalize(data):
    n_data = np.zeros(len(data))
    for i in range(len(data)):
        n_data[i] = data[i] / 255
    return n_data
def d_normalize(data):
    n_data = np.zeros(len(data), dtype=np.int64)
    for i in range(len(data)):
        n_data[i] = round(data[i] * 255)
    return n_data
def cost(a, y, beta, a_l):
    J = 1 / 2 * np.sum((a - y) ** 2) + beta * np.sum(a_l)
    return J
def o_cost(a, y):
    J = 1 / 2 * np.sum((a - y) ** 2)
    return J
def show_img(data, epoch):
    n_data = data.reshape(28, 28, 3).astype(int)
    cv2.imwrite("show/" + str(epoch) + ".jpg", n_data)
# Read and prepare the data
def generate_data(path):
    files = os.listdir(path)
    if path == "random_animal/":
        for i in range(len(files)):
            img = cv2.imread(path + files[i])
            w, h, g = img.shape
            dst = img[int((w - IMAGE_SIZE) / 2):int((w + IMAGE_SIZE) / 2),
                  int((h - IMAGE_SIZE) / 2):int((h + IMAGE_SIZE) / 2)]
            cv2.imwrite(dst_path_[0] + files[i].split('.')[0] + "0.jpg", dst)
    else:
        train_num = len(files) * 0.8
        for j in range(len(files)):
            img = cv2.imread(path + files[j])
            w, h, g = img.shape
            dst = img[int((w - IMAGE_SIZE) / 2):int((w + IMAGE_SIZE) / 2),
                  int((h - IMAGE_SIZE) / 2):int((h + IMAGE_SIZE) / 2)]
            if j < train_num:
                cv2.imwrite(dst_path_[1] + path.split('/')[0] + "_" + str(j) + ".jpg", dst)
            else:
                cv2.imwrite(dst_path_[2] + path.split('/')[0] + "_" + str(int(j - train_num)) + ".jpg", dst)
def get_blind_data():
    blind_data_ = np.zeros((7800, 2352), dtype=np.int64)
    files = os.listdir(dst_path_[0])
    for i in range(len(files)):
        img = cv2.imread(dst_path_[0] + files[i])
        img = img.flatten()
        dst = normalize(img)
        blind_data_[i] = dst
    return blind_data_.T


def get_train_data():
    train_data = np.zeros((320,2352),dtype=np.float64)
    train_label = np.zeros((320,2))
    files = os.listdir(dst_path_[0])
    for i in range(len(files)):
        img = cv2.imread(dst_path_[1] + files[i])
        img = img.flatten()
        dst = normalize(img)
        train_data[i] = dst
        if files[i].split('_')[0] == "lion":
            train_label[i]=np.array([1,0])
        else:
            train_label[i] = np.array([0,1])
    return train_data.T,train_label.T


def get_test_data():
    test_data = np.zeros((80,2352),dtype=np.float64)
    test_label = np.zeros((80,2))
    files = os.listdir(dst_path_[2])
    for i in range(len(files)):
        img = cv2.imread(dst_path_[2] + files[i])
        dst = normalize(img)
        test_data[i] = dst
        print(test_data[i])
        if files[i].split('_')[0] == "lion":
            test_label[i]=np.array([1,0])
        else:
            test_label[i] = np.array([0,1])
        print(test_label[i])
    return test_data.T,test_label.T
class Network:
    def __init__(self, L, alpha, epochs, batch_size):
        self.L = L
        self.l = int(self.L / 2)  # 第3层
        self.alpha = alpha  # initialize learning rate
        self.beta = 1e-6
        self.epochs = epochs  # training epoch
        self.batch_size = batch_size  # sample of each mini batch
        self.layer_size = []  # define number of neurons in each layer
        self.w = {}  # initialize weights
        # Network Architecture Design
        # define number of neurons in each layer
        self.layer_size = [2352,  # number of neurons in 1st layer
                           2048,  # number of neurons in 2nd layer
                           516,  # number of neurons in 3th layer
                           2048,  # number of neurons in 4th layer
                           2352,  # number of neurons in 5th layer
                           ]
        # initialize weights
        for l in range(1, self.L):
            self.w[l] = 0.1 * np.random.randn(self.layer_size[l], self.layer_size[l - 1])
        self.a = {}
        self.z = {}
        self.delta = {}
        self.J = []  # cost of each mini batch
        self.acc = []  # accuracy of each mini batch

    def train(self, x_train, train_labels):
        # Train the Network

        train_size = 7800  # number of train_set
        batch_len = math.ceil(train_size / self.batch_size)  # batch of each epoch
        for epoch in range(self.epochs):
            index = np.random.permutation(train_size)  # for divide the training set into random batch
            for k in range(batch_len):
                start_index = k * self.batch_size
                end_index = min((k + 1) * self.batch_size, train_size)
                batch_indices = index[start_index:end_index]
                # init_a = np.zeros((256,len(batch_indices)))
                # self.a[1]= np.concatenate([x_train[:, batch_indices],init_a],axis=0)  # initialize the first layer
                # print(d_normalize(x_train.T[0]))
                # print("batch_indices:",batch_indices)
                # print("test", x_train.T[0])
                # print("test",x_train.T[int(batch_indices[0])])
                self.a[1] = x_train[:, batch_indices]
                y = train_labels[:, batch_indices]  # get the according labels

                # forward computation
                for i in range(1, self.L):
                    self.a[i + 1], self.z[i + 1] = fc(self.w[i], self.a[i])

                self.delta[self.L] = (self.a[self.L] - y + self.beta) * (self.a[self.L] * (1 - self.a[self.L]))
                # backward computation

                for j in range(self.L - 1, 1, -1):
                    self.delta[j] = bc(self.w[j], self.z[j], self.delta[j + 1], self.beta)
                # update weights
                for l in range(1, self.L):
                    grad_w = np.dot(self.delta[l + 1], self.a[l].T) / self.batch_size
                    self.w[l] = self.w[l] - self.alpha * grad_w

                self.J.append(cost(self.a[self.L], y, self.beta, self.a[int(self.L / 2)]) / self.batch_size)
                # print("label", y)
                # print(self.a[1])
                # print(self.a[self.L])
                # print(self.w)
            # print(self.a[self.L].shape)
            # print(self.a[self.L].T[0].shape)
            # show_img(d_normalize(self.a[self.L].T[0]))
            if(self.J[-1]<0.01):
                self.save_model()
                print("success")
            print(epoch, "training cost:", self.J[-1])
            if epoch % 20 == 0:
                data = self.a[self.L].T
                show_img(d_normalize(data[0]), epoch)

    def predict(self, x_test, test_labels):
        self.a[1] = x_test
        y = test_labels
        for l in range(1, self.l):
            self.a[l + 1], self.z[l + 1] = fc(self.w[l], self.a[l])
        print('test cost:', cost(self.a[self.l], y, self.beta, self.a[self.l]))
        return self.a[self.l]

    def display(self, x, y):
        # display/listen to some results pairs
        self.a[1] = x
        y = y
        # forward computation
        for l in range(1, self.L):
            self.a[l + 1], self.z[l + 1] = fc(self.w[l], self.a[l])
        print(y)
        plt.plot(self.a[self.L])
        plt.plot(y)
        plt.show()
        plt.close()

    def save_model(self):
     
        model_name = 'model.pkl'
        with open(model_name, 'wb') as f:
            pickle.dump([self.w, self.layer_size], f)
        print("The model has been saved to {}".format(model_name))


class Divider:
    def __init__(self,L,alpha, epochs, batch_size):

        self.L = L
        self.alpha = alpha  # initialize learning rate
        self.beta = 1e-6
        self.epochs = epochs  # training epoch
        self.batch_size = batch_size  # sample of each mini batch

        self.layer_size = []  # define number of neurons in each layer
        self.w = {}  # initialize weights
        # Network Architecture Design
        # define number of neurons in each layer
        # - 1st column: external neurons
        # - 2nd column: internal neurons
        self.layer_size = [516,  # number of neurons in 1st layer
                           64,   # number of neurons in 2nd layer
                           2,   # number of neurons in 3th layer
        ]
        # initialize weights
        for l in range(1, self.L):
            self.w[l] = 0.1 * np.random.randn(self.layer_size[l], self.layer_size[l - 1])
        self.a = {}
        self.z = {}
        self.delta = {}
        self.J = []  # cost of each mini batch

    def train(self, x_train, train_labels):
        # Train the Divider
        train_size = 320  # number of train_set
        batch_len = math.ceil(train_size / self.batch_size)  # batch of each epoch
        for epoch in range(self.epochs):
            index = np.random.permutation(train_size)  # for divide the training set into random batch
            for k in range(batch_len):
                start_index = k * self.batch_size
                end_index = min((k + 1) * self.batch_size, train_size)
                batch_indices = index[start_index:end_index]
                self.a[1] = x_train[:, batch_indices]
                y = train_labels[:, batch_indices]  # get the according labels

                # forward computation
                for i in range(1, self.L):
                    self.a[i + 1], self.z[i + 1] = fc(self.w[i], self.a[i])

                self.delta[self.L] = (self.a[self.L] - y + self.beta) * (self.a[self.L] * (1 - self.a[self.L]))
                # backward computation

                for j in range(self.L - 1, 1, -1):
                    self.delta[j] = bc(self.w[j], self.z[j], self.delta[j + 1], self.beta)

                # update weights
                for l in range(1, self.L):
                    grad_w = np.dot(self.delta[l + 1], self.a[l].T) / self.batch_size
                    self.w[l] = self.w[l] - self.alpha * grad_w

                self.J.append(cost(self.a[self.L], y, self.beta, self.a[int(self.L / 2)]) / self.batch_size)

            print(epoch, "training cost:", self.J[-1])
            if epoch % 20 == 0:
                data = self.a[self.L].T
                show_img(d_normalize(data[0]), epoch)

    def get_weights(self):
        return self.w


class Model:
    def __init__(self, L, W):
        self.L = L
        self.w = W  # initialize weights
        self.beta = 1e-6
        self.a = {}
        self.z = {}
        self.delta = {}
        self.J = []  # cost of each mini batch

    def predict(self, x_test, test_labels):
        self.a[1] = x_test
        y = test_labels
        for l in range(1, self.L):
            self.a[l + 1], self.z[l + 1] = fc(self.w[l], self.a[l])
        print('test cost:', o_cost(self.a[self.L], y))
        return self.a[self.L]


if __name__ == '__main__':
    # Read and prepare the data
    # prepare the dir for certain shape
    print("Start read data")
    blind_data = get_blind_data()
    alpha = 0.1  # initialize learning rate
    epochs = 100
    batch_size = 50
    L = 5
    net = Network(L, alpha, epochs, batch_size)

    print('Start training')
    net.train(blind_data, blind_data)

    # save model
    net.save_model()

    # Remove the layers behind sparse representation layer after training
    W = {}
    info = pickle.load(open('model.pkl', 'rb'))
    for l in range(int(L / 2)):
        W[l] = info[0][l + 1]

    # Form a new data set in sparse representation layer by using the labeled data set (the trianing set)
    labeled_data, label = get_train_data()
    train_data = net.predict(labeled_data, label)

    # Training the network by using the new training data set
    Divider = Divider(2,alpha, epochs, batch_size)
    Divider.train(train_data, label)
    # combine the two networks
    W[3:5] = Divider.get_weights()
    # test the network with the testing set
    test_data, test_label = get_test_data()
    model = Model(5,W)
    output = model.predict(test_data, test_label)