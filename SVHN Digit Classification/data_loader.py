from scipy.io import loadmat
import numpy as np


# same function to "to_categorical", change the label to vector
def mapping(src):
    dst = np.zeros(10)
    if src == 0 or src ==10:
        dst[0] = 1
    elif src == 1:
        dst[1] = 1
    elif src == 2:
        dst[2] = 1
    elif src == 3:
        dst[3] = 1
    elif src == 4:
        dst[4] = 1
    elif src == 5:
        dst[5] = 1
    elif src == 6:
        dst[6] = 1
    elif src == 7:
        dst[7] = 1
    elif src == 8:
        dst[8] = 1
    else:
        dst[9] = 1
    return dst


def data_loader():
    m_train = loadmat("data/train.mat")
    m_test = loadmat("data/test.mat")
    train_data, train_labels = m_train['X'], m_train['y']
    test_data, test_labels = m_test['X'], m_test['y']
    train_size = 73257
    x_train = train_data.reshape(-1, train_size)
    test_size = 26032
    x_test = test_data.reshape(-1, test_size)
    # Visualization pictures
    # ====================
    # X = x_test.T
    # y = test_labels.flatten()
    # print(X.shape, y.shape)
    # pca = decomposition.PCA(n_components=3)
    # new_X = pca.fit_transform(X)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=y, cmap=plt.cm.Spectral)
    # plt.show()
    # =====================
    train_label = []
    test_label = []
    # change the label of trainset
    for i in range(len(train_labels)):
        train_label.append(mapping(train_labels[i]))
    train_label = np.array(train_label)
    # change the label of testset
    for i in range(len(test_labels)):
        test_label.append(mapping(test_labels[i]))
        # print(mapping(test_labels[i]),"**",test_labels[i])
    test_label = np.array(test_label)
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    print("x_train", x_train.shape, "x_test", x_test.shape)
    return x_train, train_label.T, x_test, test_label.T


def loader_for_conv():
    m_train = loadmat('data/train.mat')
    m_test = loadmat('data/test.mat')
    x_train_raw = m_train['X']
    x_test_raw = m_test['X']
    x_train = np.transpose(x_train_raw, (3, 2, 0, 1))
    x_test = np.transpose(x_test_raw, (3, 2, 0, 1))
    label_train_raw = m_train['y']
    label_test_raw = m_test['y']
    label_train = np.zeros((10, label_train_raw.shape[0]))
    for index in range(label_train_raw.shape[0]):  # 73257
        if label_train_raw[index][0] == 10:
            label_train[0][index] = 1
        else:
            label_train[label_train_raw[index][0]][index] = 1
    label_test = np.zeros((10, label_test_raw.shape[0]))
    for index in range(label_test_raw.shape[0]):  # 73257
        if label_test_raw[index][0] == 10:
            label_test[0][index] = 1
        else:
            label_test[label_test_raw[index][0]][index] = 1
    print(x_train.shape,label_train.shape,x_test.shape,label_test.shape)
    return x_train, label_train, x_test, label_test

