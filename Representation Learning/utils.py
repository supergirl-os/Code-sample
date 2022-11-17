import numpy as np
import math,cv2
from params import *
import matplotlib.pyplot as plt


# 对稀疏系数的one_hot类似的处理
def one_hot_likelihood(x):
    res = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > ONE_HOT_THRESHOLD:
            res[i] = 1
        else:
            res[i] = 0
    return res


def one_hot_for_dict(train_data):
    dict = {}
    index = 0
    for key in train_data.keys():
        one_hot = np.zeros(1400, dtype=np.float64)
        for i in range(14):
            one_hot[i+index*14] = 1
        index += 1
        dict[key] = one_hot
    return dict


# 归一化
def normalization(data,selection="average"):
    norm_data = np.zeros_like(data,dtype=np.float64)
    if selection == "max_min":
        min = np.min(data)
        max = np.max(data)
        for i in range(len(data)):
            norm_data[i] = (data[i]-min)/(max-min)
        return norm_data
    else:
        total = sum(data)/10
        for i in range(len(data)):
            norm_data[i] = data[i]/total
    return norm_data


def de_normalization(data):
    de_norm_data = []
    for i in range(len(data)):
        de_norm_data.append(int(np.round(data[i]*255)))
    return np.array(de_norm_data)


# 正交匹配法
def omp(y, A, sample_length):
    index_s = []    # 长度最长原子向量的索引
    A_k = np.zeros([sample_length, 0])
    e = y[:, np.newaxis]
    for i in range(sample_length):
        # 计算残差对每个原子向量投影
        predicted = np.abs(np.dot(A.T, e))
        # 选择残差投影长度最长的原子向量的索引
        longest_atom_index = np.argmax(predicted)
        index_s.append(longest_atom_index)
        # 加入新选择的基
        A_k = np.c_[A_k, A[:, longest_atom_index]]
        # 重新计算表达
        x_k = np.linalg.pinv(A_k.T.dot(A_k)).dot(A_k.T).dot(y[:, np.newaxis])
        # 重新计算残差
        e = y[:, np.newaxis] - A_k.dot(x_k)

    x = np.zeros(shape=(1400,), dtype=np.float64)
    # 按照位置的对应关系放入非零元素
    x[index_s] = x_k.flatten()
    return x


# 计算准确率
def accuracy(x,array):
    correct = 0
    for item in array:
        if item == x:
            correct += 1
    return correct/len(array)


# 图片可视化
def display(img, type):
    # type = 'image' or 'column'

    if type == 'column':
        img = de_normalization(img)
        img = img.reshape(PATCH_SIZE)
    cv2.imshow("img",img)
    cv2.waitKey(0)


def show_acc(acc, cla):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # waters = ('碳酸饮料', '绿茶', '矿泉水', '果汁', '其他')
    # buy_number = [6, 7, 6, 1, 2]
    plt.bar(cla, acc)
    plt.title('每一类别对应预测准确率情况')
    plt.show()


