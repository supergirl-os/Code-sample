from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import normalize
import cv2
from utils import *
from params import *
root_path = "AR/"


class DataLoader:
    def __init__(self):
        self.path = root_path
        self.train_data = {}
        self.test_data = {}
        # 初始化训练和测试数据集，整体为一个字典，每个人对应一个列表，总共100人，其中训练集每个列表有14条数据，测试集每个列表有12条数据
        for file in os.listdir(self.path):
            # label为该图片所属的类别
            label = file.split('-')[0] + file.split('-')[1]
            self.train_data[label] = []
            self.test_data[label] = []

    def get_data(self):
        # 读取所有图片
        for file in os.listdir(self.path):
            # seg用来判断是该图片是否有遮挡
            seg = int(file.split('.')[0].split('-')[-1])
            # label为该图片所属的类别
            label = file.split('-')[0] + file.split('-')[1]
            # 打开图片
            img = Image.open(self.path + file)
            img = np.array(img)
            # 进行降采样
            img = cv2.resize(img, DOWN_SAMPLE_SIZE)
            if IS_PATCH:
                img = self.patch_data(img)  # 获取到(9,PATCH_SAMPLE_LENGTH)
                img = np.array(img)
            else:
                # reshape
                img = img.reshape(-1, 1)
                # 对于降维取特征值
                # col_img = []
                # img = img.reshape(120, 10)
                # for i in range(len(img)):
                #     col_img.append(sum(img[i])/10)
                # img = np.array(col_img)
                # 进行归一化
                img = normalization(img)
            if (1 <= seg <= 7) or (14 <= seg <= 20):
                # 训练集
                self.get_train_data(img, label)
            else:
                self.get_test_data(img, label)
        return self.train_data, self.test_data

    def get_train_data(self, img, label):
        # 获取训练集
        self.train_data[label].append(img)

    def get_test_data(self, img, label):
        # 获取测试集
        self.test_data[label].append(img)

    def patch_data(self,img):
        every_patch = []
        for row in row_dividers:
            for col in col_dividers:
                # 根据行列分割点,分割出对应的patch
                patch = img[row:row+PATCH_SIZE[0],col:col+PATCH_SIZE[1]]
                # 将patch展平
                patch = patch.reshape(-1, 1)
                patch = np.squeeze(patch)
                # 归一化
                patch = normalization(patch)
                every_patch.append(patch)
        return every_patch

    def img_display(self,file):
        # seg用来判断是该图片是否有遮挡
        seg = int(file.split('.')[0].split('-')[-1])
        # label为该图片所属的类别
        label = file.split('-')[0] + file.split('-')[1]
        # 打开图片
        img = Image.open(self.path + file)
        img = np.array(img)
        # 进行降采样
        img = cv2.resize(img, DOWN_SAMPLE_SIZE)
        cv2.imshow(str(seg)+label,img)
        cv2.waitKey(0)


# test = DataLoader()
# a, b = test.get_data()
# print(a['m001'][0].shape)   # (1200,1)
# print(np.array(a['m001']).shape)    # (14,1200,1)
# test.img_display("m-001-01.pgm")
