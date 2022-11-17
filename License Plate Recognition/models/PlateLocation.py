# 使用传统方法实现车牌定位---依靠形状筛选
import cv2
from params import *
import numpy as np
import os


class PlateLocation:
    def __init__(self, img):
        self.src = img  # 源图像
        self.patches = []  # 存放经过定位的多个车牌候选块，类型为RotatedRect
        self.region = []  # 存放每个车牌候选块的四个角点

    def locate(self):
        """
        :return: 一系列轮廓
        """
        # 高斯模糊
        gaussian = cv2.GaussianBlur(self.src, GaussianBlurSize, 0, 0, cv2.BORDER_DEFAULT)

        # 灰度化
        gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

        # Sobel运算
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # Laplace算子
        # laplace = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        # Canny算子
        # canny = cv2.Canny(gray, 50, 150)

        # 二值化
        # 灰度值小于175的点置0，灰度值大于175的点置255
        ret, binary = cv2.threshold(sobel, 175, 255, cv2.THRESH_BINARY)

        # 闭操作
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))  # 腐蚀系数
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 9))  # 膨胀系数
        # erosion = cv2.erode(binary, element1, iterations=1)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))
        # close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 15))
        close = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        close = cv2.dilate(close, element2, iterations=3)
        close = cv2.erode(close, element1, iterations=5)

        # erosion = cv2.erode(dilation, element1, iterations=1)
        # dilation2 = cv2.dilate(close, element2, iterations=3)

        # 求轮廓
        contours_img = None
        # 查找轮廓
        contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        orig1 = self.src.copy()
        for i, c in enumerate(contours):
            if cv2.contourArea(c) < MIN_AREA:
                continue
            rect = cv2.minAreaRect(c)
            if self.judge(rect):
                # print("doing")
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                contours_img = cv2.drawContours(orig1, [box], 0, (0, 255, 0), 3)  # 在原图上绘制出各个检测到的轮廓
                self.patches.append(rect)
                self.region.append(box)

        # 调试模式才需要产生中间图片于“/tmp”中
        if debug:
            if not os.path.exists("/tmp"):
                # 不存在则创建
                os.mkdir("/tmp")
            cv2.imwrite("tmp/gaussian.png", gaussian)
            cv2.imwrite("tmp/gray.png", gray)
            cv2.imwrite("tmp/sobel.png", sobel)
            # cv2.imwrite("tmp/laplace.png", laplace)
            # cv2.imwrite("tmp/canny.png", canny)
            cv2.imwrite("tmp/binary.png", binary)
            # cv2.imwrite("tmp/erosion.png",erosion)
            cv2.imwrite("tmp/close.png", close)
            cv2.imwrite("tmp/dilation.png", dilation)
            # cv2.imwrite("tmp/dilation2.png", dilation2)
            cv2.imwrite("tmp/contours.png", contours_img)

        if contours_img is None:
            return self.src
        return contours_img

    def judge(self, patch):
        # 设置最小最大面积
        minArea = StandardArea * VerifyMin
        maxArea = StandardArea * VerifyMax
        # 设置可接受的最小最大纵横比
        r_min = Aspect - Aspect * Error
        r_max = Aspect + Aspect * Error
        # 获取候选区域的面积
        height = patch[1][0]
        width = patch[1][1]
        area = height * width  # 此处高宽不是由长短决定
        # print("area", area)
        r = height / width
        if r < 1:
            r = width / height
        # print("r", r)
        if (area < minArea or area > maxArea) or (r < r_min or r > r_max):
            return False
        else:
            return True
