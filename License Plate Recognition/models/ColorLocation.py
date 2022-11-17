# 使用传统方法实现车牌定位---依靠颜色+形状筛选
from params import *
from utils.utils import *


class ColorLocation:
    def __init__(self, img,file):
        self.src = img
        self.file = file.split('.')[0]  # 用来读取相应的标注

    def locate(self):
        img = self.src.copy()
        if blur:
            img = cv2.blur(img, (2, 2))  # 对图片进行降噪

        cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # hsvChannels = cv2.split(cvt_img)  # 将HSV格式的图片分解为3个通道
        # h = hsvChannels[0]
        # for i in range(len(h)):
        #     for j in range(len(h[0])):
        #         if h[i][j] > 150:
        #             h[i][j] = 124
        #         else:
        #             h[i][j] = 0
        # cvt_img = cv2.merge(hsvChannels)
        # cv2.namedWindow("Hue", 2)  # 创建一个窗口
        # cv2.imshow('Hue', hsvChannels[0])  # 显示Hue分量
        # cv2.namedWindow("Saturation", 2)  # 创建一个窗口
        # cv2.imshow('Saturation', hsvChannels[1])  # 显示Saturation分量
        # cv2.namedWindow("Value", 2)  # 创建一个窗口
        # cv2.imshow('Value', hsvChannels[2])  # 显示Value分量
        #
        # cv2.waitKey(0)

        mask_img = cv2.inRange(cvt_img, np.array(ColorMin), np.array(ColorMax))
        contours_img = None
        # contours= cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # 找出色域外边界
        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        orig1 = self.src.copy()
        ious = []
        label_num = 0
        for i, c in enumerate(contours):
            if cv2.contourArea(c) < MIN_AREA:
                continue
            rect = cv2.minAreaRect(c)
            # Area = max(region, key=cv2.contourArea)
            # rect = cv2.minAreaRect(Area)  # 绘制面积最大区域的最小外接矩形
            if self.judge(rect):
                # print("doing")
                box = cv2.boxPoints(rect)  # 记录矩形四点坐标
                pre = torch.Tensor(np.int0(point2xyxy(box))[np.newaxis,:])
                # print("pre",pre)
                box = np.int0(box)
                contours_img = cv2.drawContours(orig1, [box], 0, (0, 255, 0), 3)  # 在原图上绘制出检测到的轮廓
                # 计算iou
                label_points = np.int0(get_xml_labels(self.file))
                label_num = len(label_points[0])        # 标准车牌的个数
                tmp_iou = []
                for cont in range(label_num):
                    # 有cont个车牌，依次比较，取最大
                    # label = torch.Tensor(np.int0(get_xml_labels(self.file))[np.newaxis,:])   # box
                    point = np.zeros((1,4))
                    for j in range(4):
                        point[0][j] = label_points[j][cont]
                    label = torch.Tensor(point)
                    # print("label",label)
                    tmp_iou.append(box_iou(pre, label))

                iou = max(tmp_iou)
                # print("iou",iou)
                ious.append(iou)

        if len(ious) < label_num:
            # 没有检测出来, 则设为0
            for miss_out in range(label_num - len(ious)):
                ious.append(0)
        m_iou = 0   # 该图片所有检测到的车牌的总和
        for item in ious:
            m_iou += item
        aver_iou = 0.0
        if m_iou > 0:
            aver_iou = m_iou.numpy()[0][0]/len(ious)
            print(self.file + "图片的m_iou为：", aver_iou)

        if debug:
            cv2.imwrite("../color/img_blur.png", img)
            cv2.imwrite("../color/hsv.png", cvt_img)
            cv2.imwrite("../color/limi.png", mask_img)
            cv2.imwrite("../color/test.png", contours_img)

        if contours_img is None:
            return self.src, aver_iou
        return contours_img, aver_iou

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


# img = cv2.imread("../data/60.jpg")
# te = ColorLocation(img, "60.jpg")
# te.locate()



