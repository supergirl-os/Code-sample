# main
from models.PlateLocation import *
from models.ColorLocation import *
from models.YOLO_detection import *
from utils.DataLoader import *

PATH = "data/"


# Traditional shape-based method
def method_1():
    dst = "dst_1/"
    if not os.path.exists(dst):
        # create if not exist
        os.mkdir(dst)
    for file in os.listdir(PATH):
        imagePath = PATH + file
        image = cv2.imread(imagePath)
        solver = PlateLocation(image)
        patch = solver.locate()
        cv2.imwrite(dst+file, patch)


# Traditional color-based method
def method_2():
    dst = "dst_2/"
    iou = []        # save IoU for each image
    if not os.path.exists(dst):
        # create if not exist
        os.mkdir(dst)
    for file in os.listdir(PATH):
        imagePath = PATH + file
        image = cv2.imread(imagePath)
        solver = ColorLocation(image,file)
        patch, aver_iou = solver.locate()
        iou.append(aver_iou)
        cv2.imwrite(dst + file, patch)
    total = 0
    for i in iou:
        total += i
    average = total/75
    print("The IoU of dataset:", average)
    draw_iou(iou)


# Yolo
def method_3():
    test = YOLO_detection()
    test.locate()



method_3()