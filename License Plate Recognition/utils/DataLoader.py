# loader for data
import os
import torch
import numpy as np
from PIL import Image
from xml.dom.minidom import parse
import cv2
from pathlib import Path
import glob,time
from utils.utils import letterbox
from threading import Thread
PATH = "data/"


# For resnet
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
      
        self.images = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))

    def __getitem__(self, index):
        # load image and bbox
        img_path = os.path.join(self.root, "JPEGImages", self.images[index])
        bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[index])
        img = Image.open(img_path).convert("RGB")

        # read doc，VOC-xml
        dom = parse(bbox_xml_path)

        # get document element object
        data = dom.documentElement

        # get objects
        objects = data.getElementsByTagName('object')

        # get the coordinates of the bounding box
        boxes = []
        labels = []
        for object_ in objects:
            # get label content
            name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 就是label
            labels.append(np.int(name[-1]))  # 背景的label是0，mark_type的label是1
            bndbox = object_.getElementsByTagName('bndbox')[0]
            xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
            ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
            xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
            ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)


# For YOLO
class LoadImages:
    def __init__(self, path, img_size=640):
        self.path = str(Path(path))
        files = []
        if os.path.isdir(self.path):
            files = sorted(glob.glob(os.path.join(self.path, '*.*')))
        elif os.path.isfile(self.path):
            files = [self.path]
        self.img_size = img_size
        self.files = files
        self.img_num = len(files)       # 75
        self.mode = 'images'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.img_num:
            raise StopIteration
        path = self.files[self.count]

        # get image
        self.count += 1
        img0 = cv2.imread(path)
        assert img0 is not None, 'Image Not Found ' + path
        print('image %g/%g %s: \n' % (self.count, self.img_num, path), end='')

        # padding resize image
        img = letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return path, img, img0

    def __len__(self):
        return self.img_num  # number of files






