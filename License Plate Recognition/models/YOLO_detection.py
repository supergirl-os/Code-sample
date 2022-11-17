import argparse, random
import torch.backends.cudnn as cudnn
from utils.DataLoader import *
from utils.utils import *
from models.LPRNet import *
from utils import torch_utils, google_utils
from sys import platform
from params import Recognize


class YOLO_detection:
    def __init__(self):
        # self.src = "data/"
        # self.dst = "dst_3/"
        # self.model = "./trained_models/lastforyolo.pt"
        # 利用parser设置参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='./trained_models/lastforyolo.pt',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='dst_3/', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        self.opt = parser.parse_args()
        print(self.opt)

    def locate(self):
        # 此函数用来使用训练好的模型进行本数据集推理，其中模型在colab在线训练,以
        # 获取初始化时的路径，设备等参数

        source = self.opt.source    # 待预测图片的源路径
        out = self.opt.output          # 预测图片的目标存放路径
        weights = self.opt.weights  # 预训练好的模型存放路径
        view_img, img_size = self.opt.view_img, self.opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        device = torch_utils.select_device(self.opt.device)     # 表示当前推理使用的设备，一般为CPU即可
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # 加载预训练模型
        model = attempt_load(weights, map_location=device)
        img_size = check_size(img_size, stride=model.stride.max())  # 检查img_size
        if half:
            model.half()

        # 进一步的字符识别
        classify = True
        modelc = None
        plat_num = None

        if classify:
            # 加载LPRNet,对检测到的车牌进行分类，此处属于拓展功能！
            modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
            modelc.load_state_dict(
                torch.load('./trained_models/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
            print("Load pretrained model successful!")
            modelc.to(device).eval()

        # 获取模型类别和随机获取检测框颜色
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # 设置DataLoader
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

        # 开始预测推理
        t0 = time.time()
        img = torch.zeros((1, 3, img_size, img_size), device=device)  # 初始化img数据集
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s in dataset:
            # 一个epoch
            # 加载数据
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # 推理
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=self.opt.augment)[0]
            print(pred.shape)
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                            agnostic=self.opt.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # 车牌识别
            if classify:
                pred, plat_num = apply_classifier(pred, modelc, img, im0s)

            # 处理检测结果
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # 打印results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # 写结果
                    for de, lic_plat in zip(det, plat_num):
                        *xyxy, conf, cls = de

                        if save_img:  # Add bbox to image
                            lb = ""
                            for a, i in enumerate(lic_plat):
                                # if a ==0:循环
                                lb += CHARS[int(i)]
                            label = '%s %.2f' % (lb, conf)
                            print("label", label)
                            if not Recognize:
                                label = None
                            im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                print('一张图片预测结束，经过 %s . (%.3fs)' % (s, t2 - t1))

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)

        print('结果保存到%s' % os.getcwd() + os.sep + out)
        print('整个流程结束，用时.(%.3fs)' % (time.time() - t0))
