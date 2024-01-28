import torch
from ultralytics.nn.tasks import *
from utils import parse_my_detection_model
import copy


# torch.Size([16, 576, 20, 20]) backbone out shape
class MyYolo(BaseModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=False,weights=None):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""

        super().__init__()

        self.first_forward = False
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:self.yaml['nc'] = nc  # override YAML value
        self.model, self.save = parse_my_detection_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        first_head = self.model[-1]  # Detect()
        
        s = 256  # 2x min stride
        first_head.inplace = self.inplace
        first_head.stride = torch.tensor([ 8., 16., 32.])
        self.stride = first_head.stride
        first_head.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
    
        if weights != None:
            self.load_pretrained_weights(weights)
            self.enable_all_gradients()

        
        self.head_1 = self.model[10:]
        self.backbone = self.model[:10]

    def enable_all_gradients(self):
        for param in self.parameters():
            param.requires_grad = True
    """
    method used to build the stride
    explanation: the super class call one forward with a lambda x to calculate the strides of the network
    file: ultralytics/nn/task.py
    line: 246
    """
    def _build_stride(self, x, profile=False, visualize=False):
        self.first_forward = False
        x = super()._predict_once(x, profile, visualize)
        return x

    def _predict_once(self, x, profile=False, visualize=False):

        ## refer to _build_stride comment
        ## will not enter here now debugging...
        if self.first_forward:
            return self._build_stride(x,profile,visualize)

        y_1 = []
        x,y_1 = self._forward_backbone(x,y_1)
        x,y_1 = self._forward_head(x,y_1)

        return x


    def _forward_backbone(self,x,y):
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x,y

    def _forward_head(self,x,y):
            for m in self.head_1:
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output
            return x,y

    def load_pretrained_weights(self,weights):self.load(torch.load(weights))





import math

import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors

from .block import DFL
from .conv import Conv

class CircleDetectoe(nn.Module):
    """YOLOv8 Detect head for detection models."""

    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        super().__init__()

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 3 * self.reg_max, 1))
            for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat( (self.cv2[i](x[i]) , self.cv3[i](x[i])), 1)
        
        if self.training:
            return x
        elif  self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
