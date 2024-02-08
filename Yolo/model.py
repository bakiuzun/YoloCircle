import torch
from ultralytics.nn.tasks import *
from utils import parse_my_detection_model
import copy


# torch.Size([16, 576, 20, 20]) backbone out shape
class YoloCircle(BaseModel):
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

        self.head_1 = self.model[10:]
        self.backbone = self.model[:10]


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
