from model import YoloCircle
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor


img = np.array(cv2.imread("image_test.jpeg"))
img = cv2.resize(img,(640,640))
img = ToTensor()(img)
img = img.unsqueeze(0)

#img shape (1,3,640,640)
model = YoloCircle(cfg="myyolov8m.yaml",nc=1)
model.eval()
model(img)
