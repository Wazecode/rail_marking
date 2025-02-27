#!/usr/bin/env python3

import os
import sys
import torch
import torchvision
import numpy as np
import cv2

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from build.lib.segmentation.models import BiSeNetV2
    from build.lib.segmentation.data_loader import Rs19DatasetConfig
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)

data_config = Rs19DatasetConfig()
model = BiSeNetV2(n_classes=data_config.num_classes)
model.load_state_dict(torch.load('bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth')['state_dict'])
model.eval()

batch_size, channels, height, width = 1, 3, 512, 1024
dummy_input = torch.randn((batch_size, channels, height, width))

input_names = ['actual_input1']+ [ "learned_%d" % i for i in range(16) ]
output_names = ['output1']

torch.onnx.export(model, dummy_input, "rail_detect.onnx",  verbose=True, input_names=input_names, output_names=output_names)
