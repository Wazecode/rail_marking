#!/usr/bin/python3

import torch

from build.lib.segmentation.models import BiSeNetV2


def parseToOnnx():

    net = BiSeNetV2(n_classes=3)
    checkpoint = torch.load(('bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth'),map_location=torch.device('cpu'))
    net.load_state_dict(
        checkpoint['state_dict'])
    

    print(net.eval())

    batch_size, channels, height, width = 1, 3, 512, 1024
    inputs = torch.randn((batch_size, channels, height, width))

    outputs = net(inputs)
    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'

    torch.onnx.export(
        net,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        "bisenetv2.onnx",  # where to save the model (can be a file or file-like   object)
        export_params=
        True,  # store the trained parameter weights inside the model     file
        opset_version=11, # the ONNX version to export the model to
        do_constant_folding=
        False,  # whether to execute constant folding for optimization
        input_names=['inputs'],  # the model's input names
        output_names=['outputs'],  # the model's output names
        dynamic_axes={
            'inputs': {
                0: 'batch_size'
            },  # variable lenght axes
            'outputs': {
                0: 'batch_size'
            }
        })

    print("ONNX model conversion is complete.")
    return


parseToOnnx()
