import subprocess
from prettytable import PrettyTable
import numpy as np

import torch

import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_static, calibrate, CalibrationDataReader, QuantFormat

import os
import json
import argparse
import time
import copy
import numpy as np
from datetime import timedelta

import torch

from timm.utils import accuracy

from util.datasets import build_dataset, build_calib_loader
from util import misc


def onnx_convert(model, model_name="convnext_small", dummy_size=(1, 3, 224, 224), save_dir="./" ):
    model.eval()
    x = torch.randn(dummy_size, requires_grad=True)

    file_name =  model_name + ".onnx"
    torch.onnx.export(model,              
                    x,                         
                    save_dir + "/"  + file_name,
                    export_params=True,        
                    opset_version=17,          
                    do_constant_folding=True,  
                    input_names = ['input'],   
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}},
                    )
    print(f"Model {model_name} is converted to ONNX format as {save_dir}/{file_name}")
    return save_dir + "/"  + file_name

def onnx_prepro(onnx_model_path):
    
    #preprocess the model. fusion folding etc
    command = [
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", onnx_model_path,
        "--output", onnx_model_path
    ]

    result = subprocess.run(command, check=False)
    print(f"Model {onnx_model_path} is preprocessed")
    return onnx_model_path

class ONNX_calib_loader(CalibrationDataReader):
    def __init__(self, calib_loader):
        self.enum_data = None

        self.nhwc_data_list = []
        for i, (images, _) in enumerate(calib_loader):
            images = images.numpy()
            self.nhwc_data_list.append(images)

        self.input_name = "input"
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def onnx_quantize(model_path, onnx_calib_loader):

    model = onnx.load(model_path)
    graph = model.graph

    nodes_to_quantize = []
    for node in graph.node:
        if node.op_type in ["Conv", "Gemm"]: 
            nodes_to_quantize.append(node.name)
    # print(f"node to quantize {nodes_to_quantize}")
    
    quantize_static(
        model_input=model_path,
        model_output=model_path.replace(".onnx", "_quant.onnx"),
        per_channel=True,
        reduce_range=True,
        # nodes_to_quantize=nodes_to_quantize,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True,
                        "WeightSymmetric": True},
        calibration_data_reader=onnx_calib_loader,
        calibrate_method=calibrate.CalibrationMethod.MinMax,
        )
    print(f"Model {model_path} is quantized")

def ONNX_inference(calib_loader, fp_model, val_loader, output_dir, args):
    onnx_calib_loader = ONNX_calib_loader(calib_loader)
    onnx_path = onnx_convert(fp_model, model_name=args.model, dummy_size=(1, 3, args.input_size, args.input_size), save_dir=output_dir)
    onnx_path = onnx_prepro(onnx_path) 
    onnx_quantize(onnx_path, onnx_calib_loader)
    onnx_path = onnx_path.replace(".onnx", "_quant.onnx")

    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=EP_list)
    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'ONNX inference:'

    for inputs, labels in metric_logger.log_every(val_loader, 100, header):

        inputs_np = inputs.numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: inputs_np}
        ort_outs = ort_session.run(None, ort_inputs)

        logits_np = ort_outs[0]
        logits_tensor = torch.from_numpy(logits_np)

        acc1, acc5 = accuracy(logits_tensor, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} , onnx_path



    
