"""
profile the model 
"""
import os
import time
import math

import numpy as np

import torch

from fvcore.nn import FlopCountAnalysis
from thop import profile

import onnxruntime

class onnx_profiler(object):
    def __init__(self, dummy_size=(1, 3, 224, 224)):   
        self.dummy_size = dummy_size

    def model_latency(self, model_path):
        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        total = 0.0
        runs = 100
        input_data = np.zeros(self.dummy_size, np.float32)
        # Warming up
        for _ in range(50):
            _ = session.run([], {input_name: input_data})
        for i in range(runs):
            start = time.time()
            _ = session.run([], {input_name: input_data})
            end = (time.time() - start) * 1000
            total += end
        total /= runs
        return round(total, 4)
    
    def model_size(self, model_path):
        size = os.path.getsize(model_path) / 1e6  # 파일 크기를 바이트에서 MB로 변환
        return round(size, 2)
    
    def summary(self, model_path):
        summary_dict = {}
        summary_dict["latency(ms)"] = self.model_latency(model_path)
        summary_dict["size(mb)"] = self.model_size(model_path)
        return summary_dict
    


class torch_profiler(object):
    def __init__(self, dummy_size=(1, 3, 224, 224)):   
        self.dummy_size = dummy_size

    def torch_model_latency(self, model):
        device = "cpu"
        dummy_size = self.dummy_size

        dummy_input = torch.randn(*dummy_size, dtype=torch.float).to(device)
        model = model.to(device)
        model.eval()

        repetitions = 100
        warmup = 50
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

            for rep in range(repetitions):
                start = time.time()
                _ = model(dummy_input)
                end = time.time()
                timings[rep] = (end - start) * 1000  # Convert to milliseconds

        mean_time = np.mean(timings)
        name = "cpu_lantency" + "@bs" + str(1) + "_ms"
        return round(mean_time, 4)
    
    def torch_model_size(self, mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        size = os.path.getsize("tmp.pt")/1e6
        os.remove('tmp.pt')
        return round(size, 2)
    
    def _num_parameters(self, model):
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_parameters_m = n_parameters / 1e6  # Convert to millions
        return math.floor(n_parameters_m * 100) / 100

    def _flops(self, model, device):

        dummy_input = torch.randn(self.dummy_size, dtype=torch.float).to(device)
        flops = FlopCountAnalysis(model, dummy_input)
        flops_num = flops.total()
        mflops = flops_num / 1e6
        return round(mflops, 2)

    def summary(self, model, device):
        summary_dict = {}
        model = model.to(device)
        summary_dict["latency(ms)"] = self.torch_model_latency(model)
        summary_dict["size(mb)"] = self.torch_model_size(model)
        summary_dict["#parameters(M)"] = self._num_parameters(model)
        summary_dict["MFLOPs"] = self._flops(model, device)
        return summary_dict
    

def get_layer_params(model, layer_name):
    attributes = layer_name.split('.')
    layer = model
    for attr in attributes:
        layer = getattr(layer, attr)
    
    num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    return num_params

def get_layer_size(model, layer_name):
    attributes = layer_name.split('.')
    layer = model
    for attr in attributes:
        layer = getattr(layer, attr)
    try:
        return (layer.weight.size(0), layer.weight.size(1))
    except:
        return (layer.weight.size(0), None)

