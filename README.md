# Basic_Compressor
Magnitude Pruning + Post-Training Quantization with ONNX For CNN

More detail information about pruning can be found here: [Basic Pruning](https://github.com/bok3948/Basic_Pruning)

More detail information about quantization can be found here: [ONNX Quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

Original models, which are not compressed, can be downloaded from the following links:
- ResNet-18: [Download](https://drive.google.com/file/d/1iR6WdiGQ1ceWspa_jppUvklgK39k13NH/view?usp=sharing)
- ResNet-34: [Download](https://drive.google.com/file/d/1_eipZl72oBA0vBYIVwNoX1IZj5HHWk_U/view?usp=sharing)
- ResNet-50: [Download](https://drive.google.com/file/d/12UjAI5H0haUCt-JBoQO77ADMfTbdIfGh/view?usp=sharing)

# Model Compression Summary

## ResNet-34 Result

| Model Name                  | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Latency (ms) | Model Size (MB) |
|-----------------------------|--------------------|--------------------|--------------|-----------------|
| ONNX ResNet34 Compressed    | 85.88              | 98.87              | 2.4482       | 5.83            |
| Torch ResNet34 Pruned       | 86.52              | 98.95              | 7.047        | 22.73           |
| Torch ResNet34 Original     | 86.66              | 99.25              | 11.4254      | 85.29           |

## ResNet-18 Result

| Model Name                  | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Latency (ms) | Model Size (MB) |
|-----------------------------|--------------------|--------------------|--------------|-----------------|
| ONNX ResNet18 Compressed    | 85.59              | 99.2               | 1.1773       | 3.12            |
| Torch ResNet18 Pruned       | 85.98              | 99.24              | 3.8252       | 12.07           |
| Torch ResNet18 Original     | 86.42              | 99.16              | 5.5561       | 44.8            |

## ResNet-50 Result

| Model Name                  | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Latency (ms) | Model Size (MB) |
|-----------------------------|--------------------|--------------------|--------------|-----------------|
| ONNX ResNet50 Compressed    | 85.16              | 99.22              | 2.8301       | 6.72            |
| Torch ResNet50 Pruned       | 85.68              | 99.22              | 7.5003       | 25.48           |
| Torch ResNet50 Original     | 87.27              | 99.26              | 13.4592      | 94.41           |

## Requirements
- **PyTorch**: 2.2.1
- **timm**: 0.9.12
- **ONNX**: 1.16.0
- **ONNX Runtime**: 1.17.1

## Execution Instructions

### Step 1: Pruning
First, run pruning:

<pre>
python ./pruner/main.py --dataset CIFAR10 --data_path "path_to_data" --pretrained "path_to_pretrained_model" --device cuda --model resnet18 --pruning_ratio 0.7 --per_iter_pruning_ratio 0.05 --min_ratio 0.01
</pre>

### Step 2: ONNX Post-Training Quantization
After pruning, quantize the model using ONNX PTQ. Use the path to the pruned model for --pruned_pretrained and for benchmarking, use the original model path for --ori_pretrained:

<pre>
python ONNX_PTQ.py --model resnet18 --pruned_pretrained ./res18_pruned_checkpoint.pth --ori_pretrained ./pruning/res18_best_checkpoint.pth  
</pre>
