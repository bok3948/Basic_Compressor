# Basic_Compressor
Magnitude Pruning + Post-Training Quantization with ONNX For CNN
More detail information pruning can get from (https://github.com/bok3948/Basic_Pruning) 
More detail infromation about quantization can get from (https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

# Model Compression Summary

## ResNet-34 Comparison

| Model Name                  | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Latency (ms) | Model Size (MB) |
|-----------------------------|--------------------|--------------------|--------------|-----------------|
| ONNX ResNet34 Compressed    | 85.88              | 98.87              | 2.4482       | 5.83            |
| Torch ResNet34 Pruned       | 86.52              | 98.95              | 7.047        | 22.73           |
| Torch ResNet34 Original     | 86.66              | 99.25              | 11.4254      | 85.29           |

## ResNet-18 Comparison

| Model Name                  | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Latency (ms) | Model Size (MB) |
|-----------------------------|--------------------|--------------------|--------------|-----------------|
| ONNX ResNet18 Compressed    | 85.59              | 99.2               | 1.1773       | 3.12            |
| Torch ResNet18 Pruned       | 85.98              | 99.24              | 3.8252       | 12.07           |
| Torch ResNet18 Original     | 86.42              | 99.16              | 5.5561       | 44.8            |

## ResNet-50 Comparison

| Model Name                  | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Latency (ms) | Model Size (MB) |
|-----------------------------|--------------------|--------------------|--------------|-----------------|
| ONNX ResNet50 Compressed    | 85.16              | 99.22              | 2.8301       | 6.72            |
| Torch ResNet50 Pruned       | 85.68              | 99.22              | 7.5003       | 25.48           |
| Torch ResNet50 Original     | 87.27              | 99.26              | 13.4592      | 94.41           |


