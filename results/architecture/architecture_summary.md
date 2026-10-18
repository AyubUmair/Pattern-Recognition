# Custom CNN architecture and resource calculations

Both networks take RGB images with shape 3 x 64 x 64 and produce nine logits. Output shapes below omit the batch dimension. Convolution and linear MACs are calculated for one image using the actual layer outputs. One multiply-accumulate is counted as one MAC.

| Model | Trainable parameters | State tensor storage (float32 weights) | Convolution/linear MACs per image |
| --- | ---: | ---: | ---: |
| Model A | 549,161 | 2,145.16 KiB | 11,732,096 |
| Model B | 9,033 | 36.69 KiB | 1,059,392 |

## Model A

| Layer | Operation | Output shape | Configuration | Trainable parameters | MACs |
| --- | --- | --- | --- | ---: | ---: |
| features.0 | Conv2d | 16x64x64 | 3 to 16 channels; kernel 3x3; stride 1; groups 1; bias True | 448 | 1,769,472 |
| features.1 | ReLU | 16x64x64 |  | 0 | 0 |
| features.2 | MaxPool2d | 16x32x32 | kernel 2; stride 2 | 0 | 0 |
| features.3 | Conv2d | 32x32x32 | 16 to 32 channels; kernel 3x3; stride 1; groups 1; bias True | 4,640 | 4,718,592 |
| features.4 | ReLU | 32x32x32 |  | 0 | 0 |
| features.5 | MaxPool2d | 32x16x16 | kernel 2; stride 2 | 0 | 0 |
| features.6 | Conv2d | 64x16x16 | 32 to 64 channels; kernel 3x3; stride 1; groups 1; bias True | 18,496 | 4,718,592 |
| features.7 | ReLU | 64x16x16 |  | 0 | 0 |
| features.8 | MaxPool2d | 64x8x8 | kernel 2; stride 2 | 0 | 0 |
| classifier.0 | Flatten | 4096 |  | 0 | 0 |
| classifier.1 | Linear | 128 | 4096 to 128 units; bias True | 524,416 | 524,288 |
| classifier.2 | ReLU | 128 |  | 0 | 0 |
| classifier.3 | Linear | 9 | 128 to 9 units; bias True | 1,161 | 1,152 |

## Model B

| Layer | Operation | Output shape | Configuration | Trainable parameters | MACs |
| --- | --- | --- | --- | ---: | ---: |
| features.0 | Conv2d | 16x32x32 | 3 to 16 channels; kernel 3x3; stride 2; groups 1; bias False | 432 | 442,368 |
| features.1 | BatchNorm2d | 16x32x32 | 16 channels; learned scale and offset | 32 | 0 |
| features.2 | ReLU | 16x32x32 |  | 0 | 0 |
| features.3.depthwise | Conv2d | 16x16x16 | 16 to 16 channels; kernel 3x3; stride 2; groups 16; bias False | 144 | 36,864 |
| features.3.pointwise | Conv2d | 32x16x16 | 16 to 32 channels; kernel 1x1; stride 1; groups 1; bias False | 512 | 131,072 |
| features.3.bn | BatchNorm2d | 32x16x16 | 32 channels; learned scale and offset | 64 | 0 |
| features.3.relu | ReLU | 32x16x16 |  | 0 | 0 |
| features.4.depthwise | Conv2d | 32x8x8 | 32 to 32 channels; kernel 3x3; stride 2; groups 32; bias False | 288 | 18,432 |
| features.4.pointwise | Conv2d | 64x8x8 | 32 to 64 channels; kernel 1x1; stride 1; groups 1; bias False | 2,048 | 131,072 |
| features.4.bn | BatchNorm2d | 64x8x8 | 64 channels; learned scale and offset | 128 | 0 |
| features.4.relu | ReLU | 64x8x8 |  | 0 | 0 |
| features.5.depthwise | Conv2d | 64x8x8 | 64 to 64 channels; kernel 3x3; stride 1; groups 64; bias False | 576 | 36,864 |
| features.5.pointwise | Conv2d | 64x8x8 | 64 to 64 channels; kernel 1x1; stride 1; groups 1; bias False | 4,096 | 262,144 |
| features.5.bn | BatchNorm2d | 64x8x8 | 64 channels; learned scale and offset | 128 | 0 |
| features.5.relu | ReLU | 64x8x8 |  | 0 | 0 |
| features.6 | AdaptiveAvgPool2d | 64x1x1 | global average pooling to 1x1 | 0 | 0 |
| classifier.0 | Flatten | 64 |  | 0 | 0 |
| classifier.1 | Linear | 9 | 64 to 9 units; bias True | 585 | 576 |

## Parameter formulas

A standard convolution has Cout x (Cin x Kh x Kw + 1) trainable parameters when bias is enabled; remove the +1 when bias is disabled. A depthwise convolution has Cin x Kh x Kw weights when its channel multiplier is one, and a bias-free pointwise convolution has Cin x Cout weights. Batch normalization contributes two trainable values per channel: scale and offset. Its running statistics and batch counter are buffers, so they contribute to storage but not to trainable-parameter count. A fully connected layer has (input units + 1) x output units parameters when bias is enabled.

Model A: 448 + 4,640 + 18,496 + 524,416 + 1,161 = **549,161** trainable parameters.

Model B: 432 + 32 + (144 + 512 + 64) + (288 + 2,048 + 128) + (576 + 4,096 + 128) + 585 = **9,033** trainable parameters, below the 100,000 limit.

## Hardware-aware activation choice

Hidden layers use ReLU, max(0, x), which requires a comparison rather than an exponential. In-place ReLU can reuse its activation storage. The classifier returns raw logits to CrossEntropyLoss, which performs its own numerically stable log-softmax/loss calculation during training. For classification, argmax can operate directly on logits, so an explicit softmax is unnecessary when only the predicted class is needed. Avoiding an explicit softmax does not remove exponentials from the training loss itself.

## Interpretation and calculation limits

The measured architectures reduce trainable parameters by 60.8 times and convolution/linear MACs by 11.1 times when moving from Model A to Model B. These changes combine depthwise separable convolutions, earlier downsampling, batch normalization, and global average pooling. Model A's first dense layer alone has 524,416 parameters, about 95.5% of its total. Therefore, the entire saving cannot be attributed solely to replacing standard convolution with depthwise separable convolution.

MAC totals include convolution and linear operations. They exclude bias addition, batch normalization, pooling, activation functions, data movement, and framework overhead. Parameter/buffer storage is an estimate of tensor storage, not peak inference RAM or an actual serialized file size. A smaller MAC count does not by itself guarantee a proportional latency improvement; implementation and hardware support affect performance.
