# Verified values for the LaTeX report

Use RealWaste throughout a report based on the final notebook: 4,752 RGB images, nine classes, 64 by 64 input resolution, and the same stratified 3,326/713/713 train/validation/test indices for every model. ImageNet mean/std normalization is used. The final configuration uses no random augmentation.

| Model | Parameters | Test accuracy | Weighted F1 | Completed epochs |
| --- | ---: | ---: | ---: | ---: |
| Model A | 549,161 | 70.83% | 0.7080 | 30 |
| Model B | 9,033 | 71.11% | 0.7058 | 30 |
| MobileNetV2 | 2,235,401 | 77.56% | 0.7755 | 25 |
| EfficientNet-B0 | 4,019,077 | 81.63% | 0.8156 | 25 |

The highest observed test accuracy in this final comparison is 81.63%, from EfficientNet-B0. Model B has the smallest parameter count and convolution/linear MAC estimate. Describe these as an accuracy/resource tradeoff under this training setup; the test results do not provide a repeated-run significance comparison.

Both custom models use Adam at 0.001, plain cross-entropy, zero weight decay, no scheduler and 30 epochs. Their sample-order hashes match in all 30 epochs. Model B reuses the previously verified control checkpoint; its original training history and timing are retained. The final evaluation and checkpoint reload are repeated on the same split.

The pretrained models start from torchvision ImageNet weights: MobileNetV2 uses IMAGENET1K_V2 and EfficientNet-B0 uses IMAGENET1K_V1. Exact weight URLs and SHA-256 hashes are in `pretrained_weights.json`. Head training uses Adam at 0.001; fine-tuning uses 0.00001 with weight decay 0.0001. Batch-normalization statistics and parameters stay frozen during fine-tuning. Validation accuracy selects checkpoints, with early stopping patience 7 and a validation-loss learning-rate scheduler.

- MobileNetV2: 5 classifier-head epochs with a frozen backbone and 20 fine-tuning epochs. Validation retained the fine_tune checkpoint. Best head/fine validation accuracy: 71.53% / 77.28%.
- EfficientNet-B0: 5 classifier-head epochs with a frozen backbone and 20 fine-tuning epochs. Validation retained the fine_tune checkpoint. Best head/fine validation accuracy: 68.58% / 80.79%.

Model B uses 60.8 times fewer parameters and 11.1 times fewer convolution/linear MACs than Model A. In these runs the two custom test accuracies differ by only 0.28 percentage points; this single-seed result does not establish a repeatable ranking.

Model A's large dense layer contributes 524,416 parameters. The custom-model resource difference combines depthwise convolutions, earlier downsampling, batch normalization and global pooling; it cannot all be attributed to convolution type alone. ReLU avoids exponentials in the hidden activations. Cross-entropy still performs log-softmax internally during training; inference can classify using argmax on logits.

The report should compare Model B with both pretrained networks using observed accuracy, serialized size and MACs. Total parameters are reported for pretrained models, including frozen parameters. Stored weights and buffers are not peak inference RAM. Estimated MACs exclude normalization, pooling, activations and data movement. Training time is not deployment latency.

Model A's training loss decreases while later validation loss rises, even when validation accuracy changes little. Discuss overfitting and overconfident mistakes; checkpoints are selected by accuracy rather than loss. Do not claim that augmentation improved the final Model B: the combined-settings experiment reached 66.76% test accuracy, and flip-only training did not beat the control on validation.

The optimizer study in `results/optimizer_comparison/` compares six 30-epoch trials using identical starting weights and batches. Adam at 0.001 won validation selection. Momentum 0.9 retains a decaying gradient buffer; compare SGD and SGD+momentum at a fixed learning rate rather than claiming one optimizer is universally best.

All results are single runs on one split. Pretrained models have different training stages and epoch budgets from the custom models. Times include training and validation on cached tensors with four CPU threads; Model B's time comes from an earlier verified run. Machine load and stage freezing affect timing. No quantized deployment, peak-RAM or actual edge-device latency benchmark is claimed.

Use `report_tables.tex` for tables, `metrics_for_report.csv` for exact values and `accuracy_resource_tradeoff.png` for the accuracy/resource comparison figure. Confusion matrices, histories, verification results and source provenance are in this directory. The digits notebook has a different dataset, architecture and split and must not be included in these RealWaste result tables.

`references.bib` contains citation entries verified against the [UCI dataset page](https://archive.ics.uci.edu/dataset/908/realwaste), the [MobileNetV2 paper](https://arxiv.org/abs/1801.04381) and the [EfficientNet proceedings entry](https://proceedings.mlr.press/v97/tan19a.html). Cite the dataset and pretrained architectures in the LaTeX report using keys `realwaste2023`, `sandler2018mobilenetv2` and `tan2019efficientnet`.
