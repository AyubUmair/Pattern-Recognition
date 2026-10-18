# EN3150 Assignment 03

This project classifies RealWaste images using a standard CNN, a custom lightweight CNN, and pretrained lightweight networks. Model B has 9,033 trainable parameters.

Start with the [training guide](TRAINING_GUIDE.md) to run your own RealWaste experiment. The notebook now has editable settings, training-only augmentation, regularization, a learning-rate scheduler, and two-stage MobileNetV2/EfficientNet-B0 fine-tuning. All four models retain the same RealWaste split and 64 by 64 inputs. New-run results are saved separately from the historical baseline below.

[Open the notebook in Colab](https://colab.research.google.com/github/AyubUmair/Pattern-Recognition/blob/hakam/assignment-improvements/EN3150_A03.ipynb)

## Optimizer comparison

The notebook includes a self-contained experiment comparing Adam, SGD, and SGD with momentum on Model B. Each optimizer has two learning-rate candidates and receives the same training budget. Every trial starts from identical weights and uses identical batch orders. Initial-state and per-epoch sample-order hashes verify these controls.

| Optimizer | Learning rates | Momentum |
| --- | --- | --- |
| Adam | 0.001, 0.0003 | Not applicable |
| SGD | 0.01, 0.1 | 0 |
| SGD with momentum | 0.01, 0.1 | 0.9 |

All trials use 30 epochs, batch size 32, and the original stratified 70/15/15 split with split seed 42. The model initialization and sampling seed is also 42. Images are resized to 64 by 64 pixels and use the notebook's ImageNet normalization. No augmentation, scheduler, or weight decay is added in this experiment.

The best validation accuracy determines each trial's checkpoint. The final optimizer and learning rate are chosen using validation accuracy, with validation loss breaking ties. Only the selected checkpoint is evaluated on the test set. The comparison CSV reports validation performance for all six trials; the JSON also contains test metrics for the selected checkpoint.

## Run locally

Download and extract [RealWaste from UCI](https://archive.ics.uci.edu/dataset/908/realwaste). The directory passed to the runner must contain the nine class folders. The full dataset contains 4,752 images.

Use Python 3.12 and install the dependencies:

```powershell
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the experiment from this repository:

```powershell
python run_optimizer_comparison.py --data-dir "../data/realwaste-main/RealWaste" --epochs 30 --update-notebook
```

The local runner loads the notebook's definitions without mounting Google Drive or running the original training block. It caches the same normalized image tensors in `.local/` for CPU training. The cache is checked against the transform description and source-image paths, sizes, and modification times. Epoch timings include training and validation with this cache, so they should be compared within this experiment.

Use `--device cuda` when a CUDA-capable PyTorch installation and GPU are available. Reproducibility controls do not guarantee identical floating-point results across hardware and library versions.

Run a short synthetic workflow check before a full experiment:

```powershell
python run_optimizer_comparison.py --smoke-test
```

The synthetic check verifies training, identical starting states, matching sample orders, checkpoint selection, and metric export. Its results are kept in `.local/smoke_results/` and do not replace assignment results.

## Outputs

The experiment writes the comparison CSV, complete JSON histories, classification report, validation-accuracy plot, loss curves, and selected model's test confusion matrix to `results/optimizer_comparison/`. With `--update-notebook`, the same experiment summary and figures appear in the new notebook section. Split indices and software versions are recorded for reproduction. Downloaded data, cached tensors, model checkpoints, and virtual environments are excluded from Git.

The optimizer files below contain recorded baseline results from before the new training defaults. Their exact source snapshot is preserved in `results/optimizer_comparison/source_snapshot.json`. The optional six-trial notebook comparison is disabled by default; enable `RUN_OPTIMIZER_COMPARISON` to rerun it under `OUTPUT_DIR/optimizer_comparison`. Its training subset uses the deterministic transform so augmentation does not change that protocol. New training runs use their own `OUTPUT_DIR`. The optimizer CLI explicitly refreshes the recorded baseline directory and its source snapshot when run on real data.

The [architecture summary](results/architecture/architecture_summary.md) details both custom models, their layer shapes, parameter calculations, estimated convolution/linear MACs, and the activation-function choices. The [optimizer discussion](results/optimizer_comparison/discussion.md) interprets the recorded experiment for use in the assignment report.

The [custom-model evaluation](results/custom_models/evaluation.md) presents the earlier 20-epoch runs with both loss curves, labelled confusion matrices, weighted metrics, parameter counts, measured checkpoint sizes, and epoch times. Its results are kept separate from the new 30-epoch optimizer experiment.

This experiment uses one seed and a small learning-rate grid. The results describe these configurations on this split; they do not establish that one optimizer is universally best.
