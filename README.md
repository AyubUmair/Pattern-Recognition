# EN3150 Assignment 03 - final RealWaste version

Use **[EN3150_A03_Final.ipynb](EN3150_A03_Final.ipynb)** for the final RealWaste code and saved outputs. It contains all four models, the shared data pipeline, checkpoint selection, optimizer comparison, metrics and plots. The default `RUN_TRAINING=False` displays recorded results; set it to `True` to reproduce training from scratch.

[Open this branch in Colab](https://colab.research.google.com/github/AyubUmair/Pattern-Recognition/blob/hakam/assignment-improvements/EN3150_A03_Final.ipynb) | [Open main in Colab after merging](https://colab.research.google.com/github/AyubUmair/Pattern-Recognition/blob/main/EN3150_A03_Final.ipynb)

| Model | Parameters | Test accuracy | Weighted F1 | Completed epochs |
| --- | ---: | ---: | ---: | ---: |
| Model A | 549,161 | 70.83% | 0.7080 | 30 |
| Model B | 9,033 | 71.11% | 0.7058 | 30 |
| MobileNetV2 | 2,235,401 | 77.56% | 0.7755 | 25 |
| EfficientNet-B0 | 4,019,077 | 81.63% | 0.8156 | 25 |

All models use RealWaste, RGB inputs at 64 by 64, and exactly the same 70/15/15 split. Model A and Model B each complete 30 epochs. The pretrained models use classifier-head training followed by backbone fine-tuning, retaining the best validation checkpoint across both stages. Every final checkpoint passes a strict reload and deterministic-logit check.

For the LaTeX report, use [report_tables.tex](results/final/report_tables.tex), [exact metrics](results/final/metrics_for_report.csv), and [interpretation notes](results/final/notes_for_report.md). The PNG files in `results/final/` are ready to include in the report. Complete histories and provenance are in [experiment.json](results/final/experiment.json).

The [code submission ZIP](submission/EN3150_A03_RealWaste_Code.zip) contains the final notebook, source files, dependencies and result assets. Data, pretrained weights, trained checkpoints, virtual environments and Git internals are excluded. See [FINAL_HANDOFF.md](FINAL_HANDOFF.md) for merge and submission steps.

## Dataset choice

`EN3150_A03_new.ipynb` is the friend's digits experiment already on `main`; it is preserved. It uses a different dataset from this final RealWaste version. Your group should choose one dataset for the report and code submission. Use this notebook's RealWaste tables only with a RealWaste report.

## Reproduce

Use Python 3.12 and install `requirements.txt`; for local notebook tools install `requirements-notebook.txt`. In Colab, select a GPU runtime for faster full training.

```powershell
python run_final_training.py --fresh-model-b --output results/final_reproduction
```

The final notebook is self-contained for training. The CLI loads shared definitions from `EN3150_A03.ipynb`. Both locate an existing RealWaste copy, and the notebook can download the official UCI archive when data is missing. See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for the final workflow and [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for earlier tuning experiments.

## Experiment history

## Controlled horizontal-flip experiment

`run_augmentation_ablation.py` compares the baseline with training-only horizontal flips at probability 0.5. Both runs use Adam at 0.001, plain cross-entropy, 30 epochs, identical starting weights, identical batch orders, and the same dataset split. This isolates one augmentation change after the combined augmentation/regularization experiment below. Only the configuration selected using validation is evaluated on the test set. See the [training guide](TRAINING_GUIDE.md) for commands and controls.

The paired 30-epoch test reached **69.28% validation accuracy without augmentation** and **68.86% with horizontal flips**. Horizontal flips reduced best validation accuracy by 0.42 percentage points in this paired run. Validation selected **No augmentation**; that checkpoint achieved **71.11% test accuracy**. Only the selected configuration received test evaluation. Keep the no-augmentation configuration as the current Model B choice. Horizontal flips alone did not win this validation comparison.

See the [full controlled-experiment report and plots](results/augmentation_ablation/discussion.md).

## Recorded 30-epoch Model B run

The Model B-only 30-epoch run with augmentation, weight decay, label smoothing and a learning-rate scheduler achieved **66.76% test accuracy**, versus **71.11%** for the earlier recorded baseline (-4.35 percentage points). Its best validation accuracy was **68.16%** at epoch 28. Both runs use the same RealWaste split and 9,033-parameter architecture. This is a one-seed comparison with multiple training changes. The earlier baseline had higher validation accuracy than that combined-settings run. The augmented configuration is retained as an experiment, rather than a demonstrated accuracy improvement.

See the [complete run report and plots](results/model_b_improved_30epochs_seed42/discussion.md). The saved checkpoint independently reproduces the recorded test metrics; the check results are in that directory's `validation.json`.

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
