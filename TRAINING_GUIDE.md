# Run your own RealWaste experiment

Open `EN3150_A03.ipynb`, edit its settings cell, and run the cells in order. The helpers are contained in the notebook, so it can run in Colab without importing the repository's Python files. The same workflow is available through `run_training.py`.

All four architectures use the same 70/15/15 RealWaste split and 64 by 64 RGB inputs. The new defaults train both custom models and fine-tune MobileNetV2 and EfficientNet-B0. For a shorter first run, set `custom_models` to `["Model B"]` and `sota_models` to `[]`.

| Setting | Default | Purpose |
| --- | --- | --- |
| `custom_epochs` | 30 | Custom training budget; keep at least 20 for submission |
| `custom_lr` | 0.001 | Adam learning rate for custom models |
| `head_epochs` | 5 | Train only the pretrained classifier |
| `fine_tune_epochs` | 20 | Maximum backbone fine-tuning epochs |
| `head_lr` / `fine_tune_lr` | 0.001 / 0.00001 | Learning rates for the two transfer stages |
| `weight_decay` | 0.0001 | Adam L2 regularization |
| `label_smoothing` | 0.05 | Cross-entropy target smoothing |
| `AUGMENT_TRAINING` | `True` | Mild flips, rotations and colour changes on training images |
| `OUTPUT_DIR` | `results/my_run` | Use a different directory for each experiment |

Validation and test transforms are deterministic. The scheduler halves the learning rate when validation loss stalls. Custom models complete their configured epoch budget; transfer-learning stages can stop early based on validation accuracy. Each stage restores its best validation-accuracy checkpoint, and transfer learning retains the better checkpoint across both stages. The feature extractor is frozen during head training. During fine-tuning, backbone weights can update while batch-normalization parameters and running statistics stay frozen.

Keep `IMAGE_SIZE=64` for the current Model A classifier and `SPLIT_SEED=42` when comparing with the recorded baseline. Choose settings using validation results before reporting final test metrics. Keep the default `num_workers=0` for local Windows notebook runs.

## Dataset setup

Set `DATA_DIR` or the `REALWASTE_DIR` environment variable to the folder containing the nine class folders. Otherwise the notebook searches common local and Colab locations, including `../data/realwaste-main/RealWaste` and `/content/dataset/realwaste-main/RealWaste`.

If data is missing and `DOWNLOAD_DATASET=True`, it downloads the official [UCI RealWaste archive](https://archive.ics.uci.edu/dataset/908/realwaste), about 657 MB, and extracts it. Set `DOWNLOAD_DATASET=False` to require an existing copy. Google Drive mounting is optional: enable `MOUNT_GOOGLE_DRIVE` when using Drive. First-time pretrained-model use also downloads ImageNet weights.

## Terminal commands

Install the dependencies described in the README, then run Model B:

```powershell
python run_training.py --models b --epochs 30 --output results/model_b_trial_01
```

Try another learning rate and regularization setting:

```powershell
python run_training.py --models b --lr 0.0005 --weight-decay 0.0001 --label-smoothing 0.05 --output results/model_b_trial_02
```

Run both pretrained networks:

```powershell
python run_training.py --models mobilenet efficientnet --head-epochs 5 --fine-tune-epochs 20 --output results/transfer_trial_01
```

The default device selection uses CUDA when available. Use `--device cpu` to force CPU training or `--device cuda` with a CUDA-capable PyTorch installation. Full pretrained-model training is suitable for a GPU. A short CPU workflow check is available without pretrained-weight downloads:

```powershell
python run_training.py --smoke-test --device cpu
```

Synthetic results go under `.local/` and are not assignment accuracy measurements. The check covers all four architectures, frozen-backbone head training, backbone updates during fine-tuning, metrics, checkpoint export and plots.

## Outputs and baseline

Each run writes `experiment.json`, `model_comparison.csv`, per-model metrics and classification reports, labelled confusion matrices, training/validation plots, and trained state dictionaries. The notebook displays the comparison table and saved plots. Checkpoints are excluded from Git. The comparison distinguishes total parameters from final-stage trainable parameters, tensor storage from measured checkpoint size, and estimated convolution/linear MACs from timing. MAC estimates exclude pooling, normalization, activations and framework overhead.

The recorded 71.11% Model B test accuracy belongs to the earlier controlled experiment without augmentation, weight decay, label smoothing or a scheduler. Its histories and source snapshot are retained under `results/optimizer_comparison/`. Model B has now completed a full 30-epoch run with the new defaults: 66.76% test accuracy and 68.16% best validation accuracy. See the [recorded run report](results/model_b_improved_30epochs_seed42/discussion.md). Model A and the two pretrained models have passed workflow checks for the new pipeline; their full training with these new defaults is still pending. The earlier configuration remains the stronger Model B choice on validation. The augmented configuration is retained as an experiment, rather than a demonstrated accuracy improvement. Changes to several training settings prevent attributing any accuracy difference to one setting alone.

## Change one setting at a time

The horizontal-flip experiment trains a fresh control and a fresh augmented model with matching initialization and per-epoch sample orders. The only change is a 50% chance of a horizontal flip on each training image. Both use the original Model B, Adam at 0.001, batch size 32, unsmoothed cross-entropy, zero weight decay, no scheduler and no early stopping.

```powershell
python run_augmentation_ablation.py --epochs 30
```

This runner uses its explicit baseline controls rather than the notebook's exploratory regularization settings. Both trials use the same verified normalized tensors cached in RAM. Flipping after normalization gives the same pixels as flipping the resized RGB image before normalization; an example from every class checks that equivalence. Validation and test images receive no random transforms.

Each trial restores its best validation-accuracy checkpoint. The configuration with higher validation accuracy wins; lower validation loss at that checkpoint breaks ties. Only the winner receives test evaluation. A strict checkpoint reload must reproduce its confusion matrix and metrics. Initialization and sample-order hashes, all histories, source provenance, comparison CSV, plots and reloaded-checkpoint predictions are saved under `results/augmentation_ablation/`. Checkpoints stay local and are excluded from Git.

Use `--output results/flip_trial_02` for another run; the runner protects existing completed results. Run the synthetic workflow check without changing assignment results:

```powershell
python run_augmentation_ablation.py --smoke-test
```
