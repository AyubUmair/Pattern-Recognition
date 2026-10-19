import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/final"


def write_resource_plot(record):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = list(record["models"])
    colors = ["#2563eb", "#059669", "#d97706", "#9333ea"]
    for axis, field, scale, label in [
        (axes[0], "checkpoint_file_kib", 1024 / 1e6, "Serialized checkpoint size (MB, log scale)"),
        (axes[1], "estimated_conv_linear_macs", 1e-6, "Convolution and linear MACs (millions, log scale)"),
    ]:
        values = [record["models"][name][field] * scale for name in names]
        for index, name in enumerate(names):
            accuracy = record["models"][name]["test_accuracy"] * 100
            axis.scatter(values[index], accuracy, color=colors[index], s=75, zorder=3)
            offset = (8, 8) if index < 2 else (-8, -18 if index == 2 else 10)
            axis.annotate(name, (values[index], accuracy), xytext=offset, textcoords="offset points", ha="left" if index < 2 else "right", fontsize=9)
        axis.set_xscale("log")
        axis.set_xlim(min(values) / 1.8, max(values) * 1.8)
        accuracies = [model["test_accuracy"] * 100 for model in record["models"].values()]
        axis.set_ylim(min(accuracies) - 4, max(accuracies) + 4)
        axis.set(xlabel=label, ylabel="Test accuracy (%)")
        axis.grid(alpha=0.25, which="both")
    figure.suptitle("RealWaste: measured accuracy and model resource tradeoffs")
    figure.text(0.5, 0.015, "64 x 64 input; one split and seed. Stored size is not peak RAM; MACs are not measured latency.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    figure.savefig(RESULTS / "accuracy_resource_tradeoff.png", dpi=180)
    plt.close(figure)


def prepare_handoff(record):
    names = ["Model A", "Model B", "MobileNetV2", "EfficientNet-B0"]
    assert record["status"] == "complete" and list(record["models"]) == names
    assert record["dataset"] == "RealWaste"
    a, b = record["models"]["Model A"], record["models"]["Model B"]
    best_name = max(names, key=lambda name: record["models"][name]["test_accuracy"])
    best_accuracy = record["models"][best_name]["test_accuracy"]
    assert a["training_details"]["epoch_order_hashes"] == b["training_details"]["epoch_order_hashes"]
    assert b["total_parameters"] <= 100_000
    rows = []
    for name, model in record["models"].items():
        assert sum(map(sum, model["confusion_matrix"])) == 713
        assert model["verification"]["strict_reload"] and model["verification"]["probe_logits_match"]
        rows.append({
            "Model": name, "Total parameters": model["total_parameters"],
            "Tensor storage KiB": model["tensor_storage_kib"],
            "Checkpoint KB": model["checkpoint_file_kib"] * 1.024,
            "Checkpoint MB": model["checkpoint_file_kib"] * 1024 / 1e6,
            "Conv and linear MACs": model["estimated_conv_linear_macs"],
            "Completed epochs": model["epochs_completed"], "Mean epoch seconds": model["mean_epoch_seconds"],
            "Test accuracy percent": 100 * model["test_accuracy"],
            "Weighted precision": model["weighted_precision"], "Weighted recall": model["weighted_recall"],
            "Weighted F1": model["weighted_f1"], "Macro F1": model["macro_f1"],
        })
    with (RESULTS / "metrics_for_report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "\n".join([
        "| Model | Parameters | Test accuracy | Weighted F1 | Completed epochs |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[f"| {name} | {model['total_parameters']:,} | {model['test_accuracy']:.2%} | {model['weighted_f1']:.4f} | {model['epochs_completed']} |" for name, model in record["models"].items()],
    ])
    stage_notes = []
    for name in ["MobileNetV2", "EfficientNet-B0"]:
        model = record["models"][name]
        details = model["training_details"]
        stage_notes.append(f"- {name}: {details['head_epochs']} classifier-head epochs with a frozen backbone and {details['fine_tune_epochs']} fine-tuning epochs. Validation retained the {details['selected_stage']} checkpoint. Best head/fine validation accuracy: {details['head_best_val_accuracy']:.2%} / {details['fine_tune_best_val_accuracy']:.2%}.")
    notes = f"""# Verified values for the LaTeX report

Use RealWaste throughout a report based on the final notebook: 4,752 RGB images, nine classes, 64 by 64 input resolution, and the same stratified 3,326/713/713 train/validation/test indices for every model. ImageNet mean/std normalization is used. The final configuration uses no random augmentation.

{table}

The highest observed test accuracy in this final comparison is {best_accuracy:.2%}, from {best_name}. Model B has the smallest parameter count and convolution/linear MAC estimate. Describe these as an accuracy/resource tradeoff under this training setup; the test results do not provide a repeated-run significance comparison.

Both custom models use Adam at 0.001, plain cross-entropy, zero weight decay, no scheduler and 30 epochs. Their sample-order hashes match in all 30 epochs. Model B reuses the previously verified control checkpoint; its original training history and timing are retained. The final evaluation and checkpoint reload are repeated on the same split.

The pretrained models start from torchvision ImageNet weights: MobileNetV2 uses IMAGENET1K_V2 and EfficientNet-B0 uses IMAGENET1K_V1. Exact weight URLs and SHA-256 hashes are in `pretrained_weights.json`. Head training uses Adam at 0.001; fine-tuning uses 0.00001 with weight decay 0.0001. Batch-normalization statistics and parameters stay frozen during fine-tuning. Validation accuracy selects checkpoints, with early stopping patience 7 and a validation-loss learning-rate scheduler.

{chr(10).join(stage_notes)}

Model B uses {a['total_parameters'] / b['total_parameters']:.1f} times fewer parameters and {a['estimated_conv_linear_macs'] / b['estimated_conv_linear_macs']:.1f} times fewer convolution/linear MACs than Model A. In these runs the two custom test accuracies differ by only {abs(a['test_accuracy'] - b['test_accuracy']) * 100:.2f} percentage points; this single-seed result does not establish a repeatable ranking.

Model A's large dense layer contributes 524,416 parameters. The custom-model resource difference combines depthwise convolutions, earlier downsampling, batch normalization and global pooling; it cannot all be attributed to convolution type alone. ReLU avoids exponentials in the hidden activations. Cross-entropy still performs log-softmax internally during training; inference can classify using argmax on logits.

The report should compare Model B with both pretrained networks using observed accuracy, serialized size and MACs. Total parameters are reported for pretrained models, including frozen parameters. Stored weights and buffers are not peak inference RAM. Estimated MACs exclude normalization, pooling, activations and data movement. Training time is not deployment latency.

Model A's training loss decreases while later validation loss rises, even when validation accuracy changes little. Discuss overfitting and overconfident mistakes; checkpoints are selected by accuracy rather than loss. Do not claim that augmentation improved the final Model B: the combined-settings experiment reached 66.76% test accuracy, and flip-only training did not beat the control on validation.

The optimizer study in `results/optimizer_comparison/` compares six 30-epoch trials using identical starting weights and batches. Adam at 0.001 won validation selection. Momentum 0.9 retains a decaying gradient buffer; compare SGD and SGD+momentum at a fixed learning rate rather than claiming one optimizer is universally best.

All results are single runs on one split. Pretrained models have different training stages and epoch budgets from the custom models. Times include training and validation on cached tensors with four CPU threads; Model B's time comes from an earlier verified run. Machine load and stage freezing affect timing. No quantized deployment, peak-RAM or actual edge-device latency benchmark is claimed.

Use `report_tables.tex` for tables, `metrics_for_report.csv` for exact values and `accuracy_resource_tradeoff.png` for the accuracy/resource comparison figure. Confusion matrices, histories, verification results and source provenance are in this directory. The digits notebook has a different dataset, architecture and split and must not be included in these RealWaste result tables.

`references.bib` contains citation entries verified against the [UCI dataset page](https://archive.ics.uci.edu/dataset/908/realwaste), the [MobileNetV2 paper](https://arxiv.org/abs/1801.04381) and the [EfficientNet proceedings entry](https://proceedings.mlr.press/v97/tan19a.html). Cite the dataset and pretrained architectures in the LaTeX report using keys `realwaste2023`, `sandler2018mobilenetv2` and `tan2019efficientnet`.
"""
    (RESULTS / "notes_for_report.md").write_text(notes, encoding="utf-8", newline="\n")
    readme_path = ROOT / "README.md"
    current = readme_path.read_text(encoding="utf-8")
    history_start = current.index("## Controlled horizontal-flip experiment")
    history = current[history_start:]
    readme = f"""# EN3150 Assignment 03 - final RealWaste version

Use **[EN3150_A03_Final.ipynb](EN3150_A03_Final.ipynb)** for the final RealWaste code and saved outputs. It contains all four models, the shared data pipeline, checkpoint selection, optimizer comparison, metrics and plots. The default `RUN_TRAINING=False` displays recorded results; set it to `True` to reproduce training from scratch.

[Open this branch in Colab](https://colab.research.google.com/github/AyubUmair/Pattern-Recognition/blob/hakam/assignment-improvements/EN3150_A03_Final.ipynb) | [Open main in Colab after merging](https://colab.research.google.com/github/AyubUmair/Pattern-Recognition/blob/main/EN3150_A03_Final.ipynb)

{table}

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

{history}
"""
    readme_path.write_text(readme.rstrip() + "\n", encoding="utf-8", newline="\n")
    guide_path = ROOT / "TRAINING_GUIDE.md"
    experiment_guide = ROOT / "EXPERIMENT_GUIDE.md"
    if not experiment_guide.exists():
        experiment_guide.write_text(guide_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    guide = """# Final RealWaste training workflow

Open `EN3150_A03_Final.ipynb`. Its saved outputs and `RUN_TRAINING=False` setting allow review without retraining. Set `RUN_TRAINING=True`, leave `IMAGE_SIZE=64` and `SPLIT_SEED=42`, choose a fresh `OUTPUT_DIR`, and run the cells in order to reproduce all four models.

The notebook is self-contained. `DATA_DIR` points to the folder containing the nine class directories; an empty value searches local/Colab locations and can download the official UCI archive. Google Drive mounting is optional. Images are resized, normalized and cached in RAM. Validation and test processing is deterministic. First-time pretrained-model use downloads ImageNet weights.

Both custom models use 30 epochs, Adam at 0.001 and ordinary cross-entropy, without augmentation, weight decay or a scheduler. The two pretrained models use 5 classifier-head epochs followed by up to 20 fine-tuning epochs. Fine-tuning uses a smaller learning rate, frozen batch-normalization statistics/parameters, early stopping and best-checkpoint restoration.

The recorded Model B checkpoint is reused by the CLI when available; `--fresh-model-b` always retrains it. A fresh notebook run always trains it. All recorded values, including the earlier Model B timing, are labelled in `results/final/notes_for_report.md`.

```powershell
python -m pip install -r requirements.txt
python run_final_training.py --fresh-model-b --output results/final_reproduction
```

Use `--device cuda` with a CUDA-capable PyTorch installation. The CLI defaults to CPU. The notebook selects CUDA automatically when available. Full training on a GPU is recommended for speed; numerical results can vary across hardware and library versions.

```powershell
python run_final_training.py --smoke-test
```

The synthetic check covers all four models, both transfer stages, strict checkpoint reloads and metric export. It writes only to `.local/` and does not replace assignment results.

`RUN_OPTIMIZER_COMPARISON=True` optionally reproduces the six optimizer trials from the final notebook. Existing optimizer results are included as recorded outputs by default. Earlier augmentation and regularization experiments are documented in `EXPERIMENT_GUIDE.md` and kept separate from the final comparison.

Use `results/final/report_tables.tex` and the result PNGs in the LaTeX report. The code ZIP is in `submission/`; rename the report PDF and code upload using your group's required `GROUP_A03_EN3150` filename format. The final notebook has explanations in Markdown and no added inline Python comments.
"""
    guide_path.write_text(guide, encoding="utf-8", newline="\n")
    handoff = """# Final handoff

1. Agree on the report dataset as a group. This final notebook and the included LaTeX tables use RealWaste. The friend's `EN3150_A03_new.ipynb` uses digits and remains preserved in the repository. Choose one dataset for the submitted report and code; do not combine their metrics.
2. For RealWaste, use `EN3150_A03_Final.ipynb`. Its recorded outputs contain the final measured results for Model A, Model B, MobileNetV2 and EfficientNet-B0. The LaTeX table input is `results/final/report_tables.tex`; read `results/final/notes_for_report.md` for settings and caveats.
3. Merge the pull request from `hakam/assignment-improvements` into `main` after the group reviews the final notebook. The friend's latest main-branch notebook has already been incorporated into this branch; no manual copying between notebooks is required. Merging publishes this final version on the shared main branch. Do not replace the friend's digits metrics with RealWaste metrics inside that separate notebook.
4. Upload the report PDF and code separately to Moodle. A code bundle is provided at `submission/EN3150_A03_RealWaste_Code.zip`. Rename uploads to the required `GROUP_A03_EN3150` format and put the group number, names and student/index numbers in the LaTeX report. Only one group submission is required.
5. Include your GitHub profile [hakammra](https://github.com/hakammra) and the [shared repository](https://github.com/AyubUmair/Pattern-Recognition) in the report, add any other required member profiles, and confirm the group's selected dataset has been registered in the assignment's dataset sheet.

The training data, environment, pretrained weights and trained checkpoint binaries are not in the code ZIP. The notebook and scripts recreate them. Plots, metric records and source provenance are included. Opening the saved notebook does not require rerunning training.
"""
    (ROOT / "FINAL_HANDOFF.md").write_text(handoff, encoding="utf-8", newline="\n")
    print(table)


def build_bundle(record):
    folder = ROOT / "submission"
    folder.mkdir(exist_ok=True)
    files = [ROOT / name for name in ["EN3150_A03_Final.ipynb", "EN3150_A03.ipynb", "README.md", "TRAINING_GUIDE.md", "EXPERIMENT_GUIDE.md", "FINAL_HANDOFF.md", "requirements.txt", "requirements-notebook.txt", "run_final_training.py", "run_optimizer_comparison.py", "run_training.py", "run_augmentation_ablation.py", "build_final_notebook.py", "finalize_submission.py"]]
    for relative in ["results"]:
        files.extend(path for path in (ROOT / relative).rglob("*") if path.is_file() and "checkpoints" not in path.relative_to(ROOT).parts)
    assert all(path.exists() and path.resolve().is_relative_to(ROOT) for path in files)
    manifest = {
        "dataset": "RealWaste", "primary_notebook": "EN3150_A03_Final.ipynb",
        "training_source_commit": record["provenance"]["source_commit"],
        "files": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (folder / "manifest.json").write_text(manifest_text, encoding="utf-8", newline="\n")
    bundle = folder / "EN3150_A03_RealWaste_Code.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("submission_manifest.json", manifest_text)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert all(not any(part in [".git", ".local", ".venv", "checkpoints", "data"] for part in Path(name).parts) for name in archive.namelist())
    print(f"Created {bundle}; {len(files)} files; {bundle.stat().st_size:,} bytes.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()
    record = json.loads((RESULTS / "experiment.json").read_text(encoding="utf-8"))
    if not args.bundle_only:
        prepare_handoff(record)
        write_resource_plot(record)
    build_bundle(record)


if __name__ == "__main__":
    main()
