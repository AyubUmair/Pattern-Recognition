import argparse
import ast
import io
import json
import pprint
import tokenize
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent


def clean_code(source):
    tokens = [token for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type != tokenize.COMMENT]
    return "\n".join(line.rstrip() for line in tokenize.untokenize(tokens).strip().splitlines()) + "\n"


def definitions(source, names):
    tree = ast.parse(source)
    return clean_code("\n\n".join(ast.get_source_segment(source, node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names))


def latex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_latex_tables(record, optimizer_record, output_dir):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\hline", r"Model & Parameters & File (KB) & Mean s/epoch & Accuracy (\%) \\", r"\hline"]
    for name in ["Model A", "Model B"]:
        metrics = record["models"][name]
        lines.append(f"{name} & {metrics['total_parameters']:,} & {metrics['checkpoint_file_kib'] * 1.024:.2f} & {metrics['mean_epoch_seconds']:.2f} & {metrics['test_accuracy'] * 100:.2f}" + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\caption{Custom CNN comparison after 30 training epochs. KB is $10^3$ bytes. Times include training and validation on cached tensors; Model B uses its earlier verified training record.}", r"\label{tab:realwaste-custom}", r"\end{table}", ""])
    lines.extend([
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrrr}", r"\hline",
        r"Model & Parameters & File (MB) & MACs (M) & Epochs & Accuracy (\%) \\", r"\hline",
    ])
    for name, metrics in record["models"].items():
        lines.append(f"{latex_escape(name)} & {metrics['total_parameters']:,} & {metrics['checkpoint_file_kib'] * 1024 / 1e6:.3f} & {metrics['estimated_conv_linear_macs'] / 1e6:.3f} & {metrics['epochs_completed']} & {metrics['test_accuracy'] * 100:.2f}" + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\caption{RealWaste test results at $64\times64$ resolution. MB is $10^6$ bytes; MACs include convolution and linear layers. Training budgets differ between custom and pretrained networks.}", r"\label{tab:realwaste-final}", r"\end{table}", ""])
    lines.extend([r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\hline", r"Model & Accuracy (\%) & Precision (w.) & Recall (w.) & F1 (w.) \\", r"\hline"])
    for name, metrics in record["models"].items():
        lines.append(f"{latex_escape(name)} & {metrics['test_accuracy'] * 100:.2f} & {metrics['weighted_precision']:.4f} & {metrics['weighted_recall']:.4f} & {metrics['weighted_f1']:.4f}" + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\caption{RealWaste test metrics. Precision, recall and F1 use class-support weighting.}", r"\label{tab:realwaste-metrics}", r"\end{table}", ""])
    lines.extend([r"\begin{table}[htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\hline", r"Optimizer & Learning rate & Momentum & Best validation (\%) & Best epoch \\", r"\hline"])
    for trial in optimizer_record["trials"]:
        momentum = "--" if trial["momentum"] is None else str(trial["momentum"])
        lines.append(f"{latex_escape(trial['name'])} & {trial['lr']:g} & {momentum} & {trial['best_val_accuracy'] * 100:.2f} & {trial['best_epoch']}" + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\caption{Model B optimizer selection using validation results. All trials use 30 epochs with matching initialization and sample orders.}", r"\label{tab:realwaste-optimizers}", r"\end{table}", ""])
    (output_dir / "report_tables.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_notebook(record, result_dir, output_path, allow_smoke=False):
    assert record["status"] == "complete"
    expected = ["Model A", "Model B", "MobileNetV2", "EfficientNet-B0"]
    assert list(record["models"]) == expected
    assert record["models"]["Model B"]["total_parameters"] == 9033
    if not allow_smoke:
        assert record["dataset"] == "RealWaste"
        assert record["split_sizes"] == {"train": 3326, "val": 713, "test": 713}
        assert all(record["models"][name]["epochs_completed"] >= 20 for name in ["Model A", "Model B"])
    original = json.loads((ROOT / "EN3150_A03.ipynb").read_text(encoding="utf-8"))
    cell_sources = {cell.get("metadata", {}).get("id"): "".join(cell.get("source", [])) for cell in original["cells"]}
    support = cell_sources["training-support"]
    optimizer_source = cell_sources["optimizer-comparison-functions"]
    optimizer_record = json.loads((ROOT / "results/optimizer_comparison/optimizer_results.json").read_text())
    cells = []
    md = lambda source: cells.append(nbformat.v4.new_markdown_cell(source))
    code = lambda source: cells.append(nbformat.v4.new_code_cell(clean_code(source)))
    title = "EN3150 A03 - Final RealWaste notebook" if not allow_smoke else "Synthetic workflow preview - not assignment results"
    md(f"# {title}\n\nThe saved outputs contain the recorded results. Run the cells with `RUN_TRAINING=False` to view them. Set it to `True` to train all four models from scratch using the same RealWaste split. Use a fresh `OUTPUT_DIR` for each run. The two custom networks use 30 epochs; the pretrained models train the classifier head with a frozen backbone, then fine-tune the backbone. The optimizer study is optional.\n\nThe digits notebook is a separate dataset experiment and is excluded from this comparison.")
    code("""import copy
import datetime
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import Image as DisplayImage, display
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, models, transforms
""")
    settings = {key: value for key, value in record["settings"].items()}
    if allow_smoke:
        settings.update(custom_epochs=30, head_epochs=5, fine_tune_epochs=20, pretrained=True)
    relative_results = result_dir.relative_to(ROOT).as_posix()
    code(
        "RUN_TRAINING = False\nRUN_OPTIMIZER_COMPARISON = False\nMOUNT_GOOGLE_DRIVE = False\n"
        "DATA_DIR = \"\"\nDOWNLOAD_DATASET = True\nIMAGE_SIZE = 64\nSPLIT_SEED = 42\n"
        "OUTPUT_DIR = Path(\"results/final_reproduction\")\nROOT = Path.cwd()\n"
        f"RECORDED_DIR = Path({relative_results!r})\n"
        "REPOSITORY_RAW_BASES = (\"https://raw.githubusercontent.com/AyubUmair/Pattern-Recognition/hakam/assignment-improvements/\", \"https://raw.githubusercontent.com/AyubUmair/Pattern-Recognition/main/\")\n"
        "TRAINING_SETTINGS = " + pprint.pformat(settings, width=100, sort_dicts=False) + "\n"
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\nprint(\"Device:\", device)\n"
    )
    md("## Models\n\nAll inputs are RGB at 64 by 64 pixels. Model A interleaves standard convolutions and max-pooling. Model B uses depthwise separable blocks and global average pooling. It has 9,033 trainable parameters, below the 100,000 limit. Hidden custom-model activations use ReLU; classifiers return logits.")
    code("\n".join(cell_sources[key] for key in ["zy0c4XKH9TbO", "CVNoDlfXCMnA", "44DJnbQcEQyY", "ksol8ETLFDqr", "uFAw8bUsFHIG"]))
    md("## Data and training helpers\n\nThe split is stratified 70/15/15 with seed 42. Resize and ImageNet normalization are deterministic; the final configuration uses no random augmentation. Preprocessed image tensors are cached in RAM. Custom models use Adam at 0.001 with plain cross-entropy and no scheduler. Transfer learning freezes the feature extractor first, then fine-tunes at a lower learning rate with batch-normalization statistics and parameters frozen. Checkpoints are selected using validation accuracy.")
    code(definitions(optimizer_source, {"seed_experiment", "state_fingerprint", "RecordedRandomSampler"}))
    code(definitions(support, {"seed_worker", "make_image_transforms", "resolve_realwaste_path", "prepare_realwaste_datasets", "make_dataloaders"}))
    code(definitions(support, {"train_model", "build_sota_model", "train_transfer_model", "estimate_macs", "export_training_result"}))
    code(definitions((ROOT / "run_optimizer_comparison.py").read_text(encoding="utf-8"), {"cached_images"}))
    code(definitions((ROOT / "run_final_training.py").read_text(encoding="utf-8"), {"run_final_models"}))
    code(definitions(optimizer_source, {"compare_optimizers", "optimizer_summary", "save_optimizer_comparison"}))
    recorded = {
        "dataset": record["dataset"], "image_size": 64,
        "split_sizes": record["split_sizes"], "models": {},
    }
    for name, metrics in record["models"].items():
        recorded["models"][name] = {key: value for key, value in metrics.items() if key not in ["history", "confusion_matrix", "verification"]}
    optimizer_rows = [{
        "Optimizer": trial["name"], "Learning rate": trial["lr"], "Momentum": trial["momentum"],
        "Best validation accuracy (%)": trial["best_val_accuracy"] * 100,
        "Best epoch": trial["best_epoch"], "Validation loss": trial["val_loss_at_best_epoch"],
    } for trial in optimizer_record["trials"]]
    code("RECORDED_RESULTS = " + pprint.pformat(recorded, width=110, sort_dicts=False) + "\nOPTIMIZER_ROWS = " + pprint.pformat(optimizer_rows, width=110, sort_dicts=False) + "\n")
    md("## Run or load recorded results\n\nThe recorded Model B checkpoint comes from the verified no-augmentation 30-epoch comparison. Its earlier timing is preserved. A fresh notebook run trains Model B again. Both paths use the same architecture, initialization seed, split and sample-order controls. The first training run downloads the UCI dataset if missing, and ImageNet weights for the two pretrained models.")
    code("""data_transforms = make_image_transforms(image_size=IMAGE_SIZE)
if RUN_TRAINING or RUN_OPTIMIZER_COMPARISON:
    if MOUNT_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / '.local/torch_hub'))
    data_dir = resolve_realwaste_path(DATA_DIR, DOWNLOAD_DATASET).resolve()
    tensor_data, image_data = cached_images(globals(), data_dir, ROOT / '.local/realwaste_64.pt')
    _, _, split_indices = prepare_realwaste_datasets(data_dir, IMAGE_SIZE, False, SPLIT_SEED)
    split_data = {phase: Subset(tensor_data, values) for phase, values in split_indices.items()}
    print(image_data.classes)
    print({phase: len(values) for phase, values in split_data.items()})

if RUN_TRAINING:
    final_results = run_final_models(
        globals(), split_data, image_data.classes, split_indices, TRAINING_SETTINGS,
        OUTPUT_DIR, ['Model A', 'Model B', 'MobileNetV2', 'EfficientNet-B0'],
        provenance={'source': 'EN3150_A03_Final.ipynb', 'fresh_training': True},
    )
else:
    final_results = RECORDED_RESULTS
    print('Recorded results:', final_results['dataset'], final_results['split_sizes'])
""")
    md("## Test results")
    code("""metric_rows = []
for name, result in final_results['models'].items():
    metric_rows.append({
        'Model': name,
        'Test accuracy (%)': 100 * result['test_accuracy'],
        'Weighted precision': result['weighted_precision'],
        'Weighted recall': result['weighted_recall'],
        'Weighted F1': result['weighted_f1'],
        'Macro F1': result['macro_f1'],
    })
display(pd.DataFrame(metric_rows).round(4))
""")
    code("""cost_rows = []
for name, result in final_results['models'].items():
    cost_rows.append({
        'Model': name,
        'Total parameters': result['total_parameters'],
        'File size (MB)': result['checkpoint_file_kib'] * 1024 / 1e6,
        'Tensor storage (KiB)': result['tensor_storage_kib'],
        'Conv/linear MACs (M)': result['estimated_conv_linear_macs'] / 1e6,
        'Completed epochs': result['epochs_completed'],
        'Mean seconds/epoch': result['mean_epoch_seconds'],
    })
display(pd.DataFrame(cost_rows).round(4))
""")
    md("MB uses 1,000,000 bytes; KiB uses 1,024 bytes. Epoch timing includes training and validation on cached tensors. Pretrained-model means combine classifier-head training and backbone fine-tuning stages; Model B timing is from its earlier verified run. These are single-run training measurements, not deployment latency or peak RAM. MAC estimates include convolution and linear layers only.")
    code("""def show_saved_image(path):
    path = Path(path)
    if path.exists():
        display(DisplayImage(filename=str(path)))
    else:
        for base in REPOSITORY_RAW_BASES:
            try:
                with urllib.request.urlopen(base + path.as_posix(), timeout=15) as response:
                    data = response.read()
                display(DisplayImage(data=data))
                return
            except (urllib.error.HTTPError, urllib.error.URLError):
                continue
        raise FileNotFoundError(f'Plot unavailable: {path}. Download the repository assets or run training.')

plot_dir = OUTPUT_DIR if RUN_TRAINING else RECORDED_DIR
for name in final_results['models']:
    print(name)
    slug = name.lower().replace(' ', '_').replace('-', '_')
    show_saved_image(plot_dir / f'{slug}_curves.png')
    show_saved_image(plot_dir / f'{slug}_confusion.png')
""")
    md("## Optimizer comparison\n\nThe recorded study compares Adam, SGD and SGD with momentum 0.9 on Model B. Each candidate receives 30 epochs with identical starting weights and sample orders. Validation selects Adam at 0.001. Momentum retains past gradients in an update buffer; its effect depends on learning rate. Test results are used after configuration selection.")
    code("""if RUN_OPTIMIZER_COMPARISON:
    optimizer_results, selected_model = compare_optimizers(
        lambda: LightWeightModel(len(image_data.classes)),
        split_data['train'], split_data['val'], epochs=30, seed=42, batch_size=32,
    )
    optimizer_dir = OUTPUT_DIR / 'optimizer_comparison'
    optimizer_figures = save_optimizer_comparison(
        optimizer_results, selected_model, split_data['test'], image_data.classes,
        optimizer_dir, optimizer_dir / 'checkpoints',
    )
    display(optimizer_summary(optimizer_results))
else:
    optimizer_dir = Path('results/optimizer_comparison')
    display(pd.DataFrame(OPTIMIZER_ROWS).round(4))
show_saved_image(optimizer_dir / 'validation_accuracy.png')
show_saved_image(optimizer_dir / 'loss_curves.png')
""")
    md("## Files for the LaTeX report\n\nUse `results/final/report_tables.tex` for the result and optimizer tables, and the PNG files in `results/final/` for curves and confusion matrices. Layer-by-layer parameter calculations and activation justification are in `results/architecture/architecture_summary.md`. Full histories, split indices, model verification and source provenance are in `results/final/experiment.json`. Dataset and model citation entries are in `results/final/references.bib`. Previous tuning experiments remain separate from the final comparison.\n\nThe GitHub repository is https://github.com/AyubUmair/Pattern-Recognition. All final model results use RealWaste; the friend's digits notebook is retained separately.")
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}})
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(cell.source).readline))
    nbformat.validate(notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    if not allow_smoke:
        write_latex_tables(record, optimizer_record, result_dir)
    print(f"Created {output_path}; {len(cells)} cells; dataset={record['dataset']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/final")
    parser.add_argument("--output", type=Path, default=ROOT / "EN3150_A03_Final.ipynb")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    result_dir = args.results.resolve()
    record = json.loads((result_dir / "experiment.json").read_text(encoding="utf-8"))
    build_notebook(record, result_dir, args.output.resolve(), args.allow_smoke)


if __name__ == "__main__":
    main()
