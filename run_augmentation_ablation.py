import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms

from run_optimizer_comparison import NOTEBOOK, ROOT, cached_images, notebook_namespace


class HorizontalFlipView(Dataset):
    def __init__(self, dataset, probability=0.5):
        self.dataset = dataset
        self.transform = transforms.RandomHorizontalFlip(probability)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_flip_equivalence(dataset, image_dataset=None):
    image, label = dataset[0]
    original = image.clone()
    unchanged, unchanged_label = HorizontalFlipView(dataset, 0)[0]
    flipped, flipped_label = HorizontalFlipView(dataset, 1)[0]
    assert torch.equal(unchanged, original)
    assert torch.equal(flipped, original.flip(-1))
    assert label == unchanged_label == flipped_label
    assert torch.equal(dataset[0][0], original)
    checked = 0
    if image_dataset is not None:
        for label in range(len(image_dataset.classes)):
            index = image_dataset.targets.index(label)
            raw = image_dataset.loader(image_dataset.samples[index][0])
            operations = image_dataset.transform.transforms
            expected = transforms.Compose([operations[0], transforms.RandomHorizontalFlip(1), *operations[1:]])(raw)
            assert torch.equal(dataset[index][0].flip(-1), expected)
            checked += 1
    return {"identity_and_flip_checks": True, "cache_not_mutated": True, "pixel_equivalence_images": checked}


def verify_checkpoint(namespace, checkpoint, original_model, test_loader, class_names, metrics, output_dir):
    reloaded = namespace["LightWeightModel"](len(class_names)).to(namespace["device"])
    weights = torch.load(checkpoint, map_location=namespace["device"], weights_only=True)
    assert all(torch.isfinite(value).all() for value in weights.values())
    reloaded.load_state_dict(weights, strict=True)
    assert namespace["state_fingerprint"](reloaded.state_dict()) == namespace["state_fingerprint"](original_model.state_dict())
    labels, predictions = [], []
    reloaded.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            logits = reloaded(inputs.to(namespace["device"]))
            assert logits.shape == (len(inputs), len(class_names))
            assert torch.isfinite(logits).all()
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    assert np.array_equal(matrix, metrics["confusion_matrix"])
    report = classification_report(labels, predictions, labels=list(range(len(class_names))), output_dict=True, zero_division=0)
    assert abs(report["accuracy"] - metrics["test_accuracy"]) < 1e-12
    for group in ["weighted", "macro"]:
        for name, field in [("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")]:
            assert abs(report[f"{group} avg"][field] - metrics[f"{group}_{name}"]) < 1e-12
    pd.DataFrame({"dataset_index": list(test_loader.dataset.indices), "actual_class": [class_names[value] for value in labels], "predicted_class": [class_names[value] for value in predictions]}).to_csv(output_dir / "reloaded_checkpoint_predictions.csv", index=False)
    return {
        "strict_reload": True,
        "state_fingerprint_matches": True,
        "finite_weights_and_logits": True,
        "confusion_matrix_and_metrics_match": True,
        "test_images_reevaluated": len(labels),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }


def plot_comparison(trials, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for trial in trials:
        epochs = range(1, len(trial["history"]["val_acc"]) + 1)
        axes[0].plot(epochs, np.array(trial["history"]["val_acc"]) * 100, label=trial["name"])
        axes[1].plot(epochs, trial["history"]["val_loss"], label=trial["name"])
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Model B: horizontal-flip comparison")
    axes[1].set(xlabel="Epoch", ylabel="Validation cross-entropy loss", title="Same optimizer and training budget")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation_comparison.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("Epoch and thread counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = notebook_namespace(notebook)
    namespace["device"] = torch.device(args.device)
    output_dir = (ROOT / ".local/augmentation_smoke" if args.smoke_test else args.output or ROOT / "results/augmentation_ablation").resolve()
    if not args.smoke_test and (output_dir / "ablation_results.json").exists():
        parser.error("Results already exist; use --output with a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        torch.manual_seed(args.seed)
        dataset = TensorDataset(torch.randn(36, 3, 64, 64), torch.arange(36) % 9)
        split_indices = {"train": list(range(18)), "val": list(range(18, 27)), "test": list(range(27, 36))}
        classes = [f"class_{index}" for index in range(9)]
        epochs = 2
        pixel_checks = verify_flip_equivalence(dataset)
        dataset_name = "Synthetic smoke test"
    else:
        data_dir = (args.data_dir or namespace["resolve_realwaste_path"]("", False)).resolve()
        dataset, image_dataset = cached_images(namespace, data_dir, ROOT / ".local/realwaste_64.pt")
        train_indices, temporary_indices, _, temporary_labels = train_test_split(
            list(range(len(dataset))), image_dataset.targets,
            test_size=0.30, random_state=42, stratify=image_dataset.targets,
        )
        val_indices, test_indices = train_test_split(temporary_indices, test_size=0.50, random_state=42, stratify=temporary_labels)
        split_indices = {"train": train_indices, "val": val_indices, "test": test_indices}
        classes = image_dataset.classes
        epochs = args.epochs
        pixel_checks = verify_flip_equivalence(dataset, image_dataset)
        dataset_name = "RealWaste"
    assert set(split_indices["train"]).isdisjoint(split_indices["val"])
    assert set(split_indices["train"]).isdisjoint(split_indices["test"])
    assert set(split_indices["val"]).isdisjoint(split_indices["test"])
    split_data = {phase: Subset(dataset, indices) for phase, indices in split_indices.items()}
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    snapshot = json.dumps(sources)
    (output_dir / "source_snapshot.json").write_text(snapshot, encoding="utf-8")
    record = {
        "status": "training", "dataset": dataset_name, "seed": args.seed, "split_seed": 42,
        "epochs_per_trial": epochs, "batch_size": 32, "device": str(namespace["device"]),
        "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0,
        "label_smoothing": 0, "scheduler": None, "early_stopping": False,
        "image_size": 64, "classes": classes, "split_indices": split_indices,
        "split_sizes": {phase: len(values) for phase, values in split_indices.items()},
        "selection_rule": "highest validation accuracy; lower validation loss breaks ties; exact ties retain control",
        "checkpoint_rule": "first epoch attaining each trial's highest validation accuracy",
        "data_pipeline": "Identical verified normalized image tensors cached in RAM; train-only horizontal flip for the augmented trial",
        "trial_order": ["No augmentation", "Horizontal flip (p=0.5)"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "notebook_source_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        "source_file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), NOTEBOOK, ROOT / "run_optimizer_comparison.py"]},
        "python_version": platform.python_version(),
        "package_versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "pandas", "matplotlib", "scikit-learn", "Pillow"]},
        "torch_threads": args.threads, "torch_interop_threads": 2,
        "pixel_checks": pixel_checks, "trials": [],
    }
    namespace["seed_experiment"](args.seed)
    initial_state = copy.deepcopy(namespace["LightWeightModel"](len(classes)).state_dict())
    initial_hash = namespace["state_fingerprint"](initial_state)
    selected_model, selected_key, selected_index = None, None, None
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for index, probability in enumerate([0, 0.5]):
        namespace["seed_experiment"](args.seed)
        model = namespace["LightWeightModel"](len(classes)).to(namespace["device"])
        model.load_state_dict(initial_state)
        assert namespace["state_fingerprint"](model.state_dict()) == initial_hash
        train_data = split_data["train"] if probability == 0 else HorizontalFlipView(split_data["train"], probability)
        sampler = namespace["RecordedRandomSampler"](train_data, args.seed)
        loaders = {
            "train": DataLoader(train_data, batch_size=32, sampler=sampler, generator=torch.Generator().manual_seed(args.seed + 1)),
            "val": DataLoader(split_data["val"], batch_size=32, shuffle=False, generator=torch.Generator().manual_seed(args.seed + 2)),
        }
        print(f"\nTrial {index + 1}/2: {record['trial_order'][index]}", flush=True)
        model, history, mean_seconds = namespace["train_model"](
            model, loaders, torch.nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.001), num_of_epochs=epochs,
        )
        assert all(len(values) == epochs and np.isfinite(values).all() for values in history.values())
        assert len(sampler.epoch_order_hashes) == epochs
        if record["trials"]:
            assert sampler.epoch_order_hashes == record["trials"][0]["epoch_order_hashes"]
        best_index = int(np.argmax(history["val_acc"]))
        trial = {
            "name": record["trial_order"][index], "horizontal_flip_probability": probability,
            "initial_state_sha256": initial_hash, "epoch_order_hashes": sampler.epoch_order_hashes,
            "best_epoch": best_index + 1, "best_val_accuracy": history["val_acc"][best_index],
            "val_loss_at_best_epoch": history["val_loss"][best_index],
            "mean_epoch_seconds": mean_seconds, "history": history,
        }
        record["trials"].append(trial)
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, checkpoint_dir / ("control.pt" if index == 0 else "horizontal_flip.pt"))
        key = (trial["best_val_accuracy"], -trial["val_loss_at_best_epoch"])
        if selected_key is None or key > selected_key:
            selected_key, selected_index, selected_model = key, index, copy.deepcopy(model).eval()
        save_json(output_dir / "ablation_results.json", record)
        print(f"Best validation: {trial['best_val_accuracy']:.2%}, epoch {trial['best_epoch']}", flush=True)
    if not args.smoke_test:
        historical = json.loads((ROOT / "results/optimizer_comparison/optimizer_results.json").read_text())
        assert all(split_indices[phase] == historical["split_indices"]["validation" if phase == "val" else phase] for phase in split_indices)
        record["historical_split_matches"] = True
        if args.seed == historical["seed"] and epochs == historical["epochs_per_trial"]:
            baseline = historical["trials"][historical["selected_trial_index"]]
            control = record["trials"][0]
            assert initial_hash == baseline["initial_state_sha256"]
            assert control["epoch_order_hashes"] == baseline["epoch_order_hashes"]
            record["historical_control_check"] = {
                "initial_weights_match": True, "batch_orders_match": True,
                "best_validation_accuracy_difference": control["best_val_accuracy"] - baseline["best_val_accuracy"],
                "maximum_validation_history_difference": float(np.max(np.abs(np.asarray(control["history"]["val_acc"]) - baseline["history"]["val_acc"]))),
            }
    chosen = record["trials"][selected_index]
    test_loader = DataLoader(split_data["test"], batch_size=32, shuffle=False)
    metrics = namespace["export_training_result"](
        selected_model, "Selected Model B", chosen["history"], chosen["mean_epoch_seconds"],
        test_loader, classes, output_dir,
        {"selected_augmentation": chosen["name"], "selected_epoch": chosen["best_epoch"], "best_val_accuracy": chosen["best_val_accuracy"], "pretrained": False},
    )
    assert metrics["total_parameters"] == 9033
    verification = verify_checkpoint(namespace, checkpoint_dir / "selected_model_b.pt", selected_model, test_loader, classes, metrics, output_dir)
    record.update(status="complete", selected_trial_index=selected_index, selected_model=metrics, verification=verification)
    save_json(output_dir / "ablation_results.json", record)
    save_json(output_dir / "verification.json", verification)
    pd.DataFrame([{key: trial[key] for key in ["name", "horizontal_flip_probability", "best_epoch", "best_val_accuracy", "val_loss_at_best_epoch", "mean_epoch_seconds"]} for trial in record["trials"]]).to_csv(output_dir / "comparison.csv", index=False)
    plot_comparison(record["trials"], output_dir)
    print(f"\nSelected using validation only: {chosen['name']}", flush=True)
    print(f"Selected test accuracy: {metrics['test_accuracy']:.2%}", flush=True)
    print("PASS: identical initialization and batch orders, train-only flips, checkpoint reload and metric reproduction.", flush=True)
    print(f"Results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
