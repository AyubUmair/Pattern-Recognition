import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from run_optimizer_comparison import NOTEBOOK, ROOT, cached_images, notebook_namespace, source_revision


def run_final_models(namespace, split_data, class_names, split_indices, settings,
                     output_dir, model_names, reuse_model_b=None, provenance=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = namespace["device"]
    record = {
        "status": "training", "dataset": "RealWaste", "classes": class_names,
        "image_size": 64, "split_sizes": {key: len(value) for key, value in split_data.items()},
        "split_indices": split_indices, "settings": settings, "device": str(device),
        "data_pipeline": "Verified normalized 64x64 tensors cached in RAM; deterministic transforms; no augmentation",
        "provenance": provenance or {}, "models": {},
    }
    constructors = {"Model A": namespace["Standard_Model"], "Model B": namespace["LightWeightModel"]}
    for name in model_names:
        print(f"\nFinal training: {name}", flush=True)
        namespace["seed_experiment"](settings["seed"])
        if name in constructors:
            model = constructors[name](len(class_names)).to(device)
            sampler = namespace["RecordedRandomSampler"](split_data["train"], settings["seed"])
            loaders = {
                "train": DataLoader(split_data["train"], batch_size=settings["batch_size"], sampler=sampler, generator=torch.Generator().manual_seed(settings["seed"] + 1)),
                "val": DataLoader(split_data["val"], batch_size=settings["batch_size"], shuffle=False, generator=torch.Generator().manual_seed(settings["seed"] + 2)),
            }
            initial_hash = namespace["state_fingerprint"](model.state_dict())
            if name == "Model B" and reuse_model_b is not None:
                source_dir = Path(reuse_model_b)
                previous = json.loads((source_dir / "ablation_results.json").read_text(encoding="utf-8"))
                previous_trial = previous["trials"][previous["selected_trial_index"]]
                if settings["custom_epochs"] != previous["epochs_per_trial"] or settings["seed"] != previous["seed"]:
                    raise ValueError("The reused Model B budget and seed must match; use --fresh-model-b for another setup")
                assert previous_trial["name"] == "No augmentation"
                assert split_indices == previous["split_indices"]
                assert initial_hash == previous_trial["initial_state_sha256"]
                weights = torch.load(source_dir / "checkpoints/selected_model_b.pt", map_location=device, weights_only=True)
                model.load_state_dict(weights, strict=True)
                model.eval()
                history = previous_trial["history"]
                mean_seconds = previous_trial["mean_epoch_seconds"]
                order_hashes = previous_trial["epoch_order_hashes"]
                details = {"pretrained": False, "reused_training_record": "results/augmentation_ablation", "initial_state_sha256": initial_hash}
                print("Using the verified 30-epoch, validation-selected Model B checkpoint.", flush=True)
            else:
                model, history, mean_seconds = namespace["train_model"](
                    model, loaders, torch.nn.CrossEntropyLoss(),
                    torch.optim.Adam(model.parameters(), lr=settings["custom_lr"]),
                    num_of_epochs=settings["custom_epochs"],
                )
                order_hashes = sampler.epoch_order_hashes
                details = {"pretrained": False, "initial_state_sha256": initial_hash}
            details.update(
                selected_epoch=int(np.argmax(history["val_acc"])) + 1,
                best_val_accuracy=max(history["val_acc"]), epoch_order_hashes=order_hashes,
                optimizer="Adam", learning_rate=settings["custom_lr"],
                weight_decay=0, label_smoothing=0, scheduler=None,
            )
            if name == "Model B":
                assert namespace["count_parameters"](model) <= 100_000
        else:
            model = namespace["build_sota_model"](name, len(class_names), settings["pretrained"]).to(device)
            namespace["seed_experiment"](settings["seed"])
            model, history, mean_seconds, details = namespace["train_transfer_model"](model, split_data, settings)
            details.update(pretrained=settings["pretrained"], optimizer="Adam", head_lr=settings["head_lr"], fine_tune_lr=settings["fine_tune_lr"], weight_decay=settings["weight_decay"], label_smoothing=settings["label_smoothing"])
            assert details["head_backbone_unchanged"]
            assert details["fine_tune_backbone_changed"]
        test_loader = DataLoader(split_data["test"], batch_size=settings["batch_size"], shuffle=False)
        metrics = namespace["export_training_result"](model, name, history, mean_seconds, test_loader, class_names, output_dir, details)
        assert sum(map(sum, metrics["confusion_matrix"])) == len(split_data["test"])
        correct = sum(row[index] for index, row in enumerate(metrics["confusion_matrix"]))
        assert abs(correct / len(split_data["test"]) - metrics["test_accuracy"]) < 1e-12
        slug = name.lower().replace(" ", "_").replace("-", "_")
        checkpoint = output_dir / "checkpoints" / f"{slug}.pt"
        if name in constructors:
            reloaded = constructors[name](len(class_names)).to(device)
        else:
            reloaded = namespace["build_sota_model"](name, len(class_names), pretrained=False).to(device)
        reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        reloaded.eval()
        assert namespace["state_fingerprint"](reloaded.state_dict()) == namespace["state_fingerprint"](model.state_dict())
        probe, _ = next(iter(test_loader))
        with torch.no_grad():
            expected = model.eval()(probe.to(device))
            actual = reloaded(probe.to(device))
            assert torch.isfinite(actual).all()
            assert torch.equal(expected, actual)
        metrics["verification"] = {
            "strict_reload": True, "state_fingerprint_matches": True,
            "probe_images": len(probe), "probe_logits_match": True,
            "confusion_total_matches_test_split": True,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        (output_dir / f"{slug}_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        record["models"][name] = metrics
        (output_dir / "experiment.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        comparison = pd.DataFrame([{key: value for key, value in item.items() if key not in ["history", "confusion_matrix", "training_details", "verification"]} for item in record["models"].values()])
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)
        print(f"Completed {name}: test accuracy {metrics['test_accuracy']:.2%}; parameters {metrics['total_parameters']:,}", flush=True)
        del model, reloaded
    record["status"] = "complete"
    record["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output_dir / "experiment.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/final")
    parser.add_argument("--models", nargs="+", choices=["a", "b", "mobilenet", "efficientnet"], default=["a", "b", "mobilenet", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fresh-model-b", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.head_epochs, args.fine_tune_epochs, args.threads) < 1:
        parser.error("Epoch and thread counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    torch.hub.set_dir(str(ROOT / ".local/torch_hub"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = notebook_namespace(notebook)
    namespace["device"] = torch.device(args.device)
    settings = copy.deepcopy(namespace["TRAINING_SETTINGS"])
    settings.update(custom_epochs=args.epochs, head_epochs=args.head_epochs, fine_tune_epochs=args.fine_tune_epochs, label_smoothing=0, weight_decay=0.0001, early_stopping_patience=7)
    if args.smoke_test:
        torch.manual_seed(42)
        dataset = TensorDataset(torch.randn(36, 3, 64, 64), torch.arange(36) % 9)
        classes = [f"class_{index}" for index in range(9)]
        indices = {"train": list(range(18)), "val": list(range(18, 27)), "test": list(range(27, 36))}
        settings.update(custom_epochs=1, head_epochs=1, fine_tune_epochs=1, pretrained=False)
        output_dir = ROOT / ".local/final_smoke"
        reuse_model_b = None
    else:
        output_dir = args.output.resolve()
        if (output_dir / "experiment.json").exists():
            parser.error("Results already exist; choose a new --output directory")
        data_dir = (args.data_dir or namespace["resolve_realwaste_path"]("", False)).resolve()
        dataset, image_dataset = cached_images(namespace, data_dir, ROOT / ".local/realwaste_64.pt")
        _, _, indices = namespace["prepare_realwaste_datasets"](data_dir, 64, False, 42)
        classes = image_dataset.classes
        reuse_checkpoint = ROOT / "results/augmentation_ablation/checkpoints/selected_model_b.pt"
        reuse_model_b = ROOT / "results/augmentation_ablation" if not args.fresh_model_b and args.epochs == 30 and reuse_checkpoint.is_file() else None
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    snapshot = json.dumps(sources)
    (output_dir / "source_snapshot.json").write_text(snapshot, encoding="utf-8")
    provenance = {
        "source_commit": source_revision(),
        "notebook_source_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
        "source_file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), NOTEBOOK, ROOT / "run_optimizer_comparison.py"]},
        "package_versions": {name: importlib.metadata.version(name) for name in ["torch", "torchvision", "numpy", "pandas", "matplotlib", "scikit-learn", "Pillow"]},
        "torch_threads": args.threads, "torch_interop_threads": 2,
    }
    split_data = {phase: Subset(dataset, values) for phase, values in indices.items()}
    names = {"a": "Model A", "b": "Model B", "mobilenet": "MobileNetV2", "efficientnet": "EfficientNet-B0"}
    result = run_final_models(namespace, split_data, classes, indices, settings, output_dir, [names[name] for name in args.models], reuse_model_b, provenance)
    if args.smoke_test:
        result["dataset"] = "Synthetic smoke test"
        (output_dir / "experiment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("PASS: all selected models, both transfer stages, checkpoint reloads and exported metrics.", flush=True)


if __name__ == "__main__":
    main()
