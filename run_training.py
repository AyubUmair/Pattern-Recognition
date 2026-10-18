import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import Subset, TensorDataset

from run_optimizer_comparison import NOTEBOOK, ROOT, notebook_namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=["a", "b", "mobilenet", "efficientnet"], default=["a", "b", "mobilenet", "efficientnet"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--fine-tune-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = notebook_namespace(notebook)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    namespace["device"] = torch.device(device_name)
    settings = copy.deepcopy(namespace["TRAINING_SETTINGS"])
    overrides = {
        "custom_epochs": args.epochs, "head_epochs": args.head_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "custom_lr": args.lr, "fine_tune_lr": args.fine_tune_lr,
        "weight_decay": args.weight_decay, "label_smoothing": args.label_smoothing, "seed": args.seed,
    }
    settings.update({name: value for name, value in overrides.items() if value is not None})
    names = {"a": "Model A", "b": "Model B", "mobilenet": "MobileNetV2", "efficientnet": "EfficientNet-B0"}
    settings["custom_models"] = [names[name] for name in args.models if name in ["a", "b"]]
    settings["sota_models"] = [names[name] for name in args.models if name in ["mobilenet", "efficientnet"]]
    augment = namespace["AUGMENT_TRAINING"] and not args.no_augmentation
    image_size = namespace["IMAGE_SIZE"]
    if args.smoke_test:
        torch.manual_seed(settings["seed"])
        dataset = TensorDataset(torch.randn(36, 3, image_size, image_size), torch.arange(36) % 9)
        indices = {"train": list(range(18)), "val": list(range(18, 27)), "test": list(range(27, 36))}
        split_data = {name: Subset(dataset, values) for name, values in indices.items()}
        classes = [f"class_{index}" for index in range(9)]
        settings.update(custom_epochs=1, head_epochs=1, fine_tune_epochs=1, batch_size=4, pretrained=False)
        augment = False
        output_dir = ROOT / ".local" / "training_smoke"
    else:
        data_dir = args.data_dir or namespace["resolve_realwaste_path"](namespace["DATA_DIR"], namespace["DOWNLOAD_DATASET"])
        dataset, split_data, indices = namespace["prepare_realwaste_datasets"](
            data_dir, image_size, augment, namespace["SPLIT_SEED"],
        )
        classes = dataset.classes
        output_dir = args.output or ROOT / namespace["OUTPUT_DIR"]
    result = namespace["run_training_experiment"](
        split_data, classes, settings, output_dir, indices, image_size, augment,
        "Synthetic smoke test" if args.smoke_test else "RealWaste",
    )
    if args.smoke_test:
        for name, metrics in result["models"].items():
            assert sum(map(sum, metrics["confusion_matrix"])) == 9
            assert 0 <= metrics["test_accuracy"] <= 1
            if name in settings["sota_models"]:
                assert metrics["training_details"]["head_backbone_unchanged"]
                assert metrics["training_details"]["fine_tune_backbone_changed"]
                assert metrics["epochs_completed"] == 2
        print("PASS: model training, frozen heads, fine-tuning updates, metrics, checkpoints and plots.")
    print(f"Results saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
