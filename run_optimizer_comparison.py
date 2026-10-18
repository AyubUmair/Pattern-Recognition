import argparse
import ast
import base64
import hashlib
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, TensorDataset
from torchvision import datasets


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "EN3150_A03.ipynb"


def notebook_namespace(notebook):
    namespace = {"__name__": "optimizer_notebook"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", [])))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "google.colab":
                continue
            allowed = isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
            assignment = False
            if isinstance(node, ast.Assign):
                transform = any(isinstance(target, ast.Name) and target.id == "data_transforms" for target in node.targets)
                try:
                    ast.literal_eval(node.value)
                    assignment = True
                except (ValueError, TypeError):
                    assignment = transform
            if allowed or assignment:
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"notebook-cell-{index}", "exec"), namespace)
    return namespace


def cached_images(namespace, data_dir, cache_path):
    image_dataset = datasets.ImageFolder(data_dir, transform=namespace["data_transforms"])
    assert len(image_dataset) == 4752 and len(image_dataset.classes) == 9
    digest = hashlib.sha256()
    digest.update(repr(namespace["data_transforms"]).encode("utf-8"))
    for filename, label in image_dataset.samples:
        path = Path(filename)
        stat = path.stat()
        digest.update(f"{path.relative_to(data_dir)}:{label}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    fingerprint = digest.hexdigest()
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached["fingerprint"] == fingerprint:
            print("Loaded the verified 64x64 tensor cache.", flush=True)
            return TensorDataset(cached["images"], cached["labels"]), image_dataset
    print("Preparing and saving the 64x64 tensor cache...", flush=True)
    samples = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, sample in enumerate(pool.map(image_dataset.__getitem__, range(len(image_dataset)))):
            samples.append(sample[0])
            if (index + 1) % 1000 == 0:
                print(f"Prepared {index + 1}/{len(image_dataset)} images", flush=True)
    images = torch.stack(samples)
    labels = torch.tensor(image_dataset.targets)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "images": images, "labels": labels}, cache_path)
    return TensorDataset(images, labels), image_dataset


def update_notebook(notebook, result, figures, output_text):
    cell = next(cell for cell in notebook["cells"] if cell.get("metadata", {}).get("id") == "optimizer-comparison-run")
    cell["execution_count"] = 1
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output_text.splitlines(keepends=True)}]
    for path in figures:
        cell["outputs"].append({
            "output_type": "display_data",
            "metadata": {},
            "data": {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")},
        })
    NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data" / "realwaste-main" / "RealWaste")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--update-notebook", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.update_notebook and (args.smoke_test or args.epochs < 20):
        parser.error("Notebook results require at least 20 epochs on the real dataset")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; select --device cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = notebook_namespace(notebook)
    namespace["device"] = torch.device(args.device)

    if args.smoke_test:
        torch.manual_seed(args.seed)
        dataset = TensorDataset(torch.randn(36, 3, 64, 64), torch.arange(36) % 9)
        train_data, val_data, test_data = [Subset(dataset, indices) for indices in [range(18), range(18, 27), range(27, 36)]]
        classes = [f"class_{index}" for index in range(9)]
        epochs = 2
        output_dir = ROOT / ".local" / "smoke_results"
    else:
        if not args.data_dir.is_dir():
            parser.error(f"Dataset directory does not exist: {args.data_dir}")
        dataset, image_dataset = cached_images(namespace, args.data_dir.resolve(), ROOT / ".local" / "realwaste_64.pt")
        indices = list(range(len(dataset)))
        train_idx, temporary_idx, _, temporary_targets = train_test_split(
            indices, image_dataset.targets, test_size=0.30, random_state=42, stratify=image_dataset.targets,
        )
        val_idx, test_idx = train_test_split(
            temporary_idx, test_size=0.50, random_state=42, stratify=temporary_targets,
        )
        assert set(train_idx).isdisjoint(val_idx)
        assert set(train_idx).isdisjoint(test_idx)
        assert set(val_idx).isdisjoint(test_idx)
        train_data, val_data, test_data = [Subset(dataset, indices) for indices in [train_idx, val_idx, test_idx]]
        classes = image_dataset.classes
        epochs = args.epochs
        output_dir = ROOT / "results" / "optimizer_comparison"

    result, selected_model = namespace["compare_optimizers"](
        lambda: namespace["LightWeightModel"](len(classes)),
        train_data, val_data, epochs=epochs, seed=args.seed,
    )
    result["data_pipeline"] = "64x64 normalized tensors cached in RAM; no augmentation"
    result["split_seed"] = 42
    result["cpu_threads"] = args.threads
    result["base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    result["notebook_source_sha256"] = hashlib.sha256(json.dumps(sources).encode("utf-8")).hexdigest()
    result["split_indices"] = {
        "train": list(train_data.indices), "validation": list(val_data.indices), "test": list(test_data.indices),
    }
    figures = namespace["save_optimizer_comparison"](
        result, selected_model, test_data, classes, output_dir,
        ROOT / ".local" / ("smoke_checkpoints" if args.smoke_test else "optimizer_checkpoints"),
    )
    (output_dir / "source_snapshot.json").write_text(json.dumps(sources), encoding="utf-8")
    summary = io.StringIO()
    with redirect_stdout(summary):
        print(namespace["optimizer_summary"](result).to_string(index=False))
        chosen = result["trials"][result["selected_trial_index"]]
        print(f"\nSelected on validation: {chosen['name']}, lr={chosen['lr']:g}, momentum={chosen['momentum']}")
        print(f"Selected checkpoint test accuracy: {result['selected_model']['test_accuracy']:.4f}")
        print(f"All initial weights and all {epochs} epoch batch orders match across trials.")
        print(f"Dataset: {len(train_data)} train / {len(val_data)} validation / {len(test_data)} test images.")
        print(f"Training device: {args.device}; epoch timing includes training and validation.")
        print("Local runner used cached normalized image tensors; checkpoints selected by validation accuracy.")
    print(summary.getvalue(), flush=True)
    if args.update_notebook:
        update_notebook(notebook, result, figures, summary.getvalue())
    print(f"Checks passed; results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
