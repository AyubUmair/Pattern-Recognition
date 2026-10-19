# Final RealWaste training workflow

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
