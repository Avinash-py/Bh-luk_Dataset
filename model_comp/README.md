# model_comp

Stage 4 deep learning model comparison (split from the original single-file `model_comp.py`).

## File structure

- `models.py` — model definitions: `MTSAN`, `ResidualCNN`, `CNN`, `SimpleGRU`, `SimpleLSTM`, `SimpleTransformer`, `SimpleRNNAttention`
- `losses.py` — `FocalLoss`
- `dataset.py` — `extract_label`, `load_deep_learning_data`, `TimeSeriesDataset`
- `train.py` — `train_model`
- `evaluate.py` — `evaluate_model`
- `main.py` — `fix_seed`, `run_stage4_analysis`, script entrypoint

## Prerequisites

Data is expected at `helper/train_voltage`, `helper/validation_voltage`, `helper/test_voltage`, relative to the working directory. This data actually lives at `E-Nose/helper/`, so `ApprochA/` needs a symlink to it (one-time setup):

```bash
cd ApprochA
ln -s ../helper helper
```

## Running

Run as a module from inside `ApprochA/` — **not** as a plain script:

```bash
cd ApprochA
python3 -m model_comp.main
```

### Why not `python3 main.py` or `python3 model_comp/main.py`

- `main.py` uses relative imports (`from .models import ...`) to reach the other files in this package. Relative imports only resolve when the file is loaded as part of its package (`-m model_comp.main`), never when run directly as a script — running it directly raises `ImportError: attempted relative import with no known parent package`.
- The package also does `from utils.xxx import ...`, where `utils/` is a sibling folder of `model_comp/` inside `ApprochA/`. That only resolves if the working directory is `ApprochA/` when you launch Python (so `ApprochA` ends up on `sys.path`) — running from any other directory raises `ModuleNotFoundError: No module named 'utils'`.

## Outputs

Relative to wherever you ran the command from (`ApprochA/` if you followed the steps above):

- `checkpoints/` — best model weights per run (`best_model_<Model>_<timestamp>.pth`), older ones auto-pruned, keeping the last 5
- `results/<ModelName>/` — training history plots
- `embeddings/` — test-set embeddings (`.npz`) saved per model
- `confusion_matrix_<ModelName>.png` — per-model confusion matrix
- `Final_Model_Comparison_Plots.png` — comparison across all trained models
- `stage4_all_model_results_sensor_<N>.csv` — final metrics table
