# LLM prompt: generate a PE-DB / PE Ensemble plugin

Copy everything between the horizontal rules into your LLM chat. Replace the bracketed placeholders with your model details. Attach or paste your existing model code (training script, inference, checkpoint layout) so the model can adapt it.

---

You are helping a researcher add a new prime-editing prediction model to the PE Database (PE-DB) and PE Ensemble stack as a **plugin bundle**. Produce a complete, upload-ready directory — not explanations unless something is ambiguous.

## Goal

Given my model implementation (attached below), generate these files:

```
<plugin_name>/
  manifest.yaml
  convert.py
  wrapper.py
  weights/<weight_id>/   # optional, only if I ship pretrained checkpoints
    <artifact files>
```

The bundle must work with **zip-only upload** to PE Hub (Add Model → Zip bundle). No web form fields are required if `manifest.yaml` is complete.

## Naming rules

- `<plugin_name>`: lowercase slug matching `[a-z0-9_]+`, unique, must **not** be `deepprime`, `oped`, or `pridict2`.
- Use the same slug everywhere: directory name, `manifest.name`, `model.pe_db_format` (usually), and `BasePEModel(model_name=...)`.
- Wrapper class name: PascalCase, e.g. `MyModelWrapper`.

## Reference implementations

- Minimal working plugin: `testdata/plugins/dummy_model/` in the pe-db repo.
- Starter template: `plugins/_template/` in the pe-db repo.
- Human guide: `plugins/README.md`.

## File 1: `manifest.yaml` (required)

All metadata lives here. Required top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Plugin slug (see naming rules) |
| `version` | string | Semver string, e.g. `0.1.0` |
| `display_name` | string | Human title for UI |
| `authors` | list of strings | Optional but recommended |
| `description` | string | One-line catalog summary |

Required `format` section (unless `model.pe_db_format` is a built-in: `deepprime`, `pridict`, `pridict2`, `oped`):

| Field | Type | Description |
|-------|------|-------------|
| `format.module` | string | Filename, usually `convert.py` |
| `format.entrypoint` | string | Function name in that module, usually `convert` |
| `format.required_std_columns` | list | Standardized PE-DB columns your converter reads |
| `format.output_columns` | list | Columns `convert()` must produce |
| `format.label_column` | string | Ground-truth column name (e.g. `Efficiency`) |

Required `model` section:

| Field | Type | Description |
|-------|------|-------------|
| `model.module` | string | Usually `wrapper.py` |
| `model.class` | string | Wrapper class name |
| `model.pe_db_format` | string | Format key PE-DB uses when exporting training data (usually equals `name`) |
| `model.weight_format` | string | Tag returned by `save_to_registry()` for checkpoints this model writes |
| `model.hyperparameters` | list | Each entry: `{ name, type, default }` where `type` is `int`, `float`, `str`, or `bool` |
| `model.constructor_kwargs` | list | Optional extra kwargs passed to wrapper `__init__` |

Optional `weights` section (shipped pretrained checkpoints):

| Field | Type | Description |
|-------|------|-------------|
| `weights[].id` | string | Registry id (directory name under `weights/`) |
| `weights[].notes` | string | Optional description |
| `weights[].files` | list | Optional explicit filenames; if omitted, all files in `weights/<id>/` are copied |

Example skeleton:

```yaml
name: my_model
version: 0.1.0
display_name: "My PE Model"
authors: ["Your Lab"]
description: "Predicts prime-editing efficiency."

format:
  module: convert.py
  entrypoint: convert
  required_std_columns:
    - wt_sequence
    - mut_sequence
    - editing_efficiency
    - edit_len
  output_columns:
    - feature
    - Efficiency
  label_column: Efficiency

model:
  module: wrapper.py
  class: MyModelWrapper
  pe_db_format: my_model
  weight_format: my_model_weights
  hyperparameters:
    - name: epochs
      type: int
      default: 10
    - name: lr
      type: float
      default: 0.001

weights:
  - id: base
    notes: "Pretrained checkpoint"
```

## File 2: `convert.py` (required for custom formats)

PE-DB calls this when data is requested with `format=<pe_db_format>`.

```python
import pandas as pd

def convert(std_df: pd.DataFrame) -> pd.DataFrame:
    ...
```

**Contract:**

1. Input `std_df` uses the standardized PE-DB schema; columns listed in `format.required_std_columns` are present.
2. Output has the **same row count and index order** as `std_df`.
3. Every column in `format.output_columns` is present; no all-null columns.
4. `format.label_column` is included in output for supervised training/eval.
5. Do **not** split train/val/test here — PE-DB handles splits; this only encodes features.

Map sequences, tokenize, or derive features here. Keep I/O as pandas DataFrames unless your wrapper expects something else (then document that in `prepare_data`).

## File 3: `wrapper.py` (required)

Subclass `pe_common.model_interface.BasePEModel`. Implement **all** abstract methods plus weight-registry helpers used by Train, Benchmark, and CLI.

### Required imports and base

```python
from pe_common.model_interface import BasePEModel
from pe_common.training import regression_metrics  # for evaluate()
```

### Constructor

```python
class MyModelWrapper(BasePEModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(model_name="my_model", device=device)
        # kwargs may include manifest constructor_kwargs
```

### Method signatures and contracts

#### `load_model(self, model_path: str) -> None`

Load weights from a **filesystem path** (file or directory). Set `self.is_trained = True` on success. Called internally; registry ids are resolved in `load_weights_by_name`.

#### `prepare_data(self, df: pd.DataFrame, **kwargs) -> Any`

Turn native-format rows (post-`convert`) into model inputs: tensors, token batches, etc. Return type is model-specific. Called before `predict` and inside `evaluate`.

#### `predict(self, data: Any, batch_size: int = 32) -> List[float]`

Return one float prediction per row, same length as input batch.

#### `train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`

Train on native-format DataFrames. Hyperparameters come from manifest defaults, Train UI, or CLI JSON.

Common pattern when resuming from shipped weights:

```python
hp = hyperparameters or {}
if hp.get("load_pretrained"):
    self.load_weights_by_name(str(hp["weights"]))
```

Return a JSON-serializable dict of training metadata (loss history, epochs run, etc.).

#### `evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]`

`weights` is a **registry id** (e.g. `base`), not a file path.

```python
def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
    self.load_weights_by_name(weights)
    preds = self.predict(self.prepare_data(test_data))
    y_true = test_data["Efficiency"].astype(float).tolist()  # use manifest label_column
    return regression_metrics(y_true, preds)
```

Must return metrics including at least Pearson/Spearman-compatible keys (validation harness checks `pearson`, `spearman`).

#### `save_model(self, model_path: str) -> None`

Write checkpoint to an arbitrary path (low-level save).

#### `save_to_registry(self, dest_dir: Union[str, Path]) -> str`

Write trained artifacts into a weight-registry directory. **Return** `model.weight_format` string from manifest.

```python
def save_to_registry(self, dest_dir) -> str:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    # write checkpoint(s) under dest/
    return "my_model_weights"
```

#### `load_weights_by_name(self, name: str) -> None`

Resolve registry id → files on disk:

```python
def load_weights_by_name(self, name: str) -> None:
    from app.models import weights_registry

    entry_dir = weights_registry.resolve_dir("my_model", name)
    self.load_model(str(entry_dir / "model.pt"))  # or your artifact name
```

Shipped plugin weights live at `plugins/<name>/weights/<id>/` and are copied into the registry on upload/activation.

#### `list_available_weights() -> List[str]` (static)

```python
@staticmethod
def list_available_weights() -> List[str]:
    from app.models import weights_registry
    return weights_registry.list_weight_ids("my_model")
```

## Weights layout

If you ship pretrained weights in the bundle:

```
weights/
  base/
    model.pt          # or weights.txt, ensemble/, etc.
```

- Directory name `base` must match `weights[].id` in manifest.
- Document the expected artifact filename(s) in `load_weights_by_name`.
- Use the same artifact name in `save_to_registry` for newly trained weights.

Environment variable `WEIGHTS_ROOT` points at the runtime registry; do not hardcode absolute paths.

## What PE-DB vs PE Ensemble do

| Component | Uses | Role |
|-----------|------|------|
| PE-DB | `convert.py` | Standardized CSV → native columns for export |
| PE Ensemble | `wrapper.py` | Train, predict, benchmark, CLI |
| Both | `manifest.yaml` | Contract, hyperparameters, weight ids |

## After generation: local checklist

1. `name` in manifest matches directory name.
2. `convert()` preserves index; all `output_columns` populated.
3. Wrapper `model_name` matches manifest `name`.
4. `evaluate()` uses registry id via `load_weights_by_name`.
5. All imports exist in the deployment environment (no silent `pip install`).
6. Zip: `cd my_model && zip -r ../my_model.zip .` (files at zip root, not nested in an extra folder).

## Upload and activate (no form required)

1. PE Hub → Add Model → **Zip bundle** → upload zip.
2. Run **Validate** (smoke tests: convert, train, save, load, evaluate).
3. **Activate** plugin (writes `.state.json` with `"status": "active"`).

Or copy to `plugins/<name>/` on the server and activate via API.

## CLI training (after activation)

```bash
export PLUGINS_ROOT=/path/to/plugins
python -m app.train_models \
  --model my_model \
  --dataset-id <pe_db_dataset_id> \
  --hyperparameters-json '{"epochs": 10, "lr": 0.001}'
```

Active plugins load at CLI startup the same way as built-in models.

## My model implementation

[Paste or attach: model class, training loop, inference, checkpoint format, expected input columns, hyperparameters, dependencies]

Plugin name I want: `[my_model]`

Display name: `[My PE Model]`

Standardized columns I need from PE-DB: `[wt_sequence, mut_sequence, editing_efficiency, edit_len, ...]`

Native columns after convert: `[list columns]`

Label column: `[Efficiency or other]`

Hyperparameters: `[epochs, lr, batch_size, ...]`

Pretrained weights to ship: `[yes/no, ids and filenames]`

---

## Notes for humans using this prompt

- The block above is self-contained; researchers can paste model code and get upload-ready files.
- Point them to `plugins/_template/` for a minimal starting tree.
- Validation failures usually mean: row count mismatch in `convert`, missing abstract methods, or `load_weights_by_name` cannot find artifacts under `weights/<id>/`.
