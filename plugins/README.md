# Model plugins

A **plugin** is a self-contained directory that adds a new prime-editing model to PE Database and PE Ensemble without editing core source. Once **active**, the model appears in PE Hub (Train, Benchmark, Add Model) and in the training CLI (`python -m app.train_models`).

Both services read the same tree (default: `<repo>/plugins`, override with `PLUGINS_ROOT`).

```
plugins/<name>/
  manifest.yaml       # required — contract for both services
  convert.py          # PE-DB: standardized rows → model input columns
  wrapper.py          # PE Ensemble: BasePEModel subclass
  weights/<id>/       # optional shipped pretrained weights
  .state.json         # lifecycle: pending | active | rejected
```

Only plugins with `"status": "active"` in `.state.json` are registered at startup.

**Reference bundle:** `testdata/plugins/dummy_model/` (minimal working example).  
**Starter template:** `plugins/_template/` (copy and edit).  
**LLM prompt:** `docs/plugin-author-llm-prompt.md` (paste into ChatGPT/Claude to generate files).  
**Design details:** `docs/add-new-model-plugins.md`.

---

## Quick checklist (before upload)

| Item | Requirement |
|------|-------------|
| **Name** | Lowercase slug `[a-z0-9_]+`, unique, not `deepprime` / `oped` / `pridict2` |
| **manifest.yaml** | Valid YAML; `name` matches directory name |
| **convert.py** | Exposes manifest `format.entrypoint` (default `convert(std_df)`) |
| **wrapper.py** | Class in manifest subclasses `BasePEModel`; implements all abstract methods + `save_to_registry` |
| **Row order** | `convert()` returns same row count and index as input |
| **Output columns** | Every column listed in `format.output_columns` is present and non-empty after convert |
| **Label column** | Ground-truth column (e.g. `Efficiency`) listed in manifest and produced by convert |
| **Dependencies** | All imports available in the runtime env (no auto-`pip install`) |
| **Weights** (optional) | Under `weights/<id>/`; ids match manifest `weights[].id` |

Run through **Validate** in PE Hub (or `POST /models/plugins/{name}/validate`) before activating. Pending or rejected plugins are ignored by Train, Benchmark, and the CLI.

---

## 1. Prepare the bundle

### 1.1 Directory layout

Create `plugins/my_model/` (or work offline and upload later):

```bash
plugins/my_model/
  manifest.yaml
  convert.py
  wrapper.py
  weights/              # optional
    base/
      weights.pt
```

You can also upload a **zip** with the same layout via Add Model (paths must be relative; no `..`).

### 1.2 `manifest.yaml`

Minimal template (see `testdata/plugins/dummy_model/manifest.yaml`):

```yaml
name: my_model
version: 0.1.0
display_name: "My PE Model"
authors: ["Your Lab"]
description: "Short summary for the model catalog."

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
  pe_db_format: my_model          # format= requested when fetching training data
  weight_format: my_model_weights # returned by save_to_registry
  hyperparameters:
    - name: epochs
      type: int
      default: 10
    - name: lr
      type: float
      default: 0.001

weights:
  - id: base
    notes: "Optional pretrained checkpoint"
```

Notes:

- **`model.pe_db_format`** — Usually equals `name`. Set to a built-in format (`deepprime`, `oped`, …) only if the model consumes an existing native format and you **omit** `convert.py`.
- **`model.weight_format`** — Tag stored in weight manifests for checkpoints this model writes.
- **`hyperparameters`** — Drive the Train UI and can be passed via CLI as JSON (see below).

### 1.3 `convert.py` (PE Database)

```python
import pandas as pd

def convert(std_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized schema → native columns for this model."""
    out = pd.DataFrame(index=std_df.index)
    out["feature"] = pd.to_numeric(std_df["edit_len"], errors="raise")
    out["Efficiency"] = pd.to_numeric(std_df["editing_efficiency"], errors="raise")
    return out
```

Contract:

- Input: standardized columns your manifest declares in `required_std_columns`.
- Output: **same number of rows, same index order** as input.
- Include every column in `format.output_columns`.
- Deterministic, no side effects. May accept optional `progress_callback(done, total)`.

### 1.4 `wrapper.py` (PE Ensemble)

Subclass `pe_common.model_interface.BasePEModel` and implement:

| Method | Purpose |
|--------|---------|
| `load_model` | Load artifacts from a path |
| `prepare_data` | DataFrame → model input |
| `predict` | One float prediction per row |
| `train` | Fit on train (and optional val) DataFrames |
| `evaluate` | Metrics on test data using a weight id |
| `save_model` | Save to an arbitrary path |
| **`save_to_registry`** | Write trained artifacts into a weight-registry directory; **return** `weight_format` string |

Conventional helpers (used by built-ins): `load_weights_by_name`, `list_available_weights`.

Training and evaluation receive data already converted to your native columns (via PE-DB `format=`).

### 1.5 Optional weights

Ship files under `weights/<id>/`. Register ids in manifest `weights`. On activation, PE Ensemble imports them into the weight registry under your model name.

---

## 2. Register the plugin (make it active)

Plugins must be **`active`** before Train, Benchmark, or the CLI can use them.

### Option A — PE Hub (recommended)

1. Open **Add Model** (requires PE-DB and PE Ensemble running).
2. Fill metadata and upload `convert.py`, `wrapper.py`, optional weights — **or** upload a prepared **`manifest.yaml`** (recommended for hyperparameters, weight metadata, and full plugin config) plus code files.
3. **Validate** — automated harness (manifest, imports, convert round-trip, train/eval smoke on CPU).
4. **Activate** — sets `.state.json` to `active`, reloads PE Ensemble and notifies PE-DB.

### Option B — Filesystem (trusted local / CI)

1. Copy the bundle to `plugins/<name>/`.
2. Validate via API: `POST http://localhost:8001/models/plugins/{name}/validate` (after an upload or if files were placed manually with a pending state).
3. Activate: `POST http://localhost:8001/models/plugins/{name}/activate`.

Or, for local development only with a trusted bundle:

```json
// plugins/my_model/.state.json
{
  "status": "active",
  "updated_at": "2026-06-14T00:00:00Z"
}
```

Then restart services **or** reload:

- PE Ensemble: activation endpoint (or restart uvicorn)
- PE-DB: `POST /api/plugins/reload` (activation does this automatically)

Skipping validation is only appropriate for fixtures you control (e.g. `dummy_model` in tests).

---

## 3. Use the model

After activation, the model is available in **both** the web UI and the CLI. The same `execute_training()` path runs in each case.

### 3.1 Web UI (PE Hub)

- **Train** — model appears in the dropdown; hyperparameter fields come from manifest.
- **Benchmark** — same model list for evaluation jobs.
- **Add Model** — manage status, re-validate, delete.

Ensure PE-DB and PE Ensemble share the same `PLUGINS_ROOT` (default repo `plugins/`).

### 3.2 Training CLI

The CLI loads **active** plugins from `PLUGINS_ROOT` at startup (same registry as the HTTP service)
and always uses in-process PE-DB (`pe_db.library`, same path as the `pedb` CLI). No PE-DB HTTP
server is required for CLI training.

```bash
pip install -e packages/pe-common
pip install -e services/pe-db
pip install -e "services/pe-ensemble[pe-db]"

peen train \
  --model my_model \
  --dataset-name my_run \
  --dataset library2 \
  --cell-line HEK293T \
  --hyperparameters-json '{"epochs": 5, "lr": 0.001}' \
  --device cuda:0
```

The FastAPI web service (and portal) still talk to PE-DB over HTTP via `PE_DB_URL`.

Plugin-specific flags:

| Flag / env | Purpose |
|------------|---------|
| `PLUGINS_ROOT` / `--plugins-root` | Directory containing `plugins/<name>/` (default: repo `plugins/`) |
| `WEIGHTS_ROOT` / `--weights-root` | Where trained checkpoints are registered |
| `--hyperparameters-json` | JSON object; keys should match manifest `hyperparameters` names |
| `--pretrained-weights <id>` | Fine-tune from a shipped or previously trained weight set |

Fine-tune from shipped weights:

```bash
peen train \
  --model my_model \
  --dataset-name finetune_run \
  --dataset library2 \
  --pretrained-weights base \
  --hyperparameters-json '{"epochs": 3, "load_pretrained": true}'
```

Verify the model is registered:

```bash
python -c "from app.plugin_loader import load_active_plugins; from app.training.config import supported_models; load_active_plugins(); print(supported_models())"
```

---

## 4. Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `PLUGINS_ROOT` | `<repo>/plugins` | PE-DB, PE Ensemble, CLI |
| `WEIGHTS_ROOT` | `services/pe-ensemble/weights` | PE Ensemble, CLI |
| `PE_DB_URL` | `http://localhost:8000` | PE Ensemble **web service** only (HTTP to PE-DB) |
| `TRAINING_JOBS_ROOT` | `services/pe-ensemble/jobs` | CLI job state and logs |

Use the **same** `PLUGINS_ROOT` on every process that should see your model (Hub services, CLI, SLURM workers).

---

## 5. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Model missing from Train / `--model` choices | Plugin not `active`, or wrong `PLUGINS_ROOT` |
| `Unknown model` in CLI | Plugin not active; restart CLI after activation |
| PE-DB `format=my_model` fails | PE-DB has not reloaded plugins; call `/api/plugins/reload`, or use the `peen` CLI (in-process `pe_db.library` reloads plugins at startup) |
| Validation fails on convert | Row count/index changed, or missing `output_columns` |
| Validation fails on train/eval | Wrapper missing method or `save_to_registry`; smoke test timeout |
| Import error in validation | Dependency not installed in the service environment |

---

## 6. End-to-end flow

```
Prepare bundle (manifest + convert + wrapper [+ weights])
        │
        ├─► PE Hub: Upload → Validate → Activate
        └─► Or: copy to plugins/ + validate/activate via API
        │
        ▼
  .state.json status = "active"
        │
        ├─► PE Ensemble: wrapper in model registry
        └─► PE-DB: convert registered as format=
        │
        ├─► PE Hub: Train / Benchmark
        └─► CLI: python -m app.train_models --model <name> …
```

For maintainers and API details, see `docs/add-new-model-plugins.md`.
