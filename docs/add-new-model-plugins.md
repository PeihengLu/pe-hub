# Design: "Add New Model" — Hot-Pluggable Model Plugins

**Status:** Implemented

**Scope:** PE-Hub UI page + plugin system spanning PE Database and PE Ensemble  
**Audience:** maintainers and contributing study groups

**Author guide:** step-by-step bundle preparation and usage (web UI + CLI) is in [`plugins/README.md`](../plugins/README.md).

## 1. Goal

Let a collaborating study hot-plug a *new* prime-editing model into the running
system — without editing core source — by submitting a self-contained **plugin
bundle**:

1. a **data-conversion script** (standardized schema → the model's native input
  columns), consumed by PE Database;
2. a **model wrapper** implementing `BasePEModel` (train / evaluate / predict),
  consumed by PE Ensemble;
3. (optional) **pretrained weights**.

Once registered, the new model behaves like a built-in: it appears in the
`format=` export options, in the Train and Benchmark dropdowns, and in
`GET /models` — so users can train and evaluate models that are not in the
system today.

## 2. Trust model & operating assumptions

Decisions taken for this design:

- **Trusted collaborators, run locally.** The service is typically run on a
researcher's own machine or a lab server. Submitters are vetted people, not
the anonymous public.
- **Correctness-gated submission.** A plugin is only *activated* after it passes
an automated **correctness test harness** (interface compliance + round-trip
conversion + train/eval smoke tests). Submission ≠ activation.
- **In-process execution is acceptable.** Plugin Python is imported into the
service process, exactly like the existing vendor model code already loaded
via `sys.path`. We do **not** build container/subprocess sandboxing in this
design (see §12 for the residual risk and an optional hardening path).

This keeps the build pragmatic while the correctness gate prevents broken
plugins from ever becoming selectable.

## 3. Why this needs a design (current constraints)

Adding a model today requires editing ~9 hardcoded locations that must agree:

**PE Ensemble**


| Location                             | Constant / code                             |
| ------------------------------------ | ------------------------------------------- |
| `app/models/model_factory.py:13`     | `ModelFactory._models` dict                 |
| `app/training/config.py:7`           | `SUPPORTED_MODELS`                          |
| `app/training/config.py:9`           | `MODEL_FORMAT`                              |
| `app/models/weights_registry.py:22`  | `MODEL_NAMES`                               |
| `app/models/weights_registry.py:304` | per-model `register_trained_model` branches |
| `app/main.py` `GET /models`          | static JSON catalog                         |
| `app/training/model_architecture.py` | per-model hyperparameter mapping            |


**PE Database**


| Location                    | Constant / code                                                           |
| --------------------------- | ------------------------------------------------------------------------- |
| `app/utils/convert_data.py` | `standardized_to_*_dataframe()` functions                                 |
| `app/converter.py:123`      | if/elif format dispatch                                                   |
| `app/formatted_cache.py:15` | `FORMATTED_MODEL_FORMATS` frozenset                                       |
| `app/main.py:219`           | `Literal["std","oped","deepprime","pridict","pridict2"]` on `/api/filter` |


Two enabling pieces already exist and are unused:

- `ModelFactory.register_model(name, cls)` (`model_factory.py:52`) — validates
`issubclass(cls, BasePEModel)` and inserts into `_models`. **Never called.**
- `BasePEModel` ABC (`packages/pe-common/pe_common/model_interface.py`) — the
contract all wrappers already satisfy.

The runtime dispatch (`runner.py`, `device_scheduler.py`) is already generic —
it routes purely through `ModelFactory.create_model(name)` and
`weights_registry`. So the runners need **no changes**; the work is removing the
static allowlists and adding a loader + upload/validation surface.

## 4. The plugin concept

A plugin is a directory identified by a unique `name`, discovered at startup
from a **shared** plugins root that both services mount (chosen over per-service
copies: simpler, single source of truth, no cross-service file shipping):

```
plugins/                         # repo root; overridable via PLUGINS_ROOT
  <name>/
    manifest.yaml                # the contract (see §5)
    convert.py                   # standardized -> native columns (PE-DB)
    wrapper.py                   # BasePEModel subclass (PE-Ensemble)
    weights/                     # optional pretrained weights
      <weight_id>/...
    .state.json                  # managed: pending | active | rejected, hashes, validation report
```

Lifecycle states in `.state.json`:

- `pending` — uploaded, not yet validated; **not** selectable anywhere.
- `active` — passed the correctness gate; registered into both services.
- `rejected` — failed validation; report retained for the submitter.

Both services scan `PLUGINS_ROOT` on startup and register only `active`
plugins. PE Database imports `convert.py`; PE Ensemble imports `wrapper.py`.
Weights are imported into the existing weight registry.

`PLUGINS_ROOT` defaults to `<repo>/plugins` and is overridable by env var (same
pattern as `WEIGHTS_ROOT`, `TRAINING_JOBS_ROOT`).

## 5. Manifest contract (`manifest.yaml`)

```yaml
name: my_model                 # unique slug; lowercased; [a-z0-9_]+
version: 0.1.0
display_name: "My New PE Model"
authors: ["Lab X"]
description: "One-line summary shown in the UI."

# --- PE Database: data conversion ---
format:
  module: convert.py
  entrypoint: convert          # convert(std_df: pd.DataFrame) -> pd.DataFrame
  required_std_columns:        # subset of the standardized schema it depends on
    - wt_sequence
    - mut_sequence
    - protospacer_location_l
    - protospacer_location_r
    - pbs_location_l
    - pbs_location_r
    - rtt_location_l
    - rtt_location_r
  output_columns:              # declared native columns (validated post-convert)
    - MyInputSeq
    - PBSlen
    - RTlen
  label_column: Efficiency     # column holding the target during training/eval

# --- PE Ensemble: model code ---
model:
  module: wrapper.py
  class: MyModelWrapper        # must subclass BasePEModel
  pe_db_format: my_model       # which format= to request training/eval data in
                               # (usually == name; may alias an existing format)
  weight_format: my_model_state_dict
  constructor_kwargs:          # optional kwargs passed to create_model
    - {name: wsize, type: int, default: 20}
  hyperparameters:             # self-describing -> drives the Train UI form
    - {name: epochs,     type: int,   default: 10,    min: 1,   max: 200}
    - {name: lr,         type: float, default: 0.001, min: 1e-6, max: 1.0}
    - {name: batch_size, type: int,   default: 32}

# --- optional pretrained weights shipped with the plugin ---
weights:
  - id: my_model_base
    files: [weights/my_model_base/weights.pt]
    notes: "Author-provided pretrained checkpoint."
```

Notes:

- `format.pe_db_format` decouples "model name" from "data format", mirroring how
`MODEL_FORMAT` maps `deepprime → deepprime` today and how `pridict2` reuses the
`pridict` converter. A plugin may set `pe_db_format` to an existing format
(e.g. `deepprime`) and omit `convert.py` entirely if its model consumes an
existing native format.
- `hyperparameters` replaces the per-model branching in `model_architecture.py`
with a declarative schema the UI renders generically.

## 6. The three artifacts & their contracts

### 6.1 Conversion script (`convert.py`) — PE Database

```python
import pandas as pd

def convert(std_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized schema -> this model's native input columns.

    Contract:
      * Input rows follow the standardized schema (see STANDARDIZED_REQUIRED_COLUMNS).
      * MUST return a DataFrame with the SAME number of rows in the SAME order
        (the filter API aligns by index: repository.py subsets converted.loc[split_std.index]).
      * MUST include manifest.format.output_columns.
      * SHOULD be deterministic and side-effect free.
      * MAY accept an optional progress_callback(done, total) kwarg.
    """
    out = pd.DataFrame(index=std_df.index)
    out["MyInputSeq"] = std_df["wt_sequence"]
    # ... derive native columns ...
    return out
```

This is the same shape as the existing `standardized_to_*_dataframe()` functions
in `app/utils/convert_data.py`; row-order preservation is the one hard rule
(because `/api/filter` converts the full datasheet then subsets by index at
`repository.py:430`).

### 6.2 Model wrapper (`wrapper.py`) — PE Ensemble

Subclass `BasePEModel` and implement its six abstract methods plus two
conventional helpers already used across built-ins:

```python
from pe_common.model_interface import BasePEModel

class MyModelWrapper(BasePEModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(model_name="my_model", device=device)

    # --- required by the ABC ---
    def load_model(self, model_path): ...
    def prepare_data(self, df, **kwargs): ...
    def predict(self, data, batch_size=32): ...
    def train(self, train_data, val_data=None, hyperparameters=None): ...
    def evaluate(self, test_data, weights): ...
    def save_model(self, model_path): ...

    # --- conventional (used by main.py / weights flows) ---
    def load_weights_by_name(self, name): ...
    @staticmethod
    def list_available_weights(): ...

    # --- NEW generic registry hook (added to BasePEModel, see §7.1) ---
    def save_to_registry(self, dest_dir) -> str:
        """Persist trained artifacts under dest_dir; return weight_format string."""
```

### 6.3 Weights (optional)

Shipped under `weights/<id>/` and imported via the existing
`weights_registry.register_from_directory(...)`, tagged with the plugin's
`weight_format`. After training, new checkpoints are persisted through the new
`save_to_registry` hook rather than the current 3-way if/else.

## 7. Phase 1 — Registry refactor (no behavior change)

Pure refactor that converts every hardcoded allowlist into a registry, covered
by existing tests (`test_conversions.py`, `test_model_wrappers.py`,
`test_formatted_cache.py`). Adding a 4th *built-in* model becomes trivial and
the plugin loader (Phase 2) gets clean seams to register into.

### 7.1 PE Ensemble

- `**BasePEModel`** (`pe_common/model_interface.py`): add an optional concrete
method `save_to_registry(self, dest_dir) -> str`. Default raises
`NotImplementedError`; built-ins override (DeepPrime → `save_model(dest)`,
OPED → `save_model(dest/"weights.pt")`, PRIDICT2 → copy `model_statedict/` +
`config/`). This eliminates the per-model branch in
`weights_registry.register_trained_model` (`weights_registry.py:304`).
- **Model registry module** (`app/models/registry.py`, new): wraps
`ModelFactory` plus per-model metadata `{pe_db_format, weight_format, hyperparameters, display_name, source: "builtin"|"plugin"}`. Seed it with the
three built-ins.
- Derive at runtime (no more module constants):
  - `SUPPORTED_MODELS` → `registry.names()`
  - `MODEL_FORMAT` → `{name: meta.pe_db_format}`
  - `MODEL_NAMES` (weights) → `registry.names()`
  - `GET /models` → built from the registry instead of static JSON.
- `model_architecture.py` reads `meta.hyperparameters` instead of branching.

### 7.2 PE Database

- `**app/format_registry.py`** (new):

```python
from typing import Callable
import pandas as pd

FORMAT_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}

def register_format(name: str, fn: Callable[..., pd.DataFrame]) -> None:
    FORMAT_REGISTRY[name.lower()] = fn

def get_format(name: str) -> Callable[..., pd.DataFrame]:
    return FORMAT_REGISTRY[name.lower()]

def known_formats() -> frozenset[str]:
    return frozenset({"std", *FORMAT_REGISTRY})
```

  Seed with `deepprime`, `pridict`, `pridict2`, `oped` (the latter two share the
  existing functions).

- Replace the if/elif in `converter.py:123` with `get_format(target_format)(df, ...)`.
- `formatted_cache.FORMATTED_MODEL_FORMATS` → derived from `FORMAT_REGISTRY` keys.
- `/api/filter` `format_` param: drop the `Literal`; validate against
`known_formats()` at runtime, returning HTTP 400 for unknown formats. (OpenAPI
loses the static enum; acceptable since formats are now dynamic. The
`/api/filter` response and split logic are unchanged.)

**Exit criteria:** all existing tests green; `GET /models` and `/api/filter`
behave identically; no manifest/loader yet.

## 8. Phase 2 — Plugin loader & runtime registration

- `**PLUGINS_ROOT`** resolution helper (both services), default `<repo>/plugins`.
- **Shared manifest parser** in `pe-common` (`pe_common/plugins.py`): load +
schema-validate `manifest.yaml`, compute file hashes, read `.state.json`.
- **PE Database loader** (startup, after `initialize_database`): for each
`active` plugin with a `format` block, import `convert.py`, wrap its
`entrypoint` (enforce row-count/index + `output_columns`), and
`register_format(name, wrapped_fn)`. Extend `clear_formatted_cache()` to cover
plugin format names.
- **PE Ensemble loader** (startup): for each `active` plugin, import
`wrapper.py`, `ModelFactory.register_model(name, cls)`, add metadata to the
model registry, and `register_from_directory` any shipped weights.
- **Idempotent + safe:** a plugin that throws on import is logged and skipped (it
stays `active` in state but is quarantined for the session) — it can never
crash service startup.

**Exit criteria:** dropping a hand-made `active` plugin dir into `plugins/` and
restarting makes the model show up in `format=`, `GET /models`, Train, and
Benchmark — with zero source edits.

## 9. Phase 3 — Correctness gate (validation harness)

This is the mechanism behind "if their code passes the correctness test then
they can submit." A plugin is promoted `pending → active` **only** if the
harness passes; otherwise `pending → rejected` with a structured report.

`pe-common/plugin_validation.py` runs ordered checks:

1. **Manifest schema** — required fields, slug format, name uniqueness.
2. **Import safety** — `convert.py` and `wrapper.py` import without error.
3. **Interface compliance** — `issubclass(cls, BasePEModel)`; all six abstract
  methods overridden; `save_to_registry` present.
4. **Conversion round-trip** — run `convert()` on a bundled tiny standardized
  fixture (reuse `testdata/`); assert row count + index preserved and all
   `output_columns` present and non-empty.
5. **Train smoke test** — `create_model(name)` then `train()` for 1 epoch on a
  handful of converted rows on CPU; assert it returns metrics and
   `save_to_registry` writes a loadable artifact.
6. **Eval smoke test** — `evaluate()` against the just-saved weights returns the
  expected metric keys (e.g. Spearman/Pearson) as finite floats.
7. **Predict smoke test** — `predict()` returns one float per input row.

Each check yields `{id, passed, detail, duration}`; the aggregate report is
stored in `.state.json` and surfaced in the UI. Smoke tests run on CPU with a
strict timeout. Promotion is automatic on all-pass (configurable to require a
manual maintainer click — see §11).

## 10. Phase 4 — Upload & management API (PE Ensemble)

PE Ensemble owns plugin management (it already proxies PE-DB and is the
model-facing service). New endpoints, multipart where files are involved:


| Method   | Path                              | Purpose                                                                        |
| -------- | --------------------------------- | ------------------------------------------------------------------------------ |
| `POST`   | `/models/plugins`                 | Upload a bundle (zip or multi-file). Writes to `plugins/<name>/` as `pending`. |
| `POST`   | `/models/plugins/{name}/validate` | Run the §9 harness; update `.state.json`; return report.                       |
| `POST`   | `/models/plugins/{name}/activate` | Promote to `active` (only if last validation passed); register live.           |
| `GET`    | `/models/plugins`                 | List plugins with state + last report summary.                                 |
| `GET`    | `/models/plugins/{name}`          | Full manifest + validation report.                                             |
| `DELETE` | `/models/plugins/{name}`          | Deactivate/remove; unregister from factory + format registry.                  |


Implementation notes:

- Reuse the filesystem-job pattern (`jobs/`, `eval_jobs/`): validation can run
through the existing device scheduler on CPU so long imports/training don't
block the event loop, streaming a `validation.log` like training jobs.
- For the `format` half, PE Ensemble writes the bundle to the **shared**
`PLUGINS_ROOT`; PE Database picks up `convert.py` from the same directory.
Activation triggers a lightweight "reload plugins" call to PE Database
(`POST /api/plugins/reload`) so it registers the new `format=` without a
restart. (If a shared mount is unavailable in a given deployment, the upload
endpoint POSTs `convert.py` to a PE-DB ingest endpoint instead.)
- Both axios clients hardcode `Content-Type: application/json`; upload calls set
`multipart/form-data` per-request.

## 11. Phase 5 — PE-Hub "Add New Model" page

Frontend is state-routed (no react-router). Wiring:

- `HubNavbar.tsx:8`: add `'add-model'` to the `EnsemblePage` union.
- `HubNavbar.tsx` `ensembleNav`: add `{ id: 'add-model', label: 'Add Model' }`.
- `App.tsx renderContent()`: add
`{ensemblePage === 'add-model' && <AddModelPage />}` inside the existing
`ServiceGate serviceId="pe-ensemble"`.
- New `src/apps/ensemble/pages/AddModelPage.tsx`, modeled on `TrainingPage`
(`Card` sections + `useMutation` + `ErrorAlert`).

Page flow:

1. **Metadata form** — name, version, display name, description, authors,
  declared hyperparameters (repeatable rows).
2. **File pickers** — `convert.py`, `wrapper.py`, optional weight files
  (`<input type="file">`; the first uploads in the app — adds a `multipart`
   path to `api.ts`). Optionally generate `manifest.yaml` from the form so
   submitters don't hand-write it.
3. **Upload** → `POST /models/plugins` (state `pending`).
4. **Validate** → `POST /models/plugins/{name}/validate`; render the per-check
  report (green/red list with `detail`), tailing `validation.log`.
5. **Activate** — enabled only when all checks pass; calls
  `/models/plugins/{name}/activate`. On success the model auto-appears in
   Train/Benchmark because those pages already call `api.listModels()`.
6. **Manage** — list existing plugins with state badges, view report, delete.

New `api.ts` methods: `uploadPlugin`, `validatePlugin`, `activatePlugin`,
`listPlugins`, `getPlugin`, `deletePlugin`.

## 12. Security considerations

- **Arbitrary code execution is inherent.** Activating a plugin runs its Python
in the service process with full data/GPU access. This is the same trust level
as the existing vendor models. Acceptable under the trusted-collaborator,
local-run assumption (§2).
- **The correctness gate is also a tripwire**, not just a quality check: a
plugin that fails import/interface/smoke tests never becomes `active`.
- **Uploads land as `pending`** and are never auto-imported into the live
registries until validation+activation.
- **Path/name hardening:** slugify `name`; reject path traversal in `files`;
cap upload size; verify file hashes recorded in `.state.json`.
- **Optional future hardening (out of scope here):** run validation/training in
a subprocess with restricted cwd and `PYTHONPATH`, resource limits, and
no-network; or container-per-plugin. The design keeps this behind the loader
seam so it can be added without touching the UI or runners.

## 13. Persistence & data model

- Plugin truth lives on the filesystem (`plugins/<name>/` + `.state.json`),
matching the existing weights/jobs "filesystem registry" convention — no new
DB tables required.
- Trained weights continue to live in `weights/<model>/<id>/` with a manifest;
plugin-produced formats are tagged via `weight_format`.
- `registry.json` (weights index) gains plugin model names automatically once
`MODEL_NAMES` is registry-derived.

## 14. Testing strategy

- **Phase 1:** existing suites must stay green; add tests asserting
`GET /models` and `format=` come from the registries.
- **Plugin fixture:** add a tiny reference plugin under
`testdata/plugins/dummy_model/` (trivial linear model) used by:
  - a loader test (scan → register → appears in factory + format registry),
  - a validation-harness test (passes all checks),
  - a negative test (a deliberately broken plugin → `rejected` with the right
  failing check ids).
- **End-to-end:** upload → validate → activate → train → evaluate the dummy
plugin through the API.

## 15. File-by-file change checklist

**pe-common**

- `pe_common/model_interface.py` — add `save_to_registry` hook.
- `pe_common/plugins.py` (new) — manifest parse/validate + `.state.json` IO.
- `pe_common/plugin_validation.py` (new) — correctness harness.

**pe-db (`services/pe-db`)**

- `app/format_registry.py` (new).
- `app/converter.py` — use registry dispatch (replace `:123` if/elif).
- `app/formatted_cache.py` — derive formats from registry.
- `app/main.py` — drop `Literal` on `/api/filter`; runtime validation;
add `POST /api/plugins/reload`.
- `app/plugin_loader.py` (new) — startup scan + `register_format`.

**pe-ensemble (`services/pe-ensemble`)**

- `app/models/registry.py` (new) — model + metadata registry.
- `app/models/model_factory.py` — back `_models` with the registry.
- `app/training/config.py` — derive `SUPPORTED_MODELS`/`MODEL_FORMAT`.
- `app/models/weights_registry.py` — derive `MODEL_NAMES`; use
`save_to_registry` (replace `:304` branch).
- `app/training/model_architecture.py` — read declared hyperparameters.
- `app/main.py` — registry-driven `GET /models`; plugin CRUD endpoints.
- `app/plugin_loader.py` (new) — startup scan + register wrapper/weights.

**pe-hub**

- `src/components/HubNavbar.tsx` — `EnsemblePage` union + nav item.
- `src/App.tsx` — render branch.
- `src/apps/ensemble/pages/AddModelPage.tsx` (new).
- `src/apps/ensemble/services/api.ts` — plugin endpoints + multipart support.

**docs/tests**

- `testdata/plugins/dummy_model/` reference plugin + tests above.
- Update `README.md` / `services/*/README.md` with the plugin contract.

## 16. Open questions / risks

- **Auto-activate vs manual gate:** activate automatically on all-pass, or
require a maintainer click? (Recommend manual click initially.)
  - I prefer auto
- **Versioning:** allow multiple versions of one `name`, or latest-wins?
  - Last-wins make sense
- **Heavy deps:** if a plugin needs packages not installed in the env, import
fails the gate. Do we allow a declared `requirements.txt` and `pip install` at
activation, or require deps pre-installed? (Recommend pre-installed for local
runs; revisit later.)
  - deps should be pre-installed, since it is indeed local runs
  - This may need to be updated if we did decide to host pe-db online
- **PRIDICT2-style special cases:** multi-head weight expansion is bespoke in  
`main.py`; keep plugins to the generic registry path and avoid new one-offs.
- **Shared mount assumption:** the shared `PLUGINS_ROOT` is simplest locally; a
split deployment would use the PE-DB ingest fallback in §10.

