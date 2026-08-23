import type { ReactNode } from 'react'
import { BookOpen, Terminal } from 'lucide-react'
import Card from '@components/Card'

const preClass =
  'rounded-md bg-slate-900 text-slate-100 p-3 text-xs font-mono overflow-x-auto whitespace-pre'

function MethodExample({
  name,
  description,
  code,
}: {
  name: string
  description: string
  code: string
}) {
  return (
    <div className="space-y-1">
      <p className="font-medium text-slate-800">
        <code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded font-mono text-slate-900">
          {name}
        </code>
      </p>
      <p className="text-xs text-slate-500">{description}</p>
      <pre className={preClass}>{code}</pre>
    </div>
  )
}

function GuideSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details
      className="group rounded-lg border border-slate-200 bg-slate-50/80 open:bg-white"
      open={defaultOpen}
    >
      <summary className="cursor-pointer list-none px-4 py-3 text-sm font-semibold text-slate-800 [&::-webkit-details-marker]:hidden">
        <span className="inline-flex items-center gap-2">
          <span className="text-slate-400 group-open:rotate-90 transition-transform">▸</span>
          {title}
        </span>
      </summary>
      <div className="border-t border-slate-200 px-4 py-3 text-sm text-slate-600 space-y-3">
        {children}
      </div>
    </details>
  )
}

export default function AddModelInstructions() {
  return (
    <Card title="How to prepare your plugin">
      <p className="text-sm text-slate-600 mb-4">
        Prepare a bundle locally (<code className="text-xs bg-slate-100 px-1 rounded">plugins/_template/</code>
        ), zip it, and upload below — <strong className="font-medium text-slate-800">no form fields required</strong>.
        Once <strong className="font-medium text-slate-800">activated</strong>, the model works in Train,
        Benchmark, and the training CLI. Guides:{' '}
        <code className="text-xs bg-slate-100 px-1 rounded">plugins/README.md</code>,{' '}
        <code className="text-xs bg-slate-100 px-1 rounded">docs/plugin-author-llm-prompt.md</code>.
      </p>

      <div className="flex flex-col gap-2">
        <GuideSection title="Checklist before upload" defaultOpen>
          <ul className="list-disc pl-5 space-y-1.5">
            <li>
              <strong className="text-slate-800">Name</strong> — lowercase slug{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">[a-z0-9_]+</code>, unique, not
              deepprime / oped / pridict2
            </li>
            <li>
              <strong className="text-slate-800">manifest.yaml</strong> — valid;{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">name</code> matches the directory
            </li>
            <li>
              <strong className="text-slate-800">convert.py</strong> — exposes the manifest entrypoint
              (default <code className="text-xs bg-slate-100 px-1 rounded">convert</code>); same row
              count and index as input
            </li>
            <li>
              <strong className="text-slate-800">wrapper.py</strong> — class subclasses{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">BasePEModel</code>; implements
              train, evaluate, predict, and <code className="text-xs bg-slate-100 px-1 rounded">save_to_registry</code>
            </li>
            <li>
              <strong className="text-slate-800">Output columns</strong> — every column listed in the
              manifest is present and non-empty after convert
            </li>
            <li>
              <strong className="text-slate-800">Dependencies</strong> — all Python packages pre-installed
              in the service environment
            </li>
            <li>
              <strong className="text-slate-800">Weights</strong> (optional) — under{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">weights/&lt;id&gt;/</code>; ids match
              manifest entries
            </li>
          </ul>
          <p className="text-xs text-slate-500">
            Reference bundle: <code className="bg-slate-100 px-1 rounded">testdata/plugins/dummy_model/</code>
          </p>
        </GuideSection>

        <GuideSection title="Upload on Add Model" defaultOpen>
          <p className="mb-2">
            Config is defined in <strong className="text-slate-800">one of two mutually exclusive ways</strong>:
          </p>
          <ul className="list-disc pl-5 space-y-1.5 mb-3">
            <li>
              <strong className="text-slate-800">YAML manifest</strong> — upload a zip or{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">manifest.yaml</code> + scripts.
              All metadata lives in the file; web form fields are not used.
            </li>
            <li>
              <strong className="text-slate-800">Web form</strong> — fill in fields on the page; the
              server builds <code className="text-xs bg-slate-100 px-1 rounded">manifest.yaml</code>.
              Do not upload a manifest file.
            </li>
          </ul>
          <p>
            Zip the directory and upload via <strong className="text-slate-800">YAML → Zip bundle</strong>{' '}
            for the simplest path.
          </p>
        </GuideSection>

        <GuideSection title="Bundle layout">
          <pre className={preClass}>
{`plugins/my_model/
  manifest.yaml
  convert.py
  wrapper.py
  weights/           # optional
    base/
      weights.pt`}
          </pre>
          <p>
            For YAML manifest mode, zip the directory or upload files individually. For web form mode,
            use the form fields instead — do not upload a manifest.
          </p>
        </GuideSection>

        <GuideSection title="convert.py (PE Database)">
          <p>
            Maps standardized PE-DB rows to your model&apos;s native columns. PE-DB calls this when
            training or benchmarking requests <code className="text-xs bg-slate-100 px-1 rounded">format=your_model</code>.
          </p>
          <pre className={preClass}>
{`def convert(std_df):
    out = pd.DataFrame(index=std_df.index)
    out["feature"] = std_df["edit_len"]
    out["Efficiency"] = std_df["editing_efficiency"]
    return out`}
          </pre>
          <ul className="list-disc pl-5 space-y-1">
            <li>Return the same number of rows with the same index order as the input</li>
            <li>Include every column listed in manifest <code className="text-xs bg-slate-100 px-1 rounded">output_columns</code></li>
            <li>Deterministic; no side effects</li>
          </ul>
        </GuideSection>

        <GuideSection title="wrapper.py (PE Ensemble)">
          <p>
            Subclass <code className="text-xs bg-slate-100 px-1 rounded">pe_common.model_interface.BasePEModel</code>.
            Training and benchmark jobs pass DataFrames already converted by your{' '}
            <code className="text-xs bg-slate-100 px-1 rounded">convert.py</code> (native columns only).
          </p>
          <p className="text-xs text-slate-500">
            Full reference: <code className="bg-slate-100 px-1 rounded">testdata/plugins/dummy_model/wrapper.py</code>
          </p>

          <MethodExample
            name="__init__"
            description="Call super with your plugin name (must match manifest). Optional kwargs come from Train UI model-kwargs-json."
            code={`class MyModelWrapper(BasePEModel):
    def __init__(self, device=None, **kwargs):
        super().__init__(model_name="my_model", device=device)
        self.model = None  # your nn.Module, sklearn model, etc.
        # e.g. self.wsize = int(kwargs.get("wsize", 20))`}
          />

          <MethodExample
            name="prepare_data(df, **kwargs)"
            description="prepare_data turns model-format tabular rows into what your model consumes (tensors, tokenized columns, DataLoader, etc.). Not for train/test/val splitting or cross validation, this is handled by PE-DB"
            code={`def prepare_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    required = ["feature", "Efficiency"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df.copy()

# PyTorch example — build a tensor batch:
# x = torch.tensor(df["feature"].values, dtype=torch.float32, device=self.device)
# return x`}
          />

          <MethodExample
            name="predict(data, batch_size=32)"
            description="Return one float prediction per input row, in the same order. Used by evaluate, benchmark jobs, and predict API."
            code={`def predict(self, data: pd.DataFrame, batch_size: int = 32) -> List[float]:
    if "feature" not in data.columns:
        raise ValueError("Expected a 'feature' column")
    # Simple baseline: predictions = inputs (see dummy_model)
    return [float(v) for v in data["feature"].tolist()]

# PyTorch loop example:
# self.model.eval()
# preds = []
# for start in range(0, len(data), batch_size):
#     batch = data.iloc[start : start + batch_size]
#     x = self.prepare_data(batch)
#     with torch.no_grad():
#         out = self.model(x)
#     preds.extend(out.cpu().tolist())
# return preds`}
          />

          <MethodExample
            name="train(train_data, val_data=None, hyperparameters=None)"
            description="Fit on train split; optional val split for early stopping or metrics. hyperparameters dict matches manifest names (epochs, lr, …) plus load_pretrained/weights when fine-tuning."
            code={`def train(
    self,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hp = hyperparameters or {}
    epochs = int(hp.get("epochs", 10))
    lr = float(hp.get("lr", 1e-3))

    if hp.get("load_pretrained"):
        self.load_weights_by_name(str(hp["weights"]))

    train_df = self.prepare_data(train_data)
    # ... training loop ...
    self.is_trained = True

    result: Dict[str, Any] = {"epochs": epochs, "lr": lr}
    if val_data is not None:
        val_preds = self.predict(self.prepare_data(val_data))
        y_true = val_data["Efficiency"].astype(float).tolist()
        result["validation_metrics"] = regression_metrics(y_true, val_preds)
    return result`}
          />

          <MethodExample
            name="evaluate(test_data, weights)"
            description="Load a registered weight set by id, run predict on test_data, return metric dict (pearson, spearman, …). weights is always a registry id, not a file path."
            code={`def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
    from pe_common.training import regression_metrics

    self.load_weights_by_name(weights)
    preds = self.predict(self.prepare_data(test_data))
    y_true = test_data["Efficiency"].astype(float).tolist()
    return regression_metrics(y_true, preds)
    # returns e.g. {"pearson": 0.82, "spearman": 0.79, ...}`}
          />
        </GuideSection>

        <GuideSection title="Weights — how the service resolves paths">
          <p>
            All weight sets live under <code className="text-xs bg-slate-100 px-1 rounded">WEIGHTS_ROOT</code>{' '}
            (default <code className="text-xs bg-slate-100 px-1 rounded">services/pe-ensemble/weights/</code>).
            The service never passes raw paths to Benchmark or Train UI — it passes a{' '}
            <strong className="text-slate-800">weight id</strong>. Your wrapper resolves that id to files on disk.
          </p>
          <pre className={preClass}>
{`WEIGHTS_ROOT/
  my_model/
    base/                          # weight id = "base" (shipped plugin weight)
      manifest.json
      model.pt
    my_model__hek293t__20260614__a1b2c3/   # auto id after training
      manifest.json
      model.pt`}
          </pre>
          <ul className="list-disc pl-5 space-y-1.5 text-sm">
            <li>
              <code className="text-xs bg-slate-100 px-1 rounded">weights_registry.resolve_dir(&quot;my_model&quot;, weight_id)</code>{' '}
              → directory above (requires <code className="text-xs bg-slate-100 px-1 rounded">manifest.json</code>)
            </li>
            <li>
              <code className="text-xs bg-slate-100 px-1 rounded">list_weight_ids</code> /{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">list_available_weights()</code> → ids users
              pick in Benchmark or <code className="text-xs bg-slate-100 px-1 rounded">--pretrained-weights</code>
            </li>
            <li>
              On plugin <strong>activate</strong>, files from{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">plugins/my_model/weights/&lt;id&gt;/</code> are
              copied into <code className="text-xs bg-slate-100 px-1 rounded">WEIGHTS_ROOT/my_model/&lt;id&gt;/</code>
            </li>
            <li>
              After <strong>training</strong>, the runner creates a new directory, calls your{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">save_to_registry(dest_dir)</code>, then writes{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">manifest.json</code> (you do not write the manifest)
            </li>
          </ul>
        </GuideSection>

        <GuideSection title="wrapper.py — load & save weights">
          <p className="text-xs text-slate-500">
            Pick one canonical artifact name per model (e.g. <code className="bg-slate-100 px-1 rounded">model.pt</code>).
            Use the same layout in <code className="bg-slate-100 px-1 rounded">save_to_registry</code> and in{' '}
            <code className="bg-slate-100 px-1 rounded">load_weights_by_name</code>. OPED uses{' '}
            <code className="bg-slate-100 px-1 rounded">weights.pt</code>; dummy_model uses{' '}
            <code className="bg-slate-100 px-1 rounded">weights.txt</code>.
            Validation reloads weights via <code className="bg-slate-100 px-1 rounded">save_to_registry</code> then{' '}
            <code className="bg-slate-100 px-1 rounded">evaluate</code> — keep smoke training fast.
          </p>

          <MethodExample
            name="load_weights_by_name(name) — registry id → files"
            description="Benchmark, evaluate, and fine-tune pass a weight id (not a path). Resolve via weights_registry, then load your artifact(s) from that directory."
            code={`ARTIFACT = "model.pt"  # fixed name you own; document in plugin README

def load_weights_by_name(self, name: str) -> None:
    from app.models import weights_registry

    entry_dir = weights_registry.resolve_dir(self.model_name, name)
    artifact = entry_dir / ARTIFACT
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Expected {ARTIFACT} in weight entry {entry_dir}"
        )
    self.load_model(str(artifact))

# evaluate() and Benchmark always call load_weights_by_name(weights_id)`}
          />

          <MethodExample
            name="load_model(model_path) — filesystem path only"
            description="Low-level loader: a concrete file (or directory for multi-file models like DeepPrime). Called by load_weights_by_name; rarely called with a registry id directly."
            code={`def load_model(self, model_path: str) -> None:
    path = Path(model_path)
    if path.is_dir():
        checkpoint = path / "model.pt"
    else:
        checkpoint = path
    state = torch.load(checkpoint, map_location=self.device)
    self.model.load_state_dict(state)
    self.is_trained = True`}
          />

          <MethodExample
            name="save_to_registry(dest_dir) — after training"
            description="Training runner creates dest_dir, calls this, then writes manifest.json listing files. Return manifest model.weight_format (not the weight id)."
            code={`def save_to_registry(self, dest_dir) -> str:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), dest / "model.pt")
    return "my_model_weights"  # manifest model.weight_format

# Result on disk (manifest added by runner):
# WEIGHTS_ROOT/my_model/my_model__custom__20260614__abc123/
#   model.pt
#   manifest.json`}
          />

          <MethodExample
            name="save_model(model_path) — optional helper"
            description="Save to any path. save_to_registry can delegate here (OPED saves dest_dir/weights.pt this way)."
            code={`def save_model(self, model_path: str) -> None:
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), model_path)`}
          />

          <MethodExample
            name="list_available_weights()"
            description="Return registry ids for this model. Easiest: delegate to weights_registry."
            code={`@staticmethod
def list_available_weights() -> List[str]:
    from app.models import weights_registry

    return weights_registry.list_weight_ids("my_model")`}
          />
        </GuideSection>

        <GuideSection title="Register on this page" defaultOpen>
          <ol className="list-decimal pl-5 space-y-2">
            <li>
              <strong className="text-slate-800">Upload</strong> — fill the form below, upload a prepared{' '}
              <code className="text-xs bg-slate-100 px-1 rounded">manifest.yaml</code> (optional; replaces
              metadata and hyperparameter rows), or use a zip. State
              becomes <span className="text-amber-800">pending</span>.
            </li>
            <li>
              <strong className="text-slate-800">Validate</strong> — runs the harness (manifest, imports,
              convert round-trip, CPU train/eval smoke tests).
            </li>
            <li>
              <strong className="text-slate-800">Activate</strong> — only when all checks pass. State
              becomes <span className="text-emerald-800">active</span>; PE-DB and PE Ensemble reload
              plugins automatically.
            </li>
          </ol>
          <p className="text-xs text-slate-500">
            Pending or rejected plugins are not selectable in Train or Benchmark.
          </p>
        </GuideSection>

        <GuideSection title="After activation — web UI and CLI">
          <div className="flex items-start gap-2">
            <BookOpen className="w-4 h-4 text-primary-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-slate-800">PE Hub</p>
              <p>
                Open <strong>Train</strong> or <strong>Benchmark</strong> under Ensemble — your model
                appears in the dropdown with manifest hyperparameters.
              </p>
            </div>
          </div>
          <div className="flex items-start gap-2 mt-3">
            <Terminal className="w-4 h-4 text-primary-600 shrink-0 mt-0.5" />
            <div className="space-y-2 min-w-0">
              <p className="font-medium text-slate-800">Training CLI</p>
              <p>
                Active plugins load at startup from the same{' '}
                <code className="text-xs bg-slate-100 px-1 rounded">PLUGINS_ROOT</code> (default repo{' '}
                <code className="text-xs bg-slate-100 px-1 rounded">plugins/</code>).
              </p>
              <p className="text-xs text-slate-500">
                CLI always uses in-process PE-DB (no PE-DB HTTP server required):
              </p>
              <pre className={preClass}>
{`peen train \\
  --model my_model \\
  --dataset-name my_run \\
  --dataset library2 \\
  --hyperparameters-json '{"epochs": 5}' \\
  --device cuda:0`}
              </pre>
              <p className="text-xs text-slate-500">
                The FastAPI web service talks to PE-DB over HTTP via{' '}
                <code className="text-xs bg-slate-100 px-1 rounded">PE_DB_URL</code>.
              </p>
            </div>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Env vars: <code className="bg-slate-100 px-1 rounded">PLUGINS_ROOT</code>,{' '}
            <code className="bg-slate-100 px-1 rounded">WEIGHTS_ROOT</code>,{' '}
            <code className="bg-slate-100 px-1 rounded">PE_DB_URL</code> (web service only)
          </p>
        </GuideSection>
      </div>
    </Card>
  )
}
