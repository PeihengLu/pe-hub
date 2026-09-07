# PE Hub

Unified React frontend for the PE Database and PE Ensemble APIs.

## Features

- Single site with **Database** and **Ensemble** sections
- Live health checks per backend (nav status dots)
- Offline screens with copy-paste startup commands when a service is down

### Database section

- **Catalog** — browse studies, datasets, datasheets, and scaffolds; view statistics
- **Export** — multi-filter builder; download CSV in standardized or model-specific formats with optional train/val/test splits

### Ensemble section

- **Benchmark** — evaluate models on PE-DB test splits (async job queue)
- **Design** — pegRNA design placeholder (coming soon)
- **Train** — submit training jobs, pick compute device, stream logs, view job history
- **Ensemble** — combine model outputs
- **Docs** — inline API reference (full docs at `/docs` on each backend)
- **Add model** — upload, validate and activate a model plugin (see
  [`plugins/README.md`](../plugins/README.md))

## Getting started

```bash
cd pe-hub
npm install
cp .env.example .env   # optional — defaults match local backends
npm run dev
```

Open http://localhost:5173

### Run everything (recommended)

From the repository root:

```bash
./scripts/start-all.sh --install   # first time
./scripts/start-all.sh
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_PE_DB_URL` | `http://localhost:8000` | PE Database API |
| `VITE_ENSEMBLE_API_URL` | `http://localhost:8001` | PE Ensemble API |

## Project layout

```
src/
├── App.tsx                 # Top-level section switch (no react-router)
├── apps/
│   ├── database/
│   │   ├── pages/          # CatalogPage, ExportPage
│   │   ├── services/       # peDbApi client
│   │   └── components/     # ExportFilterBuilder, StatisticsCharts
│   └── ensemble/
│       ├── pages/          # Benchmark, Design, Training, Ensemble,
│       │                   #   Documentation, AddModel
│       ├── components/     # ComputeJobList, TrainingHyperparametersPanel, …
│       ├── services/       # api client (train, tune, evaluate, devices)
│       ├── utils/          # Request builders, job status/sort, result export
│       └── config/         # Split params, model formats, scheduler defaults
├── components/             # Card, ServiceGate, HubNavbar, …
├── config/                 # Service URLs from VITE_* and startup hints
├── context/                # ServiceHealthProvider (health gating)
└── pages/                  # HomePage
```

### How the UI is structured

- **Routing is state, not URLs.** `App.tsx` switches on a `HubSection`
  (`home`, `database`, `ensemble`, `add-model`); the ensemble section switches
  again between `benchmark`, `design`, `train`, `ensemble` and `docs`. There is
  no react-router, so deep links are not available.
- **Backends are gated on health.** `ServiceHealthProvider` polls each backend
  and `ServiceGate` renders an offline screen with copy-paste startup commands
  instead of letting requests fail.
- **Requests are built in `utils/`, not in pages.** `trainingRequest.ts`,
  `benchmarkRequest.ts` and `ensembleRequest.ts` translate form state into API
  payloads, so a schema change touches one file per workflow.
- **Async jobs share one polling pattern.** `utils/jobStatus.ts` decides the
  refetch interval from job status and sorts job lists newest-first;
  `ComputeJobList` renders training, tuning, evaluation and ensemble jobs alike.

The legacy standalone frontend under `services/pe-ensemble/frontend/` has been
retired; all UI development happens here.

## License

MIT
