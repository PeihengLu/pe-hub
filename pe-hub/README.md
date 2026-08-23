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
├── apps/
│   ├── database/
│   │   ├── pages/          # CatalogPage, ExportPage
│   │   ├── services/       # peDbApi client
│   │   └── components/     # ExportFilterBuilder
│   └── ensemble/
│       ├── pages/          # Prediction, Training, Ensemble, Documentation
│       ├── services/       # api client (train, evaluate, devices)
│       └── config/         # split param defaults
├── components/             # Card, ServiceGate, HubNavbar, …
├── config/                 # Service URLs and startup hints
├── context/                # ServiceHealthProvider
└── pages/                  # HomePage
```

The legacy standalone frontend under `services/pe-ensemble/frontend/` has been
retired; all UI development happens here.

## License

MIT
