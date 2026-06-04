# PE Hub

Unified React frontend for the PE Database and PE Ensemble APIs.

## Features

- Single site with **Database** and **Ensemble** sections
- Live health checks per backend (nav status dots)
- Offline screens with copy-paste startup commands when a service is down
- PE Database catalog browser (studies, datasets, datasheets, statistics)
- PE Ensemble prediction UI (moved from `services/pe-ensemble/frontend`)

## Getting Started

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
./start-all.sh --install   # first time
./start-all.sh
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
│   ├── database/     # Catalog UI + peDbApi client
│   └── ensemble/     # Prediction UI + api client
├── components/       # Shared UI (Card, ServiceGate, HubNavbar, …)
├── config/           # Service URLs and startup hints
├── context/          # ServiceHealthProvider
└── pages/            # HomePage
```

## License

MIT
