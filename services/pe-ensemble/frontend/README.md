# PE Ensemble Frontend

Modern React + TypeScript frontend for the PE Ensemble API service.

## Features

- 🎨 Modern UI with Tailwind CSS
- ⚡ Fast development with Vite
- 🔄 Real-time API integration with react-query
- 📊 Data visualization with Recharts
- 🎯 Type-safe with TypeScript
- 🎭 Component-based architecture

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

```bash
npm install
```

### Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

### Preview

```bash
npm run preview
```

## Project Structure

```
src/
├── components/        # Reusable UI components
├── pages/            # Page components
├── services/         # API integration
├── App.tsx           # Main application
└── main.tsx          # Entry point
```

## Environment Variables

Create a `.env` file based on `.env.example`:

```env
VITE_API_URL=http://localhost:8001
```

## API Integration

All API calls are centralized in `src/services/api.ts`. The service includes:

- Model listing
- Predictions
- Training jobs
- Ensemble predictions

## Technologies

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Query** - Data fetching
- **Axios** - HTTP client
- **Recharts** - Charting library
- **Lucide Icons** - Icon library

## License

MIT
