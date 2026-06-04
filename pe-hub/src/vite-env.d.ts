/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PE_DB_URL?: string
  readonly VITE_ENSEMBLE_API_URL?: string
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
