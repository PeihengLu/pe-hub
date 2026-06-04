import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

const srcDir = fileURLToPath(new URL('./src', import.meta.url))

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const peDbTarget = env.VITE_PE_DB_URL || 'http://localhost:8000'
  const ensembleTarget = env.VITE_ENSEMBLE_API_URL || 'http://localhost:8001'

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': srcDir,
        '@components': fileURLToPath(new URL('./src/components', import.meta.url)),
        '@config': fileURLToPath(new URL('./src/config', import.meta.url)),
        '@context': fileURLToPath(new URL('./src/context', import.meta.url)),
        '@apps': fileURLToPath(new URL('./src/apps', import.meta.url)),
      },
    },
    server: {
      port: 5173,
      proxy: {
        '/proxy/pe-db': {
          target: peDbTarget,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/proxy\/pe-db/, ''),
        },
        '/proxy/ensemble': {
          target: ensembleTarget,
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/proxy\/ensemble/, ''),
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
    },
  }
})
