/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // MOSAICS Studio–inspired oxblood + gold (palette only)
        canvas: '#1a0c0e',
        surface: {
          DEFAULT: '#271416',
          raised: '#3a1f24',
        },
        ink: {
          DEFAULT: '#f3ebe0',
          muted: '#c4b4a4',
          faint: '#8f8175',
        },
        line: {
          DEFAULT: '#4d3338',
          strong: '#6a454c',
        },
        primary: {
          50: '#3a2e14',
          100: '#5c4820',
          200: '#8a6c2e',
          500: '#d4af37',
          600: '#c9a227',
          700: '#a8841f',
        },
      },
    },
  },
  plugins: [],
}
