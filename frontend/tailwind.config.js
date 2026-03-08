/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neural-bg': '#0a0a0f',
        'electric-blue': '#00d4ff',
        'glowing-purple': '#a855f7',
        'neon-green': '#00ff88',
        'neon-red': '#ff0040',
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
        'sans': ['Inter', 'Roboto', 'sans-serif'],
      },
      animation: {
        'pulse-glow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'neural-pulse': 'neuralPulse 1.5s ease-in-out infinite',
      },
      keyframes: {
        neuralPulse: {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.1)' },
        }
      },
      backdropBlur: {
        'xs': '2px',
      }
    },
  },
  plugins: [],
}
