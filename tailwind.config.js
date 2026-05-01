/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./static/**/*.{html,js}",
    "./app/**/*.py",
  ],
  theme: {
    extend: {
      colors: {
        'glass': 'rgba(255, 255, 255, 0.05)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      backdropBlur: {
        'xs': '2px',
      }
    },
  },
  plugins: [],
}
