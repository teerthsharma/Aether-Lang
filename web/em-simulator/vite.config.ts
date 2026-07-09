import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/Aether-Lang/',
  plugins: [react()],
  build: {
    target: 'es2020',
    sourcemap: true,
  },
});
