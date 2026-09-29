import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['5173-i5aeut0vm5l6nxkojrge0-cc5a63a8.sg2.manus.computer'],
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
});
