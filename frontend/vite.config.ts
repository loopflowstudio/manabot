import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

const apiPort = process.env.ETUDE_API_PORT ?? '8000';
const proxy = {
  '/api': {
    target: `http://localhost:${apiPort}`,
  },
  '/ws': {
    target: `ws://localhost:${apiPort}`,
    ws: true,
  },
};

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: { proxy },
  preview: { proxy },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
