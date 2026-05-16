import { defineConfig } from 'vite';
import { resolve, join } from 'path';
import { existsSync, readFileSync } from 'fs';
import type { Plugin } from 'vite';

function serveDataDir(): Plugin {
  const dataDir = resolve(__dirname, 'data/lake_boundary_dataset');
  return {
    name: 'serve-data-dir',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url || '';
        if (!url.startsWith('/export/')) { next(); return; }
        const relPath = url.replace('/export/', '').split('?')[0];
        const filePath = join(dataDir, relPath);
        if (existsSync(filePath)) {
          const ext = (relPath.split('.').pop() || '').toLowerCase();
          const mime = ext === 'json' ? 'application/json' :
                       ext === 'png' ? 'image/png' : 'text/plain';
          res.setHeader('Content-Type', mime);
          res.setHeader('Access-Control-Allow-Origin', '*');
          res.statusCode = 200;
          res.end(ext === 'png' ? readFileSync(filePath) : readFileSync(filePath, 'utf-8'));
          return;
        }
        res.statusCode = 404; res.end('Not Found');
      });
    },
  };
}

export default defineConfig({
  root: '.',
  plugins: [serveDataDir()],
  server: { host: '0.0.0.0', port: 3000 },
  build: {
    outDir: 'dist',
    target: 'es2020',
    rollupOptions: {
      input: {
        index: resolve(__dirname, "index.html"),
        boundary: resolve(__dirname, 'boundary.html'),
        benchmark: resolve(__dirname, 'benchmark.html'),
      },
    },
  },
});
