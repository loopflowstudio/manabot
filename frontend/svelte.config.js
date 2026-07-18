import adapter from '@sveltejs/adapter-auto';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// ETUDE_STATIC_BUILD=1 packages the client build as a static SPA — build/
// with an index.html fallback — for the hosted play deployment, where the
// experience server serves it from the same origin as /ws and /api. The
// inline adapter keeps adapter-auto (and the dev workflow) untouched.
const staticSpaAdapter = () => ({
  name: 'etude-static-spa',
  async adapt(builder) {
    const out = 'build';
    builder.rimraf(out);
    builder.writeClient(out);
    await builder.generateFallback(`${out}/index.html`);
  }
});

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: process.env.ETUDE_STATIC_BUILD ? staticSpaAdapter() : adapter()
  }
};

export default config;
