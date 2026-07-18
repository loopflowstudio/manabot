# Isolate Playwright ports across worktrees

`frontend/playwright.config.ts` sets `reuseExistingServer: true`. Before treating
browser results as evidence, check whether the default ports belong to this
worktree. A sibling service on 8011 can satisfy the readiness URL while serving
different code.

Use explicit unused ports for task validation, for example:

```bash
ETUDE_API_PORT=8123 ETUDE_FRONTEND_PORT=5293 \
  npm --prefix frontend run test:e2e -- testing-house.spec.ts
```

Do not terminate a sibling worktree's server merely to reclaim the defaults.
