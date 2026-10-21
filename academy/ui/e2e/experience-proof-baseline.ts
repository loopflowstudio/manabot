import releasePromptMatrix from './release-prompt-matrix.json' with { type: 'json' };

export const REACHABLE_PROMPT_FAMILIES = releasePromptMatrix.action_spaces.reachable;

export const EXPERIENCE_PROOF_BASELINE = {
  schemaVersion: 1,
  recordedAt: '2026-07-15',
  scope: {
    heroDeck: 'ur_lessons',
    villainDeck: 'gw_allies',
    opponent: 'search-64',
    stack: 'uvicorn + Vite dev server + headless Chromium',
    launchProfile: 'warm navigation through first authoritative legal action',
  },
  referenceProfile: {
    device: 'MacBook Pro, Apple M4 Max (16 cores), 128 GB',
    architecture: 'arm64',
    operatingSystem: 'macOS 26.0.1',
    node: '25.8.0',
    playwright: '1.61.1',
    chromium: '149.0.7827.55',
  },
  samples: {
    warmLaunches: 5,
    interactions: 20,
  },
  metrics: {
    warmLaunchToPlayableMs: {
      definition: 'Navigation start until the first authoritative legal action is visible.',
      baseline: { p50: 172.5, p95: 347.91, max: 347.91 },
      budgetP95: 550,
    },
    inputAcknowledgementMs: {
      definition: 'Enter keydown on a legal action until its local hero log entry renders.',
      baseline: { p50: 0.4, p95: 0.5, max: 1.1 },
      budgetP95: 100,
    },
    authorityResponseMs: {
      definition: 'The same Enter keydown until the authoritative update sequence advances.',
      baseline: { p50: 9.3, p95: 409, max: 792.4 },
      budgetP95: 650,
    },
    frameDeltaMs: {
      definition: 'requestAnimationFrame deltas during measured play and reconnect.',
      baseline: { p50: 8.3, p95: 9.2, max: 16.4 },
      budgetP95: 34,
      budgetMax: 100,
      longFrameThresholdMs: 50,
      baselineLongFrameCount: 0,
    },
    rendererHeapMiB: {
      definition: 'Chromium renderer JSHeapUsedSize samples; this is not total process RSS.',
      baseline: { p50: 7.127, p95: 8.895, max: 9.444 },
      budgetMax: 20,
    },
  },
  exclusions: [
    'cold process or release-build launch',
    'backend, browser-process, GPU, and asset memory',
    'semantic animation quality',
    'Phase parity',
  ],
} as const;
