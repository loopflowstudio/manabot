import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { expect, test, type Browser, type Page } from '@playwright/test';

const RESUME_STORAGE_KEY = 'etude.gui.resume';
const PUBLIC_PROTOCOLS = new Set(['http:', 'https:', 'ws:', 'wss:']);
const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);
const RECEIPT_PATH = process.env.ETUDE_CLEAN_RECEIPT;
const LAUNCH_LOG_PATH = process.env.ETUDE_LAUNCH_LOG;
const LAUNCH_STARTED_MS = Number(process.env.ETUDE_CLEAN_START_MS ?? '0');

interface AssetPackReference {
  id: string;
  version: string;
  manifest_sha256: string;
}

type CleanProofWindow = Window &
  typeof globalThis & { __etudeCleanFrames?: string[] };

function isPublicRequest(rawUrl: string): boolean {
  const url = new URL(rawUrl);
  return PUBLIC_PROTOCOLS.has(url.protocol) && !LOOPBACK_HOSTS.has(url.hostname);
}

function normalize(text: string | null): string {
  return (text ?? '').replace(/\s+/g, ' ').trim();
}

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function expectedPack(): AssetPackReference {
  const manifestPath = path.resolve(
    process.cwd(),
    'src/lib/packs/tla-ur-lessons-vs-gw-allies/v1/manifest.json',
  );
  const bytes = readFileSync(manifestPath);
  const manifest = JSON.parse(bytes.toString('utf8')) as {
    pack: { id: string; version: string };
  };
  return {
    id: manifest.pack.id,
    version: manifest.pack.version,
    manifest_sha256: sha256(bytes),
  };
}

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

function packFromMessage(message: unknown): AssetPackReference | null {
  if (!message || typeof message !== 'object') {
    return null;
  }
  const record = message as Record<string, unknown>;
  const candidates = [
    record.asset_pack,
    (record.frame as Record<string, unknown> | undefined)?.asset_pack,
    (record.recovery as { frame?: Record<string, unknown> } | undefined)?.frame?.asset_pack,
    (record.update as { frame?: Record<string, unknown> } | undefined)?.frame?.asset_pack,
  ];
  for (const candidate of candidates) {
    if (
      candidate &&
      typeof candidate === 'object' &&
      typeof (candidate as AssetPackReference).id === 'string' &&
      typeof (candidate as AssetPackReference).version === 'string' &&
      typeof (candidate as AssetPackReference).manifest_sha256 === 'string'
    ) {
      return candidate as AssetPackReference;
    }
  }
  return null;
}

async function latestBrowserPack(page: Page): Promise<AssetPackReference | null> {
  const frames = await page.evaluate(
    () => (window as CleanProofWindow).__etudeCleanFrames ?? [],
  );
  for (const frame of frames.toReversed()) {
    try {
      const pack = packFromMessage(JSON.parse(frame));
      if (pack) {
        return pack;
      }
    } catch {
      // Ignore transport frames that are not JSON application messages.
    }
  }
  return null;
}

async function updateSequence(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

async function chooseDeterministicAction(page: Page): Promise<void> {
  const actions = page.getByTestId('action-option');
  const labels = await actions.allTextContents();
  const preferences = [/^Play /, /^Cast /, /^Activate /, /Pass priority/];
  let index = 0;
  for (const preference of preferences) {
    const match = labels.findIndex((label) => preference.test(label.trim()));
    if (match >= 0) {
      index = match;
      break;
    }
  }
  const before = await updateSequence(page);
  await actions.nth(index).click();
  await expect
    .poll(() => updateSequence(page), { timeout: 30_000 })
    .toBeGreaterThan(before);
}

async function visibleSignature(page: Page): Promise<string> {
  const prompts = page.getByTestId('decision-prompt');
  const prompt = (await prompts.count()) > 0 ? normalize(await prompts.first().textContent()) : '';
  const board = await page.getByTestId('game-board').evaluate((element) => {
    const canonical = element.cloneNode(true) as HTMLElement;
    canonical.querySelector('[data-testid="presentation-stage"]')?.remove();
    return canonical.textContent;
  });
  const payload = {
    deckNames: normalize(await page.getByTestId('deck-names').textContent()),
    // Optional presentation theater is transient; the proof signs the
    // authoritative board, prompt, and legal-action projection underneath it.
    board: normalize(board),
    prompt,
    actions: (await page.getByTestId('action-option').allTextContents()).map(normalize),
  };
  return sha256(JSON.stringify(payload));
}

function toolVersion(executable: string): string {
  try {
    return execFileSync(executable, ['--version'], { encoding: 'utf8' })
      .trim()
      .split('\n')[0];
  } catch {
    return 'unavailable';
  }
}

function launcherReadyRecord(): unknown {
  if (!LAUNCH_LOG_PATH) {
    return null;
  }
  const line = readFileSync(LAUNCH_LOG_PATH, 'utf8')
    .split('\n')
    .find((candidate) => candidate.startsWith('ETUDE_PLAY_READY '));
  return line ? JSON.parse(line.slice('ETUDE_PLAY_READY '.length)) : null;
}

test('clean command reaches pinned play and reloads without public network', async ({
  browser,
  page,
}: {
  browser: Browser;
  page: Page;
}, testInfo) => {
  test.setTimeout(120_000);
  expect(LAUNCH_STARTED_MS, 'clean verifier did not provide an external start time').toBeGreaterThan(0);
  const expected = expectedPack();
  const errors = collectErrors(page);
  const publicRequests: string[] = [];

  await page.addInitScript(() => {
    const proofWindow = window as CleanProofWindow;
    const nativeWebSocket = window.WebSocket;
    class TrackedWebSocket extends nativeWebSocket {
      constructor(url: string | URL, protocols?: string | string[]) {
        if (protocols === undefined) {
          super(url);
        } else {
          super(url, protocols);
        }
        this.addEventListener('message', (event: MessageEvent<unknown>) => {
          if (typeof event.data === 'string') {
            proofWindow.__etudeCleanFrames?.push(event.data);
          }
        });
      }
    }
    proofWindow.__etudeCleanFrames = [];
    Object.defineProperty(window, 'WebSocket', {
      configurable: true,
      value: TrackedWebSocket,
      writable: true,
    });
  });

  await page.route('**/*', async (route) => {
    const url = route.request().url();
    if (isPublicRequest(url)) {
      publicRequests.push(url);
      await route.abort('internetdisconnected');
      return;
    }
    await route.continue();
  });
  await page.routeWebSocket(
    (url) => isPublicRequest(url.toString()),
    async (socket) => {
      publicRequests.push(socket.url());
      await socket.close({ code: 1008, reason: 'clean-machine offline proof' });
    },
  );
  page.on('websocket', (socket) => {
    if (isPublicRequest(socket.url())) {
      publicRequests.push(socket.url());
    }
  });

  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await expect(page.getByTestId('opponent-select')).toHaveValue('search-64');
  await expect(page.getByTestId('deck-select-hero')).toHaveValue('ur_lessons');
  await expect(page.getByTestId('deck-select-villain')).toHaveValue('gw_allies');
  await page.getByRole('button', { name: 'New Game' }).first().click();

  const board = page.getByTestId('game-board');
  await expect(page.getByTestId('deck-names')).toHaveText('UR Lessons vs GW Allies', {
    timeout: 15_000,
  });
  await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 30_000 });
  const elapsedToPlayableMs = Date.now() - LAUNCH_STARTED_MS;
  expect(elapsedToPlayableMs).toBeLessThan(60_000);

  const treatments = board.getByTestId('card-treatment');
  const treatmentCount = await treatments.count();
  expect(treatmentCount).toBeGreaterThan(0);
  await expect(board.locator('[data-asset-source="pack"]')).toHaveCount(treatmentCount);
  await expect(board.locator('[data-asset-source="fallback"]')).toHaveCount(0);
  await expect.poll(() => latestBrowserPack(page), { timeout: 15_000 }).toEqual(expected);

  await chooseDeterministicAction(page);
  const credentialsBefore = await page.evaluate(
    (key) => sessionStorage.getItem(key),
    RESUME_STORAGE_KEY,
  );
  expect(credentialsBefore, 'resume credentials were not installed').not.toBeNull();
  const signatureBefore = await visibleSignature(page);

  await page.reload();
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });
  await expect(page.getByTestId('deck-names')).toHaveText('UR Lessons vs GW Allies');
  await expect(page.getByTestId('action-option').first()).toBeVisible({ timeout: 30_000 });
  const credentialsAfter = await page.evaluate(
    (key) => sessionStorage.getItem(key),
    RESUME_STORAGE_KEY,
  );
  const signatureAfter = await visibleSignature(page);
  expect(credentialsAfter).toBe(credentialsBefore);
  expect(signatureAfter).toBe(signatureBefore);
  await expect.poll(() => latestBrowserPack(page), { timeout: 15_000 }).toEqual(expected);

  const reloadedTreatments = board.getByTestId('card-treatment');
  const reloadedCount = await reloadedTreatments.count();
  expect(reloadedCount).toBeGreaterThan(0);
  await expect(board.locator('[data-asset-source="pack"]')).toHaveCount(reloadedCount);
  await expect(board.locator('[data-asset-source="fallback"]')).toHaveCount(0);
  await chooseDeterministicAction(page);

  expect(publicRequests).toEqual([]);
  expect(errors, `console errors: ${errors.join('\n')}`).toEqual([]);

  const receipt = {
    schema_version: 1,
    recorded_at: new Date().toISOString(),
    result: 'pass',
    clean_state: {
      checkout_artifacts_absent: true,
      fresh_browser_context: true,
      public_network_denied_after_install: true,
    },
    host: {
      platform: os.platform(),
      release: os.release(),
      architecture: os.arch(),
      cpu_model: os.cpus()[0]?.model ?? 'unknown',
      logical_cores: os.cpus().length,
      total_memory_bytes: os.totalmem(),
    },
    tools: {
      uv: toolVersion('uv'),
      node: process.version,
      npm: toolVersion('npm'),
      rustc: toolVersion('rustc'),
      cargo: toolVersion('cargo'),
      chromium: browser.version(),
    },
    launch: {
      argv: ['./scripts/play'],
      ready: launcherReadyRecord(),
      elapsed_to_playable_ms: elapsedToPlayableMs,
      budget_ms: 60_000,
    },
    experience: {
      opponent: 'search-64',
      hero_deck: 'ur_lessons',
      villain_deck: 'gw_allies',
      asset_pack: expected,
      same_session_after_reload: credentialsAfter === credentialsBefore,
      same_visible_state_after_reload: signatureAfter === signatureBefore,
      post_reload_action_accepted: true,
      public_requests: publicRequests,
    },
  };
  const serialized = JSON.stringify(receipt, null, 2) + '\n';
  await testInfo.attach('clean-machine-proof.json', {
    body: Buffer.from(serialized),
    contentType: 'application/json',
  });
  if (RECEIPT_PATH) {
    writeFileSync(RECEIPT_PATH, serialized, 'utf8');
  }
  console.log(`clean-machine proof: ${JSON.stringify(receipt)}`);
});
