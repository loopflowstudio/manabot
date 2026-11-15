import { expect, type Page } from '@playwright/test';

const MAX_HERO_MOVES = 2_000;

export function collectStudyConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error' && !message.text().includes('favicon')) {
      errors.push(message.text());
    }
  });
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  return errors;
}

export async function createCompletedStudyTrace(page: Page): Promise<string> {
  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });

  return page.evaluate(async (maxHeroMoves) => {
    const playerKey = 'etude.play.player-credential';
    const player = localStorage.getItem(playerKey) ?? crypto.randomUUID();
    localStorage.setItem(playerKey, player);
    const before = new Set(
      ((await (await fetch('/api/traces')).json()) as Array<{ id: string }>).map(
        ({ id }) => id,
      ),
    );
    const socket = new WebSocket(
      `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/play`,
    );
    const messages: unknown[] = [];
    let receive: ((value: unknown) => void) | null = null;
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data as string) as unknown;
      if (receive) {
        const deliver = receive;
        receive = null;
        deliver(payload);
      } else {
        messages.push(payload);
      }
    };
    const nextMessage = (): Promise<Record<string, unknown>> => {
      const queued = messages.shift();
      if (queued) return Promise.resolve(queued as Record<string, unknown>);
      return new Promise((resolve) => {
        receive = (value) => resolve(value as Record<string, unknown>);
      });
    };
    await new Promise<void>((resolve, reject) => {
      socket.onopen = () => resolve();
      socket.onerror = () => reject(new Error('Study source WebSocket failed.'));
    });
    socket.send(JSON.stringify({
      type: 'new_game',
      config: {
        player_token: player,
        villain_type: 'passive',
        seed: 7,
        hero_deck: 'ur_lessons',
        villain_deck: 'gw_allies',
        auto_pass: false,
      },
    }));
    let payload = await nextMessage();
    if (typeof payload.resume_token !== 'string') throw new Error('Study source omitted its participant credential.');
    const credentialKey = 'etude.play.record-credentials';
    const credentials: string[] = JSON.parse(localStorage.getItem(credentialKey) ?? '[]');
    localStorage.setItem(credentialKey, JSON.stringify([...new Set([...credentials, payload.resume_token])]));
    for (let move = 0; move < maxHeroMoves; move += 1) {
      if (payload.type === 'game_over') break;
      if (payload.type !== 'observation') {
        throw new Error(`Unexpected Study source payload: ${JSON.stringify(payload)}`);
      }
      const actions = payload.actions as Array<{ index: number }>;
      if (!actions.length) throw new Error('Study source produced no legal action.');
      socket.send(JSON.stringify({ type: 'action', index: actions[0].index }));
      payload = await nextMessage();
    }
    if (payload.type !== 'game_over') {
      throw new Error(`Study source did not finish within ${maxHeroMoves} moves.`);
    }
    socket.close();

    for (let attempt = 0; attempt < 100; attempt += 1) {
      const traces = (await (await fetch('/api/traces')).json()) as Array<{ id: string }>;
      const created = traces.find(({ id }) => !before.has(id));
      if (created) return created.id;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    throw new Error('Completed Study source trace did not appear.');
  }, MAX_HERO_MOVES);
}

export async function openPriorityStudyDecision(page: Page, traceId: string) {
  // Trivial decisions are skipped. The first real choice can be cleanup, so
  // choose a retained priority root for the fixture's explicit pass plan.
  const address = await page.evaluate(async id => {
    const credentials = localStorage.getItem('etude.play.record-credentials') ?? '[]';
    const response = await fetch(`/api/traces/${id}/decisions`, {headers: {'x-etude-participant-tokens': JSON.parse(credentials).join(',')}});
    if (!response.ok) throw new Error('Study fixture decisions unavailable');
    const projection = await response.json();
    const row = projection.decisions.find((decision: {frame: {offers: {verb: string}[]}}) => decision.frame.offers.some(offer => offer.verb === 'pass_priority'));
    if (!row) throw new Error('Study fixture has no recorded priority choice');
    return row.address as string;
  }, traceId);
  await page.goto(`/replay?trace=${encodeURIComponent(traceId)}&position=${encodeURIComponent(address)}`);
  const decision = page.getByRole('button', {name: 'Try another line', exact: true});
  await expect(decision).toBeEnabled();
  await decision.click();
  await expect(page.getByTestId('study-panel')).toBeVisible();
  await expect(page.getByTestId('study-reveal')).toHaveCount(0);
  return decision;
}
