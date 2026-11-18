/**
 * A game played without the mouse: Tab moves between the table's regions
 * (the opponent's field, your field, the hand, the stack, the decision, the
 * legal choices, the moves), arrows move within one, Enter acts on the focused
 * card or choice and Space is the primary. After a move, focus stays in the
 * region it was made from rather than falling back to the top of the page.
 */
import { expect, test, type Page } from '@playwright/test';

/** The region holding focus, by its accessible name, and the focused control's name. */
async function focused(page: Page): Promise<{ region: string | null; label: string }> {
  return page.evaluate(() => {
    const active = document.activeElement as HTMLElement | null;
    const region = active?.closest<HTMLElement>('[data-region]');
    return { region: region?.getAttribute('aria-label') ?? null, label: active?.getAttribute('aria-label') ?? active?.textContent?.trim() ?? '' };
  });
}

/** Tab to the region whose name matches, then arrow to a control whose name matches. */
async function reach(page: Page, region: RegExp, control: RegExp): Promise<boolean> {
  for (let tabs = 0; tabs < 16; tabs += 1) {
    if (region.test((await focused(page)).region ?? '')) break;
    await page.keyboard.press('Tab');
  }
  if (!region.test((await focused(page)).region ?? '')) return false;
  await page.keyboard.press('Home');
  const seen = new Set<string>();
  for (let steps = 0; steps < 40; steps += 1) {
    const now = await focused(page);
    if (control.test(now.label)) return true;
    if (seen.has(now.label) && steps > 0) {
      // A row ends: try the next row down before giving up.
      await page.keyboard.press('ArrowDown');
      const below = await focused(page);
      if (below.label === now.label || seen.has(below.label)) return false;
      continue;
    }
    seen.add(now.label);
    await page.keyboard.press('ArrowRight');
  }
  return false;
}

const revision = (page: Page): Promise<number> => page.locator('main').getAttribute('data-frame-revision').then(Number);

test('a game is played from the keyboard alone', async ({ page }) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 800 });
  const sent: string[] = [];
  page.on('websocket', socket => socket.on('framesent', frame => {
    if (String(frame.payload).includes('"type":"command"')) sent.push(String(frame.payload));
  }));
  // Beats are tested on their own; here every Space is a move.
  await page.addInitScript(() => localStorage.setItem('academy.table.beat-pace', 'off'));
  await page.goto('/?setup=1');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected');
  await page.getByTestId('opponent-select').focus();
  await page.getByTestId('opponent-select').selectOption('random');
  await page.getByRole('button', { name: 'Start game', exact: true }).focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('[data-testid=table][data-live=true]')).toBeVisible({ timeout: 30_000 });

  // Tab visits the regions in reading order, each once, one stop per region.
  const stops: string[] = [];
  for (let tabs = 0; tabs < 24 && stops.length < 12; tabs += 1) {
    await page.keyboard.press('Tab');
    const { region } = await focused(page);
    if (region) stops.push(region);
  }
  const regions = stops.slice(0, stops.indexOf(stops[0], 1) > 0 ? stops.indexOf(stops[0], 1) : stops.length);
  expect(new Set(regions).size).toBe(regions.length);
  for (const name of [/battlefield$/, /^Your battlefield$/, /^Your hand/, /^Decision$/, /^All legal choices$/, /^Moves$/]) {
    expect(regions.some(region => name.test(region)), `${name} in ${regions.join(' → ')}`).toBe(true);
  }

  // Arrows move within a region: the hand's cards one by one.
  expect(await reach(page, /^Your hand/, /./)).toBe(true);
  const first = (await focused(page)).label;
  await page.keyboard.press('ArrowRight');
  const second = await focused(page);
  expect(second.region).toMatch(/^Your hand/);
  expect(second.label).not.toBe(first);

  // Several decisions: a land from the hand, attacks on the field, Space for the rest.
  let byEnter = 0;
  let bySpace = 0;
  for (let decision = 0; decision < 40 && (byEnter < 3 || bySpace < 6); decision += 1) {
    await expect(page.locator('[data-testid=table][data-live=true], [data-testid=result-card]').first()).toBeVisible({ timeout: 30_000 });
    if (await page.getByTestId('result-card').isVisible()) break;
    const before = await revision(page);
    const commands = sent.length;
    const space = await page.getByTestId('action-panel').getAttribute('data-action-space-kind');
    const target = space === 'PRIORITY' ? [/^Your hand/, /Play (Island|Mountain|Forest|Plains)/] as const
      : space === 'DECLARE_ATTACKER' ? [/^Your battlefield$/, /Attack with/] as const : null;
    if (target && await reach(page, target[0], target[1])) {
      await page.keyboard.press('Enter');
      byEnter += 1;
    } else {
      // Space is the primary wherever focus is in the table; choices without one are numbered.
      if (await page.getByTestId('primary-action').count()) await page.keyboard.press('Space');
      else await page.keyboard.press('1');
      bySpace += 1;
    }
    await expect.poll(() => revision(page), { timeout: 30_000 }).toBeGreaterThan(before);
    expect(sent.length).toBe(commands + 1);
    // Focus stays on the table after the move.
    await expect.poll(async () => page.evaluate(() => document.activeElement !== document.body)).toBe(true);
  }
  expect(byEnter).toBeGreaterThanOrEqual(2);
  expect(bySpace).toBeGreaterThanOrEqual(4);
});
