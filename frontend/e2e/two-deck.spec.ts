import { expect, test, type Page } from '@playwright/test';

// Milestone-1 two-deck slice e2e: select UR Lessons vs GW Allies in the UI,
// play a full game to terminal against the random villain with the default
// main-phase stops on, and assert the new-card UI moments render — the deck
// names in the header, and at least one mid-resolution choice-kind decision
// (scry / pay-or-not [kicker] / learn / look-and-select / modal / waterbend)
// surfacing as a decision prompt with clickable, labeled options.

const MAX_ACTIONS_PER_GAME = 1500;
const MAX_GAMES = 8;

// Mid-resolution decision kinds introduced by the Stage-2/3 card machinery.
const CHOICE_KINDS = new Set([
  'SCRY',
  'LOOK_AND_SELECT',
  'PAY_OR_NOT',
  'MODAL',
  'DISCARD_THEN_DRAW',
  'WATERBEND',
]);

const TINY_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==',
  'base64',
);

function collectConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    if (message.type() !== 'error') {
      return;
    }
    const text = message.text();
    if (text.includes('favicon')) {
      return;
    }
    errors.push(text);
  });
  page.on('pageerror', (error) => {
    errors.push(`pageerror: ${error.message}`);
  });
  return errors;
}

async function updateSeq(page: Page): Promise<number> {
  return Number(await page.locator('main').getAttribute('data-update-seq'));
}

test('UR vs GW plays to terminal with deck names and choice prompts rendered', async ({
  page,
}) => {
  test.setTimeout(600_000);
  const consoleErrors = collectConsoleErrors(page);

  await page.route('https://api.scryfall.com/**', (route) =>
    route.fulfill({ contentType: 'image/png', body: TINY_PNG }),
  );

  await page.goto('/');
  await expect(page.getByTestId('connection-badge')).toHaveText('connected', {
    timeout: 15_000,
  });

  // Select the Milestone-1 matchup through the UI (also the defaults, but
  // the point is that the pickers drive the config).
  await page.getByTestId('deck-select-hero').selectOption('ur_lessons');
  await page.getByTestId('deck-select-villain').selectOption('gw_allies');
  await page.getByTestId('opponent-select').selectOption('random');
  // Default stops stay on: my main1/main2 + opponent end step + stack.

  await page.getByRole('button', { name: 'New Game' }).first().click();
  const board = page.getByTestId('game-board');
  await expect(board).toBeVisible({ timeout: 15_000 });

  // Deck names must render in the game header.
  await expect(page.getByTestId('deck-names')).toHaveText(
    'UR Lessons vs GW Allies',
    { timeout: 15_000 },
  );

  const actionButtons = page.getByTestId('action-option');
  const decisionPrompt = page.getByTestId('decision-prompt');
  const gameOverOverlay = page.getByText('Game Over', { exact: true });

  const choiceKindsSeen = new Set<string>();
  const promptKindsSeen = new Set<string>();
  let gamesFinished = 0;
  let choiceOptionClicked = false;

  for (let game = 0; game < MAX_GAMES; game += 1) {
    let finished = false;

    for (let i = 0; i < MAX_ACTIONS_PER_GAME; i += 1) {
      await expect(actionButtons.first().or(gameOverOverlay)).toBeVisible({
        timeout: 30_000,
      });
      if (await gameOverOverlay.isVisible()) {
        finished = true;
        break;
      }

      // If a mid-resolution decision surfaced, the prompt and its clickable,
      // labeled options are the new-card UI moment under test.
      if (await decisionPrompt.isVisible()) {
        const kind = (await decisionPrompt.getAttribute('data-kind')) ?? '';
        promptKindsSeen.add(kind);
        if (CHOICE_KINDS.has(kind)) {
          choiceKindsSeen.add(kind);
          const count = await actionButtons.count();
          expect(count, `no options rendered for ${kind}`).toBeGreaterThan(0);
          await expect(actionButtons.first()).toBeEnabled();
          for (const label of await actionButtons.allTextContents()) {
            expect(label.trim().length, `empty option label for ${kind}`).toBeGreaterThan(0);
          }
        }
      }

      const seqBefore = await updateSeq(page);
      const labels = await actionButtons.allTextContents();
      const kindNow = (await decisionPrompt.isVisible())
        ? ((await decisionPrompt.getAttribute('data-kind')) ?? '')
        : '';

      // Spell-biased hero: cast/play when possible so kicker, learn, and
      // scry decisions actually come up; otherwise pick at random.
      let pick = labels.findIndex((label) => label.startsWith('Cast '));
      if (pick === -1) {
        pick = labels.findIndex((label) => label.startsWith('Play '));
      }
      if (pick === -1 || Math.random() < 0.2) {
        pick = Math.floor(Math.random() * labels.length);
      }

      try {
        await actionButtons.nth(pick).click({ timeout: 2_000 });
      } catch {
        continue; // list swapped under us — resample on the next iteration
      }
      if (CHOICE_KINDS.has(kindNow)) {
        choiceOptionClicked = true;
      }

      await expect
        .poll(() => updateSeq(page), {
          timeout: 30_000,
          message: `no server response after click ${i} of game ${game}`,
        })
        .toBeGreaterThan(seqBefore);
    }

    expect(finished, `game ${game} did not finish`).toBe(true);
    gamesFinished += 1;

    if (choiceKindsSeen.size > 0 && choiceOptionClicked) {
      break;
    }
    await page.getByRole('button', { name: 'Play Again' }).click();
    await expect(board).toBeVisible({ timeout: 15_000 });
  }

  expect(gamesFinished).toBeGreaterThan(0);
  expect(
    choiceKindsSeen.size,
    `no choice-kind decision surfaced across ${gamesFinished} games ` +
      `(prompt kinds seen: ${[...promptKindsSeen].join(', ') || 'none'})`,
  ).toBeGreaterThan(0);
  expect(choiceOptionClicked, 'choice-kind option was never clicked').toBe(true);
  console.log(
    `two-deck e2e: ${gamesFinished} game(s), choice kinds seen: ` +
      [...choiceKindsSeen].join(', '),
  );

  expect(consoleErrors, `console errors: ${consoleErrors.join('\n')}`).toEqual([]);
});
