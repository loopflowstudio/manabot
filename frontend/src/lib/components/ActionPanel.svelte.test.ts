import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import ActionPanel from './ActionPanel.svelte';

describe('Learn mode availability', () => {
  it('explains empty modes while keeping decline available', () => {
    const { body } = render(ActionPanel, { props: {
      actionSpaceKind: 'LEARN',
      actions: [{ index: 0, type: 'DECLINE_CHOICE', focus: [], description: 'Decline Learn' }],
    } });
    expect(body).toContain('No eligible Lessons remaining.');
    expect(body).toContain('No cards to discard.');
    expect(body.match(/ disabled=/g)).toHaveLength(3); // two modes plus Pass Turn
    expect(body).toContain('Decline Learn');
    expect(body).not.toContain('learn-card-preview');
  });
});
