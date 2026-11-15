import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';
import ActionPanel from './ActionPanel.svelte';

describe('Learn mode availability', () => {
  it('explains empty modes while keeping decline available', () => {
    const { body } = render(ActionPanel, { props: {
      actionSpaceKind: 'LEARN',
      offers: [{ id: 0, actor: 0, verb: 'choose', source: null, choices: [], help: null, confirm_label: 'Decline Learn', action_type: 'DECLINE_CHOICE', focus: [], label: 'Decline Learn' }],
    } });
    expect(body).toContain('No eligible Lessons remaining.');
    expect(body).toContain('No cards to discard.');
    expect(body.match(/ disabled=/g)).toHaveLength(3); // two modes plus Pass Turn
    expect(body).toContain('Decline Learn');
    expect(body).not.toContain('learn-card-preview');
  });
});


describe('interrupted decision', () => {
  it('keeps the recorded offers without inviting another live choice', () => {
    const {body} = render(ActionPanel, {props: {
      interrupted: true,
      instruction: 'Choose an explicit legal offer.',
      offers: [{id: 0, actor: 0, verb: 'pass_priority', source: null, choices: [], help: null,
        confirm_label: 'Pass priority', action_type: 'PASS_PRIORITY', focus: [], label: 'Pass priority'}],
    }});
    expect(body).toContain('Last recorded choices');
    expect(body).toContain('Interrupted · read-only');
    expect(body).toContain('1 of 1 recorded choices');
    expect(body).toContain('cannot be submitted');
    expect(body).toContain('Pass priority');
    expect(body).toContain(' disabled');
    for (const text of ['Choose an explicit legal offer.', 'Your move', 'legal choices', 'Pass Turn (F6)']) {
      expect(body).not.toContain(text);
    }
  });
});
