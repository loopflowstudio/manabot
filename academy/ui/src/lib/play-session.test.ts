import { describe, expect, it } from 'vitest';
import { GameStore } from './game.svelte';
import { seedError } from './play-session';

describe('repeated play setup', () => {
  it('preserves the complete u64 seed as decimal text', () => {
    expect(seedError('18446744073709551615')).toBeNull();
    expect(seedError('18446744073709551616')).not.toBeNull();
    expect(seedError('1.5')).not.toBeNull();
    expect(seedError('-1')).not.toBeNull();
    const store = new GameStore();
    store.seed = '18446744073709551615'; store.dealMode = 'repeat';
    expect(store.newGameConfig().seed).toBe('18446744073709551615');
    store.dealMode = 'fresh';
    expect(store.newGameConfig().seed).toBeNull();
  });

  it('restores private imported setup while keeping local edits on ordinary updates', () => {
    const store = new GameStore();
    const previousDecks = {...store.decks};
    const draft = {hero_deck:'imported', villain_deck:'imported', human_seat: 1, seed: '9999999999999999', villain_type: 'random', decklists: 'closed', deck_import: {hero: {source: '8 Island', filename: 'a.txt'}, villain: {source: '8 Mountain', filename: 'b.txt'}}};
    store.applyPlaySession({id:'session', session_id:'session', game_number:2, revision:2, draft, config:draft});
    expect(store.humanSeat).toBe(1);
    expect(store.deckImport).toEqual(draft.deck_import);
    expect(store.decks).toEqual(previousDecks);
    expect(store.decklists).toBe('closed');
    store.opponentChoice = 'passive';
    store.applyPlaySession({id:'session', session_id:'session', game_number:2, revision:3, draft, config:draft});
    expect(store.opponentChoice).toBe('passive');
    expect(store.seed).toBe('9999999999999999');
  });
  it('isolates editor choices from live setup and restores the full retained Search draft', () => {
    const store = new GameStore();
    const mirror = {algorithm:'local_update', belief:'physical_deal', worlds:4, depth:2, decision_seconds:1, alpha:0, beta:1};
    const draft = {human_seat:1, seed:'18446744073709551615', villain_type:'policy', opponent_sha256:'a'.repeat(64),
      mirror, decklists:'closed', stops:{my:['main1'], opponent:['end_step']}, auto_pass:false, stop_on_stack:true,
      deck_import:{hero:{source:'8 Island',filename:'a.txt'},villain:{source:'8 Mountain',filename:'b.txt'}}};
    const editor = store.editableSetup(draft);
    expect(editor.newGameConfig()).toMatchObject(draft);
    editor.deckImport!.hero.source = '8 Forest';
    editor.searchConfig!.depth = 3;
    editor.stops.my.push('draw');
    editor.setHeroDeck('gw_allies');
    editor.watchMode = true;
    expect(draft.deck_import.hero.source).toBe('8 Island');
    expect(draft.mirror.depth).toBe(2);
    expect(draft.stops.my).toEqual(['main1']);
    expect(store.deckImport).toBeNull();
    expect(store.watchMode).toBe(false);
    expect(store.editableSetup(draft).searchConfig!.depth).toBe(2);
  });

});
