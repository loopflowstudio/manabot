import { expect, it } from 'vitest';
import { recordedGames } from './game-file';

it('preserves exact integers in formatted objects, JSONL and arrays', () => {
  const game = '{\n  "seed": 18446744073709551615, "nested": [{"name": "comma, brace } and quote \\\""}]\n}';
  expect(recordedGames(game)).toEqual([game]);
  expect(recordedGames(`[${game}, {"seed": 9007199254740993}]`)).toEqual([game, '{"seed": 9007199254740993}']);
  expect(recordedGames('{"seed": 18446744073709551615}\n{"seed": 9007199254740993}\n')).toEqual(['{"seed": 18446744073709551615}', '{"seed": 9007199254740993}']);
});
it('refuses empty or non-record input', () => {
  for (const source of ['', '[]', '[1]', 'null', 'not json']) expect(() => recordedGames(source)).toThrow();
});
