import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';

import {
  assertAddressBindsReplayDecision,
  assertViewerSafeReplayProjection,
  parseReplayDecisionAddress,
  serializeReplayDecisionAddress,
  type CanonicalReplayProjectionV1,
} from './replay-index';
import type { StudyArtifact } from './study-protocol';

function fixture<T>(name: string): T {
  return JSON.parse(
    readFileSync(new URL(`../../../protocol/fixtures/${name}`, import.meta.url), 'utf8'),
  ) as T;
}

const playerZero = fixture<CanonicalReplayProjectionV1>(
  'canonical-replay-player-0.json',
);
const playerOne = fixture<CanonicalReplayProjectionV1>(
  'canonical-replay-player-1.json',
);
const metadata = fixture<{ decisions: Array<{ ordinal: number }> }>(
  'canonical-replay-authority-metadata.json',
);
const study = fixture<StudyArtifact>('study-curated-decision.json');
const schema = fixture<object>('../canonical-replay-v1.schema.json');
const validate = new Ajv2020({ strict: false, validateFormats: false })
  .compile<CanonicalReplayProjectionV1>(schema);

describe('canonical replay viewer projection', () => {
  it('validates both isolated viewers whose ordinals form the authority timeline', () => {
    expect(validate(playerZero), JSON.stringify(validate.errors)).toBe(true);
    expect(validate(playerOne), JSON.stringify(validate.errors)).toBe(true);
    expect(() => assertViewerSafeReplayProjection(playerZero)).not.toThrow();
    expect(() => assertViewerSafeReplayProjection(playerOne)).not.toThrow();

    const union = [...playerZero.decisions, ...playerOne.decisions]
      .map(({ ordinal }) => ordinal)
      .sort((left, right) => left - right);
    expect(union).toEqual(metadata.decisions.map(({ ordinal }) => ordinal));
    expect(playerZero.decisions.every(({ viewer }) => viewer === 0)).toBe(true);
    expect(playerOne.decisions.every(({ viewer }) => viewer === 1)).toBe(true);
  });

  it('round-trips Study’s erd1 address without converting u64 fields to Number', () => {
    const encoded = study.landmarks[0].decision_id;
    const address = parseReplayDecisionAddress(encoded);
    expect(serializeReplayDecisionAddress(address)).toBe(encoded);
    const row = playerZero.decisions.find(
      ({ ordinal }) => String(ordinal) === address.ordinal,
    );
    expect(row).toBeDefined();
    if (row) {
      expect(() => assertAddressBindsReplayDecision(address, playerZero, row)).not.toThrow();
    }
    expect(() => parseReplayDecisionAddress(`${encoded}=`)).toThrow(
      /invalid replay decision address/,
    );
  });

  it('rejects mixed viewers, duplicate ordinals, and private opponent hands', () => {
    const mixed = structuredClone(playerZero);
    mixed.decisions.push(structuredClone(playerOne.decisions[0]));
    expect(() => assertViewerSafeReplayProjection(mixed)).toThrow(/mixes viewer/);

    const duplicate = structuredClone(playerZero);
    duplicate.decisions[1].ordinal = duplicate.decisions[0].ordinal;
    expect(() => assertViewerSafeReplayProjection(duplicate)).toThrow(/not increasing/);

    const privateProjection = structuredClone(playerZero);
    privateProjection.decisions[0].frame.projection.opponent.hand.push(
      structuredClone(privateProjection.decisions[0].frame.projection.agent.hand[0]),
    );
    expect(() => assertViewerSafeReplayProjection(privateProjection)).toThrow(
      /decision binding drifted/,
    );
  });

  it('rejects an empty replay namespace and drifted projection identity', () => {
    const emptyReplayId = structuredClone(playerZero);
    emptyReplayId.replay_id = '';
    expect(() => assertViewerSafeReplayProjection(emptyReplayId)).toThrow(
      /requires replay_id/,
    );

    for (const field of ['match_id', 'content_hash', 'asset_manifest_hash'] as const) {
      const drifted = structuredClone(playerZero);
      drifted[field] = field === 'match_id'
        ? 'different-match'
        : '0'.repeat(64);
      expect(() => assertViewerSafeReplayProjection(drifted)).toThrow(
        /decision binding drifted/,
      );
    }
  });
});
