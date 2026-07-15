import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';

import type { Command, ProtocolV1ConformanceBundle } from './types';

const schema = JSON.parse(
  readFileSync(new URL('../../../protocol/experience-v1.schema.json', import.meta.url), 'utf8'),
) as object;
const fixture: unknown = JSON.parse(
  readFileSync(new URL('../../../protocol/fixtures/bolt-target.json', import.meta.url), 'utf8'),
);

const validate = new Ajv2020({ strict: false, validateFormats: false })
  .compile<ProtocolV1ConformanceBundle>(schema);

describe('experience protocol v1 conformance', () => {
  it('consumes the Rust-certified Lightning Bolt fixture through TypeScript types', () => {
    expect(validate(fixture), JSON.stringify(validate.errors)).toBe(true);
    if (!validate(fixture)) {
      throw new Error('fixture did not conform');
    }

    const command: Command = fixture.command;
    expect(command.match_id).toBe(fixture.recovery.frame.match_id);
    expect(command.expected_revision).toBe(fixture.recovery.frame.revision);
    expect(command.prompt_id).toBe(fixture.recovery.frame.prompt?.id);
    expect(
      fixture.recovery.frame.offers.some((offer) => offer.id === command.offer_id),
    ).toBe(true);
  });

  it('rejects a payload from another protocol version', () => {
    const invalid = structuredClone(fixture) as Record<string, unknown>;
    (invalid.recovery as Record<string, unknown>).protocol = 2;
    expect(validate(invalid)).toBe(false);
  });
});
