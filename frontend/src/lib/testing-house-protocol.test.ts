import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';

import {
  TESTING_HOUSE_REQUEST_TYPES,
  TESTING_HOUSE_VERSION,
  type TestingHouseRequest,
  type TestingHouseV1ConformanceBundle,
} from './testing-house-protocol';

interface SchemaNode {
  $defs?: Record<string, SchemaNode>;
  discriminator?: { mapping?: Record<string, string> };
  properties?: Record<string, SchemaNode>;
  items?: SchemaNode;
}

const schema = JSON.parse(readFileSync(
  new URL('../../../protocol/testing-house-v1.schema.json', import.meta.url),
  'utf8',
)) as SchemaNode;
const fixture: unknown = JSON.parse(readFileSync(
  new URL('../../../protocol/fixtures/testing-house-control-v1.json', import.meta.url),
  'utf8',
));
const validate = new Ajv2020({ strict: false, validateFormats: false })
  .compile<TestingHouseV1ConformanceBundle>(schema);

describe('testing-house-v1 control conformance', () => {
  it('validates and round-trips the checked control fixture', () => {
    expect(validate(fixture), JSON.stringify(validate.errors)).toBe(true);
    if (!validate(fixture)) throw new Error('control fixture did not conform');
    const typed: TestingHouseV1ConformanceBundle = fixture;
    expect(typed.contract).toBe(TESTING_HOUSE_VERSION);
    expect(typed.requests.map(({ type }) => type)).toEqual(TESTING_HOUSE_REQUEST_TYPES);
    expect(JSON.parse(JSON.stringify(typed))).toEqual(fixture);
  });

  it('keeps the TypeScript operation vocabulary equal to the schema discriminator', () => {
    const mapping = schema.properties?.requests?.items?.discriminator?.mapping ?? {};
    expect(Object.keys(mapping).sort()).toEqual([...TESTING_HOUSE_REQUEST_TYPES].sort());

    const typedRequests: TestingHouseRequest[] = (
      fixture as TestingHouseV1ConformanceBundle
    ).requests;
    expect(typedRequests).toHaveLength(TESTING_HOUSE_REQUEST_TYPES.length);
  });

  it('rejects unknown operations and unmodelled room defaults', () => {
    expect(validate({
      contract: TESTING_HOUSE_VERSION,
      requests: [{ type: 'chat', message: 'hello' }],
      events: [],
    })).toBe(false);

    const sidecar = structuredClone(fixture) as TestingHouseV1ConformanceBundle & {
      room_default?: string;
    };
    sidecar.room_default = 'shared';
    expect(validate(sidecar)).toBe(false);
  });
});
