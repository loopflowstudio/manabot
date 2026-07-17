import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

import {
  STUDY_VERSION,
  assertViewerSafeStudyArtifact,
  assertViewerSafeRecordedDecisionInput,
  assertViewerSafeStudyDecisionIndex,
  type RecordedDecisionInput,
  type StudyArtifact,
  type StudyDecisionIndex,
} from './study-protocol';
import { parseReplayDecisionAddress, type CanonicalReplayProjectionV1 } from './replay-index';

interface SchemaNode {
  $defs?: Record<string, SchemaNode>;
  const?: unknown;
  maximum?: number;
  minimum?: number;
  oneOf?: SchemaNode[];
  properties?: Record<string, SchemaNode>;
  required?: string[];
}

const schema = JSON.parse(
  readFileSync(new URL('../../../protocol/study-v1.schema.json', import.meta.url), 'utf8'),
) as SchemaNode;
const fixture: unknown = JSON.parse(
  readFileSync(
    new URL('../../../protocol/fixtures/study-curated-decision.json', import.meta.url),
    'utf8',
  ),
);
const sourceReplay: unknown = JSON.parse(
  readFileSync(
    new URL('../../../protocol/fixtures/canonical-replay-player-0.json', import.meta.url),
    'utf8',
  ),
);
const recordedDecisionSchema = JSON.parse(
  readFileSync(
    new URL('../../../protocol/study-recorded-decisions-v1.schema.json', import.meta.url),
    'utf8',
  ),
) as SchemaNode;
const decisionIndexSchema = JSON.parse(
  readFileSync(new URL('../../../protocol/study-index-v1.schema.json', import.meta.url), 'utf8'),
) as SchemaNode;
const recordedDecisionFixture: unknown = JSON.parse(
  readFileSync(
    new URL(
      '../../../protocol/fixtures/recorded-match-decisions-curated.json',
      import.meta.url,
    ),
    'utf8',
  ),
);
const decisionIndexFixture: unknown = JSON.parse(
  readFileSync(
    new URL('../../../protocol/fixtures/study-decision-index-curated.json', import.meta.url),
    'utf8',
  ),
);
const source = ts.createSourceFile(
  'study-protocol.ts',
  readFileSync(new URL('./study-protocol.ts', import.meta.url), 'utf8'),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TS,
);
const validate = new Ajv2020({ strict: false, validateFormats: false })
  .compile<StudyArtifact>(schema);
const validateRecordedDecisions = new Ajv2020({ strict: false, validateFormats: false })
  .compile<RecordedDecisionInput>(recordedDecisionSchema);
const validateDecisionIndex = new Ajv2020({ strict: false, validateFormats: false })
  .compile<StudyDecisionIndex>(decisionIndexSchema);

function sorted(values: readonly string[]): string[] {
  return [...values].sort();
}

function interfaceShape(name: string): { fields: string[]; required: string[] } {
  const declaration = source.statements.find(
    (statement): statement is ts.InterfaceDeclaration =>
      ts.isInterfaceDeclaration(statement) && statement.name.text === name,
  );
  if (!declaration) {
    throw new Error(`missing TypeScript interface ${name}`);
  }
  const properties = declaration.members.filter(ts.isPropertySignature);
  const fieldName = (property: ts.PropertySignature): string => {
    if (ts.isIdentifier(property.name) || ts.isStringLiteral(property.name)) {
      return property.name.text;
    }
    return property.name.getText(source);
  };
  return {
    fields: properties.map(fieldName),
    required: properties
      .filter((property) => property.questionToken === undefined)
      .map(fieldName),
  };
}

function schemaShape(node: SchemaNode): { fields: string[]; required: string[] } {
  return {
    fields: Object.keys(node.properties ?? {}),
    required: node.required ?? [],
  };
}

function expectShape(name: string, node: SchemaNode): void {
  const actual = interfaceShape(name);
  const expected = schemaShape(node);
  expect(sorted(actual.fields), `${name} fields`).toEqual(sorted(expected.fields));
  expect(sorted(actual.required), `${name} required`).toEqual(sorted(expected.required));
}

function parsedFixture(): StudyArtifact {
  expect(validate(fixture), JSON.stringify(validate.errors)).toBe(true);
  if (!validate(fixture)) {
    throw new Error('shared study fixture did not conform');
  }
  return structuredClone(fixture);
}

describe('study protocol v1', () => {
  it('round-trips the shared historical decision with distinct evidence fields', () => {
    const artifact = parsedFixture();
    expect(() => assertViewerSafeStudyArtifact(artifact)).not.toThrow();
    expect(artifact.version).toBe(STUDY_VERSION);
    const landmark = artifact.landmarks[0];
    const replay = sourceReplay as CanonicalReplayProjectionV1;
    const address = parseReplayDecisionAddress(landmark.decision_id);
    const row = replay.decisions.find(({ ordinal }) => ordinal === Number(address.ordinal));
    expect(row).toBeDefined();
    expect(landmark.frame).toEqual(row?.frame);
    expect(landmark.played).toEqual(row?.command);
    expect(landmark.offer).toEqual(
      landmark.frame.offers.find(({ id }) => id === landmark.offer_id),
    );
    expect(landmark.played.offer_id).toBe(landmark.offer_id);
    expect(landmark.evidence.policy_mass).not.toBe(landmark.evidence.search_value);
    expect(landmark.evidence.visits).not.toBe(landmark.evidence.uncertainty);
    expect(JSON.parse(JSON.stringify(artifact))).toEqual(fixture);
  });

  it('preserves every canonical decision while ranking a separate landmark list', () => {
    expect(
      validateRecordedDecisions(recordedDecisionFixture),
      JSON.stringify(validateRecordedDecisions.errors),
    ).toBe(true);
    expect(
      validateDecisionIndex(decisionIndexFixture),
      JSON.stringify(validateDecisionIndex.errors),
    ).toBe(true);
    const recorded = structuredClone(recordedDecisionFixture) as RecordedDecisionInput;
    const index = structuredClone(decisionIndexFixture) as StudyDecisionIndex;
    expect(() => assertViewerSafeRecordedDecisionInput(recorded)).not.toThrow();
    expect(() => assertViewerSafeStudyDecisionIndex(index)).not.toThrow();
    expect(recorded.decision_count).toBe(8);
    expect(index.decisions).toHaveLength(8);
    expect(index.landmarks).toHaveLength(5);
    for (const [ordinal, decision] of index.decisions.entries()) {
      expect(decision.ordinal).toBe(ordinal);
      expect(decision.event_cursor).toBe(recorded.decisions[ordinal].event_cursor);
      expect(decision.frame).toEqual(recorded.decisions[ordinal].frame);
      expect(decision.offer).toEqual(recorded.decisions[ordinal].offer);
      expect(decision.played).toEqual(recorded.decisions[ordinal].played);
    }
    const recommended = new Set(index.landmarks.map(({ decision_id }) => decision_id));
    expect(recommended.has(index.decisions[3].id)).toBe(true);
    expect(recommended.has(index.decisions[4].id)).toBe(false);
    expect(recommended.has(index.decisions[6].id)).toBe(false);
    expect(recommended.has(index.decisions[7].id)).toBe(false);
  });

  it('keeps TypeScript fields and requiredness aligned to the Rust schema', () => {
    expectShape('StudyArtifact', schema);
    for (const name of [
      'AnalysisBudgetIdentity',
      'ContentPackIdentity',
      'DecisionAlternative',
      'DecisionEvidence',
      'EngineIdentity',
      'EvidenceProvenance',
      'ModelIdentity',
      'PolicyMass',
      'SampledWorldRobustness',
      'SearchValue',
      'StudyIdentity',
      'StudyLandmark',
      'UncertaintyEvidence',
      'VisitCount',
    ]) {
      const node = schema.$defs?.[name];
      if (!node) {
        throw new Error(`missing Rust schema definition ${name}`);
      }
      expectShape(name, node);
    }
    expect(schema.$defs?.StudyVersion.minimum).toBe(STUDY_VERSION);
    expect(schema.$defs?.StudyVersion.maximum).toBe(STUDY_VERSION);

    expectShape('RecordedDecisionInput', recordedDecisionSchema);
    expectShape('StudyDecisionIndex', decisionIndexSchema);
    for (const [name, root] of [
      ['RecordedDecision', recordedDecisionSchema],
      ['StudyDecision', decisionIndexSchema],
      ['RankedStudyLandmark', decisionIndexSchema],
    ] as const) {
      const node = root.$defs?.[name];
      if (!node) {
        throw new Error(`missing Rust schema definition ${name}`);
      }
      expectShape(name, node);
    }
  });

  it('rejects opponent-private hand identities at the study boundary', () => {
    const artifact = parsedFixture();
    artifact.landmarks[0].frame.projection.opponent.hand.push({
      id: 99,
      registry_key: 99,
      name: 'Secret Counterspell',
      zone: 'HAND',
      owner_id: 0,
      power: 0,
      toughness: 0,
      mana_value: 2,
      types: {
        is_creature: false,
        is_land: false,
        is_spell: true,
        is_artifact: false,
        is_enchantment: false,
        is_planeswalker: false,
        is_battle: false,
      },
    });
    expect(validate(artifact)).toBe(true);
    expect(() => assertViewerSafeStudyArtifact(artifact)).toThrow(/opponent-private hand/);
  });

  it('rejects RNG sidecars and historical identity drift', () => {
    const rngSecret = parsedFixture() as unknown as Record<string, unknown>;
    const landmarks = rngSecret.landmarks as Array<Record<string, unknown>>;
    const evidence = landmarks[0].evidence as Record<string, unknown>;
    const provenance = evidence.provenance as Record<string, unknown>;
    provenance.rng_seed = 377;
    expect(validate(rngSecret)).toBe(false);

    const promptDrift = parsedFixture();
    promptDrift.landmarks[0].prompt_id += 1;
    expect(validate(promptDrift)).toBe(true);
    expect(() => assertViewerSafeStudyArtifact(promptDrift)).toThrow(
      /viewer, prompt, or offer binding/,
    );
  });

  it('rejects private decisions and malformed landmark references at new boundaries', () => {
    const privateInput = structuredClone(recordedDecisionFixture) as RecordedDecisionInput;
    privateInput.decisions[0].frame.projection.opponent.hand.push({
      id: 99,
      registry_key: 99,
      name: 'Secret Counterspell',
      zone: 'HAND',
      owner_id: 0,
      power: 0,
      toughness: 0,
      mana_value: 2,
      types: {
        is_creature: false,
        is_land: false,
        is_spell: true,
        is_artifact: false,
        is_enchantment: false,
        is_planeswalker: false,
        is_battle: false,
      },
    });
    expect(validateRecordedDecisions(privateInput)).toBe(true);
    expect(() => assertViewerSafeRecordedDecisionInput(privateInput)).toThrow(
      /opponent-private hand/,
    );

    const invalidIndex = structuredClone(decisionIndexFixture) as StudyDecisionIndex;
    invalidIndex.landmarks[0].decision_id = 'missing-decision';
    expect(validateDecisionIndex(invalidIndex)).toBe(true);
    expect(() => assertViewerSafeStudyDecisionIndex(invalidIndex)).toThrow(
      /missing decision/,
    );
  });
});
