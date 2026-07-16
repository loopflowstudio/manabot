import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

import {
  AUTHORITY_STATUSES,
  OFFER_VERBS,
  PRESENTATION_IMPORTANCES,
  PRESENTATION_KIND_NAMES,
  PROTOCOL_VERSION,
  RECOVERY_REASONS,
  type Command,
  type PresentationKind,
  type ProtocolV1ConformanceBundle,
} from './types';

interface SchemaNode {
  $defs?: Record<string, SchemaNode>;
  const?: unknown;
  enum?: unknown[];
  maximum?: number;
  minimum?: number;
  oneOf?: SchemaNode[];
  properties?: Record<string, SchemaNode>;
  required?: string[];
}

interface WireShape {
  fields: string[];
  required: string[];
}

const schema = JSON.parse(
  readFileSync(new URL('../../../protocol/experience-v1.schema.json', import.meta.url), 'utf8'),
) as SchemaNode;
const fixture: unknown = JSON.parse(
  readFileSync(new URL('../../../protocol/fixtures/bolt-target.json', import.meta.url), 'utf8'),
);
const typesSource = ts.createSourceFile(
  'types.ts',
  readFileSync(new URL('./types.ts', import.meta.url), 'utf8'),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TS,
);

const validate = new Ajv2020({ strict: false, validateFormats: false })
  .compile<ProtocolV1ConformanceBundle>(schema);
const definitions = schema.$defs ?? {};

const tsPresentationKinds = [
  {
    kind: 'cast',
    object: { entity: 31, incarnation: 0 },
    controller: 0,
    stack: 4001,
  },
  {
    kind: 'targeted',
    source: { kind: 'stack', id: 4001 },
    target: { kind: 'player', id: 1 },
  },
  { kind: 'resolved', stack: 4001 },
  {
    kind: 'damage',
    source: null,
    target: { kind: 'player', id: 1 },
    amount: 3,
  },
  { kind: 'destroyed', objects: [{ entity: 77, incarnation: 0 }] },
  { kind: 'died', objects: [{ entity: 77, incarnation: 0 }] },
] as const satisfies readonly PresentationKind[];

function definition(name: string): SchemaNode {
  const value = definitions[name];
  if (!value) {
    throw new Error(`Schema definition ${name} is missing`);
  }
  return value;
}

function sorted(values: readonly string[]): string[] {
  return [...values].sort();
}

function propertyName(name: ts.PropertyName): string {
  if (ts.isIdentifier(name) || ts.isStringLiteral(name) || ts.isNumericLiteral(name)) {
    return name.text;
  }
  return name.getText(typesSource);
}

function memberShape(members: readonly ts.TypeElement[]): WireShape {
  const properties = members.filter(ts.isPropertySignature);
  return {
    fields: properties.map((property) => propertyName(property.name)),
    required: properties
      .filter((property) => property.questionToken === undefined)
      .map((property) => propertyName(property.name)),
  };
}

function interfaceDeclaration(name: string): ts.InterfaceDeclaration {
  const declaration = typesSource.statements.find(
    (statement): statement is ts.InterfaceDeclaration =>
      ts.isInterfaceDeclaration(statement) && statement.name.text === name,
  );
  if (!declaration) {
    throw new Error(`TypeScript interface ${name} is missing`);
  }
  return declaration;
}

function interfaceShape(name: string): WireShape {
  return memberShape(interfaceDeclaration(name).members);
}

function inlinePropertyShape(interfaceName: string, fieldName: string): WireShape {
  const property = interfaceDeclaration(interfaceName).members.find(
    (member): member is ts.PropertySignature =>
      ts.isPropertySignature(member) && propertyName(member.name) === fieldName,
  );
  if (!property || !property.type || !ts.isTypeLiteralNode(property.type)) {
    throw new Error(`TypeScript inline object ${interfaceName}.${fieldName} is missing`);
  }
  return memberShape(property.type.members);
}

function taggedTypeAliasShapes(name: string): Record<string, WireShape> {
  const declaration = typesSource.statements.find(
    (statement): statement is ts.TypeAliasDeclaration =>
      ts.isTypeAliasDeclaration(statement) && statement.name.text === name,
  );
  if (!declaration) {
    throw new Error(`TypeScript type alias ${name} is missing`);
  }

  const variants = ts.isUnionTypeNode(declaration.type)
    ? declaration.type.types
    : [declaration.type];
  return Object.fromEntries(
    variants.map((variant) => {
      if (!ts.isTypeLiteralNode(variant)) {
        throw new Error(`${name} contains a non-object variant`);
      }
      const kind = variant.members.find(
        (member): member is ts.PropertySignature =>
          ts.isPropertySignature(member) && propertyName(member.name) === 'kind',
      );
      if (
        !kind?.type
        || !ts.isLiteralTypeNode(kind.type)
        || !ts.isStringLiteral(kind.type.literal)
      ) {
        throw new Error(`${name} contains a variant without a string kind`);
      }
      return [kind.type.literal.text, memberShape(variant.members)];
    }),
  );
}

function schemaTaggedShapes(name: string): Record<string, WireShape> {
  return Object.fromEntries(
    (definition(name).oneOf ?? []).map((variant) => {
      const kind = variant.properties?.kind.const;
      if (typeof kind !== 'string') {
        throw new Error(`Schema ${name} contains a variant without a string kind`);
      }
      return [
        kind,
        {
          fields: Object.keys(variant.properties ?? {}),
          required: variant.required ?? [],
        },
      ];
    }),
  );
}

function expectShape(name: string, actual: WireShape, expected: WireShape): void {
  expect(sorted(actual.fields), `${name} fields`).toEqual(sorted(expected.fields));
  expect(sorted(actual.required), `${name} required fields`).toEqual(
    sorted(expected.required),
  );
}

describe('experience protocol v1 conformance', () => {
  it('round-trips the Rust-certified non-empty fixture through TypeScript types', () => {
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
    expect(fixture.recovery.presentation_tail.map((event) => event.kind.kind)).toEqual(
      PRESENTATION_KIND_NAMES,
    );
    expect(fixture.recovery.presentation_cursor).toBe(900);
    expect(fixture.recovery.presentation_tail.map((event) => event.seq)).toEqual(
      [900, 901, 902, 903, 904, 905],
    );
    const missingCursor = structuredClone(fixture) as unknown as Record<string, unknown>;
    delete (missingCursor.recovery as Record<string, unknown>).presentation_cursor;
    expect(validate(missingCursor)).toBe(false);
    expect(JSON.parse(JSON.stringify(fixture))).toEqual(fixture);
  });

  it('keeps TypeScript versions, enums, fields, and requiredness aligned to schema', () => {
    expect(definition('ProtocolVersion').minimum).toBe(PROTOCOL_VERSION);
    expect(definition('ProtocolVersion').maximum).toBe(PROTOCOL_VERSION);
    expect(definition('OfferVerb').enum).toEqual(OFFER_VERBS);
    expect(definition('AuthorityStatus').enum).toEqual(AUTHORITY_STATUSES);
    expect(definition('PresentationImportance').enum).toEqual(PRESENTATION_IMPORTANCES);
    expect(definition('RecoveryReason').enum).toEqual(RECOVERY_REASONS);

    const interfaceSchemas = {
      ProtocolV1ConformanceBundle: 'ProtocolV1ConformanceBundle',
      AssetPackReference: 'AssetPackReference',
      Candidate: 'Candidate',
      CandidateSource: 'CandidateSource',
      Command: 'Command',
      CommandReceipt: 'CommandReceipt',
      DeckNames: 'DeckNames',
      ExperienceFrame: 'ExperienceFrame',
      InteractionOffer: 'InteractionOffer',
      LegacyCardTypesView: 'CardTypes',
      LegacyCardView: 'CardState',
      LegacyHeroObservation: 'Observation',
      LegacyPermanentView: 'PermanentState',
      LegacyPlayerView: 'PlayerState',
      ObjectRenderId: 'ObjectRenderId',
      PresentationEvent: 'PresentationEvent',
      PromptView: 'PromptView',
      RecoveryEnvelope: 'RecoveryEnvelope',
      StopsConfig: 'StopsConfig',
    } as const;
    for (const [schemaName, interfaceName] of Object.entries(interfaceSchemas)) {
      const node = schemaName === 'ProtocolV1ConformanceBundle'
        ? schema
        : definition(schemaName);
      expectShape(
        schemaName,
        interfaceShape(interfaceName),
        { fields: Object.keys(node.properties ?? {}), required: node.required ?? [] },
      );
    }
    expectShape(
      'LegacyTurnView',
      inlinePropertyShape('Observation', 'turn'),
      {
        fields: Object.keys(definition('LegacyTurnView').properties ?? {}),
        required: definition('LegacyTurnView').required ?? [],
      },
    );

    for (const name of [
      'SubjectRef',
      'CandidateValue',
      'ChoiceStep',
      'ChoiceAnswer',
      'PresentationKind',
    ]) {
      const typeScriptVariants = taggedTypeAliasShapes(name);
      const schemaVariants = schemaTaggedShapes(name);
      expect(sorted(Object.keys(typeScriptVariants)), `${name} variants`).toEqual(
        sorted(Object.keys(schemaVariants)),
      );
      for (const tag of Object.keys(schemaVariants)) {
        expectShape(`${name}.${tag}`, typeScriptVariants[tag], schemaVariants[tag]);
      }
    }
  });

  it('accepts every statically typed presentation variant in the canonical schema', () => {
    expect(validate(fixture)).toBe(true);
    if (!validate(fixture)) {
      throw new Error('fixture did not conform');
    }

    for (const kind of tsPresentationKinds) {
      const candidate = structuredClone(fixture);
      candidate.recovery.presentation_tail[0].kind = kind;
      expect(validate(candidate), `${kind.kind}: ${JSON.stringify(validate.errors)}`).toBe(true);
    }
  });

  it('rejects a payload from another protocol version', () => {
    const invalid = structuredClone(fixture) as Record<string, unknown>;
    (invalid.recovery as Record<string, unknown>).protocol = 2;
    expect(validate(invalid)).toBe(false);
  });

  it('rejects missing required-nullable and unknown presentation fields', () => {
    expect(validate(fixture)).toBe(true);
    if (!validate(fixture)) {
      throw new Error('fixture did not conform');
    }

    const missingNullable = structuredClone(fixture);
    const [first] = missingNullable.recovery.presentation_tail;
    const withoutCause = { ...first } as Partial<typeof first>;
    delete withoutCause.caused_by;
    missingNullable.recovery.presentation_tail[0] = withoutCause as typeof first;
    expect(validate(missingNullable)).toBe(false);

    const unknownKindField = structuredClone(fixture) as ProtocolV1ConformanceBundle;
    const kindWithDrift = unknownKindField.recovery.presentation_tail[0].kind as unknown as Record<
      string,
      unknown
    >;
    kindWithDrift.client_only = true;
    expect(validate(unknownKindField)).toBe(false);
  });
});
