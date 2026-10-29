// content.rs
// Lower compiled semantic IR into the live rules engine's immutable content pack.

use std::{
    collections::BTreeSet,
    sync::{Arc, OnceLock},
};

use serde_json::Value;

use crate::{
    cards::content_pack::{CompiledSemanticManifest, ContentPack},
    state::{
        ability::{
            Ability, Effect, EffectValue, StaticCondition, TargetRequirement, TargetSpec,
            TriggerCondition, TriggerSubject,
        },
        card::{
            ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, CounterThreshold,
            Keywords, PowerCda, ReplacementEffect, Restriction, SacrificeCost, SpellCostReduction,
            StaticKeywordGrant, StaticPtBuff, StaticRestriction, StaticScope, TargetCostReduction,
            TriggeredManaAbility,
        },
        mana::{Color, Colors, Mana, ManaCost},
        permanent::{BecomesCreature, CounterKind},
        player::PlayerConfig,
        predicate::CardPredicate,
    },
};

use super::{
    array_field, object_field, opt_bool_field, opt_str_field, parse_condition, str_array_field,
    str_field, u64_field, Condition, Definition, IrError, Predicate, Program, SemanticPack, Step,
    ValueExpression,
};

#[derive(Clone)]
struct AuthoredRuntime {
    pack_key: String,
    content: Arc<ContentPack>,
    setups: Vec<PlayerConfig>,
}

impl AuthoredRuntime {
    fn compile(semantic: SemanticPack) -> Result<Self, IrError> {
        let content = Arc::new(semantic.compile_content_pack()?);
        let setups = semantic
            .decks
            .iter()
            .map(|deck| semantic.player_config(&deck.key, &deck.key))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pack_key: semantic.pack_key,
            content,
            setups,
        })
    }

    fn matches(&self, player_configs: &[PlayerConfig]) -> bool {
        if player_configs.len() != self.setups.len() {
            return false;
        }
        let mut unmatched = self.setups.iter().collect::<Vec<_>>();
        for config in player_configs {
            let Some(index) = unmatched.iter().position(|setup| {
                setup.decklist == config.decklist && setup.sideboard == config.sideboard
            }) else {
                return false;
            };
            unmatched.remove(index);
        }
        unmatched.is_empty()
    }
}

static AUTHORED_RUNTIME_CATALOG: OnceLock<Result<Vec<AuthoredRuntime>, IrError>> = OnceLock::new();

fn authored_runtime_catalog() -> Result<Vec<AuthoredRuntime>, IrError> {
    AUTHORED_RUNTIME_CATALOG
        .get_or_init(|| {
            SemanticPack::catalog()?
                .iter()
                .cloned()
                .map(AuthoredRuntime::compile)
                .collect()
        })
        .clone()
}

fn select_authored_runtime<'a>(
    player_configs: &[PlayerConfig],
    catalog: &'a [AuthoredRuntime],
) -> Result<Option<&'a AuthoredRuntime>, IrError> {
    if let Some(key) = player_configs.iter().find_map(|p| p.content_pack.as_ref()) {
        let runtime = catalog
            .iter()
            .find(|r| &r.pack_key == key)
            .ok_or_else(|| IrError::Malformed(format!("unknown explicit content pack {key}")))?;
        if player_configs.len() != 2
            || player_configs.iter().any(|p| {
                p.content_pack.as_ref() != Some(key)
                    || !runtime
                        .setups
                        .iter()
                        .any(|s| s.decklist == p.decklist && s.sideboard == p.sideboard)
            })
        {
            return Err(IrError::Malformed(
                "explicit pack requires two admitted authored setups".into(),
            ));
        }
        return Ok(Some(runtime));
    }
    let matches = catalog
        .iter()
        .filter(|runtime| runtime.matches(player_configs))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [runtime] => Ok(Some(runtime)),
        _ => Err(IrError::Malformed(format!(
            "authored content catalog is ambiguous for packs {:?}",
            matches
                .iter()
                .map(|runtime| runtime.pack_key.as_str())
                .collect::<Vec<_>>()
        ))),
    }
}

pub(crate) fn content_pack_for_authored_match(
    player_configs: &[PlayerConfig],
) -> Result<Option<Arc<ContentPack>>, IrError> {
    let catalog = authored_runtime_catalog()?;
    Ok(select_authored_runtime(player_configs, &catalog)?
        .map(|runtime| Arc::clone(&runtime.content)))
}

#[cfg(test)]
mod catalog_tests {
    use super::*;

    fn configs(semantic: &SemanticPack, left: &str, right: &str) -> Vec<PlayerConfig> {
        vec![
            PlayerConfig::new("left", semantic.decklist(left).expect("left deck")),
            PlayerConfig::new("right", semantic.decklist(right).expect("right deck")),
        ]
    }

    #[test]
    fn explicit_mirrors_preserve_pack_and_reject_unadmitted_setups() {
        let catalog = authored_runtime_catalog().unwrap();
        let runtime = catalog
            .iter()
            .find(|r| r.pack_key == "ur-lessons-vs-gw-allies")
            .unwrap();
        for setup in &runtime.setups {
            let mut config = setup.clone();
            config.content_pack = Some(runtime.pack_key.clone());
            let pair = vec![config.clone(), config.clone()];
            assert_eq!(
                select_authored_runtime(&pair, &catalog)
                    .unwrap()
                    .unwrap()
                    .pack_key,
                runtime.pack_key
            );
            let mut wrong = pair.clone();
            wrong[0].content_pack = None;
            assert!(select_authored_runtime(&wrong, &catalog).is_err());
            wrong = pair.clone();
            wrong[0].decklist.clear();
            assert!(select_authored_runtime(&wrong, &catalog).is_err());
            wrong = pair.clone();
            wrong[0].content_pack = Some("missing-pack".into());
            assert!(select_authored_runtime(&wrong, &catalog).is_err());
        }
    }

    #[test]
    fn complexity_roster_is_explicit_and_vanilla() {
        let catalog = authored_runtime_catalog().unwrap();
        let runtime = catalog
            .iter()
            .find(|r| r.pack_key == "mtg122-complexity-v1")
            .unwrap();
        assert_eq!(runtime.setups.len(), 6);
        let semantic = SemanticPack::current("complexity").unwrap();
        for key in ["lea.savannah_lions", "ons.glory_seeker", "lea.hill_giant"] {
            let definition = semantic
                .definitions
                .iter()
                .find(|d| d.semantic_key == key)
                .unwrap();
            assert!(!semantic
                .programs
                .iter()
                .any(|p| p.definition_index == definition.semantic_index));
        }
        for setup in &runtime.setups {
            let mut config = setup.clone();
            config.content_pack = Some(runtime.pack_key.clone());
            assert_eq!(
                select_authored_runtime(&[config.clone(), config], &catalog)
                    .unwrap()
                    .unwrap()
                    .pack_key,
                "mtg122-complexity-v1"
            );
        }
        // An ordinary product pair must still choose the original product pack.
        let pair = catalog
            .iter()
            .find(|r| r.pack_key == "ur-lessons-vs-gw-allies")
            .unwrap()
            .setups
            .clone();
        assert_eq!(
            select_authored_runtime(&pair, &catalog)
                .unwrap()
                .unwrap()
                .pack_key,
            "ur-lessons-vs-gw-allies"
        );
    }

    #[test]
    fn duplicate_exact_catalog_entries_fail_closed() {
        let semantic = SemanticPack::current("jeong_increment").expect("Jeong IR parses");
        let player_configs = configs(&semantic, "ur_lessons", "gw_allies_jeong");
        let runtime = AuthoredRuntime::compile(semantic).expect("Jeong pack compiles");
        let duplicate = runtime.clone();

        let error = select_authored_runtime(&player_configs, &[runtime, duplicate])
            .err()
            .expect("ambiguous catalog must fail closed");
        assert!(error.to_string().contains("catalog is ambiguous"));
    }
}

impl SemanticPack {
    /// Materialize the reviewed typed IR as the immutable pack consumed by a
    /// live [`Game`](crate::Game). No executable registry is consulted: text,
    /// characteristics and programs are carried by the authored content.
    pub fn compile_content_pack(&self) -> Result<ContentPack, IrError> {
        Ok(ContentPack::from_compiled_semantics(
            self.lower_definitions()?,
            CompiledSemanticManifest {
                pack_key: self.pack_key.clone(),
                ir_hash: self.ir_hash.clone(),
                source_hash: self.source_hash.clone(),
            },
        ))
    }

    pub(crate) fn lower_definitions(&self) -> Result<Vec<CardDefinition>, IrError> {
        let mut definitions = Vec::with_capacity(self.definitions.len());
        for definition in &self.definitions {
            let mut compiled = compile_characteristics(definition)?;
            // Frozen IR predating embedded text runs from its own git revision.
            compiled.text_box = definition.text_box.clone().ok_or_else(|| {
                IrError::Malformed(format!("{} has no text_box", definition.semantic_key))
            })?;

            for program in self
                .programs
                .iter()
                .filter(|program| program.definition_index == definition.semantic_index)
            {
                apply_program(self, &mut compiled, program)?;
            }
            definitions.push(compiled);
        }
        Ok(definitions)
    }

    pub fn player_config(&self, name: &str, key: &str) -> Result<PlayerConfig, IrError> {
        let deck = self
            .decks
            .iter()
            .find(|deck| deck.key == key)
            .ok_or_else(|| IrError::Malformed(format!("unknown semantic deck {key:?}")))?;
        let resolve = |entries: &std::collections::BTreeMap<usize, usize>| {
            entries
                .iter()
                .map(|(&index, &count)| {
                    let definition = self.definitions.get(index).ok_or_else(|| {
                        IrError::Malformed(format!(
                            "deck {key:?} references definition index {index}"
                        ))
                    })?;
                    Ok((definition.registry_name.clone(), count))
                })
                .collect::<Result<_, IrError>>()
        };
        Ok(
            PlayerConfig::new(name, resolve(&deck.cards)?)
                .with_sideboard(resolve(&deck.sideboard)?),
        )
    }

    pub fn decklist(
        &self,
        key: &str,
    ) -> Result<std::collections::BTreeMap<String, usize>, IrError> {
        Ok(self.player_config(key, key)?.decklist)
    }
}

fn compile_characteristics(definition: &Definition) -> Result<CardDefinition, IrError> {
    let value = &definition.characteristics;
    let mana_cost = opt_str_field(value, "mana_cost")?.map(ManaCost::parse);
    let types = optional_string_array(value, "types")?
        .into_iter()
        .map(|name| card_type(&name))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let keywords = keywords(&optional_string_array(value, "keywords")?)?;
    let colors = optional_string_array(value, "colors")?;
    let color_override = if value.get("colors").is_none()
        || (colors.is_empty() && !opt_bool_field(value, "token")?.unwrap_or(false))
    {
        None
    } else {
        Some(
            colors
                .iter()
                .map(|name| color(name))
                .collect::<Result<Colors, _>>()?,
        )
    };

    let mut replacement_effects = Vec::new();
    if opt_bool_field(value, "enters_tapped")?.unwrap_or(false) {
        replacement_effects.push(ReplacementEffect::EntersTapped);
    }
    if let Some(predicate) = value.get("enters_tapped_unless") {
        replacement_effects.push(ReplacementEffect::EntersTappedUnlessControls {
            predicate: predicate_value(predicate)?,
        });
    }

    Ok(CardDefinition {
        name: definition.registry_name.clone(),
        mana_cost,
        types: CardTypes { types },
        supertypes: optional_string_array(value, "supertypes")?,
        subtypes: optional_string_array(value, "subtypes")?,
        keywords,
        color_override,
        is_token: opt_bool_field(value, "token")?.unwrap_or(false),
        power: optional_i32(value, "power")?,
        toughness: optional_i32(value, "toughness")?,
        replacement_effects,
        ..CardDefinition::default()
    })
}

fn apply_program(
    pack: &SemanticPack,
    definition: &mut CardDefinition,
    program: &Program,
) -> Result<(), IrError> {
    let targeting = program_targeting(program)?;
    match program.kind_name.as_str() {
        "mana" => {
            let [Step::AddMana { mana, until }] = program.steps.as_slice() else {
                return unsupported(program, "mana program must contain one add_mana step");
            };
            if until.is_some() {
                return unsupported(program, "land mana cannot carry a duration");
            }
            definition
                .mana_abilities
                .push(crate::state::card::ManaAbility {
                    mana: Mana::parse(mana),
                });
        }
        "spell" => {
            let cost = program_cost(program)?;
            let mana_cost = cost
                .mana
                .ok_or_else(|| malformed(program, "spell has no mana cost"))?;
            if definition.mana_cost.as_ref() != Some(&mana_cost) {
                return unsupported(program, "spell cost disagrees with characteristics");
            }
            definition.sacrifice_cost = cost.sacrifice;
            definition.kicker = cost.kicker;
            definition.cost_reduction_per = cost.affinity;
            definition.cost_reduction_if_targets = cost.reduced_if_targets;
            definition.targeting = match opt_str_field(&program.metadata, "target_binding")? {
                None | Some("explicit") => targeting,
                Some("inferred") => Vec::new(),
                Some(_) => return unsupported(program, "unknown target binding"),
            };
            definition.spell_effects = lower_steps(pack, program, &program.steps)?;
            // An Aura spell is the one spell that attaches its card (CR 303.4f).
            let attaches = definition
                .spell_effects
                .iter()
                .any(|effect| matches!(effect, Effect::AttachSourceTo { .. }));
            let aura = definition.subtypes.iter().any(|subtype| subtype == "Aura");
            if attaches != aura || (aura && definition.spell_effects.len() != 1) {
                return unsupported(program, "an Aura spell has exactly one attach instruction");
            }
        }
        "triggered" => {
            let trigger = program_trigger(program)?;
            let effects = lower_steps(pack, program, &program.steps)?;
            if matches!(trigger, TriggerCondition::BecomesTargeted { .. }) {
                let [Effect::CounterUnlessPays { cost }] = effects.as_slice() else {
                    return unsupported(program, "ward trigger must counter unless paid");
                };
                definition.ward = Some(cost.clone());
            } else {
                definition.abilities.push(Ability::Triggered {
                    condition: trigger,
                    effects,
                });
            }
        }
        "triggered_mana" => {
            let trigger = program_trigger(program)?;
            let TriggerCondition::TappedForMana {
                subject: TriggerSubject::AnyYouControl(predicate),
            } = trigger
            else {
                return unsupported(program, "triggered mana has an unsupported trigger");
            };
            let [Step::AddMana { mana, until }] = program.steps.as_slice() else {
                return unsupported(program, "triggered mana must contain one add_mana step");
            };
            if until.is_some() {
                return unsupported(program, "triggered mana cannot carry a duration");
            }
            definition
                .triggered_mana_abilities
                .push(TriggeredManaAbility {
                    predicate,
                    mana: Mana::parse(mana),
                });
        }
        "activated" => {
            let cost = program_cost(program)?;
            let mut effects = lower_steps(pack, program, &program.steps)?;
            // An ability's sequential instructions resolve as one unconditional group.
            let effect = match effects.len() {
                0 => return unsupported(program, "activated ability has no effect"),
                1 => effects.remove(0),
                _ => Effect::IfCondition {
                    condition: StaticCondition::Always,
                    effects,
                    otherwise: Vec::new(),
                },
            };
            definition
                .activated_abilities
                .push(ActivatedAbilityDefinition {
                    sacrifice: cost.sacrifice,
                    own_turn_only: cost.own_turn_only,
                    sorcery_timing: cost.sorcery_timing,
                    life_cost: cost.life,
                    tap_source: cost.tap_source,
                    mana_cost: cost
                        .mana
                        .ok_or_else(|| malformed(program, "activated ability has no mana cost"))?,
                    sacrifice_source: cost.sacrifice_source,
                    waterbend: cost.waterbend,
                    exhaust: cost.exhaust,
                    crew: cost.crew,
                    effect,
                });
        }
        "static" => apply_static(definition, program)?,
        other => return unsupported(program, &format!("unsupported program kind {other:?}")),
    }
    Ok(())
}

#[derive(Default)]
struct ProgramCost {
    mana: Option<ManaCost>,
    kicker: Option<ManaCost>,
    affinity: Option<CardPredicate>,
    reduced_if_targets: Option<TargetCostReduction>,
    sacrifice: Option<SacrificeCost>,
    own_turn_only: bool,
    sorcery_timing: bool,
    life: u32,
    sacrifice_source: bool,
    tap_source: bool,
    waterbend: bool,
    exhaust: bool,
    crew: Option<i32>,
}

fn program_cost(program: &Program) -> Result<ProgramCost, IrError> {
    let Some(value) = program.metadata.get("cost") else {
        return Ok(ProgramCost::default());
    };
    if value.is_null() {
        return Ok(ProgramCost::default());
    }
    let _ = value
        .as_object()
        .ok_or_else(|| malformed(program, "cost must be an object"))?;
    Ok(ProgramCost {
        mana: opt_str_field(value, "mana")?.map(ManaCost::parse),
        kicker: opt_str_field(value, "kicker")?.map(ManaCost::parse),
        affinity: value.get("affinity").map(predicate_value).transpose()?,
        reduced_if_targets: value
            .get("reduced_if_targets")
            .map(|reduction| {
                Ok::<_, IrError>(TargetCostReduction {
                    predicate: predicate_value(object_field(reduction, "predicate")?)?,
                    generic: reduction
                        .get("generic")
                        .and_then(Value::as_u64)
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| malformed(program, "invalid target cost reduction"))?,
                })
            })
            .transpose()?,
        sacrifice: value
            .get("sacrifice")
            .map(|cost| {
                Ok::<_, IrError>(SacrificeCost {
                    predicate: predicate_value(object_field(cost, "predicate")?)?,
                    another: opt_bool_field(cost, "another")?
                        .ok_or_else(|| malformed(program, "missing another"))?,
                    or_mana: match cost
                        .get("or_mana")
                        .ok_or_else(|| malformed(program, "missing sacrifice alternative"))?
                    {
                        Value::Null => None,
                        Value::String(mana) => Some(ManaCost::parse(mana)),
                        _ => return Err(malformed(program, "invalid sacrifice alternative")),
                    },
                })
            })
            .transpose()?,
        own_turn_only: opt_bool_field(value, "own_turn_only")?.unwrap_or(false),
        sorcery_timing: opt_bool_field(value, "sorcery_timing")?.unwrap_or(false),
        life: value
            .get("life")
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| malformed(program, "invalid life cost"))
            })
            .transpose()?
            .unwrap_or(0),
        tap_source: opt_bool_field(value, "tap_source")?.unwrap_or(false),
        sacrifice_source: opt_bool_field(value, "sacrifice_source")?.unwrap_or(false),
        waterbend: opt_bool_field(value, "waterbend")?.unwrap_or(false),
        exhaust: opt_bool_field(value, "exhaust")?.unwrap_or(false),
        crew: value
            .get("crew")
            .map(|power| {
                power
                    .as_i64()
                    .and_then(|n| i32::try_from(n).ok())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| malformed(program, "invalid crew power"))
            })
            .transpose()?,
    })
}

fn program_targeting(program: &Program) -> Result<Vec<TargetRequirement>, IrError> {
    array_field(&program.metadata, "targets")?
        .iter()
        .map(|target| parse_target_requirement(program, target))
        .collect()
}

fn target_spec(selector: &Value) -> Result<TargetSpec, IrError> {
    let result = match str_field(selector, "kind")? {
        "creature" => TargetSpec::Creature,
        "player" => TargetSpec::Player,
        "opponent" => TargetSpec::Opponent,
        "permanent" => TargetSpec::Permanent {
            predicate: predicate_value(object_field(selector, "predicate")?)?,
        },
        "permanent_you_control" => TargetSpec::PermanentYouControl {
            predicate: predicate_value(object_field(selector, "predicate")?)?,
        },
        "graveyard_card_you_own" => TargetSpec::GraveyardCardYouOwn {
            predicate: predicate_value(object_field(selector, "predicate")?)?,
        },
        "creature_or_player" => TargetSpec::CreatureOrPlayer,
        "spell" => TargetSpec::Spell,
        "spell_or_permanent" => TargetSpec::SpellOrPermanent {
            min_mana_value: u8::try_from(u64_field(selector, "min_mana_value")?)
                .map_err(|_| IrError::Malformed("target mana value exceeds u8".to_owned()))?,
        },
        "creature_you_control" => TargetSpec::CreatureYouControl,
        "creature_opponent_controls" => TargetSpec::CreatureOpponentControls,
        "land_you_control" => TargetSpec::LandYouControl,
        "permanent_opponent_controls" => TargetSpec::PermanentOpponentControls {
            predicate: CardPredicate {
                card_types_any: optional_string_array(selector, "types_any")?
                    .iter()
                    .map(|name| card_type(name))
                    .collect::<Result<Vec<_>, _>>()?,
                min_mana_value: Some(
                    u8::try_from(u64_field(selector, "min_mana_value")?).map_err(|_| {
                        IrError::Malformed("target mana value exceeds u8".to_owned())
                    })?,
                ),
                ..CardPredicate::default()
            },
        },
        other => {
            return Err(IrError::Malformed(format!(
                "unsupported target selector {other:?}"
            )))
        }
    };
    Ok(result)
}

fn target_for(program: &Program, role: &str) -> Result<TargetSpec, IrError> {
    if role == "current_target" {
        return program_targeting(program)?
            .into_iter()
            .next()
            .map(|requirement| requirement.spec)
            .ok_or_else(|| malformed(program, "current_target has no target requirement"));
    }
    for target in array_field(&program.metadata, "targets")? {
        if str_field(target, "role")? == role {
            return target_spec(object_field(target, "selector")?);
        }
    }
    Err(malformed(
        program,
        &format!("instruction references unknown target role {role:?}"),
    ))
}

fn program_trigger(program: &Program) -> Result<TriggerCondition, IrError> {
    let trigger = object_field(&program.metadata, "trigger")?;
    let subject = trigger_subject(object_field(trigger, "subject")?)?;
    let base = match str_field(trigger, "event")? {
        "sacrificed" => TriggerCondition::Sacrificed {
            predicate: predicate_value(object_field(trigger, "predicate")?)?,
            controller_only: opt_bool_field(trigger, "controller_only")?.unwrap_or(true),
        },
        "cast_enters" => TriggerCondition::WasCast {
            condition: Box::new(TriggerCondition::EntersTheBattlefield { subject }),
        },
        "enters_battlefield" => TriggerCondition::EntersTheBattlefield { subject },
        "dies" => TriggerCondition::Dies { subject },
        "attacks" => TriggerCondition::Attacks { subject },
        "becomes_tapped" => TriggerCondition::BecomesTapped { subject },
        "tapped_for_mana" => TriggerCondition::TappedForMana { subject },
        "becomes_targeted" => TriggerCondition::BecomesTargeted { subject },
        "upkeep_begins" => TriggerCondition::BeginningOfYourUpkeep,
        "combat_begins" => TriggerCondition::BeginningOfYourCombat,
        "cast_spell" => TriggerCondition::YouCastSpell {
            predicate: predicate_value(object_field(trigger, "predicate")?)?,
        },
        "draw_nth_card" => TriggerCondition::YouDrawNthCardThisTurn {
            n: u32::try_from(u64_field(trigger, "n")?)
                .map_err(|_| malformed(program, "draw count exceeds u32"))?,
        },
        other => return unsupported(program, &format!("unsupported trigger event {other:?}")),
    };
    let base = if let Some(condition) = trigger.get("if") {
        TriggerCondition::ActiveIf {
            active_if: lower_condition(&parse_condition(condition)?)?,
            condition: Box::new(base),
        }
    } else {
        base
    };
    let base = if opt_bool_field(trigger, "own_turn_only")?.unwrap_or(false) {
        TriggerCondition::ActiveIf {
            active_if: StaticCondition::YourTurn,
            condition: Box::new(base),
        }
    } else {
        base
    };
    Ok(
        if opt_bool_field(trigger, "once_each_turn")?.unwrap_or(false) {
            TriggerCondition::OnceEachTurn {
                condition: Box::new(base),
            }
        } else {
            base
        },
    )
}

fn trigger_subject(value: &Value) -> Result<TriggerSubject, IrError> {
    match str_field(value, "kind")? {
        "this" => Ok(TriggerSubject::This),
        "another_you_control" => Ok(TriggerSubject::AnotherYouControl(predicate_value(
            object_field(value, "predicate")?,
        )?)),
        "any_you_control" => Ok(TriggerSubject::AnyYouControl(predicate_value(
            object_field(value, "predicate")?,
        )?)),
        other => Err(IrError::Malformed(format!(
            "unsupported trigger subject {other:?}"
        ))),
    }
}

fn apply_static(definition: &mut CardDefinition, program: &Program) -> Result<(), IrError> {
    match program.steps.as_slice() {
        [Step::GrantFlashback] => definition.grants_flashback = true,
        [Step::GrantMana { predicates }] => {
            definition.mana_grant = predicates
                .iter()
                .map(predicate_card)
                .collect::<Result<_, _>>()?
        }
        [Step::UnblockableIf { condition }] => {
            definition.unblockable_if = Some(lower_condition(condition)?);
        }
        [Step::PreventLifeGain] => definition.prevents_life_gain = true,
        [Step::CombatRequiresOther { predicate }] => {
            definition.combat_requires_other = Some(predicate_card(predicate)?)
        }
        [Step::BlockingRestriction {
            predicate,
            direction,
        }] => match direction.as_str() {
            "blocking" => definition.blocking_restriction = Some(predicate_card(predicate)?),
            "blocked_by" => definition.block_restriction = Some(predicate_card(predicate)?),
            _ => return unsupported(program, "unknown blocking direction"),
        },
        [Step::PowerFromLandCounters] => {
            definition.power_cda = Some(PowerCda::CountersOnLandsYouControl)
        }
        [Step::SpellCostReduction {
            predicate,
            condition,
            generic,
            first_on_own_turn,
        }] => definition.spell_cost_reductions.push(SpellCostReduction {
            predicate: predicate_card(predicate)?,
            condition: lower_condition(condition)?,
            generic: u8::try_from(*generic)
                .map_err(|_| malformed(program, "cost reduction exceeds u8"))?,
            first_on_own_turn: *first_on_own_turn,
        }),
        [Step::StaticPt {
            power,
            toughness,
            condition,
        }] => definition.static_pt_buffs.push(StaticPtBuff {
            scope: StaticScope::This,
            condition: Some(lower_condition(condition)?),
            power: *power,
            toughness: *toughness,
            per_creature_type_max: None,
        }),
        [Step::SetPowerFromCount {
            zone,
            controller,
            predicate,
        }] if controller == "you" && zone == "battlefield" => {
            if predicate_card(predicate)? != CardPredicate::creature() {
                return unsupported(program, "battlefield power count must count creatures");
            }
            definition.power_cda = Some(PowerCda::CreaturesYouControl);
        }
        [Step::SetPowerFromCount {
            zone,
            controller,
            predicate,
        }] if controller == "you" && zone == "your_graveyard" => {
            definition.power_cda = Some(PowerCda::GraveyardMatching(predicate_card(predicate)?));
        }
        [Step::Branch {
            condition: Condition::SourceCountersAtLeast { count, counter },
            then,
            otherwise,
        }] if otherwise.is_empty()
            && matches!(
                then.as_slice(),
                [Step::BecomeArtifactCreature { target, duration }]
                    if target == "source" && duration.as_deref() == Some("static")
            ) =>
        {
            definition.creature_with_counters = Some(CounterThreshold {
                kind: counter_kind(program, counter)?,
                count: i32::try_from(*count)
                    .map_err(|_| malformed(program, "counter threshold exceeds i32"))?,
            });
        }
        [Step::Branch {
            condition,
            then,
            otherwise,
        }] if otherwise.is_empty() => {
            let [step] = then.as_slice() else {
                return unsupported(program, "conditional static must have one effect");
            };
            apply_static_effect(definition, program, step, Some(lower_condition(condition)?))?;
        }
        [step @ (Step::ModifyPt { .. } | Step::GrantKeywords { .. })] => {
            apply_static_effect(definition, program, step, None)?
        }
        [Step::Branch {
            condition,
            then,
            otherwise,
        }] if then.is_empty() => {
            let [Step::CantAttackOrBlock { target }] = otherwise.as_slice() else {
                return unsupported(
                    program,
                    "an unless-static must restrict attacking and blocking",
                );
            };
            definition.static_restrictions.push(StaticRestriction {
                scope: static_scope(program, target)?,
                kind: Restriction::CantAttackOrBlock,
                unless: Some(lower_condition(condition)?),
            })
        }
        [Step::CantAttackOrBlock { target }] => {
            definition.static_restrictions.push(StaticRestriction {
                scope: static_scope(program, target)?,
                kind: Restriction::CantAttackOrBlock,
                unless: None,
            })
        }
        [Step::DoesntUntap { target }] => definition.static_restrictions.push(StaticRestriction {
            scope: static_scope(program, target)?,
            kind: Restriction::DoesntUntap,
            unless: None,
        }),
        _ => return unsupported(program, "unsupported static program shape"),
    }
    Ok(())
}

/// A static P/T modification or keyword grant over an affected set.
fn apply_static_effect(
    definition: &mut CardDefinition,
    program: &Program,
    step: &Step,
    condition: Option<StaticCondition>,
) -> Result<(), IrError> {
    match step {
        Step::ModifyPt {
            target,
            power,
            toughness,
            duration,
            per_creature_type_max,
        } if duration == "static" => definition.static_pt_buffs.push(StaticPtBuff {
            scope: static_scope(program, target)?,
            condition,
            power: i32_value(program, *power, "power")?,
            toughness: i32_value(program, *toughness, "toughness")?,
            per_creature_type_max: per_creature_type_max
                .map(|max| i32_value(program, max, "creature-type maximum"))
                .transpose()?,
        }),
        Step::GrantKeywords {
            target,
            keywords: names,
            duration,
        } if duration == "static" => definition.static_keyword_grants.push(StaticKeywordGrant {
            scope: static_scope(program, target)?,
            condition,
            keywords: keywords(names)?,
        }),
        _ => {
            return unsupported(
                program,
                "static effect must be a static P/T or keyword grant",
            )
        }
    }
    Ok(())
}

/// "Equipped creature" and "enchanted creature" name the exact object the
/// source is attached to.
const ATTACHED_EQUIPMENT: &str = "each:equipped_creature";
const ATTACHED_AURA: &str = "each:enchanted_creature";

/// The permanents a static effect's target reference covers.
fn static_scope(program: &Program, target: &str) -> Result<StaticScope, IrError> {
    let creature = CardPredicate::creature;
    Ok(match target {
        "source" => StaticScope::This,
        "each:other_allies_you_control" => {
            StaticScope::OtherYouControl(CardPredicate::subtype("Ally"))
        }
        "each:other_permanent_you_control" => {
            StaticScope::OtherYouControl(CardPredicate::default())
        }
        "each:creature_you_control_with_flying" => StaticScope::YouControl(CardPredicate {
            flying: true,
            ..creature()
        }),
        "each:creature_you_control_with_counters" => {
            StaticScope::YouControlWithCounters(creature())
        }
        "each:non_human_creature_you_control" => StaticScope::YouControl(CardPredicate {
            not_subtypes: vec!["Human".into()],
            ..creature()
        }),
        ATTACHED_EQUIPMENT | ATTACHED_AURA => StaticScope::Attached,
        _ => return unsupported(program, &format!("unsupported static scope {target:?}")),
    })
}

/// Authored counter kind; absence is the +1/+1 counter.
fn counter_kind(program: &Program, counter: &Option<String>) -> Result<CounterKind, IrError> {
    match counter.as_deref() {
        None => Ok(CounterKind::PlusOnePlusOne),
        Some(name) => CounterKind::parse(name)
            .ok_or_else(|| malformed(program, &format!("unknown counter kind {name:?}"))),
    }
}

fn lower_steps(
    pack: &SemanticPack,
    program: &Program,
    steps: &[Step],
) -> Result<Vec<Effect>, IrError> {
    steps
        .iter()
        .map(|step| lower_step(pack, program, step))
        .collect()
}

fn lower_step(pack: &SemanticPack, program: &Program, step: &Step) -> Result<Effect, IrError> {
    let effect = match step {
        Step::GrantFlashback | Step::GrantMana { .. } => {
            return unsupported(
                program,
                "mana and flashback grants require a static program",
            )
        }
        Step::CantAttackOrBlock { .. } | Step::DoesntUntap { .. } => {
            return unsupported(program, "restrictions require a static program")
        }
        Step::BecomeArtifactCreature { target, duration } if target == "source" => {
            Effect::BecomeCreatureSource {
                duration: match duration.as_deref() {
                    Some("end_of_turn") => BecomesCreature::EndOfTurn,
                    None => BecomesCreature::Incarnation,
                    Some(other) => {
                        return unsupported(
                            program,
                            &format!("unsupported animation duration {other:?}"),
                        )
                    }
                },
            }
        }
        Step::BecomeArtifactCreature { .. } => {
            return unsupported(program, "only the source becomes an artifact creature")
        }
        Step::Exile { target } if target == ATTACHED_AURA => Effect::ExileAttached,
        Step::ShuffleIntoLibrary { target } if target == ATTACHED_AURA => {
            Effect::ShuffleAttachedIntoLibrary
        }
        Step::ShuffleIntoLibrary { .. } => {
            return unsupported(program, "shuffle_into_library reads the enchanted creature")
        }
        Step::AddLessonMana { mana } => {
            let mut mana = Mana::parse(mana);
            mana.lesson = mana.mana;
            mana.mana = [0; 6];
            Effect::AddMana {
                mana,
                until_end_of_combat: false,
            }
        }
        Step::CreateNonlegendaryCopy => Effect::CreateNonlegendaryCopy,
        Step::ModifyControlled {
            predicate,
            power,
            toughness,
        } => Effect::ModifyControlled {
            predicate: predicate_card(predicate)?,
            power: i32::try_from(*power).map_err(|_| malformed(program, "power exceeds i32"))?,
            toughness: i32::try_from(*toughness)
                .map_err(|_| malformed(program, "toughness exceeds i32"))?,
        },
        Step::Discard => Effect::Discard,
        Step::ExileIfDies { target } => Effect::ExileIfDiesThisTurn {
            target: target_for(program, target)?,
        },
        Step::AddSubtype { target, subtype } => Effect::AddSubtypeToTarget {
            target: target_for(program, target)?,
            subtype: subtype.clone(),
        },
        Step::OptionalPayment {
            cost,
            if_paid,
            if_declined,
        } => Effect::OptionalPayment {
            cost: cost.clone(),
            if_paid: lower_steps(pack, program, if_paid)?,
            if_declined: lower_steps(pack, program, if_declined)?,
        },
        Step::UnblockableIf { .. } => {
            return unsupported(program, "unblockable_if requires static program")
        }
        Step::AddMana { mana, until } => Effect::AddMana {
            mana: Mana::parse(mana),
            until_end_of_combat: match until.as_deref() {
                None => false,
                Some("end_of_combat") => true,
                Some(other) => {
                    return unsupported(program, &format!("unsupported duration {other:?}"))
                }
            },
        },
        Step::DrawCards { count } => Effect::DrawCards {
            count: usize_value(program, *count, "draw count")?,
        },
        Step::PutTopCardsInHand { count } => Effect::PutTopCardsInHand {
            count: usize_value(program, *count, "hand count")?,
        },
        Step::CreateToken {
            count,
            attach_source: true,
            ..
        } if *count != 1 => {
            return unsupported(program, "an Equipment attaches to one created token")
        }
        Step::CreateToken {
            definition_index,
            count,
            tapped_and_attacking,
            attach_source,
        } => Effect::CreateToken {
            token_name: token_name(pack, *definition_index)?,
            count: usize_value(program, *count, "token count")?,
            tapped_and_attacking: *tapped_and_attacking,
            attach_source: *attach_source,
        },
        Step::PutCounters {
            target,
            count,
            counter,
        } if target == "source" => Effect::PutCountersOnSource {
            count: i32_value(program, *count, "counter count")?,
            kind: counter_kind(program, counter)?,
        },
        Step::PutCounters {
            target,
            count,
            counter: None,
        } if target == "each:other_ally_creature_you_control" => {
            Effect::PutCountersOnEachMatching {
                count: i32_value(program, *count, "counter count")?,
                predicate: CardPredicate {
                    card_type: Some(CardType::Creature),
                    subtype: Some("Ally".to_owned()),
                    ..CardPredicate::default()
                },
                other: true,
            }
        }
        Step::PutCounters {
            target,
            count,
            counter: None,
        } if target == "each:creature_you_control" => Effect::PutCountersOnEachMatching {
            count: i32_value(program, *count, "counter count")?,
            predicate: CardPredicate::creature(),
            other: false,
        },
        Step::PutCounters {
            target,
            count,
            counter,
        } if !target.starts_with("each:") => Effect::PutCounters {
            count: i32_value(program, *count, "counter count")?,
            kind: counter_kind(program, counter)?,
            target: target_for(program, target)?,
        },
        Step::PutCounters { .. } => {
            return unsupported(program, "typed counters on each matching permanent")
        }
        Step::GainLife { amount } => Effect::GainLife {
            amount: i32_value(program, *amount, "life amount")?,
        },
        Step::Scry { count } => Effect::Scry {
            count: usize_value(program, *count, "scry count")?,
        },
        Step::Tap { target } if target == "source" => Effect::TapSource,
        Step::RemoveCounters { target, count } => Effect::RemoveCounters {
            up_to: i32_value(program, *count, "counter count")?,
            chosen: None,
            target: target_for(program, target)?,
        },
        Step::Tap { target } => Effect::TapTarget {
            target: target_for(program, target)?,
        },
        Step::Attach { target } => Effect::AttachSourceTo {
            target: target_for(program, target)?,
        },
        Step::Untap { target } if target == "source" => Effect::UntapSource,
        Step::Untap { target } => Effect::UntapTarget {
            target: target_for(program, target)?,
        },
        Step::ModifyPt {
            target,
            power,
            toughness,
            duration,
            per_creature_type_max: None,
        } if duration == "end_of_turn" && target == "source" => Effect::ModifyUntilEot {
            power_delta: i32_value(program, *power, "power")?,
            toughness_delta: i32_value(program, *toughness, "toughness")?,
        },
        Step::ModifyPt {
            target,
            power,
            toughness,
            duration,
            per_creature_type_max: None,
        } if duration == "end_of_turn" => Effect::BuffTarget {
            power: i32_value(program, *power, "power")?,
            toughness: i32_value(program, *toughness, "toughness")?,
            target: target_for(program, target)?,
        },
        Step::SetBasePt {
            target,
            power,
            toughness,
            duration,
        } if duration == "end_of_turn" && target == "source" => Effect::SetBasePtUntilEot {
            power: power
                .map(|p| i32_value(program, p, "base power"))
                .transpose()?,
            toughness: toughness
                .map(|t| i32_value(program, t, "base toughness"))
                .transpose()?,
        },
        Step::SetBasePt { .. } => {
            return unsupported(
                program,
                "base P/T is set only on the source until end of turn",
            )
        }
        Step::ModifyPt { duration, .. } => {
            return unsupported(program, &format!("unsupported P/T duration {duration:?}"))
        }
        Step::GrantKeywords {
            target,
            keywords: names,
            duration,
        } if duration == "end_of_turn" && target == "source" => Effect::GrantKeywordsToSource {
            keywords: keywords(names)?,
        },
        Step::GrantKeywords {
            target,
            keywords: names,
            duration,
        } if duration == "end_of_turn" => Effect::GrantKeywordsToTarget {
            keywords: keywords(names)?,
            target: target_for(program, target)?,
        },
        Step::GrantKeywords { duration, .. } => {
            return unsupported(
                program,
                &format!("unsupported keyword duration {duration:?}"),
            )
        }
        Step::RestrictBlocking {
            target,
            predicate,
            duration,
        } if target == "source"
            && duration == "end_of_turn"
            && matches!(predicate, Predicate::Any) =>
        {
            Effect::CantBeBlockedThisTurnSource
        }
        Step::RestrictBlocking { .. } => {
            return unsupported(program, "unsupported blocking restriction")
        }
        Step::DealDamage { amount, target } => Effect::DealDamage {
            amount: i32_value(program, *amount, "damage")?,
            target: target_for(program, target)?,
        },
        Step::ReturnToHand { target } => Effect::ReturnToHand {
            target: target_for(program, target)?,
        },
        Step::Learn => Effect::Learn,
        Step::CounterUnlessPays { cost, .. } => Effect::CounterUnlessPays {
            cost: ManaCost::parse(cost),
        },
        Step::LookAndSelect {
            reveal,
            look,
            min_select,
            max_select,
            destination,
            predicate,
        } if destination == "hand" => Effect::LookAndSelect {
            reveal: *reveal,
            look: usize_value(program, *look, "look count")?,
            min_select: usize_value(program, *min_select, "minimum selection")?,
            max_select: usize_value(program, *max_select, "maximum selection")?,
            predicate: predicate_card(predicate)?,
        },
        Step::LookAndSelect { destination, .. } => {
            return unsupported(program, &format!("unsupported destination {destination:?}"))
        }
        Step::Branch {
            condition: Condition::Kicked,
            then,
            otherwise,
        } => Effect::IfKicked {
            then: lower_steps(pack, program, then)?,
            otherwise: lower_steps(pack, program, otherwise)?,
        },
        Step::Branch {
            condition: Condition::NthResolution(n),
            then,
            otherwise,
        } if otherwise.is_empty() => {
            let effects = lower_steps(pack, program, then)?;
            let [effect] = effects.as_slice() else {
                return unsupported(program, "nth-resolution branch must contain one effect");
            };
            Effect::OnNthResolutionThisTurn {
                n: u32::try_from(*n)
                    .map_err(|_| malformed(program, "resolution count exceeds u32"))?,
                effect: Box::new(effect.clone()),
            }
        }
        Step::Branch {
            condition: Condition::GraveyardAtLeast { count, predicate },
            then,
            otherwise,
        } => Effect::IfGraveyardAtLeast {
            count: usize::try_from(*count)
                .map_err(|_| malformed(program, "graveyard count exceeds usize"))?,
            predicate: predicate_card(predicate)?,
            then: lower_steps(pack, program, then)?,
            otherwise: lower_steps(pack, program, otherwise)?,
        },
        Step::Branch {
            condition: Condition::TargetMatches { predicate, .. },
            then,
            otherwise,
        } if otherwise.is_empty() => Effect::IfTargetMatches {
            predicate: predicate_card(predicate)?,
            then: lower_steps(pack, program, then)?,
        },
        Step::Branch {
            condition: condition @ Condition::ControlsMatching { .. },
            then,
            otherwise,
        } => Effect::IfCondition {
            condition: lower_condition(condition)?,
            effects: lower_steps(pack, program, then)?,
            otherwise: lower_steps(pack, program, otherwise)?,
        },
        Step::Branch { .. } => return unsupported(program, "unsupported branch shape"),
        Step::ForEachTarget { body, .. } => Effect::ForEachTarget {
            effects: lower_steps(pack, program, body)?,
        },
        Step::Earthbend { target, count } => Effect::Earthbend {
            target: target_for(program, target)?,
            count: i32_value(program, *count, "earthbend count")?,
        },
        Step::ExileUntilSourceLeaves { target } => Effect::ExileUntilSourceLeaves {
            target: target_for(program, target)?,
        },
        Step::DealPowerDamage { .. } => Effect::TargetCreaturesDealPowerDamageToLastTarget,
        Step::SearchBasicLands {
            count,
            tapped,
            untap_if_lands,
            bottom_source,
        } => Effect::SearchBasicLand {
            subtype: None,
            count: *count,
            tapped: *tapped,
            untap_if_lands: *untap_if_lands,
            bottom_source: *bottom_source,
        },
        Step::Sacrifice { predicate } => Effect::Sacrifice {
            predicate: predicate_card(predicate)?,
        },
        Step::DealDamageToController { amount } => Effect::DealDamageToController {
            amount: i32::try_from(*amount).map_err(|_| malformed(program, "damage exceeds i32"))?,
        },
        Step::DrainLife { amount } => Effect::DrainLife {
            amount: i32::try_from(*amount).map_err(|_| malformed(program, "drain exceeds i32"))?,
        },
        Step::SearchBasicLand { subtype } => Effect::SearchBasicLand {
            subtype: subtype.clone(),
            count: 1,
            tapped: true,
            untap_if_lands: None,
            bottom_source: false,
        },
        Step::OptionalDiscardDraw { count } => Effect::OptionalDiscardDraw { count: *count },
        Step::DiscardTargetPlayer { count, target } => Effect::DiscardTargetPlayer {
            count: *count,
            target: target_for(program, target)?,
        },
        Step::Destroy { target } => Effect::Destroy {
            target: target_for(program, target)?,
        },
        Step::Exile { target } => Effect::Exile {
            target: target_for(program, target)?,
        },
        Step::BuffTargetValue {
            power,
            toughness,
            target,
        } => Effect::BuffTargetValue {
            power: power.lower()?,
            toughness: *toughness,
            target: target_for(program, target)?,
        },
        Step::ReturnToHandDrawIfControlled { target } => Effect::ReturnToHandDrawIfControlled {
            target: target_for(program, target)?,
        },
        Step::ModifyTriggeringPermanent {
            counters,
            keywords: names,
        } => Effect::ModifyTriggeringPermanent {
            counters: *counters,
            keywords: keywords(names)?,
        },
        Step::AddManaOfOneColor { amount } => Effect::AddOneColorManaValue {
            amount: amount.lower()?,
        },
        Step::DealDamageValue { amount, target } => Effect::DealDamageValue {
            amount: amount.lower()?,
            target: target_for(program, target)?,
        },
        Step::PutCountersValue { count, target } => Effect::PutCountersValue {
            count: count.lower()?,
            target: target_for(program, target)?,
        },
        Step::CreateTokensValue {
            definition_index,
            count,
        } => Effect::CreateTokensValue {
            token_name: token_name(pack, *definition_index)?,
            count: count.lower()?,
        },
        Step::ExileWithControllerToken {
            target,
            definition_index,
        } => Effect::ExileWithControllerToken {
            target: target_for(program, target)?,
            token_name: token_name(pack, *definition_index)?,
        },
        Step::DamageOpponentCreatures { amount } => {
            Effect::DamageOpponentCreatures { amount: *amount }
        }
        Step::MillCards { count } => Effect::MillCards { count: *count },
        Step::MillAndSelect {
            count,
            predicate,
            if_declined,
        } => Effect::MillAndSelect {
            count: *count,
            predicate: predicate_card(predicate)?,
            if_declined: lower_steps(pack, program, if_declined)?,
        },
        Step::DrainLifeValue { amount } => Effect::DrainLifeValue {
            amount: amount.lower()?,
        },
        Step::CounterSpell { target } => Effect::CounterSpell {
            target: target_for(program, target)?,
        },
        Step::CantBeBlockedTarget { target } => Effect::CantBeBlockedTarget {
            target: target_for(program, target)?,
        },
        Step::ReturnGraveyardTarget { target } => Effect::ReturnGraveyardTarget {
            target: target_for(program, target)?,
        },
        Step::Modal { modes } => Effect::Modal {
            modes: modes
                .iter()
                .map(|mode| lower_steps(pack, program, mode))
                .collect::<Result<_, _>>()?,
        },
        Step::TargetClause { role, body } => Effect::TargetClause {
            requirement: target_requirement(program, role)?,
            effects: lower_steps(pack, program, body)?,
        },
        Step::PreventLifeGain
        | Step::CombatRequiresOther { .. }
        | Step::BlockingRestriction { .. }
        | Step::PowerFromLandCounters
        | Step::SpellCostReduction { .. }
        | Step::StaticPt { .. } => {
            return unsupported(program, "static instruction in effect program")
        }
        Step::SetPowerFromCount { .. } => {
            return unsupported(
                program,
                "set_power_from_count is only valid in a static program",
            )
        }
    };
    Ok(effect)
}

fn predicate_value(value: &Value) -> Result<CardPredicate, IrError> {
    let parsed = super::parse_predicate(value)?;
    predicate_card(&parsed)
}

fn predicate_card(predicate: &Predicate) -> Result<CardPredicate, IrError> {
    let mut result = CardPredicate::default();
    merge_predicate(&mut result, predicate)?;
    Ok(result)
}

fn merge_predicate(result: &mut CardPredicate, predicate: &Predicate) -> Result<(), IrError> {
    match predicate {
        Predicate::Flying => result.flying = true,
        Predicate::NoCounters => result.no_counters = true,
        Predicate::PowerAtLeast(value) => set_once(
            &mut result.min_power,
            i32::try_from(*value).map_err(|_| IrError::Malformed("power exceeds i32".into()))?,
            "min_power",
        )?,
        Predicate::ManaValueAtLeast(value) => {
            set_once(&mut result.min_mana_value, *value, "min_mana_value")?
        }
        Predicate::TypesAny(names) => {
            result.card_types_any = names
                .iter()
                .map(|name| card_type(name))
                .collect::<Result<_, _>>()?
        }
        Predicate::SubtypesAny(names) => result.subtypes_any = names.clone(),
        Predicate::NotSubtypes(names) => result.not_subtypes = names.clone(),
        Predicate::Any => {}
        Predicate::All(predicates) => {
            for predicate in predicates {
                merge_predicate(result, predicate)?;
            }
        }
        Predicate::CardType(name) => {
            set_once(&mut result.card_type, card_type(name)?, "card_type")?
        }
        Predicate::Subtype(name) => set_once(&mut result.subtype, name.clone(), "subtype")?,
        Predicate::Supertype(name) => set_once(&mut result.supertype, name.clone(), "supertype")?,
        Predicate::NotCardTypes(names) => {
            for name in names {
                let value = card_type(name)?;
                if !result.not_card_types.contains(&value) {
                    result.not_card_types.push(value);
                }
            }
        }
        Predicate::PowerAtMost(value) => set_once(
            &mut result.max_power,
            i32::try_from(*value)
                .map_err(|_| IrError::Malformed("power predicate exceeds i32".to_owned()))?,
            "max_power",
        )?,
    }
    Ok(())
}

fn set_once<T: PartialEq>(slot: &mut Option<T>, value: T, name: &str) -> Result<(), IrError> {
    if slot.as_ref().is_some_and(|existing| existing != &value) {
        return Err(IrError::Malformed(format!(
            "predicate contains conflicting {name} values"
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn keywords(names: &[String]) -> Result<Keywords, IrError> {
    let mut result = Keywords::default();
    for name in names {
        match name.as_str() {
            "flying" => result.flying = true,
            "reach" => result.reach = true,
            "haste" => result.haste = true,
            "flash" => result.flash = true,
            "vigilance" => result.vigilance = true,
            "trample" => result.trample = true,
            "first_strike" => result.first_strike = true,
            "double_strike" => result.double_strike = true,
            "deathtouch" => result.deathtouch = true,
            "lifelink" => result.lifelink = true,
            "defender" => result.defender = true,
            "menace" => result.menace = true,
            "hexproof" => result.hexproof = true,
            "indestructible" => result.indestructible = true,
            other => return Err(IrError::Malformed(format!("unsupported keyword {other:?}"))),
        }
    }
    Ok(result)
}

fn card_type(name: &str) -> Result<CardType, IrError> {
    match name {
        "creature" => Ok(CardType::Creature),
        "instant" => Ok(CardType::Instant),
        "sorcery" => Ok(CardType::Sorcery),
        "planeswalker" => Ok(CardType::Planeswalker),
        "land" => Ok(CardType::Land),
        "enchantment" => Ok(CardType::Enchantment),
        "artifact" => Ok(CardType::Artifact),
        "kindred" => Ok(CardType::Kindred),
        "battle" => Ok(CardType::Battle),
        other => Err(IrError::Malformed(format!(
            "unsupported card type {other:?}"
        ))),
    }
}

fn color(name: &str) -> Result<Color, IrError> {
    match name {
        "W" => Ok(Color::White),
        "U" => Ok(Color::Blue),
        "B" => Ok(Color::Black),
        "R" => Ok(Color::Red),
        "G" => Ok(Color::Green),
        "C" => Ok(Color::Colorless),
        other => Err(IrError::Malformed(format!("unsupported color {other:?}"))),
    }
}

fn optional_string_array(value: &Value, name: &str) -> Result<Vec<String>, IrError> {
    if value.get(name).is_none() {
        Ok(Vec::new())
    } else {
        str_array_field(value, name)
    }
}

fn optional_i32(value: &Value, name: &str) -> Result<Option<i32>, IrError> {
    let Some(inner) = value.get(name) else {
        return Ok(None);
    };
    if inner.is_null() {
        return Ok(None);
    }
    let number = inner
        .as_i64()
        .ok_or_else(|| IrError::Malformed(format!("field {name:?} must be an integer or null")))?;
    i32::try_from(number)
        .map(Some)
        .map_err(|_| IrError::Malformed(format!("field {name:?} exceeds i32")))
}

fn usize_value(program: &Program, value: i64, name: &str) -> Result<usize, IrError> {
    usize::try_from(value).map_err(|_| malformed(program, &format!("{name} exceeds usize")))
}

fn i32_value(program: &Program, value: i64, name: &str) -> Result<i32, IrError> {
    i32::try_from(value).map_err(|_| malformed(program, &format!("{name} exceeds i32")))
}

fn malformed(program: &Program, message: &str) -> IrError {
    IrError::Malformed(format!("program {}: {message}", program.semantic_key))
}

fn unsupported<T>(program: &Program, message: &str) -> Result<T, IrError> {
    Err(malformed(program, message))
}

fn target_requirement(program: &Program, role: &str) -> Result<TargetRequirement, IrError> {
    for target in array_field(&program.metadata, "targets")? {
        if str_field(target, "role")? == role {
            return parse_target_requirement(program, target);
        }
    }
    Err(malformed(program, "unknown target clause role"))
}

fn parse_target_requirement(
    program: &Program,
    target: &Value,
) -> Result<TargetRequirement, IrError> {
    Ok(TargetRequirement {
        another: opt_bool_field(target, "another")?.unwrap_or(false),
        spec: target_spec(object_field(target, "selector")?)?,
        min: usize::try_from(u64_field(target, "min")?)
            .map_err(|_| malformed(program, "target min exceeds usize"))?,
        max: usize::try_from(u64_field(target, "max")?)
            .map_err(|_| malformed(program, "target max exceeds usize"))?,
    })
}

fn token_name(pack: &SemanticPack, index: usize) -> Result<String, IrError> {
    let definition = pack
        .definitions
        .get(index)
        .ok_or_else(|| IrError::Malformed("token index out of bounds".into()))?;
    if !opt_bool_field(&definition.characteristics, "token")?.unwrap_or(false) {
        return Err(IrError::Malformed("token reference is not a token".into()));
    }
    Ok(definition.registry_name.clone())
}

fn lower_condition(condition: &Condition) -> Result<StaticCondition, IrError> {
    Ok(match condition {
        Condition::Always => StaticCondition::Always,
        Condition::CardsDrawnAtLeast(count) => StaticCondition::CardsDrawnAtLeast { count: *count },
        Condition::GraveyardAtLeast { count, predicate } => StaticCondition::GraveyardAtLeast {
            count: usize::try_from(*count)
                .map_err(|_| IrError::Malformed("count exceeds usize".into()))?,
            predicate: predicate_card(predicate)?,
        },
        Condition::ControlsMatching { predicate } => StaticCondition::ControlsMatching {
            predicate: predicate_card(predicate)?,
        },
        _ => return Err(IrError::Malformed("unsupported static condition".into())),
    })
}

impl ValueExpression {
    fn lower(&self) -> Result<EffectValue, IrError> {
        Ok(match self {
            Self::CreaturesWithCounters => EffectValue::CreaturesWithCounters,
            Self::BattlefieldMatching(predicate) => EffectValue::BattlefieldMatching {
                predicate: predicate_card(predicate)?,
            },
            Self::GraveyardMatching { base, predicate } => EffectValue::GraveyardMatching {
                base: *base,
                predicate: predicate_card(predicate)?,
            },
            Self::CardsDrawnMinus(subtract) => EffectValue::CardsDrawnMinus {
                subtract: *subtract,
            },
            Self::GreatestSharedCreatureType => EffectValue::GreatestSharedCreatureType,
        })
    }
}
