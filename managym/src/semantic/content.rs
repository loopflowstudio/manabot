// content.rs
// Lower checked-in semantic IR into the live rules engine's immutable content pack.

use std::{
    collections::BTreeSet,
    sync::{Arc, OnceLock},
};

use serde_json::Value;

use crate::{
    cardsets::alpha::{CompiledSemanticManifest, ContentPack},
    state::{
        ability::{
            Ability, Effect, StaticCondition, TargetRequirement, TargetSpec, TriggerCondition,
            TriggerSubject,
        },
        card::{
            ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords, PowerCda,
            StaticPtBuff, StaticScope, TriggeredManaAbility,
        },
        mana::{Color, Colors, Mana, ManaCost},
        player::PlayerConfig,
        predicate::CardPredicate,
    },
};

use super::{
    array_field, i64_field, object_field, opt_bool_field, opt_str_field, str_array_field,
    str_field, u64_field, Condition, Definition, IrError, Predicate, Program, SemanticPack, Step,
};

#[derive(Clone)]
struct AuthoredRuntime {
    pack_key: String,
    content: Arc<ContentPack>,
    decklists: Vec<std::collections::BTreeMap<String, usize>>,
}

impl AuthoredRuntime {
    fn compile(semantic: SemanticPack) -> Result<Self, IrError> {
        let content = Arc::new(semantic.compile_content_pack()?);
        let decklists = semantic
            .decks
            .iter()
            .map(|deck| semantic.decklist(&deck.key))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pack_key: semantic.pack_key,
            content,
            decklists,
        })
    }

    fn matches(&self, player_configs: &[PlayerConfig]) -> bool {
        if player_configs.len() != self.decklists.len() {
            return false;
        }
        let mut unmatched = self.decklists.iter().collect::<Vec<_>>();
        for config in player_configs {
            let Some(index) = unmatched
                .iter()
                .position(|decklist| *decklist == &config.decklist)
            else {
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
            [SemanticPack::two_deck()?, SemanticPack::jeong_increment()?]
                .into_iter()
                .map(AuthoredRuntime::compile)
                .collect()
        })
        .clone()
}

fn select_authored_runtime<'a>(
    player_configs: &[PlayerConfig],
    catalog: &'a [AuthoredRuntime],
) -> Result<Option<&'a AuthoredRuntime>, IrError> {
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
    fn duplicate_exact_catalog_entries_fail_closed() {
        let semantic = SemanticPack::jeong_increment().expect("Jeong IR parses");
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
    /// live [`Game`](crate::Game). Registry names are used only to carry
    /// presentation text across the migration boundary; every rules-relevant
    /// characteristic and program is lowered from this IR.
    pub fn compile_content_pack(&self) -> Result<ContentPack, IrError> {
        let presentation = ContentPack::default();
        let mut definitions = Vec::with_capacity(self.definitions.len());
        for definition in &self.definitions {
            let mut compiled = compile_characteristics(definition)?;
            let presentation_id = presentation
                .definition_id(&definition.registry_name)
                .ok_or_else(|| IrError::UnadmittedDefinition {
                    semantic_key: definition.semantic_key.clone(),
                    registry_name: definition.registry_name.clone(),
                })?;
            compiled.text_box = presentation
                .definition(presentation_id)
                .expect("definition id came from this content pack")
                .text_box
                .clone();

            for program in self
                .programs
                .iter()
                .filter(|program| program.definition_index == definition.semantic_index)
            {
                apply_program(self, &mut compiled, program)?;
            }
            definitions.push(compiled);
        }

        Ok(ContentPack::from_compiled_semantics(
            definitions,
            CompiledSemanticManifest {
                pack_key: self.pack_key.clone(),
                ir_hash: self.ir_hash.clone(),
                source_hash: self.source_hash.clone(),
            },
        ))
    }

    pub fn decklist(
        &self,
        key: &str,
    ) -> Result<std::collections::BTreeMap<String, usize>, IrError> {
        let deck = self
            .decks
            .iter()
            .find(|deck| deck.key == key)
            .ok_or_else(|| IrError::Malformed(format!("unknown semantic deck {key:?}")))?;
        let mut decklist = std::collections::BTreeMap::new();
        for (&definition_index, &count) in &deck.cards {
            let definition = self.definitions.get(definition_index).ok_or_else(|| {
                IrError::Malformed(format!(
                    "deck {key:?} references definition index {definition_index}"
                ))
            })?;
            decklist.insert(definition.registry_name.clone(), count);
        }
        Ok(decklist)
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
    let color_override = if colors.is_empty() {
        None
    } else {
        Some(
            colors
                .iter()
                .map(|name| color(name))
                .collect::<Result<Colors, _>>()?,
        )
    };

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
            definition.kicker = cost.kicker;
            definition.cost_reduction_per = cost.affinity;
            definition.targeting = targeting;
            definition.spell_effects = lower_steps(pack, program, &program.steps)?;
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
            let effects = lower_steps(pack, program, &program.steps)?;
            let [effect] = effects.as_slice() else {
                return unsupported(program, "activated ability must lower to one effect");
            };
            definition
                .activated_abilities
                .push(ActivatedAbilityDefinition {
                    mana_cost: cost
                        .mana
                        .ok_or_else(|| malformed(program, "activated ability has no mana cost"))?,
                    sacrifice_source: cost.sacrifice_source,
                    waterbend: cost.waterbend,
                    effect: effect.clone(),
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
    sacrifice_source: bool,
    waterbend: bool,
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
        sacrifice_source: opt_bool_field(value, "sacrifice_source")?.unwrap_or(false),
        waterbend: opt_bool_field(value, "waterbend")?.unwrap_or(false),
    })
}

fn program_targeting(program: &Program) -> Result<Vec<TargetRequirement>, IrError> {
    array_field(&program.metadata, "targets")?
        .iter()
        .map(|target| {
            Ok(TargetRequirement {
                spec: target_spec(object_field(target, "selector")?)?,
                min: usize::try_from(u64_field(target, "min")?)
                    .map_err(|_| malformed(program, "target min exceeds usize"))?,
                max: usize::try_from(u64_field(target, "max")?)
                    .map_err(|_| malformed(program, "target max exceeds usize"))?,
            })
        })
        .collect()
}

fn target_spec(selector: &Value) -> Result<TargetSpec, IrError> {
    let result = match str_field(selector, "kind")? {
        "creature" => TargetSpec::Creature,
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
        "enters_battlefield" => TriggerCondition::EntersTheBattlefield { subject },
        "dies" => TriggerCondition::Dies { subject },
        "attacks" => TriggerCondition::Attacks { subject },
        "becomes_tapped" => TriggerCondition::BecomesTapped { subject },
        "tapped_for_mana" => TriggerCondition::TappedForMana { subject },
        "becomes_targeted" => TriggerCondition::BecomesTargeted { subject },
        "upkeep_begins" => TriggerCondition::BeginningOfYourUpkeep,
        "draw_nth_card" => TriggerCondition::YouDrawNthCardThisTurn {
            n: u32::try_from(u64_field(trigger, "n")?)
                .map_err(|_| malformed(program, "draw count exceeds u32"))?,
        },
        other => return unsupported(program, &format!("unsupported trigger event {other:?}")),
    };
    if let Some(condition) = trigger.get("if") {
        Ok(TriggerCondition::ActiveIf {
            active_if: static_condition(condition)?,
            condition: Box::new(base),
        })
    } else {
        Ok(base)
    }
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

fn static_condition(value: &Value) -> Result<StaticCondition, IrError> {
    match str_field(value, "kind")? {
        "graveyard_at_least" => Ok(StaticCondition::GraveyardAtLeast {
            count: usize::try_from(u64_field(value, "count")?)
                .map_err(|_| IrError::Malformed("condition count exceeds usize".to_owned()))?,
            predicate: predicate_value(object_field(value, "predicate")?)?,
        }),
        other => Err(IrError::Malformed(format!(
            "unsupported static condition {other:?}"
        ))),
    }
}

fn apply_static(definition: &mut CardDefinition, program: &Program) -> Result<(), IrError> {
    match program.steps.as_slice() {
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
            condition: Condition::GraveyardAtLeast { count, predicate },
            then,
            otherwise,
        }] if otherwise.is_empty() => {
            let [Step::ModifyPt {
                target,
                power,
                toughness,
                duration,
            }] = then.as_slice()
            else {
                return unsupported(program, "conditional static must modify P/T once");
            };
            if target != "source" || duration != "static" {
                return unsupported(program, "conditional static must modify its source");
            }
            definition.static_pt_buffs.push(StaticPtBuff {
                scope: StaticScope::This,
                condition: Some(StaticCondition::GraveyardAtLeast {
                    count: usize::try_from(*count)
                        .map_err(|_| malformed(program, "static count exceeds usize"))?,
                    predicate: predicate_card(predicate)?,
                }),
                power: i32_value(program, *power, "power")?,
                toughness: i32_value(program, *toughness, "toughness")?,
            });
        }
        [Step::ModifyPt {
            target,
            power,
            toughness,
            duration,
        }] if target == "each:other_allies_you_control" && duration == "static" => {
            definition.static_pt_buffs.push(StaticPtBuff {
                scope: StaticScope::OtherYouControl(CardPredicate::subtype("Ally")),
                condition: None,
                power: i32_value(program, *power, "power")?,
                toughness: i32_value(program, *toughness, "toughness")?,
            });
        }
        _ => return unsupported(program, "unsupported static program shape"),
    }
    Ok(())
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
            definition_index,
            count,
            tapped_and_attacking,
        } => Effect::CreateToken {
            token_name: pack
                .definitions
                .get(*definition_index)
                .ok_or_else(|| malformed(program, "token definition index is out of bounds"))?
                .registry_name
                .clone(),
            count: usize_value(program, *count, "token count")?,
            tapped_and_attacking: *tapped_and_attacking,
        },
        Step::PutCounters { target, count } if target == "source" => Effect::PutCountersOnSource {
            count: i32_value(program, *count, "counter count")?,
        },
        Step::PutCounters { target, count } if target == "each:other_ally_creature_you_control" => {
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
        Step::PutCounters { target, count } => Effect::PutCounters {
            count: i32_value(program, *count, "counter count")?,
            target: target_for(program, target)?,
        },
        Step::GainLife { amount } => Effect::GainLife {
            amount: i32_value(program, *amount, "life amount")?,
        },
        Step::Scry { count } => Effect::Scry {
            count: usize_value(program, *count, "scry count")?,
        },
        Step::Tap { target } if target == "source" => Effect::TapSource,
        Step::Tap { target } => {
            return unsupported(program, &format!("cannot tap target {target:?}"))
        }
        Step::Untap { target } if target == "source" => Effect::UntapSource,
        Step::Untap { target } => Effect::UntapTarget {
            target: target_for(program, target)?,
        },
        Step::ModifyPt {
            target,
            power,
            toughness,
            duration,
        } if duration == "end_of_turn" && target == "source" => Effect::ModifyUntilEot {
            power_delta: i32_value(program, *power, "power")?,
            toughness_delta: i32_value(program, *toughness, "toughness")?,
        },
        Step::ModifyPt {
            target,
            power,
            toughness,
            duration,
        } if duration == "end_of_turn" => Effect::BuffTarget {
            power: i32_value(program, *power, "power")?,
            toughness: i32_value(program, *toughness, "toughness")?,
            target: target_for(program, target)?,
        },
        Step::ModifyPt { duration, .. } => {
            return unsupported(program, &format!("unsupported P/T duration {duration:?}"))
        }
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
            look,
            min_select,
            max_select,
            destination,
            predicate,
        } if destination == "hand" => Effect::LookAndSelect {
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
    let parsed = match str_field(value, "kind")? {
        "any" => Predicate::Any,
        "all" => Predicate::All(
            array_field(value, "predicates")?
                .iter()
                .map(|item| match str_field(item, "kind")? {
                    "card_type" => Ok(Predicate::CardType(str_field(item, "value")?.to_owned())),
                    "subtype" => Ok(Predicate::Subtype(str_field(item, "value")?.to_owned())),
                    "not_card_types" => {
                        Ok(Predicate::NotCardTypes(str_array_field(item, "values")?))
                    }
                    "power_at_most" => Ok(Predicate::PowerAtMost(i64_field(item, "value")?)),
                    other => Err(IrError::Malformed(format!(
                        "unsupported predicate {other:?}"
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        "card_type" => Predicate::CardType(str_field(value, "value")?.to_owned()),
        "subtype" => Predicate::Subtype(str_field(value, "value")?.to_owned()),
        "not_card_types" => Predicate::NotCardTypes(str_array_field(value, "values")?),
        "power_at_most" => Predicate::PowerAtMost(i64_field(value, "value")?),
        other => {
            return Err(IrError::Malformed(format!(
                "unsupported predicate {other:?}"
            )))
        }
    };
    predicate_card(&parsed)
}

fn predicate_card(predicate: &Predicate) -> Result<CardPredicate, IrError> {
    let mut result = CardPredicate::default();
    merge_predicate(&mut result, predicate)?;
    Ok(result)
}

fn merge_predicate(result: &mut CardPredicate, predicate: &Predicate) -> Result<(), IrError> {
    match predicate {
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
