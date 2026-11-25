//! Human descriptions derived from the same ability/effect definitions that resolve.
//! Offers, public stack inspection and text capture share this projection. It never
//! evaluates hidden state or predicts an outcome; unbound quantities remain symbolic.

use crate::cards::content_pack::default_content_pack;
use crate::state::{
    ability::{
        Ability, Actor, AttackSize, Condition, DamageKind, DamageRecipientFilter, DelayedEvent,
        DelayedExpiry, DelayedFrequency, DistinctSearchEnd, Effect, EffectValue, EntryCounters,
        Except, KeywordAction, LinkedChooser, LinkedExile, LinkedGrant, ObjectSelector,
        PaymentCost, Per, PlayerSelector, Scope, SpellTargetFilter, TargetSpec, TriggerCondition,
        TriggerSubject, TurnStep,
    },
    card::{CardType, Keyword, Keywords},
    effect::{CopyException, EffectDuration, Modification},
    mana::{Color, Mana, ManaCost, SpendEffect},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Bindings available when describing a particular occurrence of a trigger.
#[derive(Clone, Copy)]
struct TriggerText {
    noun: &'static str,
    amount: i32,
    x: Option<(&'static str, u8)>,
}

impl TriggerText {
    fn symbolic(noun: &'static str) -> Self {
        Self {
            noun,
            amount: 0,
            x: None,
        }
    }
}

impl crate::flow::game::Game {
    pub(crate) fn pending_trigger_description(
        &self,
        trigger: &crate::flow::trigger::PendingTrigger,
    ) -> Option<String> {
        let face = self.source_face(
            trigger.source_object(),
            self.pending_trigger_definition(trigger),
        )?;
        let chosen = trigger.modes_chosen || !trigger.chosen_modes.is_empty();
        Some(trigger_description(
            face,
            self.unnamed_label(trigger.source_object()),
            trigger.ability_index,
            trigger.inline_effects.as_deref(),
            chosen.then_some(trigger.chosen_modes.as_slice()),
            trigger.x,
            trigger.triggering_amount,
        ))
    }
}

pub(crate) fn pending_trigger_description(
    game: &crate::flow::game::Game,
    trigger: &crate::flow::trigger::PendingTrigger,
) -> Option<String> {
    game.pending_trigger_description(trigger)
}

pub(super) fn stacked_trigger_description(
    game: &crate::flow::game::Game,
    trigger: &crate::state::stack_object::TriggeredAbilityOnStack,
) -> String {
    trigger_description(
        crate::flow::resolution::stacked_trigger_face(trigger),
        game.unnamed_label(trigger.source_object()),
        trigger.ability_index,
        trigger.inline_effects.as_deref(),
        Some(&trigger.chosen_modes),
        trigger.x,
        trigger.triggering_amount,
    )
}

/// A triggered ability's text with its X filled in where its event binds
/// one (`bound_x`). A trigger announces no X (CR 107.3); an entering
/// permanent's own trigger reads its spell's X (CR 107.3m).
/// Inline delayed payloads take precedence: their ability index is not a card slot.
pub(crate) fn trigger_description(
    object: &crate::state::card::PrintedCharacteristics,
    unnamed: &str,
    index: usize,
    inline: Option<&[Effect]>,
    chosen_modes: Option<&[usize]>,
    x: u8,
    triggering_amount: i32,
) -> String {
    let name = if object.name.is_empty() {
        unnamed
    } else {
        &object.name
    };
    if let Some(effects) = inline {
        return format!(
            "{name} — delayed ability: {}",
            effects_description_in(
                TriggerText {
                    noun: "permanent",
                    amount: triggering_amount,
                    x: Some(("that many", x))
                },
                effects
            )
        );
    }
    let Ability::Triggered {
        condition,
        intervening_if,
        effects,
    } = &object.abilities[index];
    let triggering = TriggerText {
        noun: triggering_noun(condition),
        amount: triggering_amount,
        x: bound_x(condition).map(|label| (label, x)),
    };
    let intervening = intervening_if
        .as_ref()
        .map(|condition| format!(", if {}", condition_text_in(triggering, condition)))
        .unwrap_or_default();
    let attached = if object.subtypes.iter().any(|subtype| subtype == "Equipment") {
        "equipped creature"
    } else {
        "enchanted creature"
    };
    // A chosen modal trigger reads as the modes it will run (CR 700.2).
    let program = match (effects.as_slice(), chosen_modes) {
        ([Effect::TriggerModes { .. }], Some([])) => "no modes chosen".into(),
        ([Effect::TriggerModes { .. }], Some(chosen_modes)) => {
            let mut chosen = chosen_modes.to_vec();
            chosen.sort_unstable();
            let modes: Vec<_> = chosen.iter().map(|mode| (mode + 1).to_string()).collect();
            format!(
                "chosen mode {}: {}",
                modes.join(", "),
                effects_description_in(
                    triggering,
                    &crate::state::ability::chosen_trigger_effects(effects, chosen_modes)
                )
            )
        }
        _ => effects_description_in(triggering, effects),
    };
    format!(
        "{name} — {}{intervening}: {program}",
        condition_description(condition, attached),
    )
}

/// What a trigger's event calls its triggering object: a cast's spell, a
/// graveyard arrival's card, otherwise a permanent.
fn triggering_noun(condition: &TriggerCondition) -> &'static str {
    match condition.leaf() {
        TriggerCondition::Casts { .. }
        | TriggerCondition::YouCastThis
        | TriggerCondition::SpellTargetsOnlyThis { .. } => "spell",
        TriggerCondition::PutIntoGraveyard { .. } => "card",
        _ => "permanent",
    }
}

/// An entering permanent reads its spell's X. Counter triggers retain their
/// printed X binding for existing programs.
fn bound_x(condition: &TriggerCondition) -> Option<&'static str> {
    match condition.leaf() {
        TriggerCondition::EntersTheBattlefield {
            subject: TriggerSubject::This,
        } => Some("X"),
        TriggerCondition::PutsCounters { .. }
        | TriggerCondition::NthCounter { .. }
        | TriggerCondition::Chapter { .. } => Some("that many"),
        _ => None,
    }
}

/// `attached` names the object an attached source watches ("enchanted
/// creature" for an Aura, "equipped creature" for Equipment).
fn condition_description(condition: &TriggerCondition, attached: &str) -> String {
    match condition {
        TriggerCondition::WasCast { condition } => {
            format!("{} (if this was cast)", condition_description(condition, attached))
        }
        TriggerCondition::OnceEachTurn { condition } => {
            format!("{} (once each turn)", condition_description(condition, attached))
        }
        TriggerCondition::ActiveIf {
            active_if,
            condition,
        } => format!(
            "{} (while {})",
            condition_description(condition, attached),
            condition_text(triggering_noun(condition), active_if)
        ),
        TriggerCondition::Sacrificed { subject } => {
            format!("Whenever {} is sacrificed", subject_description(subject, attached))
        }
        TriggerCondition::Expend(amount) => format!("Whenever you expend {amount}"),
        TriggerCondition::YouCastThis => "When you cast this spell".into(),
        TriggerCondition::NoCounters { kind } => {
            format!("When this has no {} counters on it", counter_description(*kind))
        }
        TriggerCondition::State(condition) => {
            format!("When {}", condition_text("this", condition))
        }
        TriggerCondition::SpellTargetsOnlyThis { predicate } => format!(
            "Whenever a player casts a {} that targets only this",
            predicate_description(predicate, "spell")
        ),
        TriggerCondition::EntersTheBattlefield { subject } => format!(
            "When {} enters the battlefield",
            subject_description(subject, attached)
        ),
        TriggerCondition::Dies { subject } => format!("When {} dies", subject_description(subject, attached)),
        TriggerCondition::Attacks {
            subject,
            per,
            attackers, nth,
        } => {
            let attack = match per {
                Per::Object => format!("{} attacks", subject_description(subject, attached)),
                Per::Batch => format!("{} attack", one_or_more(subject, "permanents", attached)),
            };
            let prefix = nth.map_or_else(|| "Whenever".to_string(), |n| format!("On attack {n} this turn, when"));
            match attackers {
                AttackSize::Any => format!("{prefix} {attack}"),
                AttackSize::Alone => format!("{prefix} {attack} alone"),
                AttackSize::AtLeast(n) => {
                    format!("{prefix} {attack} with at least {n} attackers")
                }
            }
        }
        TriggerCondition::FromZone { zone, condition } => format!(
            "{} (from {})",
            condition_description(condition, attached),
            format!("{zone:?}").to_lowercase()
        ),
        TriggerCondition::Prepared { condition } => {
            format!("{} (a prepared spell)", condition_description(condition, attached))
        }
        TriggerCondition::BecomesTapped { subject } => {
            format!("When {} becomes tapped", subject_description(subject, attached))
        }
        TriggerCondition::BecomesUntapped { subject } => {
            format!("When {} becomes untapped", subject_description(subject, attached))
        }
        TriggerCondition::TappedForMana { subject } => {
            format!("When {} is tapped for mana", subject_description(subject, attached))
        }
        TriggerCondition::BecomesTargeted { subject } => format!(
            "When {} becomes the target of an opponent's spell or ability",
            subject_description(subject, attached)
        ),
        TriggerCondition::YourAttackingCreatureTriggers => "Whenever a creature you control attacking causes a triggered ability of that creature to trigger".into(),
        TriggerCondition::WhileSourceAttacking { condition } => format!(
            "{} while this is attacking",
            condition_description(condition, attached)
        ),
        TriggerCondition::NthCounter { kind, nth } => format!("When the number of {} counters on this reaches {}", kind.name(), nth),
        TriggerCondition::Chapter { chapter } => format!("Chapter {}", roman(*chapter)),
        TriggerCondition::LeavesTheBattlefield { subject } => {
            format!(
                "When {} leaves the battlefield",
                subject_description(subject, attached)
            )
        }
        TriggerCondition::PutsCounters { who, kind, on } => format!(
            "Whenever {} one or more {} counters on a {}",
            actor_does(*who, "put", "puts"),
            counter_description(*kind),
            predicate_description(on, "permanent")
        ),
        TriggerCondition::PutsCountersOn { who, kind, subject } => format!(
            "Whenever {} one or more {} counters on {}",
            actor_does(*who, "put", "puts"),
            counter_description(*kind),
            subject_description(subject, attached)
        ),
        TriggerCondition::Casts {
            who,
            predicate,
            nth,
            targets,
        } => {
            let spell = predicate_description(predicate, "spell");
            let spell = match targets {
                SpellTargetFilter::Any => spell,
                SpellTargetFilter::TargetsThis => format!("{spell} that targets this"),
            };
            match nth {
                None => format!("Whenever {} a {spell}", actor_does(*who, "cast", "casts")),
                Some(n) => format!(
                    "Whenever {} {} {} {spell} each turn",
                    actor_does(*who, "cast", "casts"),
                    actor_possessive(*who),
                    ordinal(*n),
                ),
            }
        }
        TriggerCondition::LosesLife { who } => format!("Whenever {} life", actor_does(*who, "lose", "loses")),
        TriggerCondition::GainsLife { who, nth } => format!("Whenever {} life{}", actor_does(*who,"gain","gains"), nth.map_or_else(String::new, |n| format!(" for the {} time each turn",ordinal(n)))),
        TriggerCondition::Draws { who, nth: None } => {
            format!("Whenever {} a card", actor_does(*who, "draw", "draws"))
        }
        TriggerCondition::Draws { who, nth: Some(n) } => format!(
            "Whenever {} {} {} card each turn",
            actor_does(*who, "draw", "draws"),
            actor_possessive(*who),
            ordinal(*n)
        ),
        TriggerCondition::BeginningOf { step, whose } => {
            let whose = match whose {
                Actor::You => "your",
                Actor::Opponent => "each opponent's",
                Actor::AnyPlayer => "each player's",
            };
            let step = match step {
                TurnStep::Upkeep => "upkeep",
                TurnStep::Draw => "draw step",
                TurnStep::FirstMain => "first main phase",
                TurnStep::Combat => "combat",
                TurnStep::End => "end step",
            };
            format!("At the beginning of {whose} {step}")
        }
        TriggerCondition::Performs { who, any_of } => {
            let actions: Vec<String> = any_of
                .iter()
                .map(|action| keyword_action_text(*who, *action))
                .collect();
            format!(
                "Whenever {} {}",
                actor_noun(*who),
                alternatives(&actions)
            )
        }
        TriggerCondition::PutIntoGraveyard { whose, predicate } => format!(
            "Whenever a {} is put into {} from anywhere",
            predicate_description(predicate, "card"),
            match whose {
                Actor::You => "your graveyard",
                Actor::Opponent => "an opponent's graveyard",
                Actor::AnyPlayer => "a graveyard",
            }
        ),
        TriggerCondition::CardsLeaveYourGraveyard => "Whenever one or more cards leave your graveyard".into(),
        TriggerCondition::SourceDealsDamage => "this source deals damage".into(),
        TriggerCondition::SourceDealsCombatDamage => {
            "When this creature deals combat damage".into()
        }
        TriggerCondition::DealsCombatDamage { subject, to, per } => {
            let to = to
                .map(|to| format!(" to {}", actor_noun(to)))
                .unwrap_or_default();
            match per {
                Per::Object => format!(
                    "Whenever {} deals combat damage{to}",
                    subject_description(subject, attached)
                ),
                Per::Batch => format!(
                    "Whenever {} deal combat damage{to}",
                    one_or_more(subject, "creatures", attached)
                ),
            }
        }
        TriggerCondition::DealsDamage { kind, source, to } => {
            let kind = match kind {
                DamageKind::Any => "",
                DamageKind::Combat => "combat ",
                DamageKind::Noncombat => "noncombat ",
            };
            let to = match to {
                DamageRecipientFilter::Player(who) => actor_noun(*who),
                DamageRecipientFilter::Any => "a player or permanent",
            };
            format!(
                "Whenever {} deals {kind}damage to {to}",
                subject_phrase(source, ["a", "another"], "source", attached)
            )
        }
        TriggerCondition::InGraveyard { condition } => {
            format!(
                "{} (while this is in your graveyard)",
                condition_description(condition, attached)
            )
        }
    }
}

fn subject_description(subject: &TriggerSubject, attached: &str) -> String {
    subject_phrase(subject, ["a", "another"], "permanent", attached)
}

/// A subject read as a batch ("one or more other Allies you control"); a
/// single object reads as itself.
fn one_or_more(subject: &TriggerSubject, plural: &str, attached: &str) -> String {
    subject_phrase(
        subject,
        ["one or more", "one or more other"],
        plural,
        attached,
    )
}

/// "[quantity] [objects] [controller controls]", where `quantity` is the
/// words for any object and for another one; any player's objects name no
/// controller.
fn subject_phrase(
    subject: &TriggerSubject,
    quantity: [&str; 2],
    objects: &str,
    attached: &str,
) -> String {
    match subject {
        TriggerSubject::ThisOr(other) => {
            format!(
                "this or {}",
                subject_phrase(other, quantity, objects, attached)
            )
        }
        TriggerSubject::This => "this".into(),
        TriggerSubject::CreatureDamagedByThis => {
            "a creature dealt damage by this creature this turn".into()
        }
        TriggerSubject::Attached => attached.into(),
        TriggerSubject::Matching {
            controlled_by,
            predicate,
            another,
        } => {
            let phrase = format!(
                "{} {}",
                quantity[usize::from(*another)],
                predicate_description(predicate, objects)
            );
            match controlled_by {
                Actor::AnyPlayer => phrase,
                _ => format!(
                    "{phrase} {}",
                    actor_does(*controlled_by, "control", "controls")
                ),
            }
        }
    }
}

/// A trigger actor as the subject of a sentence.
fn actor_noun(who: Actor) -> &'static str {
    match who {
        Actor::You => "you",
        Actor::Opponent => "an opponent",
        Actor::AnyPlayer => "a player",
    }
}

fn actor_possessive(who: Actor) -> &'static str {
    match who {
        Actor::You => "your",
        Actor::Opponent | Actor::AnyPlayer => "their",
    }
}

/// The verb form agreeing with its actor: "cast" for you, "casts" otherwise.
fn agree<'a>(who: Actor, plain: &'a str, third_person: &'a str) -> &'a str {
    if who == Actor::You {
        plain
    } else {
        third_person
    }
}

/// "you cast", "an opponent casts".
fn actor_does(who: Actor, plain: &str, third_person: &str) -> String {
    format!("{} {}", actor_noun(who), agree(who, plain, third_person))
}

/// A keyword action's verb phrase agreeing with its actor.
fn keyword_action_text(who: Actor, action: KeywordAction) -> String {
    let (plain, third_person) = match action {
        KeywordAction::Search => {
            return format!(
                "{} {} library",
                agree(who, "search", "searches"),
                actor_possessive(who)
            );
        }
        KeywordAction::Scry => ("scry", "scries"),
        KeywordAction::Surveil => ("surveil", "surveils"),
        KeywordAction::Waterbend => ("waterbend", "waterbends"),
        KeywordAction::Earthbend => ("earthbend", "earthbends"),
        KeywordAction::Firebend => ("firebend", "firebends"),
        KeywordAction::Airbend => ("airbend", "airbends"),
        KeywordAction::Discard => ("discard a card", "discards a card"),
    };
    agree(who, plain, third_person).into()
}

/// "a or b"; "a, b, or c".
fn alternatives(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [first, second] => format!("{first} or {second}"),
        [rest @ .., last] => format!("{}, or {last}", rest.join(", ")),
    }
}

/// "second", as printed in "your second spell each turn".
fn ordinal(n: u32) -> String {
    match n {
        1 => "first".into(),
        2 => "second".into(),
        3 => "third".into(),
        4 => "fourth".into(),
        5 => "fifth".into(),
        n => format!("{n}th"),
    }
}

pub(crate) fn condition_text(noun: &'static str, condition: &Condition) -> String {
    condition_text_in(TriggerText::symbolic(noun), condition)
}

fn condition_text_in(triggering: TriggerText, condition: &Condition) -> String {
    match condition {
        Condition::YourTurn => "it is your turn".into(),
        Condition::Always => "always".into(),
        Condition::CardsDrawnAtLeast { count } => {
            format!("you have drawn at least {count} cards this turn")
        }
        Condition::AnyGraveyardAtLeast { count } => {
            format!("a graveyard has {count} or more cards in it")
        }
        Condition::GraveyardAtLeast { count, predicate } => format!(
            "your graveyard has at least {count} {} cards",
            predicate_description(predicate, "matching")
        ),
        Condition::ControlsAtLeast {
            distinct_names: false,
            count: 1,
            predicate,
        } => format!(
            "you control a {}",
            predicate_description(predicate, "permanent")
        ),
        Condition::ControlsAtLeast {
            count,
            predicate,
            distinct_names,
        } => format!(
            "you control {count} or more {}s{}",
            predicate_description(predicate, "permanent"),
            if *distinct_names {
                " with different names"
            } else {
                ""
            }
        ),
        Condition::OpponentControlsMoreCreatures => {
            "an opponent controls more creatures than you".into()
        }
        Condition::BattlefieldAtLeast { count, predicate } => format!(
            "there are {count} or more {}s on the battlefield",
            predicate_description(predicate, "permanent")
        ),
        Condition::ControlsUnownedAtLeast { count } => {
            format!("you control {count} or more permanents you don't own")
        }
        Condition::OpponentHasMoreCardsInHand => {
            "an opponent has more cards in hand than you".into()
        }
        Condition::CardsInHandAtLeast { count } => {
            format!("you have {count} or more cards in hand")
        }
        Condition::LifeAtLeast { amount } => format!("you have {amount} or more life"),
        Condition::UnspentManaAtLeast { count } => {
            format!("you have {count} or more unspent mana")
        }
        Condition::AttackedThisTurn => "you attacked this turn".into(),
        Condition::CastThisTurn(predicate) => {
            let spell = predicate_description(predicate, "spell");
            let article = if spell.starts_with(['a', 'e', 'i', 'o', 'u']) {
                "an"
            } else {
                "a"
            };
            format!("you've cast {article} {spell} this turn")
        }
        Condition::BentAllFourThisTurn => {
            "you've waterbent, earthbent, firebent and airbent this turn".into()
        }
        Condition::PermanentLeftThisTurn { who, nonland } => format!(
            "a {}permanent{} left the battlefield this turn",
            if *nonland { "nonland " } else { "" },
            match who {
                Actor::You => " you controlled",
                Actor::Opponent => " an opponent controlled",
                Actor::AnyPlayer => "",
            }
        ),
        Condition::WarpedThisTurn { who } => {
            format!("{} cast a spell for warp this turn", actor_noun(*who))
        }
        Condition::LandEnteredThisTurn => {
            "a land entered the battlefield under your control this turn".into()
        }
        Condition::SacrificedThisTurnAtLeast { count } => {
            format!("you have sacrificed {count} or more permanents this turn")
        }
        Condition::OpponentLostLifeThisTurnAtLeast { amount } => {
            format!("an opponent lost {amount} or more life this turn")
        }
        Condition::CreatureDiedThisTurn => "a creature died this turn".into(),
        Condition::Compare { left, op, right } => format!(
            "{} {} {}",
            value_description(triggering, left),
            op.symbol(),
            value_description(triggering, right)
        ),
        Condition::AllOf(conditions) => conditions
            .iter()
            .map(|condition| condition_text_in(triggering, condition))
            .collect::<Vec<_>>()
            .join(" and "),
        Condition::AnyOf(conditions) => conditions
            .iter()
            .map(|condition| condition_text_in(triggering, condition))
            .collect::<Vec<_>>()
            .join(" or "),
        Condition::CastFromGraveyard => "this spell was cast from a graveyard".into(),
        Condition::Not(condition) => match condition.as_ref() {
            Condition::YourTurn => "it's an opponent's turn".into(),
            condition => format!("not ({})", condition_text_in(triggering, condition)),
        },
        Condition::Kicked => "it was kicked".into(),
        Condition::TargetMatches(predicate) => format!(
            "that target is a {}",
            predicate_description(predicate, "permanent")
        ),
        Condition::OwnsMeldPair { partner } => {
            format!("you own and control this and a creature named {partner}")
        }
        Condition::AtMostHalfStartingLife => {
            "your life total is half your starting life total or less".into()
        }
        Condition::TargetDealtDamageThisTurn => "that target was dealt damage this turn".into(),
        Condition::SourceMatches(predicate) => {
            format!("this is {}", predicate_description(predicate, "permanent"))
        }
        Condition::SourceUnprepared => "this is unprepared".into(),
        Condition::SourceTapped => "this is tapped".into(),
        Condition::SourceSaddled => "this is saddled".into(),
        Condition::TriggeringHadCounters => "it had counters on it".into(),
        Condition::TriggeringMatches(predicate) => {
            format!("it was {}", predicate_description(predicate, "permanent"))
        }
        Condition::CardsThisWayMatch(predicate) => format!(
            "the cards found this way include a {}",
            predicate_description(predicate, "card")
        ),
        Condition::TriggeringSpellMatches(predicate) => format!(
            "that spell is a {}",
            predicate_description(predicate, "spell")
        ),
        Condition::TargetCardExiled(predicate) => format!(
            "a {} was exiled this way",
            predicate_description(predicate, "card")
        ),
        Condition::SourceCountersAtLeast { kind, count } => format!(
            "this has {count} or more {} counters",
            kind.name().replace('_', " ")
        ),
        Condition::AttachedCountersAtLeast { kind, count } => format!(
            "the permanent this is attached to has {count} or more {} counters",
            kind.name().replace('_', " ")
        ),
        Condition::Moved => "you do".into(),
        Condition::TargetControlledByYou => "you controlled that target".into(),
    }
}

fn card_type(t: CardType) -> &'static str {
    match t {
        CardType::Creature => "creature",
        CardType::Instant => "instant",
        CardType::Sorcery => "sorcery",
        CardType::Planeswalker => "planeswalker",
        CardType::Land => "land",
        CardType::Enchantment => "enchantment",
        CardType::Artifact => "artifact",
        CardType::Kindred => "kindred",
        CardType::Battle => "battle",
    }
}

fn color_word(color: Color) -> &'static str {
    match color {
        Color::White => "white",
        Color::Blue => "blue",
        Color::Black => "black",
        Color::Red => "red",
        Color::Green => "green",
        Color::Colorless => "colorless",
        Color::Generic => "generic",
    }
}

pub(crate) fn predicate_description(p: &CardPredicate, noun: &str) -> String {
    let CardPredicate {
        namesake,
        chosen_color,
        chosen_subtype,
        chosen_name,
        named,
        flying,
        not_flying,
        deathtouch,
        no_counters,
        in_combat,
        token,
        with_counter,
        attacking,
        tapped,
        from_battlefield_this_turn,
        subtypes_any,
        colors_any,
        not_subtypes,
        supertype,
        nonbasic,
        nonlegendary,
        any_of,
        min_power,
        min_toughness,
        card_type: kind,
        card_types_any,
        not_card_types,
        subtype,
        max_power,
        min_mana_value,
        max_mana_value,
    } = p;
    let mut terms = Vec::new();
    if *tapped {
        terms.push("tapped".into());
    }
    if !colors_any.is_empty() {
        terms.push(
            colors_any
                .iter()
                .map(|color| color_word(*color))
                .collect::<Vec<_>>()
                .join(" or "),
        );
    }
    if *deathtouch {
        terms.push("deathtouch".into());
    }
    if *flying {
        terms.push("flying".into());
    }
    terms.extend(supertype.iter().cloned());
    if *nonlegendary {
        terms.push("nonlegendary".into());
    }
    if *nonbasic {
        terms.push("nonbasic".into());
    }
    if *token == Some(false) {
        terms.push("nontoken".into());
    }
    terms.extend(
        not_card_types
            .iter()
            .map(|t| format!("non{}", card_type(*t))),
    );
    terms.extend(not_subtypes.iter().map(|t| format!("non-{t}")));
    terms.extend(subtype.iter().cloned());
    if !subtypes_any.is_empty() {
        terms.push(format!("({})", subtypes_any.join(" or ")));
    }
    if let Some(t) = kind {
        terms.push(card_type(*t).into());
    }
    if !any_of.is_empty() {
        terms.push(format!(
            "({})",
            any_of
                .iter()
                .map(|alternative| predicate_description(alternative, ""))
                .map(|text| text.trim().to_string())
                .collect::<Vec<_>>()
                .join(" or ")
        ));
    }
    if !card_types_any.is_empty() {
        terms.push(format!(
            "({})",
            card_types_any
                .iter()
                .map(|t| card_type(*t))
                .collect::<Vec<_>>()
                .join(" or ")
        ));
    }
    if kind.is_none() || noun != "permanent" {
        terms.push(noun.into());
    }
    if *chosen_color {
        terms.push("of the chosen color".into());
    }
    if *chosen_subtype {
        terms.push("of the chosen type".into());
    }
    if *chosen_name {
        terms.push("with the chosen name".into());
    }
    if *namesake {
        terms.push("with the same name as this permanent".into());
    }
    if let Some(name) = named {
        terms.push(format!("named {name}"));
    }
    if *token == Some(true) {
        terms.push("token".into());
    }
    if let Some(n) = min_power {
        terms.push(format!("with power at least {n}"));
    }
    if let Some(n) = max_power {
        terms.push(format!("with power at most {n}"));
    }
    if let Some(n) = min_toughness {
        terms.push(format!("with toughness {n} or greater"));
    }
    if *not_flying {
        terms.push("without flying".into());
    }
    if let Some(n) = max_mana_value {
        terms.push(format!("with mana value {n} or less"));
    }
    if let Some(n) = min_mana_value {
        terms.push(format!("with mana value at least {n}"));
    }
    if *no_counters {
        terms.push("with no counters on it".into());
    }
    if *with_counter {
        terms.push("with a +1/+1 counter on it".into());
    }
    if *from_battlefield_this_turn {
        terms.push("put into a graveyard from the battlefield this turn".into());
    }
    if *attacking {
        terms.insert(0, "attacking".into());
    }
    if *in_combat {
        terms.insert(0, "attacking or blocking".into());
    }
    terms.join(" ")
}

pub(crate) fn counter_description(kind: CounterKind) -> &'static str {
    match kind {
        CounterKind::PlusOnePlusOne => "+1/+1",
        CounterKind::MinusOneMinusOne => "-1/-1",
        CounterKind::DoubleStrike => "double strike",
        _ => kind.name(),
    }
}

fn relation_description(relation: Option<crate::state::ability::TargetRelation>) -> String {
    use crate::state::ability::TargetRelation;
    match relation {
        None => String::new(),
        Some(TargetRelation::DistinctFromEarlier) => ", other than the earlier targets".into(),
        Some(TargetRelation::AttachedToEarlier(clause)) => {
            format!(" attached to the target of clause {}", clause + 1)
        }
        Some(TargetRelation::DifferentControllerFrom(clause)) => {
            format!(
                " controlled by a different player than the target of clause {}",
                clause + 1
            )
        }
    }
}

fn target_description(t: &TargetSpec) -> String {
    let noun = match t {
        TargetSpec::PermanentYouOwn { predicate } => {
            format!("{} you own", predicate_description(predicate, "permanent"))
        }
        TargetSpec::PermanentYouControl { predicate } => format!(
            "{} you control",
            predicate_description(predicate, "permanent")
        ),
        TargetSpec::PermanentOpponentControls { predicate } => format!(
            "{} an opponent controls",
            predicate_description(predicate, "permanent")
        ),
        TargetSpec::GraveyardCardYouOwn { predicate } => format!(
            "{} in your graveyard",
            predicate_description(predicate, "card")
        ),
        TargetSpec::GraveyardCardYouOwnWithinSourcePower { predicate } => format!(
            "{} in your graveyard with mana value at most this creature's power",
            predicate_description(predicate, "card")
        ),
        TargetSpec::GraveyardCard { predicate } => format!(
            "{} from a graveyard",
            predicate_description(predicate, "card")
        ),
        TargetSpec::GraveyardCardOpponentOwns { predicate } => format!(
            "{} from that player's graveyard",
            predicate_description(predicate, "card")
        ),
        TargetSpec::Permanent { predicate } => predicate_description(predicate, "permanent"),
        TargetSpec::Player => "player".into(),
        TargetSpec::Opponent => "opponent".into(),
        TargetSpec::Creature => "creature".into(),
        TargetSpec::AnyTarget => "any target".into(),
        TargetSpec::PlayerOrPlaneswalker => "player or planeswalker".into(),
        TargetSpec::Spell => "spell".into(),
        TargetSpec::SpellMatching { predicate } => predicate_description(predicate, "spell"),
        TargetSpec::CreatureOrSpell => "creature or spell".into(),
        TargetSpec::SpellOrAbilityWithSingleTarget => {
            "spell or ability with a single target".into()
        }
        TargetSpec::Ability => "activated or triggered ability".into(),
        TargetSpec::SpellOrPermanent { min_mana_value } => {
            format!("spell or permanent with mana value at least {min_mana_value}")
        }
        TargetSpec::CreatureYouControl => "creature you control".into(),
        TargetSpec::CreatureOpponentControls => "creature an opponent controls".into(),
        TargetSpec::LandYouControl => "land you control".into(),
    };
    format!("target {noun}")
}

fn zone_phrase(zone: ZoneType) -> &'static str {
    match zone {
        ZoneType::Graveyard => "a graveyard",
        ZoneType::Hand => "a hand",
        ZoneType::Library => "a library",
        ZoneType::Exile => "exile",
        ZoneType::Battlefield => "the battlefield",
        ZoneType::Stack => "the stack",
        ZoneType::Command => "the command zone",
    }
}

fn player_description(player: &PlayerSelector) -> String {
    match player {
        PlayerSelector::You => "you".into(),
        PlayerSelector::EachOpponent => "each opponent".into(),
        PlayerSelector::EachPlayer => "each player".into(),
        PlayerSelector::TriggeringPlayer => "that player".into(),
        PlayerSelector::TriggeringPlayersOpponents => "each of that player's opponents".into(),
        PlayerSelector::TargetController => "its controller".into(),
        PlayerSelector::Target(target) => target_description(target),
        PlayerSelector::Owner => "its owner".into(),
    }
}

fn selector_description(triggering: TriggerText, selector: &ObjectSelector) -> String {
    match selector {
        ObjectSelector::ClauseTarget(index) => format!("the targets of clause {}", index + 1),
        ObjectSelector::ExactSource => "this object".into(),
        ObjectSelector::This => "this".into(),
        ObjectSelector::Target(target) => target_description(target),
        ObjectSelector::Targets(target) => format!("each chosen {}", target_description(target)),
        ObjectSelector::TargetAndSameName(target) => format!(
            "{} and all other permanents with the same name as that permanent",
            target_description(target)
        ),
        ObjectSelector::Triggering => format!("that {}", triggering.noun),
        ObjectSelector::Attached => "the permanent this is attached to".into(),
        ObjectSelector::CreatedThisWay => "that token".into(),
        ObjectSelector::ExiledWithSource => "the cards exiled with this permanent".into(),
        ObjectSelector::Remembered => "the remembered objects".into(),
        ObjectSelector::Grantor => "the permanent granting this ability".into(),
        ObjectSelector::Moved => "it".into(),
        ObjectSelector::CardsThisWay => "cards moved by the latest mill or discard".into(),
        ObjectSelector::Selected => "the selected cards".into(),
        ObjectSelector::Paid(how) => {
            format!("the permanent {} to pay", format!("{how:?}").to_lowercase())
        }
        ObjectSelector::Matching { of, predicate } => format!(
            "all {} among {}",
            predicate_description(predicate, "card"),
            selector_description(triggering, of)
        ),
        ObjectSelector::Zone(zone) => format!("your {}", format!("{zone:?}").to_lowercase()),
        ObjectSelector::TargetPlayersZone { spec, zone } => format!(
            "{}'s {}",
            target_description(spec),
            format!("{zone:?}").to_lowercase()
        ),
        ObjectSelector::You => "you".into(),
        ObjectSelector::Chosen { predicate, except } => format!(
            "{}{}",
            if except.is_some() { "another " } else { "a " },
            predicate_description(predicate, "permanent")
        ),
        ObjectSelector::Each {
            of,
            predicate,
            except,
            max_mana_value,
            exact_mana_value,
        } => format!(
            "each {}{}{}{}",
            if except.is_some() { "other " } else { "" },
            predicate_description(predicate, "permanent"),
            match max_mana_value {
                Some(value) => format!(
                    " with mana value {} or less",
                    value_description(triggering, value)
                ),
                None => exact_mana_value
                    .as_ref()
                    .map(|value| format!(
                        " with mana value {}",
                        value_description(triggering, value)
                    ))
                    .unwrap_or_default(),
            },
            match of {
                Scope::YouControl => " you control",
                Scope::OpponentsControl => " your opponents control",
                Scope::Battlefield => "",
                Scope::DamagedBySourceInCombat =>
                    " controlled by a player dealt combat damage by this source this turn",
                Scope::TargetPlayerControls => " target player controls",
            }
        ),
        ObjectSelector::ZoneMatching {
            except,
            zone,
            owner,
            predicate,
        } => {
            let zone = format!("{zone:?}").to_lowercase();
            format!(
                "each {} in {}{}",
                predicate_description(predicate, "card"),
                match owner {
                    Actor::You => format!("your {zone}"),
                    Actor::Opponent => format!("your opponents' {zone}s"),
                    Actor::AnyPlayer => format!("all {zone}s"),
                },
                match except {
                    Some(Except::Moved) => " except the cards just moved",
                    Some(Except::This) => " except this card",
                    Some(Except::Target) => " except the target",
                    None => "",
                }
            )
        }
    }
}

/// A selector as the subject of a sentence.
fn selector_subject(triggering: TriggerText, selector: &ObjectSelector) -> String {
    let mut text = match selector {
        ObjectSelector::Triggering => "It".to_string(),
        other => selector_description(triggering, other),
    };
    if let Some(first) = text.get(..1) {
        text.replace_range(..1, &first.to_uppercase());
    }
    text
}

/// One sentence for a `Move`: the verb the destination implies, then its
/// terms.
fn move_description(
    triggering: TriggerText,
    what: &ObjectSelector,
    to: ZoneType,
    tapped: bool,
    transformed: bool,
    under: Option<&PlayerSelector>,
    with_counter: Option<&EntryCounters>,
    linked: Option<LinkedExile>,
) -> String {
    // A card returning itself (Earthbend, flicker) reads as a return; its
    // owner's control is the natural reading and goes unsaid.
    let returns_itself = matches!(what, ObjectSelector::This);
    let what = selector_description(triggering, what);
    match to {
        ZoneType::Hand => format!("Return {what} to its owner's hand"),
        ZoneType::Battlefield => {
            let mut text = if returns_itself {
                "Return this card to the battlefield".to_string()
            } else {
                format!("Put {what} onto the battlefield")
            };
            if tapped {
                text.push_str(" tapped");
            }
            if transformed {
                text.push_str(" transformed");
            }
            if let Some(entry) = with_counter {
                let kind = counter_description(entry.kind);
                if entry.count == EffectValue::Fixed(1) {
                    text.push_str(&format!(" with a {kind} counter on it"));
                } else {
                    text.push_str(&format!(
                        " with {} {kind} counters on it",
                        value_description(triggering, &entry.count)
                    ));
                }
            }
            match under {
                None => {}
                Some(PlayerSelector::Owner) if returns_itself => {}
                Some(PlayerSelector::Owner) => text.push_str(" under its owner's control"),
                Some(player) => text.push_str(&format!(
                    " under the control of {}",
                    player_description(player)
                )),
            }
            text
        }
        ZoneType::Exile => {
            let mut text = format!("Exile {what}");
            match linked {
                Some(LinkedExile::WithSource) => text.push_str(" with this"),
                Some(LinkedExile::UntilSourceLeaves) => {
                    text.push_str(" until this leaves the battlefield")
                }
                None => {}
            }
            text
        }
        ZoneType::Library => format!("Shuffle {what} into its owner's library"),
        ZoneType::Graveyard => format!("Put {what} into its owner's graveyard"),
        ZoneType::Stack | ZoneType::Command => format!("Put {what} into the {to:?} zone"),
    }
}

fn value_description(triggering: TriggerText, value: &EffectValue) -> String {
    match value {
        EffectValue::Fixed(value) => value.to_string(),
        EffectValue::NegativeX => {
            format!("minus {}", value_description(triggering, &EffectValue::X))
        }
        EffectValue::X => match triggering.x {
            Some((label, x)) => format!("{label} ({x})"),
            None => "the announced value of X".into(),
        },
        EffectValue::Property {
            of,
            property,
            aggregate,
        } => {
            use crate::state::ability::{Aggregate, ObjectProperty};
            let noun = match property {
                ObjectProperty::Power => "power".into(),
                ObjectProperty::Toughness => "toughness".into(),
                ObjectProperty::ManaValue => "mana value".into(),
                ObjectProperty::Colors => "number of colors".into(),
                ObjectProperty::Counters(kind) => {
                    format!("number of {} counters", counter_description(*kind))
                }
            };
            let how = match of.as_ref() {
                ObjectSelector::This | ObjectSelector::Target(_) | ObjectSelector::Triggering => "",
                _ if *aggregate == Aggregate::Sum => "total ",
                _ => "greatest ",
            };
            format!(
                "the {how}{noun} of {}",
                selector_description(triggering, of)
            )
        }
        EffectValue::DistinctManaValues { of } => format!(
            "the number of different mana values among {}",
            selector_description(triggering, of)
        ),
        EffectValue::Objects {
            of,
            predicate,
            in_zone,
        } if matches!(
            (of.as_ref(), in_zone),
            (ObjectSelector::Moved, _) | (ObjectSelector::CardsThisWay, Some(_))
        ) =>
        {
            let objects = match predicate {
                None if in_zone.is_some() => "cards".into(),
                None => "objects".into(),
                Some(predicate) => predicate_description(predicate, "cards"),
            };
            match in_zone {
                None => format!("the number of {objects} moved this way"),
                Some(zone) => format!(
                    "the number of {objects} put into {} this way",
                    zone_phrase(*zone)
                ),
            }
        }
        EffectValue::Objects {
            of,
            predicate,
            in_zone,
        } => {
            let objects = match in_zone {
                Some(zone) => format!(
                    "{} now in {}",
                    selector_description(triggering, of),
                    zone_phrase(*zone)
                ),
                None => selector_description(triggering, of),
            };
            match predicate {
                None => format!("the number of {objects}"),
                Some(predicate) => format!(
                    "the number of {} among {objects}",
                    predicate_description(predicate, "cards"),
                ),
            }
        }
        EffectValue::Half(value) => format!(
            "half of {}, rounded down",
            value_description(triggering, value)
        ),
        EffectValue::Scaled { factor, value } => {
            format!("{factor} times {}", value_description(triggering, value))
        }
        EffectValue::OwnedPermanentsOpponentsControl => {
            "the number of permanents you own that your opponents control".into()
        }
        EffectValue::CreaturesWithCounters => {
            "the number of creatures you control with +1/+1 counters".into()
        }
        EffectValue::BattlefieldMatching { predicate } => format!(
            "the number of {} you control",
            predicate_description(predicate, "permanents")
        ),
        EffectValue::GraveyardMatching { base, predicate } => format!(
            "{base} plus the number of {} in your graveyard",
            predicate_description(predicate, "cards")
        ),
        EffectValue::GreatestSharedCreatureType => {
            "the greatest number of creatures you control that share a creature type".into()
        }
        EffectValue::GreatestPowerYouControl { predicate } => format!(
            "the greatest power among {} you control",
            predicate_description(predicate, "creatures")
        ),
        EffectValue::CardsInHand => "the number of cards in your hand".into(),
        EffectValue::CardsDrawnMinus { subtract } => {
            format!("the number of cards you have drawn this turn minus {subtract} (minimum zero)")
        }
        EffectValue::TargetShare => "the share assigned to that target".into(),
        EffectValue::TriggerDamage => "the damage dealt".into(),
        EffectValue::ExcessDamage => "the excess damage dealt this way".into(),
        EffectValue::DeathCountersMinus { kind, subtract } => format!(
            "the number of {} counters it had as it died minus {subtract}",
            counter_description(*kind)
        ),
        EffectValue::TriggeringAmount if triggering.amount > 0 => {
            format!("that many ({})", triggering.amount)
        }
        EffectValue::TriggeringAmount => {
            "the amount of the triggering event (damage dealt or life lost)".into()
        }
        EffectValue::LifeTotal(who) => format!("life total of {}", player_description(who)),
        EffectValue::LifeAboveStarting(who) => format!(
            "how much more life {} has than their starting life total",
            player_description(who)
        ),
        EffectValue::LifeGainedThisTurn(who) => {
            format!("life gained this turn by {}", player_description(who))
        }
        EffectValue::LifeLostThisTurn(who) => {
            format!("life lost this turn by {}", player_description(who))
        }
        EffectValue::SpellManaValue(of) => {
            format!("mana value of {}", selector_description(triggering, of))
        }
        EffectValue::SpellManaSpent(of) => {
            format!(
                "mana spent to cast {}",
                selector_description(triggering, of)
            )
        }
        EffectValue::LifeLostThisWay => "the life lost this way".into(),
        EffectValue::SourcePower => "this permanent's power".into(),
        EffectValue::AttachedCreaturePower => "equipped creature's power".into(),
        EffectValue::ManaSpentToCast => "the amount of mana spent to cast it".into(),
        EffectValue::ColorsSpentToCast => "the number of colors of mana spent to cast it".into(),
        EffectValue::TriggeringPower => "that creature's power".into(),
        EffectValue::PowerTappedToPay => "the power of the creatures tapped to pay".into(),
        EffectValue::TriggeringManaValue => "that spell's mana value".into(),
        EffectValue::TargetPower => "its power".into(),
        EffectValue::TargetToughness => "its toughness".into(),
        EffectValue::AttackingMatching { predicate } => format!(
            "the number of attacking {} you control",
            predicate_description(predicate, "creatures")
        ),
        EffectValue::PermanentsSacrificedThisTurn => {
            "the number of permanents you have sacrificed this turn".into()
        }
        EffectValue::SpellsCastBeforeTriggering(predicate) => format!(
            "the number of {} you cast before the triggering spell this turn",
            predicate_description(predicate, "spells")
        ),
        EffectValue::SpellsCastThisTurn => "the number of spells you've cast this turn".into(),
    }
}

pub(crate) fn mana_description(mana: &Mana) -> String {
    let restricted = mana.restricted_colors();
    let unrestricted = mana.unrestricted_colors();
    let mut text: String = ["W", "U", "B", "R", "G", "C"]
        .into_iter()
        .enumerate()
        .flat_map(|(i, s)| {
            std::iter::repeat_n(
                format!("{{{s}}}"),
                unrestricted[i] as usize + restricted[i] as usize,
            )
        })
        .collect();
    for (class, _) in mana.classes() {
        if class.combat {
            text.push_str(" (expires at end of combat)");
        }
        if let Some(restriction) = &class.restriction {
            let spend = restriction.describe();
            // An exclusion already limits the mana: "(anything but spells from hand)".
            if matches!(restriction, crate::state::mana::ManaRestriction::Only(_)) {
                text.push_str(&format!(" ({spend} only)"));
            } else {
                text.push_str(&format!(" ({spend})"));
            }
        }
        if let Some(effect) = &class.effect {
            text.push_str(&format!(" (puts {})", spend_effect_description(effect)));
        }
    }
    text
}

fn payment_description(cost: &PaymentCost) -> String {
    match cost {
        PaymentCost::Mana(cost) => cost_description(cost),
        PaymentCost::Waterbend { generic } => format!("waterbend {generic}"),
        // Read after "pays" and "You may pay": the amount only.
        PaymentCost::Discard => "a discarded card".into(),
        PaymentCost::Life(life) => format!("{life} life"),
    }
}

pub(crate) fn cost_description(cost: &ManaCost) -> String {
    let mut text = "{X}".repeat(usize::from(cost.x));
    if cost.cost[6] > 0 {
        text.push_str(&format!("{{{}}}", cost.cost[6]));
    }
    for (i, symbol) in ["W", "U", "B", "R", "G", "C"].into_iter().enumerate() {
        for _ in 0..cost.cost[i] {
            text.push_str(&format!("{{{symbol}}}"));
        }
    }
    for [a, b] in &cost.hybrid {
        text.push_str(&format!("{{{}/{}}}", a.symbol(), b.symbol()));
    }
    if text.is_empty() {
        text.push_str("{0}");
    }
    text
}

/// What spending a unit of mana adds, as the object of "puts": "a +1/+1
/// counter on a non-Human creature spell it casts".
pub(crate) fn spend_effect_description(effect: &SpendEffect) -> String {
    match effect {
        SpendEffect::CreatureEntersWithPlusOneCounter { without_subtype } => {
            format!("a +1/+1 counter on a non-{without_subtype} creature spell it casts")
        }
        SpendEffect::SpellGainsKeyword { spell, keyword } => format!(
            "{} until end of turn on {} it casts",
            keyword.printed(),
            predicate_description(spell, "spell").trim()
        ),
        SpendEffect::CopySpell { spell } => format!(
            "a copy of {} it casts on the stack",
            predicate_description(spell, "spell").trim()
        ),
    }
}

fn keywords_description(k: &Keywords) -> String {
    k.iter()
        .map(Keyword::printed)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Effect text outside a trigger, where the triggering object (if any) is a
/// permanent.
pub(crate) fn effects_description(effects: &[Effect]) -> String {
    effects_description_in(TriggerText::symbolic("permanent"), effects)
}

/// Effect text whose `ObjectSelector::Triggering` reads "that [triggering]"
/// (see [`triggering_noun`]).
fn effects_description_in(triggering: TriggerText, effects: &[Effect]) -> String {
    let parts: Vec<_> = effects
        .iter()
        .filter_map(|effect| effect_description(triggering, effect))
        .collect();
    if parts.is_empty() {
        "Do nothing".into()
    } else {
        parts.join(". ")
    }
}

/// Exhaustive matching makes new effect vocabulary require a human description.
/// The internal target-clause cursor is not itself an effect a person chooses.
fn effect_description(triggering: TriggerText, effect: &Effect) -> Option<String> {
    use Effect::*;
    Some(match effect {
        SelectTargetRequirement { .. } => return None,
        CreateTokenCopy { of, count, exceptions } => copy_description(triggering, of, count, exceptions),
        PutCounters { on, kind, amount } => format!("Put {} {} counters on {}", value_description(triggering, amount), counter_description(*kind), selector_description(triggering, on)),
        PutCountersFrom { from, on } => format!("Put the counters of {} on {}", selector_description(triggering, from), selector_description(triggering, on)),
        Sacrifice { what, optional } => format!("{}acrifice {}", if *optional { "You may s" } else { "S" }, selector_description(triggering, what)),
        Move { what, to, tapped, transformed, under, with_counter, linked, library_end, unrevealed } => {
            if let Some(end) = library_end { format!("Put {} on the {} of its owner's library", selector_description(triggering, what), format!("{end:?}").to_lowercase()) }
            else if *unrevealed && *to == ZoneType::Hand { format!("Put {} into its owner's hand without revealing it", selector_description(triggering, what)) }
            else { move_description(triggering, what, *to, *tapped, *transformed, under.as_ref(), with_counter.as_ref(), *linked) }
        },
        Prepare { what } => format!("Prepare {}", selector_description(triggering, what)),
        Unprepare { what } => format!("Unprepare {}", selector_description(triggering, what)),
        Tap { what } => format!("Tap {}", selector_description(triggering, what)),
        Untap { what } => format!("Untap {}", selector_description(triggering, what)),
        RemoveAllCounters { from } => format!("Remove all counters from {}", selector_description(triggering, from)),
        RemoveCounters { up_to, chosen: None, from } => format!("Remove up to {up_to} counters from {}", selector_description(triggering, from)),
        RemoveCounters { chosen: Some(kind), from, .. } => format!("Remove a {} counter from {}", counter_description(*kind), selector_description(triggering, from)),
        ApplyContinuous { affected, modification, duration } => continuous_description_in(triggering, affected, modification, *duration),
        OptionalPayment { cost, if_paid, if_declined } => {
            let cost = payment_description(cost);
            format!("You may pay {cost}. If paid: {}. Otherwise: {}", effects_description_in(triggering, if_paid), effects_description_in(triggering, if_declined))
        },
        GivePlayerCounters {player, kind, count} => format!("Give {} {count} {} counters", player_description(player), kind.name()),
        DealDamageToPlayer { player, amount } => format!("Deal {} damage to {}",value_description(triggering, amount),player_description(player)),
        DrainLife { amount } => format!("Each opponent loses {0} life and you gain {0} life",value_description(triggering, amount)),
        MillCards { count, player: Some(player), .. } => format!("{} mills {} cards", player_description(player), value_description(triggering, count)),
        MillCards { count, target: None, .. } => format!("Mill {} cards", value_description(triggering, count)),
        MillCards { count, target: Some(target), .. } => format!("{} mills {} cards", target_description(target), value_description(triggering, count)),
        MillUntil { player, predicate } => format!("{} reveals cards from the top of their library until they reveal {}, then puts those cards into their graveyard", player_description(player), predicate_description(predicate, "card")),
        SelectCards { from, predicate, min, max, effects, if_declined } => format!("Choose {min}–{} {} from {}: {}. If none chosen: {}", value_description(triggering, max), predicate_description(predicate, "cards"), selector_description(triggering, from), effects_description_in(triggering, effects), effects_description_in(triggering, if_declined)),
        TargetClause { requirement, effects } => format!("Choose {}–{} {}{}{} for this clause: {}",requirement.min,if requirement.max_x { "X".to_string() } else { requirement.max.to_string() },target_description(&requirement.spec),if requirement.same_graveyard { " from a single graveyard" } else { "" },relation_description(requirement.relation),effects_description_in(triggering, effects)),
        Discard { what: ObjectSelector::You, count: 1 } => "Discard a card".into(),
        Discard { what, count } => format!("{} discards {count} card{}", selector_subject(triggering, what), if *count == 1 { "" } else { "s" }),
        Destroy { what } => format!("Destroy {}",selector_description(triggering, what)),
        Airbend { what: what @ ObjectSelector::Each { .. } } => format!("Airbend {}. Their owners may cast them from exile for {{2}}", selector_description(triggering, what)),
        Airbend { what } => format!("Airbend {}. Its owner may cast it from exile for {{2}}",selector_description(triggering, what)),
        GrantGraveyardCastThisTurn { target, flashback } => if *flashback {
            format!("{} gains flashback at its mana cost until end of turn", target_description(target))
        } else { format!("You may cast {} this turn", target_description(target)) },
        OpponentsCastOnlyFromHand => "Until your next turn, your opponents can't cast spells from anywhere other than their hands".to_string(),
        ExileTopCardsToPlay { count, until, face_down, condition } => {
            let them = if *count == 1 { "that card" } else { "those cards" };
            let window = match until {
                crate::state::ability::PlayWindow::ThisTurn => "this turn",
                crate::state::ability::PlayWindow::ThroughYourNextTurn => "until the end of your next turn",
                crate::state::ability::PlayWindow::WhileExiled => "for as long as they remain exiled",
            };
            let cards = if *count == 1 { "card".to_string() } else { format!("{count} cards") };
            let condition = condition.as_ref().map(|condition| format!(" if {}", condition_text_in(triggering, condition))).unwrap_or_default();
            if *face_down {
                format!("Look at the top {cards} of your library and exile {} face down. You may play {them} {window}{condition}", if *count == 1 { "it" } else { "them" })
            } else {
                format!("Exile the top {cards} of your library. You may play {them} {window}{condition}")
            }
        }
        ExileLinked { chooser, zone, predicate, up_to, grant, target } => format!("{} {} {} {} from {}; {}", match chooser { LinkedChooser::Controller => "You exile", LinkedChooser::TargetPlayer => "That player exiles" }, if *up_to { "up to one" } else { "a" }, predicate_description(predicate, "card"), match zone { ZoneType::Battlefield => "they control", _ => "" }, target_description(target), match grant { LinkedGrant::WaterbendManaValue => "for as long as you control this, you may cast it during your turn by waterbending {X}, where X is its mana value", LinkedGrant::FlashAnyMana => "during your turn, you may cast it as though it had flash, spending mana of any type" }),
        CumulativeUpkeep { each } => format!("Sacrifice this unless you pay {} for each age counter on it", cost_description(each)),
        ExileEachLibraryTopCastFree => "Exile the top card of each library. You may cast any number of spells from among those cards without paying their mana costs".into(),
        Cascade => "Cascade: exile cards from the top of your library until you exile a nonland card that costs less. You may cast it without paying its mana cost. Put the exiled cards on the bottom in a random order".into(),
        ExileUntilNonlandCastFree { below } => format!("Exile cards from the top of your library until you exile a nonland card. You may cast that card without paying its mana cost if its mana value is less than {}. If you don't cast it this way, put it into your hand", value_description(triggering, below)),
        TriggerModes { modes, up_to, unchosen } => format!(
            "Choose {} — {}",
            match (up_to, unchosen) {
                (Some(value), _) => format!("up to {}", value_description(triggering, value)),
                (None, true) => "one that hasn't been chosen".into(),
                (None, false) => "one".into(),
            },
            modes.iter().map(|mode| effects_description_in(triggering, mode)).collect::<Vec<_>>().join("; or ")
        ),
        ExileMeldPair { partner } => format!("Exile this and {partner}"),
        MeldExiled { result, .. } => format!("Meld them into {result}"),
        TransformSource => "Transform this".into(),
        ControlPlayer { what, duration } => format!("You control {} during their next {}", selector_description(triggering, what), match duration {
            crate::state::ability::PlayerControlDuration::NextCombat => "combat phase",
            crate::state::ability::PlayerControlDuration::NextTurn => "turn",
        }),
        TakeExtraTurn => "Take an extra turn after this one".into(),
        WinGame => "You win the game".into(),
        LoseGame => "You lose the game".into(),
        EndTheTurn => "End the turn".into(),
        AdditionalCombatPhase { attackers } => match attackers {
            None => "After this phase, there is an additional combat phase".into(),
            Some(predicate) => format!(
                "After this phase, there is an additional combat phase. Only {} can attack during that combat phase",
                predicate_description(predicate, "creatures")
            ),
        },
        MayExileTriggeringWithSource => "You may exile it".into(),
        ChooseExiledWithSource { predicate } => format!("Choose a {} exiled with this", predicate_description(predicate, "card")),
        HaveAbilitiesOf { .. } => "This has all activated and triggered abilities of the chosen card".into(),
        NameAndExileFromTargetPlayer { count } => format!("Choose a card name. Search target opponent's graveyard, hand, and library for up to {count} cards with that name and exile them. Then that player shuffles"),
        ExileNamed { name, count } => format!("Exile up to {count} cards named {name} from that player's graveyard, hand, and library, then they shuffle"),
        OptionalManaPaymentX { if_paid } => format!("Choose X. You may pay {{X}}. If you do, {}", effects_description_in(triggering, if_paid)),
        EachPlayerChoosesToExile { predicate } => format!(
            "Starting with you, each player chooses up to one {} from among permanents your opponents control. Exile those permanents",
            predicate_description(predicate, "permanent")
        ),
        GainControl { player, what, duration } => format!("{} gains control of {}{}", player_description(player), selector_description(triggering, what), match duration { EffectDuration::EndOfTurn => " until end of turn", _ => "" }),
        CopyTriggeringObject => "Copy it. You may choose new targets for the copy".into(),
        CopyTriggeringSpellForEach { each } => format!("Copy that spell for {} it could target. Each copy targets a different one of them", selector_description(triggering, each)),
        CastFromGraveyard { predicate, exile_instants_and_sorceries } => format!("You may cast a {} from your graveyard.{}", predicate_description(predicate, "spell"), if *exile_instants_and_sorceries { " Exile an instant or sorcery cast this way if it would go to your graveyard." } else { "" }),
        CastExiledWithSource { predicate } => format!("You may cast a {} from among cards you own exiled with this", predicate_description(predicate, "spell")),
        ChangeTarget { target } => format!("Change the target of {}", target_description(target)),
        WhenYouDo { effects } => format!("When you do: {}", effects_description_in(triggering, effects)),
        CreateDelayedTrigger { event, frequency, expiry, remember, effects } => format!(
            "{}When {}{} ({}), {}",
            remember.as_ref().map_or_else(String::new, |what| format!("Remember {}. ", selector_description(triggering, what))),
            match event {
                DelayedEvent::Dies => "that creature dies".into(),
                DelayedEvent::CreatureDies => "a creature dies".into(),
                DelayedEvent::EndStep { your_turn: false } => "the end step begins".into(),
                DelayedEvent::EndStep { your_turn: true } => "your end step begins".into(),
                DelayedEvent::Upkeep => "upkeep begins".into(),
                DelayedEvent::YouCastSpell(predicate) => format!("you cast a {}", predicate_description(predicate, "spell")),
            },
            match expiry { DelayedExpiry::ThisTurn => " this turn", DelayedExpiry::Never => "" },
            match frequency { DelayedFrequency::Once => "once", DelayedFrequency::EachMatch => "each time" },
            effects_description_in(triggering, effects)
        ),
        ForPlayer { player, effects } => format!(
            "{} does this: {}",
            player_description(player),
            effects_description_in(triggering, effects)
        ),
        RestoreController(_) => return None,
        EachPlayerKeepsCreatures { max_total_power } => format!("Each player chooses any number of creatures they control with total power {max_total_power} or less, then sacrifices all other creatures they control"),
        EachPlayerSacrifices { predicate, count } => format!("Each player sacrifices {count} of their choice ({})", predicate_description(predicate, "permanent")),
        CastLibraryTopFree => "Look at the top card of your library. You may cast it without paying its mana cost".into(),
        CastFromHandFree { predicate } => format!("You may cast a {} from your hand without paying its mana cost", predicate_description(predicate, "spell")),
        ShuffleGraveyardIntoLibrary => "Shuffle your graveyard into your library".to_string(),
        Reveal { what } => format!("Reveal {}", selector_description(triggering, what)),
        RevealUntilToHand { predicate } => format!("Reveal cards from the top of your library until you reveal a {} card. Put that card into your hand and the rest on the bottom of your library in a random order", predicate_description(predicate, "card")),
        RevealHandDiscardChosen { target, predicate } => format!("{} reveals their hand. You choose a {} from it; that player discards it", target_description(target), predicate_description(predicate, "card")),
        SearchDistinctNames { predicate, count, end } => {
            let search = format!("Search your library for up to {count} {} with different names and reveal them", predicate_description(predicate, "card"));
            match end {
                DistinctSearchEnd::OpponentSplits { picks } => format!("{search}. An opponent chooses {picks}; put those into your graveyard and the rest onto the battlefield tapped, then shuffle"),
                DistinctSearchEnd::Hand => format!("{search}, put them into your hand, then shuffle"),
            }
        }
        OwnerPutsIntoLibrary { target, top: true } => format!("The owner of {} puts it on their choice of the top or bottom of their library", target_description(target)),
        OwnerPutsIntoLibrary { target, top: false } => format!("The owner of {} puts it into their library second from the top or on the bottom", target_description(target)),
        PlaceTargetInLibrary { second_from_top: true } => "Put it into its owner's library second from the top".into(),
        PlaceTargetInLibrary { second_from_top: false } => "Put it on the bottom of its owner's library".into(),
        MayOnceEachTurn { effects } => format!(
            "You may: {}. Do this only once each turn",
            effects_description_in(triggering, effects)
        ),
        If { condition: Condition::Always, then, .. } => effects_description_in(triggering, then),
        If { condition, then, otherwise } if otherwise.is_empty() => format!("If {}: {}",condition_text_in(triggering, condition),effects_description_in(triggering, then)),
        If { condition, then, otherwise } => format!("If {}: {}. Otherwise: {}",condition_text_in(triggering, condition),effects_description_in(triggering, then),effects_description_in(triggering, otherwise)),
        DealDamage { amount, what, by: Some(by) } => format!("{} deals X damage to {}, where X is {}",selector_description(triggering, by),selector_description(triggering, what),value_description(triggering, amount)),
        DealDamage { amount: EffectValue::Fixed(amount), what, by: None } => format!("Deal {amount} damage to {}",selector_description(triggering, what)),
        DealDamage { amount, what, by: None } => format!("Deal X damage to {}, where X is {}",selector_description(triggering, what),value_description(triggering, amount)),
        CounterSpell { what, exile } => format!("Counter {}{}", selector_description(triggering, what), if *exile { ". If it is countered this way, exile it instead of putting it into its owner's graveyard" } else { "" }),
        CounterAbility { what } => format!("Counter {}",selector_description(triggering, what)),
        DrawCards { count } => format!("Draw {} cards", value_description(triggering, count)),
        LoseLife { player, amount } => format!("{} loses {} life",player_description(player),value_description(triggering, amount)),
        CreateToken { token, count: EffectValue::Fixed(count), tapped_and_attacking, tapped } => format!("Create {count} {}{} token{}{}",if *tapped { "tapped " } else { "" },default_content_pack().token_name(token),if *count == 1 { "" } else { "s" },if *tapped_and_attacking { " tapped and attacking" } else { "" }),
        CreateToken { token, count, tapped_and_attacking, .. } => format!("Create X {} tokens, where X is {}{}",default_content_pack().token_name(token),value_description(triggering, count),if *tapped_and_attacking { ". They enter tapped and attacking" } else { "" }),
        GainLife { amount: EffectValue::Fixed(amount) } => format!("You gain {amount} life"),
        GainLife { amount } => format!("You gain life equal to {}", value_description(triggering, amount)),
        Fight { first, second } => format!("{} fights {}", selector_description(triggering, first), selector_description(triggering, second)),
        ShuffleLibrary => "Shuffle your library".into(),
        Repeat { count, effect } => format!("Repeat {} times: {}", value_description(triggering, count), effects_description_in(triggering, std::slice::from_ref(effect))),
        OnNthResolutionThisTurn { n, effect } => format!("If this is resolution {n} of this ability this turn: {}",effects_description_in(triggering, std::slice::from_ref(effect))),
        Scry { count } => format!("Scry {count}"),
        ManifestDread => "Manifest dread".into(),
        SeparatePiles { look } => format!("Separate the top {look} cards into face-up and face-down piles. An opponent chooses one for your hand. Put the other in your graveyard."),
        Surveil { count } => format!("Surveil {count}"),
        LookAndSelect { random_order, reveal, to_battlefield, look, min_select, max_select, predicate, max_mana_value } => format!("{} the top {} cards of your library. Put {min_select}–{max_select} {}{} from among them {} and the rest on the bottom in {} order",if *reveal { "Reveal" } else { "Look at" },value_description(triggering, look),predicate_description(predicate,"cards"), max_mana_value.as_ref().map(|value| format!(" with mana value at most {}", value_description(triggering, value))).unwrap_or_default(), if *to_battlefield { "onto the battlefield" } else { "into your hand" }, if *random_order { "a random" } else { "your chosen" }),
        RevealLibraryTop => "Reveal the top card of your library".into(),
        PutTopCardsInHand { count } => format!("Put the top {count} cards of your library into your hand"),
        Learn => "Learn: you may reveal a Lesson you own from outside the game and put it into your hand, or discard a card to draw a card, or decline".into(),
        Modal { modes } => format!("Choose one — {}",modes.iter().enumerate().map(|(i,m)| format!("({}) {}",i+1,effects_description_in(triggering, m))).collect::<Vec<_>>().join("; ")),
        CounterUnlessPays { cost } => format!("Counter the spell unless its controller pays {}",payment_description(cost)),
        TargetCreaturesDealPowerDamageToLastTarget => "Each chosen creature except the last target deals damage equal to its power to the last target".into(),
        Earthbend { count, what } => { let count = value_description(triggering, count); format!("Earthbend {count} {}: it becomes a 0/0 creature with haste that's still a land; put {count} +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped",selector_description(triggering, what)) },
        AddOneColorManaValue { amount, chosen_color, creature_purposes } => {
            let color = if *chosen_color { "the chosen color" } else { "any one color" };
            let mut description = format!("Add X mana of {color}, where X is {}", value_description(triggering, amount));
            if !creature_purposes.is_empty() {
                let uses: Vec<_> = creature_purposes.iter().map(|p| match p {
                    crate::state::mana::CreatureManaPurpose::Cast => "cast a creature spell",
                    crate::state::mana::CreatureManaPurpose::Activate => "activate an ability of a creature source",
                }).collect();
                description.push_str(&format!("; spend only to {} of the chosen type", uses.join(" or ")));
            }
            description
        },
        AddMana { mana, amount: EffectValue::Fixed(1), until_end_of_combat } => format!("Add {}{}",mana_description(mana),if *until_end_of_combat { ". This mana lasts until end of combat" } else { "" }),
        AddMana { mana, amount, until_end_of_combat } => format!("Add {} for each of {}{}", mana_description(mana), value_description(triggering, amount), if *until_end_of_combat { ". This mana lasts until end of combat" } else { "" }),
        AttachSourceTo { what } => format!("Attach this to {}",selector_description(triggering, what)),
        DistributeCountersToTappedOpponentCreatures { kind, count } => format!("Distribute {count} {} counters among any number of tapped creatures your opponents control",counter_description(*kind)),
        ForEachTarget { effects } => format!("For each chosen target: {}",effects_description_in(triggering, effects)),
        Simultaneously { effects } => format!("At the same time: {}",effects_description_in(triggering, effects)),
    })
}

/// "Create a token that's a copy of this[, except …]" (CR 707.9).
fn copy_description(
    triggering: TriggerText,
    of: &ObjectSelector,
    count: &EffectValue,
    exceptions: &[CopyException<EffectValue>],
) -> String {
    let tokens = if *count == EffectValue::Fixed(1) {
        "a token that's a copy".into()
    } else {
        format!(
            "{} tokens that are copies",
            value_description(triggering, count)
        )
    };
    let mut text = match of {
        ObjectSelector::Each { .. } => {
            format!(
                "For {}, create {tokens} of it",
                selector_description(triggering, of)
            )
        }
        _ => format!(
            "Create {tokens} of {}",
            selector_description(triggering, of)
        ),
    };
    for (index, exception) in exceptions.iter().enumerate() {
        // "It …", lowercased after "except".
        let phrase = continuous_description_in(
            triggering,
            &ObjectSelector::Triggering,
            &exception.modification(),
            EffectDuration::Indefinite,
        );
        text.push_str(if index == 0 { ", except " } else { " and " });
        text.push_str(&phrase[..1].to_lowercase());
        text.push_str(&phrase[1..]);
    }
    text
}

/// One sentence for a continuous effect: whom, what, how long.
pub(super) fn continuous_description(
    affected: &ObjectSelector,
    modification: &Modification<EffectValue>,
    duration: EffectDuration,
) -> String {
    continuous_description_in(
        TriggerText::symbolic("permanent"),
        affected,
        modification,
        duration,
    )
}

fn continuous_description_in(
    triggering: TriggerText,
    affected: &ObjectSelector,
    modification: &Modification<EffectValue>,
    duration: EffectDuration,
) -> String {
    let subject = selector_subject(triggering, affected);

    let until = match duration {
        EffectDuration::EndOfTurn => " until end of turn",
        EffectDuration::Indefinite => "",
        EffectDuration::WhileYouControlSource => " for as long as you control this",
        EffectDuration::UntilYourNextTurn => " until your next turn",
    };
    let amount = |value: &EffectValue| match value {
        EffectValue::Fixed(n) => format!("{n:+}"),
        other => format!("+({})", value_description(triggering, other)),
    };
    let base = |value: &Option<EffectValue>| match value {
        Some(EffectValue::Fixed(n)) => n.to_string(),
        Some(other) => value_description(triggering, other),
        None => "unchanged".to_string(),
    };
    match modification {
        Modification::SetName(name) => format!("{subject} is named {name}{until}"),
        Modification::SetCreatureTypes(types) => {
            format!("{subject} becomes {}{until}", types.join(" "))
        }
        Modification::AddTypes(types) => format!(
            "{subject} is also {}{until}",
            types
                .iter()
                .map(|t| card_type(*t))
                .collect::<Vec<_>>()
                .join(" ")
        ),
        Modification::SetTypes { types, subtypes } => format!(
            "{subject} becomes {} — {}{until}",
            types
                .iter()
                .map(|t| card_type(*t))
                .collect::<Vec<_>>()
                .join(" "),
            subtypes.join(" ")
        ),
        Modification::SetColors(colors) => format!(
            "{subject} becomes {}{until}",
            colors
                .iter()
                .map(|c| format!("{c:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(" and ")
        ),
        Modification::AddColors(colors) => format!(
            "{subject} is also {}{until}",
            colors
                .iter()
                .map(|c| format!("{c:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(" and ")
        ),
        Modification::GrantActivated(ability) => {
            format!("{subject} gains an activated ability: {ability:?}{until}")
        }
        Modification::GrantTriggered(ability) => {
            let crate::state::ability::Ability::Triggered {
                condition,
                intervening_if,
                effects,
            } = ability.as_ref();
            // The granted ability names its own triggering object.
            let triggering = TriggerText::symbolic(triggering_noun(condition));
            let intervening = intervening_if
                .as_ref()
                .map(|c| format!(", if {}", condition_text_in(triggering, c)))
                .unwrap_or_default();
            format!(
                "{subject} gains ‘{}{intervening}, {}’{until}",
                condition_description(condition, "attached creature"),
                effects_description_in(triggering, effects)
            )
        }
        Modification::GrantStatic(ability) => {
            format!("{subject} gains a static ability: {ability:?}{until}")
        }
        Modification::SwitchPt => format!("Switch {subject}'s power and toughness{until}"),
        Modification::AddSubtypes(subtypes) => format!(
            "{subject} becomes {} in addition to its other types{until}",
            subtypes.join(" ")
        ),
        Modification::BecomeCreature => {
            format!("{subject} becomes a creature in addition to its other types{until}")
        }
        Modification::Nonlegendary => format!("{subject} isn't legendary{until}"),
        Modification::AddKeywords(keywords) => {
            format!("{subject} gains {}{until}", keywords_description(keywords))
        }
        Modification::LoseNonManaAbilities => {
            format!("{subject} loses all abilities except mana abilities{until}")
        }
        Modification::LoseAllAbilities => format!("{subject} loses all abilities{until}"),
        Modification::Firebending(n) => format!("{subject} gains firebending {n}{until}"),
        Modification::SetBasePt { power, toughness } => format!(
            "{subject} has base power and toughness {}/{}{until}",
            base(power),
            base(toughness)
        ),
        Modification::ModifyPt { power, toughness } => {
            format!(
                "{subject} gets {}/{}{until}",
                amount(power),
                amount(toughness)
            )
        }
        Modification::CantBeBlocked => format!("{subject} can't be blocked{until}"),
        Modification::CantBlock => format!("{subject} can't block{until}"),
        Modification::Saddled => format!("{subject} becomes saddled{until}"),
        Modification::DoesntUntap => {
            format!("{subject} doesn't untap during its controller's untap step{until}")
        }
        Modification::CantActivateAbilities => {
            format!("{subject}'s activated abilities can't be activated{until}")
        }
        Modification::MustBeBlockedByAllAble => {
            format!(
                "All creatures able to block {}{until} do so",
                subject.to_lowercase()
            )
        }
        Modification::LoseKeywords(keywords) => {
            format!("{subject} loses {}{until}", keywords_description(keywords))
        }
        Modification::ExileIfWouldLeave => format!(
            "If {} would leave the battlefield{until}, exile it instead",
            subject.to_lowercase()
        ),
        Modification::PreventCombatDamage => format!(
            "Prevent all combat damage that would be dealt to {}{until}",
            subject.to_lowercase()
        ),
        Modification::ExileIfWouldDie => {
            format!(
                "If {} would die{until}, exile it instead",
                subject.to_lowercase()
            )
        }
        Modification::KeepUnspentMana(color) => format!(
            "{subject} don't lose unspent {} mana as steps and phases end{until}",
            format!("{color:?}").to_lowercase()
        ),
        Modification::AdditionalLandPlay => format!("{subject} may play an additional land{until}"),
        Modification::NoMaximumHandSize => format!("{subject} have no maximum hand size{until}"),
        Modification::NextCreatureCastFlash => {
            let this_turn = match duration {
                EffectDuration::EndOfTurn => " this turn",
                _ => until,
            };
            format!(
                "the next creature card {subject} cast{this_turn} can be cast as though it had flash"
            )
        }
        Modification::SkipsCombatPhasesNextTurn => {
            format!("{subject}: skip all combat phases of the next turn")
        }
        Modification::LifeTotalCantChange => {
            format!("{subject}: life total can't change{until}")
        }
        Modification::ProtectionFromEverything => {
            format!("{subject}: protection from everything{until}")
        }
        Modification::PhasedOut => format!("{subject} phase out{until}"),
        Modification::ControlChanged { .. } => {
            format!("You control {}{until}", subject.to_lowercase())
        }
        Modification::ModeChosen { mode } => format!("{subject} has chosen mode {}", mode + 1),
        Modification::HasAbilitiesOfSource => {
            format!("{subject} has all activated and triggered abilities of the chosen card")
        }
        Modification::AllCreatureTypes => format!("{subject} has all creature types{until}"),
    }
}

/// A chapter number as printed (CR 714.2a).
fn roman(chapter: u8) -> &'static str {
    match chapter {
        1 => "I",
        2 => "II",
        3 => "III",
        4 => "IV",
        5 => "V",
        6 => "VI",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ability::TargetRequirement;

    #[test]
    fn restricted_mana_says_only_for_a_positive_class_and_not_an_exclusion() {
        use crate::state::mana::{ManaRestriction, ManaRestriction::ExceptSpellsFromHand};
        let colorless = [0, 0, 0, 0, 0, 1];
        assert_eq!(
            mana_description(&Mana::restricted_to(ExceptSpellsFromHand, colorless)),
            "{C} (anything but spells from hand)"
        );
        assert!(mana_description(&Mana::restricted_to(
            ManaRestriction::spell(["Lesson"]),
            colorless
        ))
        .ends_with(" only)"));
    }

    #[test]
    fn distinct_name_threshold_describes_the_required_permanents() {
        let condition = Condition::ControlsAtLeast {
            count: 10,
            predicate: CardPredicate::subtype("Gate"),
            distinct_names: true,
        };
        assert_eq!(
            condition_text("permanent", &condition),
            "you control 10 or more Gate permanents with different names"
        );
    }

    #[test]
    fn trigger_descriptions_fill_one_template_per_event() {
        let cases = [
            (
                TriggerCondition::Casts {
                    targets: SpellTargetFilter::Any,
                    who: Actor::Opponent,
                    predicate: CardPredicate::creature(),
                    nth: Some(2),
                },
                "Whenever an opponent casts their second creature spell each turn",
            ),
            (
                TriggerCondition::casts(Actor::You, CardPredicate::default()),
                "Whenever you cast a spell",
            ),
            (
                TriggerCondition::Draws {
                    who: Actor::You,
                    nth: Some(2),
                },
                "Whenever you draw your second card each turn",
            ),
            (
                TriggerCondition::beginning_of(TurnStep::Draw, Actor::AnyPlayer),
                "At the beginning of each player's draw step",
            ),
            (
                TriggerCondition::Performs {
                    who: Actor::Opponent,
                    any_of: vec![KeywordAction::Search],
                },
                "Whenever an opponent searches their library",
            ),
            (
                TriggerCondition::Performs {
                    who: Actor::You,
                    any_of: KeywordAction::BENDING.to_vec(),
                },
                "Whenever you waterbend, earthbend, firebend, or airbend",
            ),
            (
                TriggerCondition::PutIntoGraveyard {
                    whose: Actor::Opponent,
                    predicate: CardPredicate::default(),
                },
                "Whenever a card is put into an opponent's graveyard from anywhere",
            ),
            (
                TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::you_control(CardPredicate::subtype("Ally")),
                    to: Some(Actor::AnyPlayer),
                    per: Per::Batch,
                },
                "Whenever one or more Ally creatures you control deal combat damage to a player",
            ),
            (
                TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::any(CardPredicate::creature()),
                    to: Some(Actor::You),
                    per: Per::Object,
                },
                "Whenever a creature deals combat damage to you",
            ),
            (
                TriggerCondition::attacks(
                    TriggerSubject::you_control(CardPredicate::creature()),
                    Per::Batch,
                ),
                "Whenever one or more creature permanents you control attack",
            ),
            (
                TriggerCondition::Attacks {
                    subject: TriggerSubject::you_control(CardPredicate::creature()),
                    per: Per::Object,
                    attackers: AttackSize::Alone,
                    nth: None,
                },
                "Whenever a creature you control attacks alone",
            ),
            (
                TriggerCondition::Sacrificed {
                    subject: TriggerSubject::another_you_control(CardPredicate::default()),
                },
                "Whenever another permanent you control is sacrificed",
            ),
            (
                TriggerCondition::Dies {
                    subject: TriggerSubject::matching(
                        Actor::Opponent,
                        CardPredicate::creature(),
                        false,
                    ),
                },
                "When a creature an opponent controls dies",
            ),
            (
                TriggerCondition::PutsCounters {
                    who: Actor::Opponent,
                    kind: CounterKind::PlusOnePlusOne,
                    on: CardPredicate::creature(),
                },
                "Whenever an opponent puts one or more +1/+1 counters on a creature",
            ),
        ];
        for (condition, expected) in cases {
            assert_eq!(
                condition_description(&condition, "enchanted creature"),
                expected
            );
        }
    }

    #[test]
    fn descriptions_keep_target_clauses_and_conditional_branches() {
        let description = effects_description(&[
            Effect::TargetClause {
                requirement: TargetRequirement::up_to(2, TargetSpec::CreatureYouControl),
                effects: vec![Effect::ForEachTarget {
                    effects: vec![Effect::untap(ObjectSelector::Target(
                        TargetSpec::CreatureYouControl,
                    ))],
                }],
            },
            Effect::OptionalPayment {
                cost: PaymentCost::Mana(ManaCost::parse("1W")),
                if_paid: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(2),
                }],
                if_declined: vec![Effect::DealDamageToPlayer {
                    player: PlayerSelector::You,
                    amount: EffectValue::Fixed(1),
                }],
            },
        ]);
        assert_eq!(description, "Choose 0–2 target creature you control for this clause: For each chosen target: Untap target creature you control. You may pay {1}{W}. If paid: Draw 2 cards. Otherwise: Deal 1 damage to you");
    }
    #[test]
    fn object_counts_describe_the_filter_and_resolution_result() {
        let count = EffectValue::Objects {
            of: Box::new(ObjectSelector::Moved),
            predicate: Some(CardPredicate::creature()),
            in_zone: None,
        };
        assert_eq!(
            value_description(TriggerText::symbolic("permanent"), &count),
            "the number of creature cards moved this way"
        );
        assert_eq!(
            value_description(
                TriggerText::symbolic("permanent"),
                &ObjectSelector::Moved.count()
            ),
            "the number of objects moved this way"
        );
    }

    #[test]
    fn flicker_descriptions_keep_the_moved_object_pronoun() {
        assert_eq!(
            effects_description(&[
                Effect::exile(ObjectSelector::Target(TargetSpec::CreatureYouControl)),
                Effect::Move {
                    unrevealed: false,
                    what: ObjectSelector::Moved,
                    to: ZoneType::Battlefield,
                    tapped: false,
                    transformed: false,
                    under: Some(PlayerSelector::Owner),
                    with_counter: None,
                    linked: None,
                    library_end: None,
                },
            ]),
            "Exile target creature you control. Put it onto the battlefield under its owner's control"
        );
    }

    /// A cast trigger names its own triggering object "that spell"; a
    /// permanent trigger it grants keeps "that permanent".
    #[test]
    fn each_trigger_names_its_own_triggering_object() {
        let grant = Ability::Triggered {
            condition: TriggerCondition::Dies {
                subject: TriggerSubject::This,
            },
            intervening_if: None,
            effects: vec![Effect::return_to_hand(ObjectSelector::Triggering)],
        };
        let object = crate::state::card::PrintedCharacteristics {
            name: "Probe".into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Casts {
                    who: Actor::You,
                    predicate: CardPredicate::default(),
                    nth: None,
                    targets: SpellTargetFilter::Any,
                },
                intervening_if: None,
                effects: vec![
                    Effect::exile(ObjectSelector::Triggering),
                    Effect::ApplyContinuous {
                        affected: ObjectSelector::Target(TargetSpec::CreatureYouControl),
                        modification: Modification::GrantTriggered(Box::new(grant)),
                        duration: EffectDuration::EndOfTurn,
                    },
                ],
            }],
            ..Default::default()
        };
        let text = trigger_description(&object, "", 0, None, None, 0, 0);
        assert!(text.contains("Exile that spell"), "{text}");
        assert!(
            text.contains("Return that permanent to its owner's hand"),
            "{text}"
        );
        assert!(!text.contains("that spell to its owner's hand"), "{text}");
    }
}
