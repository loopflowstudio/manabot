//! Human descriptions derived from the same ability/effect definitions that resolve.
//! Offers, public stack inspection and text capture share this projection. It never
//! evaluates hidden state or predicts an outcome; quantities remain symbolic.

use crate::state::{
    ability::{
        Ability, Condition, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue,
        LinkedChooser, LinkedExile, LinkedGrant, ObjectSelector, PaymentCost, PlayerSelector,
        Scope, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{Card, CardType, Keyword, Keywords},
    effect::{EffectDuration, Modification},
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Inline delayed payloads take precedence: their ability index is not a card slot.
pub(crate) fn trigger_description(card: &Card, index: usize, inline: Option<&[Effect]>) -> String {
    if let Some(effects) = inline {
        return format!(
            "{} — delayed ability: {}",
            card.name,
            effects_description(effects)
        );
    }
    let Ability::Triggered {
        condition,
        intervening_if,
        effects,
    } = &card.abilities[index];
    let intervening = intervening_if
        .as_ref()
        .map(|condition| format!(", if {}", condition_text(condition)))
        .unwrap_or_default();
    format!(
        "{} — {}{intervening}: {}",
        card.name,
        condition_description(condition),
        effects_description(effects)
    )
}

fn condition_description(condition: &TriggerCondition) -> String {
    match condition {
        TriggerCondition::WasCast { condition } => {
            format!("{} (if this was cast)", condition_description(condition))
        }
        TriggerCondition::OnceEachTurn { condition } => {
            format!("{} (once each turn)", condition_description(condition))
        }
        TriggerCondition::ActiveIf {
            active_if,
            condition,
        } => format!(
            "{} (while {})",
            condition_description(condition),
            condition_text(active_if)
        ),
        TriggerCondition::Sacrificed {
            predicate,
            controller_only,
            another,
        } => format!(
            "When {}{} {} sacrificed",
            if *another { "another " } else { "" },
            predicate_description(predicate, "permanent"),
            if *controller_only {
                "you control is"
            } else {
                "is"
            }
        ),
        TriggerCondition::YouCastSpell { predicate } => format!(
            "When you cast a {}",
            predicate_description(predicate, "spell")
        ),
        TriggerCondition::PlayerCastsSpell { predicate } => format!(
            "When a player casts a {}",
            predicate_description(predicate, "spell")
        ),
        TriggerCondition::AttacksWithAnother => {
            "When this and at least one other creature attack".into()
        }
        TriggerCondition::EntersTheBattlefield { subject } => format!(
            "When {} enters the battlefield",
            subject_description(subject)
        ),
        TriggerCondition::Dies { subject } => format!("When {} dies", subject_description(subject)),
        TriggerCondition::Attacks {
            subject: TriggerSubject::This,
        } => "When this attacks".into(),
        TriggerCondition::Attacks {
            subject: TriggerSubject::EnchantedCreature,
        } => "When enchanted creature attacks".into(),
        TriggerCondition::Attacks {
            subject: TriggerSubject::AnotherYouControl(predicate),
        } => format!(
            "When one or more other {} you control attack",
            predicate_description(predicate, "permanents")
        ),
        TriggerCondition::Attacks {
            subject: TriggerSubject::AnyYouControl(predicate),
        } => format!(
            "When one or more {} you control attack",
            predicate_description(predicate, "permanents")
        ),
        TriggerCondition::Attacks {
            subject: TriggerSubject::Another(predicate),
        } => format!(
            "When one or more other {} attack",
            predicate_description(predicate, "permanents")
        ),
        TriggerCondition::AttacksAlone { subject } => {
            format!("When {} attacks alone", subject_description(subject))
        }
        TriggerCondition::OpponentSearchesLibrary => {
            "When an opponent searches their library".into()
        }
        TriggerCondition::FromZone { zone, condition } => format!(
            "{} (from {})",
            condition_description(condition),
            format!("{zone:?}").to_lowercase()
        ),
        TriggerCondition::CreatureDealsCombatDamageToPlayer { subject } => format!(
            "When {} deals combat damage to a player",
            subject_description(subject)
        ),
        TriggerCondition::OneOrMoreDealCombatDamageToPlayer { predicate } => format!(
            "When one or more {} you control deal combat damage to a player",
            predicate_description(predicate, "creatures")
        ),
        TriggerCondition::BecomesTapped { subject } => {
            format!("When {} becomes tapped", subject_description(subject))
        }
        TriggerCondition::TappedForMana { subject } => {
            format!("When {} is tapped for mana", subject_description(subject))
        }
        TriggerCondition::BecomesTargeted { subject } => format!(
            "When {} becomes the target of an opponent's spell or ability",
            subject_description(subject)
        ),
        TriggerCondition::BeginningOfYourUpkeep => "At the beginning of your upkeep".into(),
        TriggerCondition::BeginningOfEachUpkeep => "At the beginning of each upkeep".into(),
        TriggerCondition::YouBend => {
            "Whenever you waterbend, earthbend, firebend, or airbend".into()
        }
        TriggerCondition::BeginningOfYourCombat => "At the beginning of your combat".into(),
        TriggerCondition::BeginningOfYourEndStep => "At the beginning of your end step".into(),
        TriggerCondition::YourAttackingCreatureTriggers => "Whenever a creature you control attacking causes a triggered ability of that creature to trigger".into(),
        TriggerCondition::BeginningOfYourFirstMainPhase => {
            "At the beginning of your first main phase".into()
        }
        TriggerCondition::WhileSourceAttacking { condition } => format!(
            "{} while this is attacking",
            condition_description(condition)
        ),
        TriggerCondition::YouCastNthSpellThisTurn { n } => {
            format!("Whenever you cast your spell number {n} each turn")
        }
        TriggerCondition::Chapter { chapter } => format!("Chapter {}", roman(*chapter)),
        TriggerCondition::YouDrawNthCardThisTurn { n } => {
            format!("When you draw card {n} this turn")
        }
        TriggerCondition::YouScry => "When you scry".into(),
        TriggerCondition::YouScryOrSurveil => "When you scry or surveil".into(),
        TriggerCondition::LeavesTheBattlefield { subject } => {
            format!(
                "When {} leaves the battlefield",
                subject_description(subject)
            )
        }
        TriggerCondition::YouPutCountersOnCreature { kind } => format!(
            "When you put one or more {} counters on a creature",
            counter_description(*kind)
        ),
        TriggerCondition::OpponentDrawsNthCardThisTurn { n } => {
            format!("When an opponent draws card {n} this turn")
        }
        TriggerCondition::InGraveyard { condition } => {
            format!(
                "{} (while this is in your graveyard)",
                condition_description(condition)
            )
        }
    }
}

fn subject_description(subject: &TriggerSubject) -> String {
    match subject {
        TriggerSubject::This => "this".into(),
        TriggerSubject::AnotherYouControl(p) => format!(
            "another {} you control",
            predicate_description(p, "permanent")
        ),
        TriggerSubject::EnchantedCreature => "enchanted creature".into(),
        TriggerSubject::Another(p) => format!("another {}", predicate_description(p, "permanent")),
        TriggerSubject::AnyYouControl(p) => {
            format!("a {} you control", predicate_description(p, "permanent"))
        }
    }
}

fn condition_text(condition: &Condition) -> String {
    match condition {
        Condition::YourTurn => "it is your turn".into(),
        Condition::Always => "always".into(),
        Condition::CardsDrawnAtLeast { count } => {
            format!("you have drawn at least {count} cards this turn")
        }
        Condition::GraveyardAtLeast { count, predicate } => format!(
            "your graveyard has at least {count} {} cards",
            predicate_description(predicate, "matching")
        ),
        Condition::ControlsAtLeast {
            count: 1,
            predicate,
        } => format!(
            "you control a {}",
            predicate_description(predicate, "permanent")
        ),
        Condition::ControlsAtLeast { count, predicate } => format!(
            "you control {count} or more {}s",
            predicate_description(predicate, "permanent")
        ),
        Condition::UnspentManaAtLeast { count } => {
            format!("you have {count} or more unspent mana")
        }
        Condition::AttackedThisTurn => "you attacked this turn".into(),
        Condition::BentAllFourThisTurn => {
            "you've waterbent, earthbent, firebent and airbent this turn".into()
        }
        Condition::LandEnteredThisTurn => {
            "a land entered the battlefield under your control this turn".into()
        }
        Condition::SacrificedThisTurnAtLeast { count } => {
            format!("you have sacrificed {count} or more permanents this turn")
        }
        Condition::AnyOf(conditions) => conditions
            .iter()
            .map(condition_text)
            .collect::<Vec<_>>()
            .join(" or "),
        Condition::Kicked => "it was kicked".into(),
        Condition::TargetMatches(predicate) => format!(
            "that target is a {}",
            predicate_description(predicate, "permanent")
        ),
        Condition::TargetDealtDamageThisTurn => "that target was dealt damage this turn".into(),
        Condition::SourceMatches(predicate) => {
            format!("this is {}", predicate_description(predicate, "permanent"))
        }
        Condition::SourceTapped => "this is tapped".into(),
        Condition::TriggeringHadCounters => "it had counters on it".into(),
        Condition::CardsThisWayMatch(predicate) => format!(
            "a {} was milled or discarded this way",
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

pub(crate) fn predicate_description(p: &CardPredicate, noun: &str) -> String {
    let CardPredicate {
        namesake,
        flying,
        no_counters,
        in_combat,
        token,
        with_counter,
        attacking,
        tapped,
        subtypes_any,
        not_subtypes,
        supertype,
        nonbasic,
        any_of,
        min_power,
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
    if *flying {
        terms.push("flying".into());
    }
    terms.extend(supertype.iter().cloned());
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
    if *namesake {
        terms.push("with the same name as this permanent".into());
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
        CounterKind::DoubleStrike => "double strike",
        _ => kind.name(),
    }
}

fn target_description(t: &TargetSpec) -> String {
    let noun = match t {
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
        TargetSpec::Spell => "spell".into(),
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

fn player_description(player: &PlayerSelector) -> String {
    match player {
        PlayerSelector::You => "you".into(),
        PlayerSelector::EachOpponent => "each opponent".into(),
        PlayerSelector::TriggeringPlayer => "that player".into(),
        PlayerSelector::TargetController => "its controller".into(),
        PlayerSelector::Target(target) => target_description(target),
        PlayerSelector::Owner => "its owner".into(),
    }
}

fn selector_description(selector: &ObjectSelector) -> String {
    match selector {
        ObjectSelector::This => "this".into(),
        ObjectSelector::Target(target) => target_description(target),
        ObjectSelector::Targets(target) => format!("each chosen {}", target_description(target)),
        ObjectSelector::Triggering => "that permanent".into(),
        ObjectSelector::Attached => "the permanent this is attached to".into(),
        ObjectSelector::CreatedThisWay => "that token".into(),
        ObjectSelector::Remembered => "the remembered objects".into(),
        ObjectSelector::Grantor => "the permanent granting this ability".into(),
        ObjectSelector::Moved => "it".into(),
        ObjectSelector::CardsThisWay => "cards moved by the latest mill or discard".into(),
        ObjectSelector::Selected => "the selected cards".into(),
        ObjectSelector::Zone(zone) => format!("your {}", format!("{zone:?}").to_lowercase()),
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
        } => format!(
            "each {}{}{}{}",
            if except.is_some() { "other " } else { "" },
            predicate_description(predicate, "permanent"),
            match max_mana_value {
                Some(value) => format!(" with mana value {} or less", value_description(value)),
                None => String::new(),
            },
            match of {
                Scope::YouControl => " you control",
                Scope::OpponentsControl => " your opponents control",
                Scope::Battlefield => "",
                Scope::TargetPlayerControls => " target player controls",
            }
        ),
    }
}

/// A selector as the subject of a sentence.
fn selector_subject(selector: &ObjectSelector) -> String {
    let mut text = match selector {
        ObjectSelector::Triggering => "It".to_string(),
        other => selector_description(other),
    };
    if let Some(first) = text.get(..1) {
        text.replace_range(..1, &first.to_uppercase());
    }
    text
}

/// One sentence for a `Move`: the verb the destination implies, then its
/// terms.
fn move_description(
    what: &ObjectSelector,
    to: ZoneType,
    tapped: bool,
    transformed: bool,
    under: Option<&PlayerSelector>,
    with_counter: Option<CounterKind>,
    linked: Option<LinkedExile>,
) -> String {
    // A card returning itself (Earthbend, flicker) reads as a return; its
    // owner's control is the natural reading and goes unsaid.
    let returns_itself = matches!(what, ObjectSelector::This);
    let what = selector_description(what);
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
            if let Some(kind) = with_counter {
                text.push_str(&format!(
                    " with a {} counter on it",
                    counter_description(kind)
                ));
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

fn value_description(value: &EffectValue) -> String {
    match value {
        EffectValue::Fixed(value) => value.to_string(),
        EffectValue::X => "the announced value of X".into(),
        EffectValue::Objects(of) => format!("the number of {}", selector_description(of)),
        EffectValue::Half(value) => format!("half of {}, rounded down", value_description(value)),
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
        EffectValue::GreatestPowerYouControl => {
            "the greatest power among creatures you control".into()
        }
        EffectValue::CardsInHand => "the number of cards in your hand".into(),
        EffectValue::CardsDrawnMinus { subtract } => {
            format!("the number of cards you have drawn this turn minus {subtract} (minimum zero)")
        }
        EffectValue::ExcessDamage => "the excess damage dealt this way".into(),
        EffectValue::SourcePower => "this permanent's power".into(),
        EffectValue::TriggeringPower => "that creature's power".into(),
        EffectValue::AttackingMatching { predicate } => format!(
            "the number of attacking {} you control",
            predicate_description(predicate, "creatures")
        ),
        EffectValue::PermanentsSacrificedThisTurn => {
            "the number of permanents you have sacrificed this turn".into()
        }
    }
}

pub(crate) fn mana_description(mana: &Mana) -> String {
    let restricted = mana.restricted_colors();
    let mut text: String = ["W", "U", "B", "R", "G", "C"]
        .into_iter()
        .enumerate()
        .flat_map(|(i, s)| {
            std::iter::repeat_n(
                format!("{{{s}}}"),
                mana.mana[i] as usize + restricted[i] as usize,
            )
        })
        .collect();
    for (restriction, _) in mana.restricted() {
        text.push_str(&format!(" ({} only)", restriction.describe()));
    }
    text
}

fn payment_description(cost: &PaymentCost) -> String {
    match cost {
        PaymentCost::Mana(cost) => cost_description(cost),
        PaymentCost::Waterbend { generic } => format!("waterbend {generic}"),
    }
}

fn cost_description(cost: &ManaCost) -> String {
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

fn keywords_description(k: &Keywords) -> String {
    k.iter()
        .map(Keyword::printed)
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn effects_description(effects: &[Effect]) -> String {
    let parts: Vec<_> = effects.iter().filter_map(effect_description).collect();
    if parts.is_empty() {
        "Do nothing".into()
    } else {
        parts.join(". ")
    }
}

/// Exhaustive matching makes new effect vocabulary require a human description.
/// The internal target-clause cursor is not itself an effect a person chooses.
fn effect_description(effect: &Effect) -> Option<String> {
    use Effect::*;
    Some(match effect {
        SelectTargetRequirement { .. } => return None,
        CreateTokenCopy { of, exceptions } => copy_description(of.as_ref(), exceptions),
        PutCounters { on, kind, amount } => format!("Put {} {} counters on {}", value_description(amount), counter_description(*kind), selector_description(on)),
        PutCountersFrom { from, on } => format!("Put the counters of {} on {}", selector_description(from), selector_description(on)),
        Sacrifice { what, optional } => format!("{}acrifice {}", if *optional { "You may s" } else { "S" }, selector_description(what)),
        Move { what, to, tapped, transformed, under, with_counter, linked, library_end } => {
            if let Some(end) = library_end { format!("Put {} on the {} of its owner's library", selector_description(what), format!("{end:?}").to_lowercase()) }
            else { move_description(what, *to, *tapped, *transformed, under.as_ref(), *with_counter, *linked) }
        },
        Tap { what } => format!("Tap {}", selector_description(what)),
        Untap { what } => format!("Untap {}", selector_description(what)),
        RemoveAllCounters { from } => format!("Remove all counters from {}", selector_description(from)),
        RemoveCounters { up_to, chosen: None, from } => format!("Remove up to {up_to} counters from {}", selector_description(from)),
        RemoveCounters { chosen: Some(kind), from, .. } => format!("Remove a {} counter from {}", counter_description(*kind), selector_description(from)),
        ApplyContinuous { affected, modification, duration } => continuous_description(affected, modification, *duration),
        OptionalPayment { cost, if_paid, if_declined } => {
            let cost = payment_description(cost);
            format!("You may pay {cost}. If paid: {}. Otherwise: {}", effects_description(if_paid), effects_description(if_declined))
        },
        DealDamageToPlayer { player, amount } => format!("Deal {} damage to {}",value_description(amount),player_description(player)),
        DrainLife { amount } => format!("Each opponent loses {0} life and you gain {0} life",value_description(amount)),
        MillCards { count, target: None } => format!("Mill {} cards", value_description(count)),
        MillCards { count, target: Some(target) } => format!("{} mills {} cards", target_description(target), value_description(count)),
        SelectCards { from, predicate, min, max, effects, if_declined } => format!("Choose {min}–{} {} from {}: {}. If none chosen: {}", value_description(max), predicate_description(predicate, "cards"), selector_description(from), effects_description(effects), effects_description(if_declined)),
        TargetClause { requirement, effects } => format!("Choose {}–{} {} for this clause: {}",requirement.min,if requirement.max_x { "X".to_string() } else { requirement.max.to_string() },target_description(&requirement.spec),effects_description(effects)),
        Discard { what: ObjectSelector::You, count: 1 } => "Discard a card".into(),
        Discard { what, count } => format!("{} discards {count} card{}", selector_subject(what), if *count == 1 { "" } else { "s" }),
        Destroy { what } => format!("Destroy {}",selector_description(what)),
        Airbend { what: what @ ObjectSelector::Each { .. } } => format!("Airbend {}. Their owners may cast them from exile for {{2}}", selector_description(what)),
        Airbend { what } => format!("Airbend {}. Its owner may cast it from exile for {{2}}",selector_description(what)),
        OpponentsCastOnlyFromHand => "Until your next turn, your opponents can't cast spells from anywhere other than their hands".to_string(),
        ExileTopCardsToPlay { count, until } => format!(
            "Exile the top {} of your library. You may play {} {}",
            if *count == 1 { "card".to_string() } else { format!("{count} cards") },
            if *count == 1 { "that card" } else { "those cards" },
            match until {
                crate::state::ability::PlayWindow::ThisTurn => "this turn",
                crate::state::ability::PlayWindow::ThroughYourNextTurn => "until the end of your next turn",
            }
        ),
        ExileLinked { chooser, zone, predicate, up_to, grant, target } => format!("{} {} {} {} from {}; {}", match chooser { LinkedChooser::Controller => "You exile", LinkedChooser::TargetPlayer => "That player exiles" }, if *up_to { "up to one" } else { "a" }, predicate_description(predicate, "card"), match zone { ZoneType::Battlefield => "they control", _ => "" }, target_description(target), match grant { LinkedGrant::WaterbendManaValue => "for as long as you control this, you may cast it during your turn by waterbending {X}, where X is its mana value", LinkedGrant::FlashAnyMana => "during your turn, you may cast it as though it had flash, spending mana of any type" }),
        ExileUntilNonlandCastFree { below } => format!("Exile cards from the top of your library until you exile a nonland card. You may cast that card without paying its mana cost if its mana value is less than {}. If you don't cast it this way, put it into your hand", value_description(below)),
        TriggerModes { modes, up_to, unchosen } => format!(
            "Choose {} — {}",
            match (up_to, unchosen) {
                (Some(value), _) => format!("up to {}", value_description(value)),
                (None, true) => "one that hasn't been chosen".into(),
                (None, false) => "one".into(),
            },
            modes.iter().map(|mode| effects_description(mode)).collect::<Vec<_>>().join("; or ")
        ),
        TransformSource => "Transform this".into(),
        ControlPlayer { what, duration } => format!("You control {} during their next {}", selector_description(what), match duration {
            crate::state::ability::PlayerControlDuration::NextCombat => "combat phase",
            crate::state::ability::PlayerControlDuration::NextTurn => "turn",
        }),
        TakeExtraTurn => "Take an extra turn after this one".into(),
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
        OptionalManaPaymentX { if_paid } => format!("Choose X. You may pay {{X}}. If you do, {}", effects_description(if_paid)),
        EachPlayerChoosesToExile { predicate } => format!(
            "Starting with you, each player chooses up to one {} from among permanents your opponents control. Exile those permanents",
            predicate_description(predicate, "permanent")
        ),
        GainControl { player, what, duration } => format!("{} gains control of {}{}", player_description(player), selector_description(what), match duration { EffectDuration::EndOfTurn => " until end of turn", _ => "" }),
        CopyTriggeringObject => "Copy it. You may choose new targets for the copy".into(),
        CastExiledWithSource { predicate } => format!("You may cast a {} from among cards you own exiled with this", predicate_description(predicate, "spell")),
        ChangeTarget { target } => format!("Change the target of {}", target_description(target)),
        WhenYouDo { effects } => format!("When you do: {}", effects_description(effects)),
        CreateDelayedTrigger { event, frequency, expiry, remember, effects } => format!(
            "{}When {}{} ({}), {}",
            remember.as_ref().map_or_else(String::new, |what| format!("Remember {}. ", selector_description(what))),
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
            effects_description(effects)
        ),
        ForPlayer { player, effects } => format!(
            "{} does this: {}",
            player_description(player),
            effects_description(effects)
        ),
        RestoreController(_) => return None,
        EachPlayerKeepsCreatures { max_total_power } => format!("Each player chooses any number of creatures they control with total power {max_total_power} or less, then sacrifices all other creatures they control"),
        CastLibraryTopFree => "Look at the top card of your library. You may cast it without paying its mana cost".into(),
        CastFromHandFree { predicate } => format!("You may cast a {} from your hand without paying its mana cost", predicate_description(predicate, "spell")),
        ShuffleGraveyardIntoLibrary => "Shuffle your graveyard into your library".to_string(),
        RevealHandDiscardChosen { target, predicate } => format!("{} reveals their hand. You choose a {} from it; that player discards it", target_description(target), predicate_description(predicate, "card")),
        SearchLandsOpponentSplits { count, opponent_picks } => format!("Search your library for up to {count} land cards with different names and reveal them. An opponent chooses {opponent_picks}; put those into your graveyard and the rest onto the battlefield tapped, then shuffle"),
        OwnerPutsIntoLibrary { target } => format!("The owner of {} puts it into their library second from the top or on the bottom", target_description(target)),
        PlaceTargetInLibrary { second_from_top: true } => "Put it into its owner's library second from the top".into(),
        PlaceTargetInLibrary { second_from_top: false } => "Put it on the bottom of its owner's library".into(),
        MayOnceEachTurn { effects } => format!(
            "You may: {}. Do this only once each turn",
            effects_description(effects)
        ),
        If { condition: Condition::Always, then, .. } => effects_description(then),
        If { condition, then, otherwise } if otherwise.is_empty() => format!("If {}: {}",condition_text(condition),effects_description(then)),
        If { condition, then, otherwise } => format!("If {}: {}. Otherwise: {}",condition_text(condition),effects_description(then),effects_description(otherwise)),
        DealDamage { amount: EffectValue::Fixed(amount), what } => format!("Deal {amount} damage to {}",selector_description(what)),
        DealDamage { amount, what } => format!("Deal X damage to {}, where X is {}",selector_description(what),value_description(amount)),
        CounterSpell { what } => format!("Counter {}",selector_description(what)),
        CounterAbility { what } => format!("Counter {}",selector_description(what)),
        DrawCards { count } => format!("Draw {} cards", value_description(count)),
        LoseLife { player, amount } => format!("{} loses {} life",player_description(player),value_description(amount)),
        CreateToken { token_name, count: EffectValue::Fixed(count), tapped_and_attacking } => format!("Create {count} {token_name} token{}{}",if *count == 1 { "" } else { "s" },if *tapped_and_attacking { " tapped and attacking" } else { "" }),
        CreateToken { token_name, count, tapped_and_attacking } => format!("Create X {token_name} tokens, where X is {}{}",value_description(count),if *tapped_and_attacking { ". They enter tapped and attacking" } else { "" }),
        GainLife { amount: EffectValue::Fixed(amount) } => format!("You gain {amount} life"),
        GainLife { amount } => format!("You gain life equal to {}", value_description(amount)),
        FightLastTarget => "The first target fights the last target".into(),
        ShuffleLibrary => "Shuffle your library".into(),
        OnNthResolutionThisTurn { n, effect } => format!("If this is resolution {n} of this ability this turn: {}",effects_description(std::slice::from_ref(effect))),
        Scry { count } => format!("Scry {count}"),
        Surveil { count } => format!("Surveil {count}"),
        LookAndSelect { random_order, reveal, to_battlefield, look, min_select, max_select, predicate } => format!("{} the top {} cards of your library. Put {min_select}–{max_select} {} from among them {} and the rest on the bottom in {} order",if *reveal { "Reveal" } else { "Look at" },value_description(look),predicate_description(predicate,"cards"), if *to_battlefield { "onto the battlefield" } else { "into your hand" }, if *random_order { "a random" } else { "your chosen" }),
        PutTopCardsInHand { count } => format!("Put the top {count} cards of your library into your hand"),
        Learn => "Learn: you may reveal a Lesson you own from outside the game and put it into your hand, or discard a card to draw a card, or decline".into(),
        Modal { modes } => format!("Choose one — {}",modes.iter().enumerate().map(|(i,m)| format!("({}) {}",i+1,effects_description(m))).collect::<Vec<_>>().join("; ")),
        CounterUnlessPays { cost } => format!("Counter the spell unless its controller pays {}",payment_description(cost)),
        TargetCreaturesDealPowerDamageToLastTarget => "Each chosen creature except the last target deals damage equal to its power to the last target".into(),
        Earthbend { count, what } => { let count = value_description(count); format!("Earthbend {count} {}: it becomes a 0/0 creature with haste that's still a land; put {count} +1/+1 counters on it. When it dies or is exiled, return it to the battlefield tapped",selector_description(what)) },
        AddOneColorManaValue { amount } => format!("Add X mana of any one color, where X is {}",value_description(amount)),
        AddMana { mana, amount: EffectValue::Fixed(1), until_end_of_combat } => format!("Add {}{}",mana_description(mana),if *until_end_of_combat { ". This mana lasts until end of combat" } else { "" }),
        AddMana { mana, amount, until_end_of_combat } => format!("Add {} for each of {}{}", mana_description(mana), value_description(amount), if *until_end_of_combat { ". This mana lasts until end of combat" } else { "" }),
        AttachSourceTo { what } => format!("Attach this to {}",selector_description(what)),
        DistributeCountersToTappedOpponentCreatures { kind, count } => format!("Distribute {count} {} counters among any number of tapped creatures your opponents control",counter_description(*kind)),
        ForEachTarget { effects } => format!("For each chosen target: {}",effects_description(effects)),
    })
}

/// "Create a token that's a copy of this[, except …]" (CR 707.9).
fn copy_description(of: Option<&TargetSpec>, exceptions: &[Modification<EffectValue>]) -> String {
    let mut text = match of {
        None => "Create a token that's a copy of this".to_string(),
        Some(spec) => format!(
            "Create a token that's a copy of {}",
            target_description(spec)
        ),
    };
    for (index, exception) in exceptions.iter().enumerate() {
        // "It …", lowercased after "except".
        let phrase = continuous_description(
            &ObjectSelector::Triggering,
            exception,
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
    let subject = selector_subject(affected);

    let until = match duration {
        EffectDuration::EndOfTurn => " until end of turn",
        EffectDuration::Indefinite => "",
        EffectDuration::WhileYouControlSource => " for as long as you control this",
    };
    let amount = |value: &EffectValue| match value {
        EffectValue::Fixed(n) => format!("{n:+}"),
        other => format!("+({})", value_description(other)),
    };
    let base = |value: &Option<EffectValue>| match value {
        Some(EffectValue::Fixed(n)) => n.to_string(),
        Some(other) => value_description(other),
        None => "unchanged".to_string(),
    };
    match modification {
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
            let intervening = intervening_if
                .as_ref()
                .map(|c| format!(", if {}", condition_text(c)))
                .unwrap_or_default();
            format!(
                "{subject} gains ‘{}{intervening}, {}’{until}",
                condition_description(condition),
                effects_description(effects)
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
        Modification::DoesntUntap => {
            format!("{subject} doesn't untap during its controller's untap step{until}")
        }
        Modification::LoseKeywords(keywords) => {
            format!("{subject} loses {}{until}", keywords_description(keywords))
        }
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
        Modification::NoMaximumHandSize => format!("{subject} have no maximum hand size{until}"),
        Modification::ControlChanged { .. } => {
            format!("You control {}{until}", subject.to_lowercase())
        }
        Modification::ModeChosen { mode } => format!("{subject} has chosen mode {}", mode + 1),
        Modification::HasAbilitiesOfSource => {
            format!("{subject} has all activated and triggered abilities of the chosen card")
        }
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
}
