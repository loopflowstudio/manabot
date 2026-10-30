//! Every mechanic reaches the policy three ways: match state through the
//! observation tensor (`agent::observation_encoder`, policy input v5), card
//! rules through learning tokens (`cards::tokens`), and both through the
//! LLM text (`manabot/semantic/text_observation.py`, guarded in Python). The
//! guards destructure or match exhaustively, with no wildcard, so a new
//! keyword, counter kind, `Observation`, `Permanent` or `Player` field, or
//! `Effect`, `Ability` or `TriggerCondition` variant does not compile until it
//! names where the policy sees it; every listed column is checked against the
//! v5 widths and every v5 row column must be claimed. Directed tests put a game
//! into each state and check that the named column or token changes, against
//! an otherwise equal baseline v4 cannot tell apart where v5 adds the column.
//! Viewer-safety tests check hidden cards change neither.

use crate::support::{
    game::{deck, five_color_setup, five_color_unfunded_setup, move_to_hand, Scenario},
    inspect::{card_of, library_card},
    play::{activate, cast, choose, choose_mode, pass_until, settle},
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        continuation::{
            ClauseTarget, ContinuationFrame, ManaOutput, VisibleReference, RELATION_DIM_V5,
            RELATION_KEYWORDS_INDEX, RELATION_X_INDEX,
        },
        observation::{
            ActionOption, ActionSpaceData, CardData, CardTypeData, CastPermissionData,
            ContinuousEffectData, EventData, Observation, PermanentData, PlayerData,
            RestrictedManaData, SideboardCardData, StackObjectData, StackTargetData, TurnData,
            KEYWORD_COUNT,
        },
        observation_encoder::{
            encode, EncodedObservation, FactKind, ObservationEncoderConfig, CARD_DIM,
            CARD_PERMISSION_INDEX, CARD_X_INDEX, FACT_DETAIL_INDEX, FACT_DIM, FACT_DURATION_INDEX,
            FACT_ID_INDEX, FACT_KIND_INDEX, FACT_OBJECT_INDEX, FACT_ORDER_INDEX,
            FACT_SOURCE_INDEX, FACT_VALID_INDEX, FACT_VALUE_INDEX, PERMANENT_ABILITIES_LOST_INDEX,
            PERMANENT_ATTACHED_INDEX, PERMANENT_BLOCKING_INDEX, PERMANENT_DIM,
            PERMANENT_EXHAUSTED_INDEX, PERMANENT_FIREBENDING_INDEX, PERMANENT_GRANTED_LAND_INDEX,
            PERMANENT_ONCE_EACH_TURN_INDEX, PERMANENT_TIMESTAMP_INDEX, PLAYER_DIM,
            PLAYER_HAND_ONLY_INDEX, PLAYER_KEEPS_RED_MANA_INDEX, PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
            PLAYER_RESTRICTED_INDEX, PLAYER_SEAT_INDEX, PLAYER_TURN_INDEX,
        },
    },
    flow::turn::StepKind,
    state::{
        effect::{Affected, EffectDuration, Modification},
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::Color,
        permanent::CounterKind,
        zone::ZoneType,
    },
};
use managym::{
    cards::content_pack::default_content_pack,
    cards::tokens::{definition_tokens, name_value, TokenKind},
    state::{
        ability::{
            AffectedSpec, Ability, Effect, EffectValue, LinkedChooser, LinkedGrant, PaymentCost, StaticCondition,
            TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
        },
        card::{CardDefinition, CardType, Keywords},
        mana::{Mana, ManaCost},
        permanent::Permanent,
        player::Player,
        predicate::CardPredicate,
    },
    Game, PlayerConfig,
};
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Where each piece of state is observed.

/// How the policy sees one state field.
#[derive(Clone, Copy, Debug)]
enum Seen {
    /// Permanent tensor column.
    Permanent(usize),
    /// Player tensor columns `start..end`.
    Player(usize, usize),
    /// Fact rows of this kind (`FACT_*`).
    Fact(FactKind),
    /// Observed through another field named here.
    Via(&'static str),
    /// Never visible at a decision point, with the reason.
    Transient(&'static str),
    /// Object identity or bookkeeping rather than game state.
    Identity,
}

/// Keyword → (card column, permanent column). The token is the `Keywords`
/// field of the same name on the definition.
fn keyword_columns(keywords: Keywords) -> [(&'static str, bool, usize, usize); 14] {
    let Keywords {
        flying,
        reach,
        haste,
        flash,
        vigilance,
        trample,
        first_strike,
        double_strike,
        deathtouch,
        lifelink,
        defender,
        menace,
        hexproof,
        indestructible,
    } = keywords;
    [
        ("flying", flying, 17, 10),
        ("reach", reach, 18, 11),
        ("haste", haste, 19, 12),
        ("flash", flash, 28, 13),
        ("vigilance", vigilance, 20, 14),
        ("trample", trample, 21, 15),
        ("first_strike", first_strike, 22, 16),
        ("double_strike", double_strike, 23, 17),
        ("deathtouch", deathtouch, 24, 18),
        ("lifelink", lifelink, 25, 19),
        ("defender", defender, 26, 20),
        ("menace", menace, 27, 21),
        ("hexproof", hexproof, 36, 22),
        ("indestructible", indestructible, 38, 25),
    ]
}

fn permanent_fields(permanent: &Permanent) -> [(&'static str, Seen); 15] {
    let Permanent {
        id: _,
        card: _,
        controller: _,
        tapped: _,
        summoning_sick: _,
        damage: _,
        deathtouch_damage: _,
        timestamp: _,
        counters: _,
        animated: _,
        attacking: _,
        was_cast: _,
        attached_to: _,
        exhausted: _,
        activated_this_turn: _,
    } = permanent;
    [
        ("id", Seen::Identity),
        ("card", Seen::Permanent(24)),
        ("controller", Seen::Permanent(0)),
        ("tapped", Seen::Permanent(1)),
        ("summoning_sick", Seen::Permanent(3)),
        ("damage", Seen::Permanent(2)),
        (
            "deathtouch_damage",
            Seen::Transient("state-based actions destroy the creature before priority"),
        ),
        ("timestamp", Seen::Permanent(PERMANENT_TIMESTAMP_INDEX)),
        ("counters", Seen::Fact(FactKind::Counter)),
        ("animated", Seen::Permanent(8)),
        ("attacking", Seen::Permanent(27)),
        (
            "was_cast",
            Seen::Transient("read only by entering-the-battlefield triggers as it enters"),
        ),
        ("attached_to", Seen::Permanent(PERMANENT_ATTACHED_INDEX)),
        ("exhausted", Seen::Fact(FactKind::UsedAbility)),
        ("activated_this_turn", Seen::Fact(FactKind::UsedAbility)),
    ]
}

fn player_fields(player: &Player) -> [(&'static str, Seen); 11] {
    let Player {
        id: _,
        index: _,
        deck: _,
        sideboard: _,
        known_hand: _,
        name: _,
        life: _,
        drew_when_empty: _,
        alive: _,
        mana_pool: _,
        combat_mana_pool: _,
    } = player;
    [
        ("id", Seen::Identity),
        ("index", Seen::Identity),
        ("deck", Seen::Via("zone counts (library size)")),
        ("sideboard", Seen::Via("sideboard cards and counts")),
        ("known_hand", Seen::Via("known_hand tensor")),
        ("name", Seen::Identity),
        ("life", Seen::Player(0, 1)),
        (
            "drew_when_empty",
            Seen::Transient("the player loses at the next state-based check"),
        ),
        ("alive", Seen::Transient("the game ends")),
        ("mana_pool", Seen::Player(28, 40)),
        ("combat_mana_pool", Seen::Player(27, 28)),
    ]
}

/// Where each `Modification` shows besides its fact rows: the derived
/// column that already applies it. No wildcard: a new variant must be named
/// here and given a witness in [`modification_witnesses`].
fn modification_seen<N>(modification: &Modification<N>) -> (&'static str, Seen) {
    match modification {
        Modification::AddSubtypes(_) => {
            ("Modification::AddSubtypes", Seen::Fact(FactKind::AddedSubtype))
        }
        Modification::BecomeArtifactCreature => {
            ("Modification::BecomeArtifactCreature", Seen::Permanent(8))
        }
        Modification::AddKeywords(_) => (
            "Modification::AddKeywords",
            Seen::Via("keyword_columns: effective keywords"),
        ),
        Modification::LoseAllAbilities => (
            "Modification::LoseAllAbilities",
            Seen::Permanent(PERMANENT_ABILITIES_LOST_INDEX),
        ),
        Modification::Firebending(_) => (
            "Modification::Firebending",
            Seen::Permanent(PERMANENT_FIREBENDING_INDEX),
        ),
        Modification::SetBasePt { .. } => ("Modification::SetBasePt", Seen::Permanent(6)),
        Modification::ModifyPt { .. } => ("Modification::ModifyPt", Seen::Permanent(6)),
        Modification::CantBeBlocked => ("Modification::CantBeBlocked", Seen::Permanent(5)),
        Modification::ExileIfWouldDie => ("Modification::ExileIfWouldDie", Seen::Permanent(23)),
        Modification::KeepUnspentMana(_) => (
            "Modification::KeepUnspentMana",
            Seen::Player(PLAYER_KEEPS_RED_MANA_INDEX, PLAYER_KEEPS_RED_MANA_INDEX + 1),
        ),
        Modification::NoMaximumHandSize => (
            "Modification::NoMaximumHandSize",
            Seen::Player(
                PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
                PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX + 1,
            ),
        ),
    }
}

/// A witness's record form: fixed amounts only.
fn resolved(modification: &Modification<EffectValue>) -> Modification {
    modification.resolve(|value| match value {
        EffectValue::Fixed(amount) => *amount,
        other => panic!("witness amounts are fixed: {other:?}"),
    })
}

/// One value of every `Modification` variant, with whether it affects a
/// player.
fn modification_witnesses() -> Vec<(bool, Modification<EffectValue>)> {
    let permanent = |modification| (false, modification);
    vec![
        permanent(Modification::AddSubtypes(vec!["Ally".into()])),
        permanent(Modification::BecomeArtifactCreature),
        permanent(Modification::AddKeywords(Keywords {
            flying: true,
            ..Default::default()
        })),
        permanent(Modification::LoseAllAbilities),
        permanent(Modification::Firebending(2)),
        permanent(Modification::base_pt(Some(4), None)),
        permanent(Modification::pump(2, 1)),
        permanent(Modification::CantBeBlocked),
        permanent(Modification::ExileIfWouldDie),
        (true, Modification::KeepUnspentMana(Color::Red)),
        (true, Modification::NoMaximumHandSize),
    ]
}

const MODIFICATION_VARIANTS: usize = 11;

/// The fact-row duration code of each `EffectDuration`; no wildcard.
fn duration_code(duration: EffectDuration) -> f32 {
    match duration {
        EffectDuration::EndOfTurn => 1.0,
        EffectDuration::Indefinite => 2.0,
    }
}

/// The token each `AffectedSpec` contributes; no wildcard.
fn affected_token(affected: &AffectedSpec) -> &'static str {
    match affected {
        AffectedSpec::You => "AffectedSpec::You",
        AffectedSpec::Source => "AffectedSpec::Source",
        AffectedSpec::Target(_) => "AffectedSpec::Target",
        AffectedSpec::TriggeringPermanent => "AffectedSpec::TriggeringPermanent",
        AffectedSpec::EachYouControl(_) => "AffectedSpec::EachYouControl",
    }
}

fn affected_witnesses() -> [AffectedSpec; 5] {
    [
        AffectedSpec::You,
        AffectedSpec::Source,
        AffectedSpec::Target(TargetSpec::Creature),
        AffectedSpec::TriggeringPermanent,
        AffectedSpec::EachYouControl(CardPredicate::creature()),
    ]
}

/// The token every `Effect` variant contributes. No wildcard: a new variant
/// must be named here (and given a witness below).
fn effect_token(effect: &Effect) -> &'static str {
    match effect {
        Effect::CreateNonlegendaryCopy => "Effect::CreateNonlegendaryCopy",
        Effect::ApplyContinuous { .. } => "Effect::ApplyContinuous",
        Effect::PutCountersOnTriggeringPermanent { .. } => {
            "Effect::PutCountersOnTriggeringPermanent"
        }
        Effect::OptionalPayment { .. } => "Effect::OptionalPayment",
        Effect::Sacrifice { .. } => "Effect::Sacrifice",
        Effect::DealDamageToController { .. } => "Effect::DealDamageToController",
        Effect::DrainLife { .. } => "Effect::DrainLife",
        Effect::ReturnGraveyardTarget { .. } => "Effect::ReturnGraveyardTarget",
        Effect::MillCards { .. } => "Effect::MillCards",
        Effect::MillAndSelect { .. } => "Effect::MillAndSelect",
        Effect::DrainLifeValue { .. } => "Effect::DrainLifeValue",
        Effect::TargetClause { .. } => "Effect::TargetClause",
        Effect::SelectTargetRequirement { .. } => "Effect::SelectTargetRequirement",
        Effect::SearchBasicLand { .. } => "Effect::SearchBasicLand",
        Effect::OptionalDiscardDraw { .. } => "Effect::OptionalDiscardDraw",
        Effect::DiscardTargetPlayer { .. } => "Effect::DiscardTargetPlayer",
        Effect::Destroy { .. } => "Effect::Destroy",
        Effect::Exile { .. } => "Effect::Exile",
        Effect::ReturnToHandDrawIfControlled { .. } => "Effect::ReturnToHandDrawIfControlled",
        Effect::Discard => "Effect::Discard",
        Effect::DealDamageValue { .. } => "Effect::DealDamageValue",
        Effect::PutCountersValue { .. } => "Effect::PutCountersValue",
        Effect::IfCondition { .. } => "Effect::IfCondition",
        Effect::ReturnToHand { .. } => "Effect::ReturnToHand",
        Effect::DealDamage { .. } => "Effect::DealDamage",
        Effect::CounterSpell { .. } => "Effect::CounterSpell",
        Effect::DrawCards { .. } => "Effect::DrawCards",
        Effect::MassDamage { .. } => "Effect::MassDamage",
        Effect::DamageOpponentCreatures { .. } => "Effect::DamageOpponentCreatures",
        Effect::CreateTokensValue { .. } => "Effect::CreateTokensValue",
        Effect::ExileWithControllerToken { .. } => "Effect::ExileWithControllerToken",
        Effect::CreateToken { .. } => "Effect::CreateToken",
        Effect::PutCountersOnSource { .. } => "Effect::PutCountersOnSource",
        Effect::PutCounters { .. } => "Effect::PutCounters",
        Effect::TapSource => "Effect::TapSource",
        Effect::UntapSource => "Effect::UntapSource",
        Effect::GainLife { .. } => "Effect::GainLife",
        Effect::OnNthResolutionThisTurn { .. } => "Effect::OnNthResolutionThisTurn",
        Effect::Scry { .. } => "Effect::Scry",
        Effect::LookAndSelect { .. } => "Effect::LookAndSelect",
        Effect::PutTopCardsInHand { .. } => "Effect::PutTopCardsInHand",
        Effect::Learn => "Effect::Learn",
        Effect::Modal { .. } => "Effect::Modal",
        Effect::CounterUnlessPays { .. } => "Effect::CounterUnlessPays",
        Effect::IfKicked { .. } => "Effect::IfKicked",
        Effect::IfGraveyardAtLeast { .. } => "Effect::IfGraveyardAtLeast",
        Effect::TargetCreaturesDealPowerDamageToLastTarget => {
            "Effect::TargetCreaturesDealPowerDamageToLastTarget"
        }
        Effect::Earthbend { .. } => "Effect::Earthbend",
        Effect::ReturnSourceToBattlefieldTapped => "Effect::ReturnSourceToBattlefieldTapped",
        Effect::ExileUntilSourceLeaves { .. } => "Effect::ExileUntilSourceLeaves",
        Effect::AddMana { .. } => "Effect::AddMana",
        Effect::UntapTarget { .. } => "Effect::UntapTarget",
        Effect::IfTargetMatches { .. } => "Effect::IfTargetMatches",
        Effect::ForEachTarget { .. } => "Effect::ForEachTarget",
        Effect::PutCountersOnEachMatching { .. } => "Effect::PutCountersOnEachMatching",
        Effect::Airbend { .. } => "Effect::Airbend",
        Effect::AirbendAll { .. } => "Effect::AirbendAll",
        Effect::OpponentsCastOnlyFromHand => "Effect::OpponentsCastOnlyFromHand",
        Effect::ExileTopCardPlayThisTurn => "Effect::ExileTopCardPlayThisTurn",
        Effect::ExileLinked { .. } => "Effect::ExileLinked",
        Effect::ExileUntilNonlandCastFree { .. } => "Effect::ExileUntilNonlandCastFree",
        Effect::CastFromHandFree { .. } => "Effect::CastFromHandFree",
        Effect::ShuffleGraveyardIntoLibrary => "Effect::ShuffleGraveyardIntoLibrary",
        Effect::OwnerPutsIntoLibrary { .. } => "Effect::OwnerPutsIntoLibrary",
        Effect::RevealHandDiscardChosen { .. } => "Effect::RevealHandDiscardChosen",
        Effect::SearchLandsOpponentSplits { .. } => "Effect::SearchLandsOpponentSplits",
        Effect::PlaceTargetInLibrary { .. } => "Effect::PlaceTargetInLibrary",
        Effect::SacrificeSource { .. } => "Effect::SacrificeSource",
        Effect::ExileSourceSpell => "Effect::ExileSourceSpell",
        Effect::DestroyCreaturesLosingAbilities { .. } => "Effect::DestroyCreaturesLosingAbilities",
        Effect::AttachSourceTo { .. } => "Effect::AttachSourceTo",
        Effect::ExileAttached => "Effect::ExileAttached",
        Effect::TapAttached => "Effect::TapAttached",
        Effect::RemoveAllCountersFromAttached => "Effect::RemoveAllCountersFromAttached",
        Effect::DistributeCountersToTappedOpponentCreatures { .. } => {
            "Effect::DistributeCountersToTappedOpponentCreatures"
        }
        Effect::ReturnGrantorToHand => "Effect::ReturnGrantorToHand",
        Effect::ShuffleAttachedIntoLibrary => "Effect::ShuffleAttachedIntoLibrary",
        Effect::ReturnSourceToHand => "Effect::ReturnSourceToHand",
        Effect::GainLifeValue { .. } => "Effect::GainLifeValue",
        Effect::AddOneColorManaValue { .. } => "Effect::AddOneColorManaValue",
        Effect::AddManaValue { .. } => "Effect::AddManaValue",
        Effect::FightLastTarget => "Effect::FightLastTarget",
        Effect::SearchLibraryToHand { .. } => "Effect::SearchLibraryToHand",
        Effect::TapTarget { .. } => "Effect::TapTarget",
        Effect::RemoveCounters { .. } => "Effect::RemoveCounters",
    }
}

/// Number of `Effect` variants named in [`effect_token`].
const EFFECT_VARIANTS: usize = 85;

/// One value of every `Effect` variant.
fn effect_witnesses() -> Vec<Effect> {
    let creature = CardPredicate::creature;
    let target = || TargetSpec::Creature;
    let value = || EffectValue::CreaturesWithCounters;
    let draw = || vec![Effect::DrawCards { count: 1 }];
    vec![
        Effect::CreateNonlegendaryCopy,
        Effect::until_end_of_turn(AffectedSpec::Source, Modification::pump(1, 1)),
        Effect::PutCountersOnTriggeringPermanent { count: 1 },
        Effect::OptionalPayment {
            cost: PaymentCost::Waterbend { generic: 2 },
            if_paid: draw(),
            if_declined: vec![Effect::Discard],
        },
        Effect::Sacrifice {
            predicate: creature(),
        },
        Effect::DealDamageToController { amount: 1 },
        Effect::DrainLife { amount: 1 },
        Effect::ReturnGraveyardTarget {
            target: TargetSpec::GraveyardCardYouOwn {
                predicate: creature(),
            },
            tapped_onto_battlefield: false,
        },
        Effect::MillCards {
            count: 2,
            target: None,
        },
        Effect::MillAndSelect {
            count: 2,
            predicate: creature(),
            if_declined: draw(),
            if_milled: None,
        },
        Effect::DrainLifeValue { amount: value() },
        Effect::TargetClause {
            requirement: TargetRequirement::one(target()),
            effects: draw(),
        },
        Effect::SelectTargetRequirement { index: 0 },
        Effect::SearchBasicLand {
            subtype: None,
            count: 1,
            tapped: true,
            untap_if_lands: None,
            bottom_source: false,
            to_hand: false,
        },
        Effect::OptionalDiscardDraw { count: 1, up_to: 1 },
        Effect::DiscardTargetPlayer {
            count: 1,
            target: TargetSpec::Opponent,
        },
        Effect::Destroy { target: target() },
        Effect::Exile { target: target() },
        Effect::ReturnToHandDrawIfControlled { target: target() },
        Effect::Discard,
        Effect::DealDamageValue {
            amount: value(),
            target: target(),
        },
        Effect::PutCountersValue {
            count: value(),
            target: target(),
        },
        Effect::IfCondition {
            condition: StaticCondition::YourTurn,
            effects: draw(),
            otherwise: vec![],
        },
        Effect::ReturnToHand { target: target() },
        Effect::DealDamage {
            amount: 1,
            target: target(),
        },
        Effect::CounterSpell {
            target: TargetSpec::Spell,
        },
        Effect::DrawCards { count: 1 },
        Effect::MassDamage { amount: 1 },
        Effect::DamageOpponentCreatures { amount: 1 },
        Effect::CreateTokensValue {
            token_name: "Clue".into(),
            count: value(),
        },
        Effect::ExileWithControllerToken {
            target: target(),
            token_name: "Clue".into(),
        },
        Effect::CreateToken {
            token_name: "Clue".into(),
            count: 1,
            tapped_and_attacking: false,
            attach_source: false,
        },
        Effect::PutCountersOnSource {
            count: 1,
            kind: CounterKind::PlusOnePlusOne,
        },
        Effect::PutCounters {
            count: 1,
            target: target(),
            kind: CounterKind::Stun,
        },
        Effect::TapSource,
        Effect::UntapSource,
        Effect::GainLife { amount: 1 },
        Effect::OnNthResolutionThisTurn {
            n: 2,
            effect: Box::new(Effect::DrawCards { count: 1 }),
        },
        Effect::Scry { count: 1 },
        Effect::LookAndSelect {
            reveal: false,
            look: EffectValue::Fixed(3),
            min_select: 0,
            max_select: 1,
            predicate: creature(),
        },
        Effect::PutTopCardsInHand { count: 1 },
        Effect::Learn,
        Effect::Modal {
            modes: vec![draw(), vec![Effect::Discard]],
        },
        Effect::CounterUnlessPays {
            cost: ManaCost::parse("{2}"),
        },
        Effect::IfKicked {
            then: draw(),
            otherwise: vec![],
        },
        Effect::IfGraveyardAtLeast {
            count: 2,
            predicate: creature(),
            then: draw(),
            otherwise: vec![],
        },
        Effect::TargetCreaturesDealPowerDamageToLastTarget,
        Effect::Earthbend {
            count: EffectValue::Fixed(1),
            target: TargetSpec::LandYouControl,
        },
        Effect::ReturnSourceToBattlefieldTapped,
        Effect::ExileUntilSourceLeaves { target: target() },
        Effect::AddMana {
            mana: Mana::parse("R"),
            until_end_of_combat: true,
        },
        Effect::UntapTarget { target: target() },
        Effect::IfTargetMatches {
            predicate: creature(),
            then: draw(),
        },
        Effect::ForEachTarget { effects: draw() },
        Effect::PutCountersOnEachMatching {
            count: 1,
            predicate: creature(),
            other: true,
        },
        Effect::Airbend {
            target: TargetSpec::Permanent {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Land],
                    ..Default::default()
                },
            },
        },
        Effect::AirbendAll {
            predicate: CardPredicate {
                card_type: Some(CardType::Creature),
                ..Default::default()
            },
        },
        Effect::OpponentsCastOnlyFromHand,
        Effect::ExileTopCardPlayThisTurn,
        Effect::ExileLinked {
            chooser: LinkedChooser::TargetPlayer,
            zone: ZoneType::Battlefield,
            predicate: CardPredicate {
                card_type: Some(CardType::Creature),
                ..Default::default()
            },
            up_to: false,
            grant: LinkedGrant::FlashAnyMana,
            target: TargetSpec::Opponent,
        },
        Effect::ExileUntilNonlandCastFree {
            below: EffectValue::BattlefieldMatching {
                predicate: CardPredicate {
                    subtype: Some("Mountain".into()),
                    ..Default::default()
                },
            },
        },
        Effect::CastFromHandFree {
            predicate: CardPredicate {
                not_card_types: vec![CardType::Creature, CardType::Land],
                ..Default::default()
            },
        },
        Effect::ShuffleGraveyardIntoLibrary,
        Effect::OwnerPutsIntoLibrary {
            target: TargetSpec::Permanent {
                predicate: CardPredicate {
                    card_types_any: vec![CardType::Creature, CardType::Enchantment],
                    ..Default::default()
                },
            },
        },
        Effect::RevealHandDiscardChosen {
            target: TargetSpec::Opponent,
            predicate: CardPredicate {
                not_card_types: vec![CardType::Land, CardType::Instant, CardType::Sorcery],
                ..Default::default()
            },
        },
        Effect::SearchLandsOpponentSplits {
            count: 4,
            opponent_picks: 2,
        },
        Effect::PlaceTargetInLibrary {
            second_from_top: true,
        },
        Effect::SacrificeSource {
            then: vec![Effect::Scry { count: 2 }],
        },
        Effect::ExileSourceSpell,
        Effect::DestroyCreaturesLosingAbilities {
            max_mana_value: EffectValue::X,
        },
        Effect::AttachSourceTo {
            target: TargetSpec::CreatureYouControl,
        },
        Effect::ExileAttached,
        Effect::TapAttached,
        Effect::RemoveAllCountersFromAttached,
        Effect::DistributeCountersToTappedOpponentCreatures {
            kind: CounterKind::Stun,
            count: 3,
        },
        Effect::ReturnGrantorToHand,
        Effect::ShuffleAttachedIntoLibrary,
        Effect::ReturnSourceToHand,
        Effect::GainLifeValue {
            amount: EffectValue::ExcessDamage,
        },
        Effect::AddOneColorManaValue {
            amount: EffectValue::GreatestSharedCreatureType,
        },
        Effect::AddManaValue {
            mana: Mana::parse("R"),
            amount: EffectValue::ExcessDamage,
        },
        Effect::FightLastTarget,
        Effect::SearchLibraryToHand {
            predicate: CardPredicate {
                subtypes_any: vec!["Lesson".into(), "Noble".into()],
                ..Default::default()
            },
        },
        Effect::TapTarget {
            target: TargetSpec::Creature,
        },
        Effect::RemoveCounters {
            up_to: 3,
            chosen: None,
            target: TargetSpec::Creature,
        },
    ]
}

/// The token every triggered-ability condition contributes.
fn trigger_token(condition: &TriggerCondition) -> &'static str {
    match condition {
        TriggerCondition::WasCast { .. } => "TriggerCondition::WasCast",
        TriggerCondition::Sacrificed { .. } => "TriggerCondition::Sacrificed",
        TriggerCondition::OnceEachTurn { .. } => "TriggerCondition::OnceEachTurn",
        TriggerCondition::YouCastSpell { .. } => "TriggerCondition::YouCastSpell",
        TriggerCondition::EntersTheBattlefield { .. } => "TriggerCondition::EntersTheBattlefield",
        TriggerCondition::Dies { .. } => "TriggerCondition::Dies",
        TriggerCondition::Attacks { .. } => "TriggerCondition::Attacks",
        TriggerCondition::BecomesTapped { .. } => "TriggerCondition::BecomesTapped",
        TriggerCondition::TappedForMana { .. } => "TriggerCondition::TappedForMana",
        TriggerCondition::BecomesTargeted { .. } => "TriggerCondition::BecomesTargeted",
        TriggerCondition::BeginningOfYourUpkeep => "TriggerCondition::BeginningOfYourUpkeep",
        TriggerCondition::BeginningOfYourCombat => "TriggerCondition::BeginningOfYourCombat",
        TriggerCondition::YouDrawNthCardThisTurn { .. } => {
            "TriggerCondition::YouDrawNthCardThisTurn"
        }
        TriggerCondition::ActiveIf { .. } => "TriggerCondition::ActiveIf",
        TriggerCondition::AttacksAlone { .. } => "TriggerCondition::AttacksAlone",
        TriggerCondition::BeginningOfYourEndStep => "TriggerCondition::BeginningOfYourEndStep",
    }
}

fn trigger_witnesses() -> Vec<TriggerCondition> {
    let this = || TriggerSubject::This;
    let enters = || Box::new(TriggerCondition::EntersTheBattlefield { subject: this() });
    vec![
        TriggerCondition::WasCast {
            condition: enters(),
        },
        TriggerCondition::Sacrificed {
            predicate: CardPredicate::creature(),
            controller_only: true,
        },
        TriggerCondition::OnceEachTurn {
            condition: enters(),
        },
        TriggerCondition::YouCastSpell {
            predicate: CardPredicate::default(),
        },
        TriggerCondition::EntersTheBattlefield { subject: this() },
        TriggerCondition::Dies { subject: this() },
        TriggerCondition::Attacks { subject: this() },
        TriggerCondition::BecomesTapped { subject: this() },
        TriggerCondition::TappedForMana { subject: this() },
        TriggerCondition::BecomesTargeted { subject: this() },
        TriggerCondition::BeginningOfYourUpkeep,
        TriggerCondition::BeginningOfYourCombat,
        TriggerCondition::YouDrawNthCardThisTurn { n: 2 },
        TriggerCondition::ActiveIf {
            active_if: StaticCondition::YourTurn,
            condition: enters(),
        },
        TriggerCondition::AttacksAlone {
            subject: TriggerSubject::AnyYouControl(CardPredicate {
                card_type: Some(CardType::Creature),
                ..Default::default()
            }),
        },
        TriggerCondition::BeginningOfYourEndStep,
    ]
}

fn ability_token(ability: &Ability) -> &'static str {
    match ability {
        Ability::Triggered { .. } => "Ability::Triggered",
    }
}

// ---------------------------------------------------------------------------
// Fixtures.

/// Player 0 controls one Gray Ogre; both libraries are otherwise Mountains.
fn ogre_game() -> (Game, PermanentId) {
    let mut game = Game::new(
        vec![
            PlayerConfig::new("agent", deck(&[("Gray Ogre", 4), ("Mountain", 36)])),
            PlayerConfig::new("opponent", deck(&[("Mountain", 40)])),
        ],
        7,
        false,
    );
    let ogre = game
        .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
        .unwrap();
    (game, ogre)
}

fn observe(game: &Game, viewer: usize) -> (Observation, EncodedObservation) {
    let observation = Observation::for_player(game, PlayerId(viewer));
    let encoded = encode(&observation, &ObservationEncoderConfig::default()).unwrap();
    (observation, encoded)
}

fn permanent_row(encoded: &EncodedObservation) -> &[f32] {
    &encoded.agent_permanents[..PERMANENT_DIM]
}

fn card_row<'a>(
    game: &Game,
    observation: &Observation,
    encoded: &'a EncodedObservation,
    permanent: PermanentId,
) -> &'a [f32] {
    let card = game.state.permanents[permanent].as_ref().unwrap().card;
    let id = game.state.cards[card].id.0 as i32;
    let row = observation
        .agent_cards
        .iter()
        .position(|card| card.id == id)
        .unwrap();
    &encoded.agent_cards[row * CARD_DIM..(row + 1) * CARD_DIM]
}

fn changed(before: &[f32], after: &[f32]) -> Vec<usize> {
    (0..before.len())
        .filter(|&column| before[column] != after[column])
        .collect()
}

fn tokens_of(definition: &CardDefinition) -> Vec<(TokenKind, i32)> {
    definition_tokens(definition, &default_content_pack()).unwrap()
}

fn spell(effects: Vec<Effect>) -> CardDefinition {
    CardDefinition {
        name: "Witness".into(),
        spell_effects: effects,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// (a) Guards name a real place for every variant and field.

#[test]
fn every_effect_variant_reaches_the_learning_tokens() {
    let witnesses = effect_witnesses();
    assert_eq!(witnesses.len(), EFFECT_VARIANTS);
    let mut names = witnesses.iter().map(effect_token).collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), EFFECT_VARIANTS, "one witness per variant");
    for effect in witnesses {
        let tokens = tokens_of(&spell(vec![effect.clone()]));
        let expected = (TokenKind::Variant, name_value(effect_token(&effect)));
        assert!(tokens.contains(&expected), "{effect:?}");
    }
}

#[test]
fn every_trigger_condition_and_ability_reaches_the_learning_tokens() {
    for condition in trigger_witnesses() {
        let ability = Ability::Triggered {
            condition: condition.clone(),
            effects: vec![Effect::DrawCards { count: 1 }],
        };
        let definition = CardDefinition {
            name: "Witness".into(),
            abilities: vec![ability.clone()],
            ..Default::default()
        };
        let tokens = tokens_of(&definition);
        for name in [trigger_token(&condition), ability_token(&ability)] {
            assert!(
                tokens.contains(&(TokenKind::Variant, name_value(name))),
                "{name}"
            );
        }
    }
}

#[test]
fn every_permanent_and_player_field_names_where_it_is_seen() {
    let (game, ogre) = ogre_game();
    let permanent = game.state.permanents[ogre].as_ref().unwrap();
    let player = &game.state.players[0];
    let fields = permanent_fields(permanent)
        .into_iter()
        .map(|(field, seen)| (field, seen, PERMANENT_DIM - 1))
        .chain(
            player_fields(player)
                .into_iter()
                .map(|(field, seen)| (field, seen, PLAYER_DIM)),
        );
    for (field, seen, width) in fields {
        match seen {
            Seen::Permanent(column) => assert!(column < width, "{field}"),
            Seen::Player(start, end) => assert!(start < end && end <= width, "{field}"),
            Seen::Via(place) | Seen::Transient(place) => assert!(!place.is_empty(), "{field}"),
            Seen::Fact(_) | Seen::Identity => {}
        }
    }
}

#[test]
fn every_continuous_effect_variant_reaches_the_learning_tokens() {
    let witnesses = modification_witnesses();
    assert_eq!(witnesses.len(), MODIFICATION_VARIANTS);
    let names: BTreeSet<_> = witnesses
        .iter()
        .map(|(_, modification)| modification_seen(modification).0)
        .collect();
    assert_eq!(names.len(), MODIFICATION_VARIANTS, "one witness per variant");
    for (on_player, modification) in witnesses {
        let affected = if on_player {
            AffectedSpec::You
        } else {
            AffectedSpec::Source
        };
        let effect = Effect::ApplyContinuous {
            affected,
            modification: modification.clone(),
            duration: EffectDuration::EndOfTurn,
        };
        let tokens = tokens_of(&spell(vec![effect]));
        let name = modification_seen(&modification).0;
        assert!(
            tokens.contains(&(TokenKind::Variant, name_value(name))),
            "{name}"
        );
    }
    for duration in [EffectDuration::EndOfTurn, EffectDuration::Indefinite] {
        let effect = Effect::ApplyContinuous {
            affected: AffectedSpec::Source,
            modification: Modification::CantBeBlocked,
            duration,
        };
        let name = format!("EffectDuration::{duration:?}");
        assert!(tokens_of(&spell(vec![effect])).contains(&(TokenKind::Variant, name_value(&name))));
        assert!(duration_code(duration) > 0.0);
    }
    for affected in affected_witnesses() {
        let name = affected_token(&affected);
        let effect = Effect::ApplyContinuous {
            affected,
            modification: Modification::CantBeBlocked,
            duration: EffectDuration::EndOfTurn,
        };
        assert!(
            tokens_of(&spell(vec![effect])).contains(&(TokenKind::Variant, name_value(name))),
            "{name}"
        );
    }
}

// ---------------------------------------------------------------------------
// (b) Directed: each mechanic moves its named column or token.

#[test]
fn printed_and_granted_keywords_set_their_card_and_permanent_columns_and_tokens() {
    let (base_game, ogre) = ogre_game();
    let (base_obs, base) = observe(&base_game, 0);
    for index in 0..14 {
        let (name, _, card_column, permanent_column) = keyword_columns(Keywords::default())[index];
        let mut keywords = Keywords::default();
        set_keyword(&mut keywords, name);
        assert!(keyword_columns(keywords)[index].1, "{name}");

        let mut granted = base_game.clone();
        granted.state.permanents[ogre]
            .as_mut()
            .unwrap()
            .temp_keywords = keywords;
        let (_, after) = observe(&granted, 0);
        assert_eq!(
            changed(permanent_row(&base), permanent_row(&after)),
            [permanent_column],
            "{name}"
        );

        let mut printed = base_game.clone();
        let card = printed.state.permanents[ogre].as_ref().unwrap().card;
        printed.state.cards[card].keywords = keywords;
        let (obs, after) = observe(&printed, 0);
        assert!(
            changed(
                card_row(&base_game, &base_obs, &base, ogre),
                card_row(&printed, &obs, &after, ogre)
            )
            .contains(&card_column),
            "{name}"
        );

        let definition = CardDefinition {
            name: "Witness".into(),
            keywords,
            ..Default::default()
        };
        let tokens = tokens_of(&definition);
        let at = tokens
            .iter()
            .position(|token| *token == (TokenKind::Field, name_value(name)))
            .unwrap();
        assert_eq!(tokens[at + 1], (TokenKind::Boolean, 1), "{name}");
    }
}

fn set_keyword(keywords: &mut Keywords, name: &str) {
    let flag = match name {
        "flying" => &mut keywords.flying,
        "reach" => &mut keywords.reach,
        "haste" => &mut keywords.haste,
        "flash" => &mut keywords.flash,
        "vigilance" => &mut keywords.vigilance,
        "trample" => &mut keywords.trample,
        "first_strike" => &mut keywords.first_strike,
        "double_strike" => &mut keywords.double_strike,
        "deathtouch" => &mut keywords.deathtouch,
        "lifelink" => &mut keywords.lifelink,
        "defender" => &mut keywords.defender,
        "menace" => &mut keywords.menace,
        "hexproof" => &mut keywords.hexproof,
        "indestructible" => &mut keywords.indestructible,
        other => panic!("unknown keyword {other}"),
    };
    *flag = true;
}

#[test]
fn each_permanent_status_moves_its_named_column() {
    let (base_game, ogre) = ogre_game();
    let (_, base) = observe(&base_game, 0);
    let fields = permanent_fields(base_game.state.permanents[ogre].as_ref().unwrap());
    let mutate = |field: &str, permanent: &mut Permanent| match field {
        "tapped" => permanent.tapped = !permanent.tapped,
        "summoning_sick" => permanent.summoning_sick = !permanent.summoning_sick,
        "damage" => permanent.damage = 1,
        "temp_power" => permanent.temp_power = 2,
        "temp_toughness" => permanent.temp_toughness = 2,
        "plus1_counters" => permanent.plus1_counters = 1,
        "temp_base_power" => permanent.temp_base_power = Some(5),
        "temp_base_toughness" => permanent.temp_base_toughness = Some(5),
        "cant_be_blocked_this_turn" => permanent.cant_be_blocked_this_turn = true,
        "firebending_this_turn" => permanent.firebending_this_turn = 2,
        "animated" => permanent.animated = true,
        "abilities_lost" => permanent.abilities_lost = true,
        "temp_subtypes" => permanent.temp_subtypes = vec!["Avatar".into()],
        "exile_if_dies_this_turn" => permanent.exile_if_dies_this_turn = true,
        "attacking" => permanent.attacking = true,
        other => panic!("no directed mutation for {other}"),
    };
    for (field, seen) in fields {
        let Seen::Permanent(column) = seen else {
            continue;
        };
        if matches!(field, "card" | "controller") {
            continue; // Identity of the row itself; covered by the encoder's own tests.
        }
        if matches!(
            field,
            "attached_to" | "becomes_creature" | "exhausted" | "activated_this_turn"
        ) {
            // Need a real attachment, animation or ability; the directed v5
            // tests below reach these states through play.
            continue;
        }
        let mut game = base_game.clone();
        mutate(field, game.state.permanents[ogre].as_mut().unwrap());
        let (_, after) = observe(&game, 0);
        assert!(
            changed(permanent_row(&base), permanent_row(&after)).contains(&column),
            "{field} did not move column {column}"
        );
    }
}

#[test]
fn player_life_and_mana_move_their_named_columns() {
    let (base_game, _) = ogre_game();
    let (_, base) = observe(&base_game, 0);
    let fields = player_fields(&base_game.state.players[0]);
    for (field, seen) in fields {
        let Seen::Player(start, end) = seen else {
            continue;
        };
        let mut game = base_game.clone();
        let player = &mut game.state.players[0];
        match field {
            "life" => player.life = 7,
            "mana_pool" => player.mana_pool = Mana::parse("R"),
            "combat_mana_pool" => player.combat_mana_pool = Mana::parse("R"),
            "no_maximum_hand_size" => player.no_maximum_hand_size = true,
            "keeps_red_mana" => player.keeps_red_mana = true,
            other => panic!("no directed mutation for {other}"),
        }
        let (_, after) = observe(&game, 0);
        let moved = changed(&base.agent_player, &after.agent_player);
        assert!(
            moved.iter().any(|column| (start..end).contains(column)),
            "{field} moved {moved:?}, expected {start}..{end}"
        );
    }
}

// ---------------------------------------------------------------------------
// (c) Viewer safety: hidden cards change neither the tensor nor the catalog.

#[test]
fn opponent_hidden_cards_do_not_reach_the_viewer_tensor() {
    let game = |opponent: &[(&str, usize)]| {
        Game::new(
            vec![
                PlayerConfig::new("agent", deck(&[("Gray Ogre", 4), ("Mountain", 36)])),
                PlayerConfig::new("opponent", deck(opponent)),
            ],
            11,
            false,
        )
    };
    let mountains = game(&[("Mountain", 40)]);
    let ogres = game(&[("Gray Ogre", 20), ("Mountain", 20)]);
    let hand = |game: &Game| {
        let mut names = game
            .state
            .zones
            .zone_cards(ZoneType::Hand, PlayerId(1))
            .iter()
            .map(|card| game.state.cards[*card].name.clone())
            .collect::<Vec<_>>();
        names.sort();
        names
    };
    assert_ne!(hand(&mountains), hand(&ogres), "the hidden hands differ");
    let (_, left) = observe(&mountains, 0);
    let (_, right) = observe(&ogres, 0);
    assert_eq!(left.agent_player, right.agent_player);
    assert_eq!(left.opponent_player, right.opponent_player);
    assert_eq!(left.agent_cards, right.agent_cards);
    assert_eq!(left.opponent_cards, right.opponent_cards);
    assert_eq!(left.opponent_permanents, right.opponent_permanents);
    assert_eq!(left.known_hand, right.known_hand);
    assert_eq!(left.semantic_cards, right.semantic_cards);
}

#[test]
fn learning_tokens_never_carry_names_or_printed_text() {
    let pack = default_content_pack();
    for (_, definition) in pack.definition_entries() {
        let tokens = definition_tokens(definition, &pack).unwrap();
        for field in ["key", "scryfall_id", "name", "text_box"] {
            assert!(
                !tokens.contains(&(TokenKind::Field, name_value(field))),
                "{}: {field}",
                definition.name
            );
        }
        if !definition.text_box.is_empty() {
            assert!(!tokens.contains(&(TokenKind::Text, name_value(&definition.text_box))));
        }
    }
}

// ---------------------------------------------------------------------------
// Policy input v5: every `Observation` field.

// ---------------------------------------------------------------------------
// Guard
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Buffer {
    Player,
    Card,
    Permanent,
    Relation,
    Action,
    Fact,
}

impl Buffer {
    /// V5 row width.
    fn width(self) -> usize {
        match self {
            Buffer::Player => PLAYER_DIM,
            Buffer::Card => CARD_DIM,
            Buffer::Permanent => PERMANENT_DIM,
            Buffer::Relation => RELATION_DIM_V5,
            Buffer::Action => managym::agent::observation_encoder::ACTION_DIM,
            Buffer::Fact => FACT_DIM,
        }
    }

    /// The row's validity column; player validity is a buffer of its own.
    fn validity(self) -> Option<usize> {
        match self {
            Buffer::Player => None,
            Buffer::Relation => Some(31),
            Buffer::Fact => Some(FACT_VALID_INDEX),
            _ => Some(self.width() - 1),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Encoded {
    /// Columns `start..start + len` of the named row buffer.
    Columns(Buffer, usize, usize),
    /// Object index + 1 of another object: a join, never a magnitude.
    Join(Buffer, usize),
    /// Fixed by the card definition, which its row's definition reference names.
    Definition,
    /// Which seat's buffer holds the row, or the agent-relative view itself.
    Side,
    /// The object-table key that focus and joins resolve to.
    Identity,
    /// Rows of their own: cards, permanents, relations, events, actions,
    /// action focus, known hand or definition references.
    Rows(&'static str),
    /// Fact rows of this kind, joined to the field's object.
    Fact(FactKind),
    /// Not a decision input, for the stated reason.
    Excluded(&'static str),
}

/// The field each fact kind states; no wildcard, so a new kind must name it.
fn fact_field(kind: FactKind) -> &'static str {
    match kind {
        FactKind::Counter => "PermanentData.counters",
        FactKind::AddedSubtype => "PermanentData.temp_subtypes",
        FactKind::SetName => "PermanentData.name",
        FactKind::SetCreatureType => "PermanentData.creature_types",
        FactKind::UsedAbility => "PermanentData.used_abilities",
        FactKind::Effect => "ContinuousEffectData.modification",
    }
}

fn column(buffer: Buffer, index: usize) -> Encoded {
    Encoded::Columns(buffer, index, 1)
}

/// `(field, encodings)` for every field of `$ty`. The pattern is exhaustive,
/// so a new field fails to compile until it is listed here.
macro_rules! encoding {
    ($ty:ident { $($field:ident => [$($encoded:expr),+ $(,)?]),* $(,)? }) => {{
        #[allow(dead_code)]
        fn exhaustive(value: &$ty) {
            let $ty { $($field: _),* } = value;
        }
        vec![$((concat!(stringify!($ty), ".", stringify!($field)), vec![$($encoded),+])),*]
    }};
}

const PERMISSION: usize = CARD_PERMISSION_INDEX;

const HIDDEN_SIDEBOARD: &str = "the agent's remaining copies are outside rows of \
    agent_cards and the ones in play are visible cards; the opponent's sideboard is \
    hidden information (scratch/questions.md)";
const HISTORY: &str = "zone and incarnation metadata belong to the public history rows \
    (policy_history_version 1); the v0 event tensor is frozen";

fn guard() -> Vec<(&'static str, Vec<Encoded>)> {
    use Buffer::*;
    use Encoded::*;
    let mut fields = encoding!(Observation {
        continuation => [Rows("relations")],
        game_over => [Excluded("no decision follows the end of the game")],
        won => [Excluded("no decision follows the end of the game")],
        turn => [Rows("player")],
        action_space => [Rows("actions")],
        agent => [Rows("agent_player")],
        agent_cards => [Rows("agent_cards")],
        agent_sideboard => [Rows("agent_cards")],
        agent_permanents => [Rows("agent_permanents")],
        opponent => [Rows("opponent_player")],
        opponent_cards => [Rows("opponent_cards")],
        opponent_permanents => [Rows("opponent_permanents")],
        stack_objects => [Rows("relations")],
        recent_events => [Rows("events")],
        policy_history => [Rows("events")],
    });
    fields.extend(encoding!(TurnData {
        turn_number => [column(Player, PLAYER_TURN_INDEX)],
        phase => [Columns(Player, 9, 5)],
        step => [Columns(Player, 14, 12)],
        active_player_id => [column(Player, 1)],
        agent_player_id => [Side],
    }));
    fields.extend(encoding!(PlayerData {
        player_index => [column(Player, PLAYER_SEAT_INDEX)],
        id => [Identity],
        is_agent => [Side],
        is_active => [column(Player, 1)],
        life => [column(Player, 0)],
        zone_counts => [Columns(Player, 2, 7)],
        graveyard_lessons => [column(Player, 26)],
        combat_mana => [column(Player, 27)],
        mana => [Columns(Player, 28, 6)],
        lesson_mana => [Columns(Player, 34, 6)],
        restricted_mana => [Rows("restricted mana below")],
        hand_only_until => [column(Player, PLAYER_HAND_ONLY_INDEX)],
        known_hand => [Rows("known_hand")],
        sideboard_counts => [Excluded(HIDDEN_SIDEBOARD)],
        remaining_sideboard_counts => [Excluded(HIDDEN_SIDEBOARD)],
        keeps_red_mana => [column(Player, PLAYER_KEEPS_RED_MANA_INDEX)],
        no_maximum_hand_size => [column(Player, PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX)],
        effects => [Rows("Effect fact rows, ContinuousEffectData below")],
    }));
    fields.extend(encoding!(CardData {
        zone => [Columns(Card, 0, 7)],
        owner_id => [column(Card, 7), Side],
        id => [Identity],
        registry_key => [Rows("semantic_cards")],
        name => [Definition],
        text_box => [Definition],
        power => [column(Card, 8)],
        toughness => [column(Card, 9)],
        is_token => [column(Card, 29)],
        is_legendary => [column(Card, 39)],
        is_ally => [column(Card, 30)],
        is_lesson => [column(Card, 31)],
        ward_cost => [Columns(Card, 32, 2)],
        kicker_cost => [Columns(Card, 34, 2)],
        card_types => [Rows("card types below")],
        keywords => [Rows("keywords below")],
        mana_cost => [column(Card, 10), Definition],
        // Face-down cards are foretold, and only their owner sees them.
        face_down => [column(Card, PERMISSION + 11)],
        cast_permission => [Rows("cast permission below")],
        x => [Columns(Card, CARD_X_INDEX, 2)],
    }));
    fields.extend(encoding!(CastPermissionData {
        player_id => [Columns(Card, PERMISSION, 2)],
        cost => [Columns(Card, PERMISSION + 2, 2)],
        this_turn => [column(Card, PERMISSION + 4)],
        own_turn_only => [column(Card, PERMISSION + 5)],
        after_turn => [column(Card, PERMISSION + 6)],
        source_permanent_id => [column(Card, PERMISSION + 7)],
        flash => [column(Card, PERMISSION + 8)],
        any_mana => [column(Card, PERMISSION + 9)],
        waterbend => [column(Card, PERMISSION + 10)],
        foretold => [column(Card, PERMISSION + 11)],
    }));
    fields.extend(encoding!(RestrictedManaData {
        spells => [Columns(Player, PLAYER_RESTRICTED_INDEX, 18)],
        abilities => [Columns(Player, PLAYER_RESTRICTED_INDEX, 18)],
        mana => [Columns(Player, PLAYER_RESTRICTED_INDEX, 18), Columns(Player, 34, 6)],
    }));
    fields.extend(encoding!(CardTypeData {
        is_castable => [Definition],
        is_permanent => [Definition],
        is_non_land_permanent => [Definition],
        is_non_creature_permanent => [Definition],
        is_spell => [Definition],
        is_creature => [column(Card, 12)],
        is_land => [column(Card, 11)],
        is_planeswalker => [column(Card, 15)],
        is_enchantment => [column(Card, 14)],
        is_artifact => [column(Card, 13)],
        is_kindred => [Definition],
        is_battle => [column(Card, 16)],
    }));
    fields.extend(encoding!(SideboardCardData {
        candidate_id => [Identity, column(Card, 37)],
        owner_id => [column(Card, 7)],
        registry_key => [Rows("semantic_cards")],
        name => [Definition],
        text_box => [Definition],
    }));
    fields.extend(encoding!(PermanentData {
        id => [Identity],
        registry_key => [Join(Permanent, 24)],
        controller_id => [column(Permanent, 0), Side],
        tapped => [column(Permanent, 1)],
        damage => [column(Permanent, 2)],
        is_summoning_sick => [column(Permanent, 3)],
        plus1_counters => [column(Permanent, 4), Rows("Counter fact rows")],
        cant_be_blocked_this_turn => [column(Permanent, 5)],
        power => [column(Permanent, 6)],
        toughness => [column(Permanent, 7)],
        is_animated => [column(Permanent, 8)],
        has_exile_link => [column(Permanent, 9)],
        exile_if_dies_this_turn => [column(Permanent, 23)],
        attacking => [column(Permanent, 27)],
        exhausted_abilities => [column(Permanent, PERMANENT_EXHAUSTED_INDEX)],
        once_each_turn_used => [column(Permanent, PERMANENT_ONCE_EACH_TURN_INDEX)],
        granted_firebending => [column(Permanent, PERMANENT_FIREBENDING_INDEX)],
        temp_subtypes => [
            column(Permanent, 26),
            Fact(FactKind::AddedSubtype),
            Columns(Fact, FACT_ID_INDEX, 2),
        ],
        keywords => [Rows("keywords below")],
        counters => [
            Fact(FactKind::Counter),
            Columns(Fact, FACT_ID_INDEX, 2),
            column(Fact, FACT_VALUE_INDEX),
        ],
        attached_to => [Join(Permanent, PERMANENT_ATTACHED_INDEX)],
        name => [Fact(FactKind::SetName)],
        creature_types => [Fact(FactKind::SetCreatureType)],
        is_granted_land => [column(Permanent, PERMANENT_GRANTED_LAND_INDEX)],
        blocking => [Join(Permanent, PERMANENT_BLOCKING_INDEX)],
        abilities_lost => [column(Permanent, PERMANENT_ABILITIES_LOST_INDEX)],
        used_abilities => [Fact(FactKind::UsedAbility)],
        effects => [Rows("Effect fact rows, ContinuousEffectData below")],
        timestamp => [column(Permanent, PERMANENT_TIMESTAMP_INDEX)],
    }));
    fields.extend(encoding!(ContinuousEffectData {
        modification => [Fact(FactKind::Effect), Columns(Fact, FACT_ID_INDEX, 2)],
        power => [column(Fact, FACT_VALUE_INDEX)],
        toughness => [column(Fact, FACT_VALUE_INDEX + 1)],
        amount => [column(Fact, FACT_VALUE_INDEX)],
        keywords => [Columns(Fact, FACT_DETAIL_INDEX, 2)],
        subtypes => [Columns(Fact, FACT_DETAIL_INDEX, 2)],
        color => [Columns(Fact, FACT_DETAIL_INDEX, 2)],
        duration => [column(Fact, FACT_DURATION_INDEX)],
        source_registry_key => [Join(Fact, FACT_SOURCE_INDEX)],
        timestamp => [column(Fact, FACT_ORDER_INDEX)],
    }));
    fields.extend(encoding!(StackObjectData {
        description => [Definition],
        stack_object_id => [Rows("relations")],
        kind => [column(Relation, 3)],
        controller_id => [column(Relation, 4)],
        source_card_registry_key => [Join(Relation, 5)],
        source_permanent_id => [Join(Relation, 6)],
        chosen_mode => [column(Relation, 7)],
        ability_index => [column(Relation, 8)],
        targets => [Rows("relations")],
        x => [column(Relation, RELATION_X_INDEX)],
        granted_keywords => [Columns(Relation, RELATION_KEYWORDS_INDEX, KEYWORD_COUNT)],
    }));
    fields.extend(encoding!(StackTargetData {
        kind => [column(Relation, 25), Join(Relation, 24)],
        player_id => [column(Relation, 25)],
        permanent_id => [Join(Relation, 24)],
        stack_object_id => [Join(Relation, 24)],
        card_id => [Join(Relation, 24)],
    }));
    fields.extend(encoding!(ContinuationFrame {
        kind => [column(Relation, 3)],
        decision => [column(Relation, 28)],
        parent => [column(Relation, 2)],
        controller => [column(Relation, 4)],
        source => [Join(Relation, 5), Join(Relation, 6)],
        mode => [column(Relation, 7)],
        ability => [column(Relation, 8)],
        kicked => [column(Relation, 9)],
        active_clause => [column(Relation, 10)],
        targets => [column(Relation, 0), column(Relation, 29)],
        context => [column(Relation, 0)],
        sacrifice => [column(Relation, 0)],
        cost => [column(Relation, 11), Columns(Relation, 12, 7)],
        hybrid => [Columns(Relation, 26, 2)],
        lesson => [column(Relation, 19)],
        waterbend => [column(Relation, 20)],
        can_decline => [column(Relation, 21)],
        mana_outputs => [column(Relation, 7), Columns(Relation, 12, 6), column(Relation, 19)],
        resolutions_this_turn => [column(Relation, 30)],
        x => [column(Relation, RELATION_X_INDEX)],
        granted_keywords => [Columns(Relation, RELATION_KEYWORDS_INDEX, KEYWORD_COUNT)],
    }));
    fields.extend(encoding!(ClauseTarget {
        reference => [Rows("VisibleReference below")],
        clause => [column(Relation, 22)],
        legal_at_resolution_start => [column(Relation, 32)],
    }));
    fields.extend(encoding!(VisibleReference {
        object => [Join(Relation, 24)],
        definition => [Join(Relation, 23)],
        player => [column(Relation, 25)],
    }));
    fields.extend(encoding!(ManaOutput {
        mode => [column(Relation, 7)],
        mana => [Columns(Relation, 12, 6)],
        lesson => [Columns(Relation, 12, 6), column(Relation, 19)],
    }));
    fields.extend(encoding!(ActionSpaceData {
        action_space_type => [Excluded("implied by the offered action types")],
        actions => [Rows("actions")],
        focus => [Rows("action_focus")],
    }));
    fields.extend(encoding!(ActionOption {
        description => [Excluded("text for the selectors encoded beside it")],
        action_type => [Columns(Action, 0, 17)],
        focus => [Rows("action_focus")],
        declared => [column(Action, 17)],
        mode_index => [column(Action, 18)],
        ability_index => [column(Action, 19)],
        continuation_index => [column(Action, 20)],
    }));
    fields.extend(encoding!(EventData {
        event_type => [Rows("events")],
        source_kind => [Rows("events")],
        source_id => [Rows("events")],
        target_kind => [Rows("events")],
        target_id => [Rows("events")],
        amount => [Rows("events")],
        controller_id => [Rows("events")],
        from_zone => [Excluded(HISTORY)],
        to_zone => [Excluded(HISTORY)],
        source_incarnation => [Excluded(HISTORY)],
        target_incarnation => [Excluded(HISTORY)],
    }));
    fields
}

/// Structural fact columns: the object join and the row kind.
const FACT_STRUCTURE: [usize; 2] = [FACT_OBJECT_INDEX, FACT_KIND_INDEX];

/// Structural relation columns: row kind, frame index and reference ordinal.
const RELATION_STRUCTURE: [usize; 3] = [0, 1, 29];

#[test]
fn every_observed_field_names_its_v5_encoding_and_every_v5_column_is_claimed() {
    let mut claimed: BTreeSet<(Buffer, usize)> = BTreeSet::new();
    let mut fact_kinds: BTreeSet<u8> = BTreeSet::new();
    for (field, encodings) in guard() {
        for encoded in encodings {
            let (buffer, start, len) = match encoded {
                Encoded::Fact(kind) => {
                    assert_eq!(fact_field(kind), field, "{kind:?} states another field");
                    fact_kinds.insert(kind as u8);
                    continue;
                }
                Encoded::Columns(buffer, start, len) => (buffer, start, len),
                Encoded::Join(buffer, index) => (buffer, index, 1),
                Encoded::Rows(reason) | Encoded::Excluded(reason) => {
                    assert!(!reason.is_empty(), "{field}: name its rows or reason");
                    continue;
                }
                Encoded::Definition | Encoded::Side | Encoded::Identity => continue,
            };
            assert!(len > 0, "{field}: empty column run");
            for index in start..start + len {
                assert!(
                    index < buffer.width() && Some(index) != buffer.validity(),
                    "{field}: {buffer:?} column {index} is not a v5 feature column"
                );
                claimed.insert((buffer, index));
            }
        }
    }
    for (keyword, _, card, permanent) in keyword_columns(Keywords::default()) {
        for (buffer, index) in [(Buffer::Card, card), (Buffer::Permanent, permanent)] {
            assert!(
                index < buffer.width() && Some(index) != buffer.validity(),
                "{keyword}: {buffer:?} {index}"
            );
            claimed.insert((buffer, index));
        }
    }
    for offset in 0..KEYWORD_COUNT {
        claimed.insert((Buffer::Relation, RELATION_KEYWORDS_INDEX + offset));
    }
    assert_eq!(
        fact_kinds,
        FactKind::ALL.iter().map(|kind| *kind as u8).collect(),
        "every fact kind is claimed by the field it states"
    );
    for index in FACT_STRUCTURE {
        claimed.insert((Buffer::Fact, index));
    }
    for index in RELATION_STRUCTURE {
        claimed.insert((Buffer::Relation, index));
    }
    for buffer in [
        Buffer::Player,
        Buffer::Card,
        Buffer::Permanent,
        Buffer::Relation,
        Buffer::Action,
        Buffer::Fact,
    ] {
        for index in (0..buffer.width()).filter(|i| Some(*i) != buffer.validity()) {
            assert!(
                claimed.contains(&(buffer, index)),
                "{buffer:?} column {index} encodes no listed field"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Directed states
// ---------------------------------------------------------------------------

fn config(version: u8) -> ObservationEncoderConfig {
    ObservationEncoderConfig {
        policy_input_version: version,
        ..ObservationEncoderConfig::default()
    }
}

fn encoded(s: &Scenario, viewer: usize, version: u8) -> EncodedObservation {
    let observation = Observation::for_player(s.game(), PlayerId(viewer));
    encode(&observation, &config(version)).expect("encodes")
}

/// The flattened policy input; v4 names every buffer this test compares.
fn flat(encoded: &EncodedObservation) -> Vec<f32> {
    [
        &encoded.relations,
        &encoded.semantic_cards,
        &encoded.known_hand,
        &encoded.agent_player,
        &encoded.opponent_player,
        &encoded.agent_cards,
        &encoded.opponent_cards,
        &encoded.agent_permanents,
        &encoded.opponent_permanents,
        &encoded.actions,
        &encoded.events,
    ]
    .into_iter()
    .flatten()
    .copied()
    .chain(encoded.action_focus.iter().map(|v| *v as f32))
    .collect()
}

/// `(side, row)` of a permanent in `viewer`'s observation; side 0 is the agent.
fn permanent_row_of(s: &Scenario, viewer: usize, id: PermanentId) -> (usize, usize) {
    let observed = s.game().state.permanents[id].as_ref().unwrap().id.0 as i32;
    let observation = Observation::for_player(s.game(), PlayerId(viewer));
    let found = [
        &observation.agent_permanents,
        &observation.opponent_permanents,
    ]
    .into_iter()
    .enumerate()
    .find_map(|(side, rows)| {
        rows.iter()
            .position(|p| p.id == observed)
            .map(|row| (side, row))
    })
    .expect("permanent observed");
    found
}

fn permanent_value(s: &Scenario, viewer: usize, id: PermanentId, column: usize) -> f32 {
    let (side, row) = permanent_row_of(s, viewer, id);
    let encoded = encoded(s, viewer, 5);
    let rows = if side == 0 {
        &encoded.agent_permanents
    } else {
        &encoded.opponent_permanents
    };
    rows[row * PERMANENT_DIM + column]
}

/// Object index of a permanent row, as action focus and joins use it.
fn object_index(s: &Scenario, viewer: usize, id: PermanentId) -> usize {
    let defaults = ObservationEncoderConfig::default();
    let (side, row) = permanent_row_of(s, viewer, id);
    2 + 2 * defaults.max_cards_per_player + side * defaults.max_permanents_per_player + row
}

/// Two states v4 cannot tell apart; v5 differs exactly at `column` of `id`.
fn assert_v5_alone_sees(
    baseline: &Scenario,
    changed: &Scenario,
    viewer: usize,
    id: PermanentId,
    column: usize,
) {
    assert_v5_sees(baseline, changed, viewer, id, column, 1);
}

/// As [`assert_v5_alone_sees`], when the change also shows in other v5 columns.
fn assert_v5_sees(
    baseline: &Scenario,
    changed: &Scenario,
    viewer: usize,
    id: PermanentId,
    column: usize,
    v5_differences: usize,
) {
    assert_eq!(
        flat(&encoded(baseline, viewer, 4)),
        flat(&encoded(changed, viewer, 4)),
        "v4 tells the states apart"
    );
    let (before, after) = (
        permanent_value(baseline, viewer, id, column),
        permanent_value(changed, viewer, id, column),
    );
    assert_ne!(before, after, "v5 column {column} misses the change");
    let differing = flat(&encoded(baseline, viewer, 5))
        .iter()
        .zip(flat(&encoded(changed, viewer, 5)))
        .filter(|(a, b)| **a != *b)
        .count();
    assert_eq!(differing, v5_differences, "v5 columns that differ");
}

/// Record a continuous effect on `affected`, as a resolving spell would.
fn apply(s: &mut Scenario, affected: Affected, modification: Modification) {
    s.game_mut().apply_continuous(
        None,
        PlayerId(0),
        [affected],
        modification,
        EffectDuration::EndOfTurn,
    );
}

fn on_permanent(s: &Scenario, id: PermanentId) -> Affected {
    s.game().affected_permanent(id).expect("on the battlefield")
}

/// `viewer`'s v5 fact rows about the object at `object_index`.
fn facts_about(s: &Scenario, viewer: usize, object_index: usize) -> Vec<Vec<f32>> {
    encoded(s, viewer, 5)
        .facts
        .chunks_exact(FACT_DIM)
        .filter(|row| row[FACT_VALID_INDEX] == 1.0 && row[FACT_OBJECT_INDEX] == (object_index + 1) as f32)
        .map(<[f32]>::to_vec)
        .collect()
}

/// A vocabulary id in fact-row form (16-bit halves).
fn id_columns(id: i32) -> [f32; 2] {
    let id = id as u32;
    [(id >> 16) as f32, (id & 0xffff) as f32]
}

/// The fact row of `kind` naming `id` among `rows`.
fn fact_row(rows: &[Vec<f32>], kind: FactKind, id: i32) -> Vec<f32> {
    rows.iter()
        .find(|row| {
            row[FACT_KIND_INDEX] == kind as u8 as f32
                && row[FACT_ID_INDEX..FACT_ID_INDEX + 2] == id_columns(id)
        })
        .unwrap_or_else(|| panic!("no {kind:?} fact {id}: {rows:?}"))
        .clone()
}

/// Two states only v5's fact rows tell apart: v4 and every other v5 buffer
/// are equal.
fn assert_facts_alone_see(baseline: &Scenario, changed: &Scenario, viewer: usize) {
    assert_eq!(
        flat(&encoded(baseline, viewer, 4)),
        flat(&encoded(changed, viewer, 4)),
        "v4 tells the states apart"
    );
    assert_eq!(
        flat(&encoded(baseline, viewer, 5)),
        flat(&encoded(changed, viewer, 5)),
        "a v5 column tells the states apart"
    );
    assert_ne!(
        encoded(baseline, viewer, 5).facts,
        encoded(changed, viewer, 5).facts,
        "the fact rows miss the change"
    );
}

/// Gray Ogres for both players, tapped, in player 0's main phase.
fn ogre_state() -> (Scenario, PermanentId) {
    let mut s = five_color_unfunded_setup(&["Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[ogre].as_mut().unwrap().tapped = true;
    (s, ogre)
}

#[test]
fn each_counter_kind_reaches_a_fact_row() {
    for kind in CounterKind::ALL {
        let (mut baseline, ogre) = ogre_state();
        let (mut changed, _) = ogre_state();
        if kind == CounterKind::DoubleStrike {
            // CR 122.1b: the counter grants double strike, which v4 sees; a
            // granted keyword is the state v4 cannot tell apart.
            let grant = Modification::AddKeywords(Keywords {
                double_strike: true,
                ..Default::default()
            });
            for s in [&mut baseline, &mut changed] {
                let affected = on_permanent(s, ogre);
                apply(s, affected, grant.clone());
            }
        }
        changed.game_mut().state.permanents[ogre]
            .as_mut()
            .unwrap()
            .change_counters(kind, 1);
        let index = object_index(&changed, 0, ogre);
        let id = name_value(&format!("CounterKind::{}", kind.name()));
        let row = fact_row(&facts_about(&changed, 0, index), FactKind::Counter, id);
        assert_eq!(row[FACT_VALUE_INDEX], 0.1, "{kind:?}");
        if kind == CounterKind::PlusOnePlusOne {
            continue; // v4 sees +1/+1 counters and the power they add.
        }
        assert_facts_alone_see(&baseline, &changed, 0);
    }
}

#[test]
fn a_stunned_tapped_creature_differs_from_an_unstunned_one_only_in_v5() {
    let (baseline, ogre) = ogre_state();
    let (mut stunned, _) = ogre_state();
    stunned.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .change_counters(CounterKind::Stun, 1);
    assert_facts_alone_see(&baseline, &stunned, 0);
}

fn equipped(on_first: bool) -> (Scenario, PermanentId, [PermanentId; 2]) {
    let mut s = five_color_unfunded_setup(&["Trusty Boomerang", "Gray Ogre"]);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let boomerang = s.force_permanent_on_battlefield(0, "Trusty Boomerang");
    let host = s
        .game()
        .permanent_object_ref(if on_first { first } else { second });
    s.game_mut().state.permanents[boomerang]
        .as_mut()
        .unwrap()
        .attached_to = host;
    (s, boomerang, [first, second])
}

#[test]
fn equipment_on_one_creature_differs_from_equipment_on_another_only_in_v5() {
    let (on_first, boomerang, [first, second]) = equipped(true);
    let (on_second, ..) = equipped(false);
    assert_eq!(
        permanent_value(&on_first, 0, boomerang, PERMANENT_ATTACHED_INDEX),
        (object_index(&on_first, 0, first) + 1) as f32
    );
    assert_eq!(
        permanent_value(&on_second, 0, boomerang, PERMANENT_ATTACHED_INDEX),
        (object_index(&on_second, 0, second) + 1) as f32
    );
    assert_v5_alone_sees(
        &on_first,
        &on_second,
        0,
        boomerang,
        PERMANENT_ATTACHED_INDEX,
    );
}

fn united_front_with_x(x: usize) -> Scenario {
    let mut s = five_color_setup(&["United Front", "Gray Ogre"]);
    s.set_player_mana_pool(0, &format!("{x}WW"));
    cast(&mut s, "United Front");
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    choose_mode(&mut s, x);
    s
}

#[test]
fn stack_x_one_and_three_differ_only_in_v5() {
    let one = united_front_with_x(1);
    let three = united_front_with_x(3);
    assert_eq!(flat(&encoded(&one, 0, 4)), flat(&encoded(&three, 0, 4)));
    let x = |s: &Scenario| encoded(s, 0, 5).relations[RELATION_X_INDEX];
    assert_eq!((x(&one), x(&three)), (1.0, 3.0), "the stack frame is row 0");
    // The opponent sees the announced X of a spell on the stack too.
    let opponent_x = |s: &Scenario| encoded(s, 1, 5).relations[RELATION_X_INDEX];
    assert_eq!((opponent_x(&one), opponent_x(&three)), (1.0, 3.0));
}

fn toph_and_tile(toph_has_abilities: bool) -> (Scenario, PermanentId) {
    let mut s = five_color_unfunded_setup(&["Toph, the First Metalbender", "White Lotus Tile"]);
    let toph = s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    if !toph_has_abilities {
        let affected = on_permanent(&s, toph);
        apply(&mut s, affected, Modification::LoseAllAbilities);
    }
    (s, tile)
}

#[test]
fn a_toph_granted_land_differs_from_an_ungranted_one_in_v5() {
    let (granted, tile) = toph_and_tile(true);
    let (plain, _) = toph_and_tile(false);
    assert_eq!(
        permanent_value(&granted, 0, tile, PERMANENT_GRANTED_LAND_INDEX),
        1.0
    );
    // Toph's own lost abilities are the other difference.
    assert_v5_sees(&plain, &granted, 0, tile, PERMANENT_GRANTED_LAND_INDEX, 2);
}

#[test]
fn granted_firebending_reaches_its_permanent_column_and_an_effect_row() {
    let (baseline, ogre) = ogre_state();
    let (mut changed, _) = ogre_state();
    let affected = on_permanent(&changed, ogre);
    apply(&mut changed, affected, Modification::Firebending(4));
    assert_eq!(
        permanent_value(&changed, 0, ogre, PERMANENT_FIREBENDING_INDEX),
        0.4
    );
    let row = fact_row(
        &facts_about(&changed, 0, object_index(&changed, 0, ogre)),
        FactKind::Effect,
        name_value("Modification::Firebending"),
    );
    assert_eq!(row[FACT_VALUE_INDEX], 0.4);
    assert_v5_alone_sees(&baseline, &changed, 0, ogre, PERMANENT_FIREBENDING_INDEX);
}

#[test]
fn each_added_subtype_reaches_a_fact_row() {
    for subtype in ["Plains", "Construct", "Citizen"] {
        let (baseline, ogre) = ogre_state();
        let (mut changed, _) = ogre_state();
        let affected = on_permanent(&changed, ogre);
        apply(
            &mut changed,
            affected,
            Modification::AddSubtypes(vec![subtype.into()]),
        );
        let rows = facts_about(&changed, 0, object_index(&changed, 0, ogre));
        fact_row(&rows, FactKind::AddedSubtype, name_value(subtype));
        let effect = fact_row(&rows, FactKind::Effect, name_value("Modification::AddSubtypes"));
        assert_eq!(
            effect[FACT_DETAIL_INDEX..FACT_DETAIL_INDEX + 2],
            id_columns(name_value(subtype))
        );
        assert_facts_alone_see(&baseline, &changed, 0);
    }
}

#[test]
fn honest_work_marks_set_characteristics_and_joins_the_aura_to_its_host() {
    let mut s = five_color_setup(&["Honest Work", "Wind Drake"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    let set_facts = |s: &Scenario| {
        facts_about(s, 0, object_index(s, 0, drake))
            .into_iter()
            .filter(|row| {
                [FactKind::SetName, FactKind::SetCreatureType]
                    .map(|kind| kind as u8 as f32)
                    .contains(&row[FACT_KIND_INDEX])
            })
            .count()
    };
    assert_eq!(set_facts(&s), 0);
    s.set_player_mana_pool(0, "U");
    cast(&mut s, "Honest Work");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(managym::state::game_object::Target::Permanent(drake)));
    settle(&mut s);
    let work = s.battlefield_permanents_named(0, "Honest Work")[0];
    let rows = facts_about(&s, 0, object_index(&s, 0, drake));
    fact_row(&rows, FactKind::SetName, name_value("Humble Merchant"));
    fact_row(&rows, FactKind::SetCreatureType, name_value("Citizen"));
    assert_eq!(set_facts(&s), 2, "the name and its creature type");
    assert_eq!(
        permanent_value(&s, 0, work, PERMANENT_ATTACHED_INDEX),
        (object_index(&s, 0, drake) + 1) as f32
    );
}

/// Card and permanent keyword columns are checked by
/// `printed_and_granted_keywords_set_their_card_and_permanent_columns_and_tokens`;
/// this checks the v5 relation block for keywords granted to a stack object.
#[test]
fn every_keyword_reaches_its_relation_column() {
    let mut s = five_color_unfunded_setup(&["Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let observation = Observation::for_player(s.game(), PlayerId(0));
    for position in 0..KEYWORD_COUNT {
        let mut flags = [false; KEYWORD_COUNT];
        flags[position] = true;
        let mut changed = observation.clone();
        let mut frame = ContinuationFrame {
            kind: 1,
            granted_keywords: flags,
            ..ContinuationFrame::default()
        };
        frame.controller = 0;
        changed.continuation = vec![frame];
        let encoded = encode(&changed, &config(5)).unwrap();
        let relation = &encoded.relations[RELATION_KEYWORDS_INDEX..][..KEYWORD_COUNT];
        assert_eq!(relation.iter().position(|v| *v == 1.0), Some(position));
        assert_eq!(relation.iter().sum::<f32>(), 1.0, "{position}");
    }
}

#[test]
fn lo_and_li_grant_lifelink_to_a_lesson_on_the_stack_in_v5() {
    let lifelink = RELATION_KEYWORDS_INDEX + 9;
    let lesson = |with_lo_and_li: bool| {
        let mut s = five_color_unfunded_setup(&[
            "Lo and Li, Twin Tutors",
            "Firebending Lesson",
            "Gray Ogre",
        ]);
        if with_lo_and_li {
            s.force_permanent_on_battlefield(0, "Lo and Li, Twin Tutors");
        }
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.set_player_mana_pool(0, "R");
        cast(&mut s, "Firebending Lesson");
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(managym::state::game_object::Target::Permanent(ogre)));
        encoded(&s, 1, 5).relations
    };
    let (granted, plain) = (lesson(true), lesson(false));
    assert_eq!(granted[3], 1.0, "the top frame is the spell");
    assert_eq!((granted[lifelink], plain[lifelink]), (1.0, 0.0));
}

#[test]
fn turn_number_and_seat_reach_the_player_rows() {
    let mut s = five_color_unfunded_setup(&[]);
    let turn = |s: &Scenario, viewer: usize| {
        let encoded = encoded(s, viewer, 5);
        (
            encoded.agent_player[PLAYER_TURN_INDEX],
            encoded.agent_player[PLAYER_SEAT_INDEX],
            encoded.opponent_player[PLAYER_SEAT_INDEX],
        )
    };
    let first = s.game().state.turn.turn_number;
    assert_eq!(turn(&s, 0), (first as f32 / 20.0, 0.0, 1.0));
    assert_eq!(turn(&s, 1).1, 1.0, "seat 1 is on the draw");
    s.advance_to_active_step(1, StepKind::Main);
    assert_ne!(turn(&s, 0).0, first as f32 / 20.0, "a new turn");
}

// ---------------------------------------------------------------------------
// Viewer safety
// ---------------------------------------------------------------------------

/// Every definition reference anywhere in `viewer`'s v5 encoding.
fn definitions_seen(s: &Scenario, viewer: usize) -> BTreeSet<u32> {
    let encoded = encoded(s, viewer, 5);
    let mut seen = BTreeSet::new();
    let mut add = |value: f32| {
        if value > 0.0 {
            seen.insert(value as u32 - 1);
        }
    };
    encoded.semantic_cards.iter().copied().for_each(&mut add);
    encoded
        .known_hand
        .iter()
        .step_by(2)
        .copied()
        .for_each(&mut add);
    for rows in [&encoded.agent_permanents, &encoded.opponent_permanents] {
        rows.as_chunks::<PERMANENT_DIM>()
            .0
            .iter()
            .for_each(|row| add(row[24]));
    }
    encoded
        .relations
        .as_chunks::<RELATION_DIM_V5>()
        .0
        .iter()
        .for_each(|row| {
            add(row[5]);
            add(row[23]);
        });
    seen
}

#[test]
fn foretold_and_hand_cards_never_reach_the_opponents_encoding() {
    let mut s = five_color_setup(&["Sozin's Comet", "United Front"]);
    let comet = library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    let front = move_to_hand(&mut s, "United Front");
    s.set_player_mana_pool(0, "3");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { card, .. } if *card == comet),
    );
    assert_eq!(s.game().state.zones.zone_of(comet), Some(ZoneType::Exile));
    let hidden = [comet, front].map(|card| s.game().state.cards[card].definition_id.0);

    let owner = encoded(&s, 0, 5);
    let face_down = owner
        .agent_cards
        .as_chunks::<CARD_DIM>()
        .0
        .iter()
        .filter(|row| row[PERMISSION + 11] == 1.0)
        .count();
    assert_eq!(face_down, 1, "its owner sees the foretold card");
    for definition in hidden {
        assert!(definitions_seen(&s, 0).contains(&definition));
        assert!(
            !definitions_seen(&s, 1).contains(&definition),
            "definition {definition} leaks to the opponent"
        );
    }
    let opponent = encoded(&s, 1, 5);
    assert!(opponent
        .opponent_cards
        .as_chunks::<CARD_DIM>()
        .0
        .iter()
        .all(|row| row[PERMISSION + 11] == 0.0));
}

// ---------------------------------------------------------------------------
// Combat, lost abilities, used activations and player state
// ---------------------------------------------------------------------------

/// Two ogres for player 1 and one for player 0, in player 0's main phase.
fn combat_state() -> (Scenario, PermanentId, [PermanentId; 2]) {
    let mut s = five_color_unfunded_setup(&["Gray Ogre"]);
    let blocker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let first = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(1, "Gray Ogre");
    (s, blocker, [first, second])
}

fn declare(
    s: &mut Scenario,
    attackers: &[PermanentId],
    blocks: Option<(PermanentId, PermanentId)>,
) {
    for attacker in attackers {
        s.game_mut().state.permanents[*attacker]
            .as_mut()
            .unwrap()
            .attacking = true;
    }
    s.game_mut().state.combat = Some(managym::flow::combat::CombatState {
        attackers: attackers.to_vec(),
        attacker_to_blockers: blocks
            .map(|(attacker, blocker)| (attacker, vec![blocker]))
            .into_iter()
            .collect(),
        ..Default::default()
    });
}

/// Attacking has been a permanent column since v4 (column 27).
#[test]
fn attacking_creatures_differ_from_idle_ones_at_column_27() {
    let (idle, _, [first, _]) = combat_state();
    let (mut attacking, ..) = combat_state();
    declare(&mut attacking, &[first], None);
    assert_eq!(permanent_value(&idle, 0, first, 27), 0.0);
    assert_eq!(permanent_value(&attacking, 0, first, 27), 1.0);
}

#[test]
fn blocking_one_attacker_differs_from_blocking_another_only_in_v5() {
    let (mut on_first, blocker, [first, second]) = combat_state();
    let (mut on_second, ..) = combat_state();
    declare(&mut on_first, &[first, second], Some((first, blocker)));
    declare(&mut on_second, &[first, second], Some((second, blocker)));
    assert_eq!(
        permanent_value(&on_first, 0, blocker, PERMANENT_BLOCKING_INDEX),
        (object_index(&on_first, 0, first) + 1) as f32
    );
    assert_v5_alone_sees(&on_first, &on_second, 0, blocker, PERMANENT_BLOCKING_INDEX);
}

#[test]
fn a_creature_that_lost_its_abilities_differs_only_in_v5() {
    let (baseline, ogre) = ogre_state();
    let (mut changed, _) = ogre_state();
    let affected = on_permanent(&changed, ogre);
    apply(&mut changed, affected, Modification::LoseAllAbilities);
    assert_v5_alone_sees(&baseline, &changed, 0, ogre, PERMANENT_ABILITIES_LOST_INDEX);
}

fn mai(exhausted: bool) -> (Scenario, PermanentId) {
    let mut s = five_color_unfunded_setup(&["Mai, Jaded Edge"]);
    let mai = s.force_permanent_on_battlefield(1, "Mai, Jaded Edge");
    if exhausted {
        s.game_mut().state.permanents[mai]
            .as_mut()
            .unwrap()
            .exhausted
            .push(0);
    }
    (s, mai)
}

#[test]
fn a_spent_exhaust_ability_differs_from_an_unused_one_only_in_v5() {
    let (unused, mai_id) = mai(false);
    let (spent, _) = mai(true);
    let rows = facts_about(&spent, 0, object_index(&spent, 0, mai_id));
    // Mai's only activated ability is exhaust: index 0.
    fact_row(&rows, FactKind::UsedAbility, 0);
    assert_eq!(
        permanent_value(&spent, 0, mai_id, PERMANENT_EXHAUSTED_INDEX),
        1.0,
        "the exhausted count moves with it"
    );
    assert_v5_alone_sees(&unused, &spent, 0, mai_id, PERMANENT_EXHAUSTED_INDEX);
}

#[test]
fn kept_red_mana_and_no_hand_size_reach_the_player_rows_only_in_v5() {
    for column in [
        PLAYER_KEEPS_RED_MANA_INDEX,
        PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
    ] {
        let (baseline, _) = ogre_state();
        let (mut changed, _) = ogre_state();
        let modification = if column == PLAYER_KEEPS_RED_MANA_INDEX {
            Modification::KeepUnspentMana(Color::Red)
        } else {
            Modification::NoMaximumHandSize
        };
        apply(&mut changed, Affected::Player(PlayerId(1)), modification);
        assert_eq!(
            flat(&encoded(&baseline, 0, 4)),
            flat(&encoded(&changed, 0, 4))
        );
        let (before, after) = (encoded(&baseline, 0, 5), encoded(&changed, 0, 5));
        assert_eq!(before.agent_player, after.agent_player);
        assert_eq!(
            (
                before.opponent_player[column],
                after.opponent_player[column]
            ),
            (0.0, 1.0)
        );
    }
}

// ---------------------------------------------------------------------------
// Cast permissions, restricted mana and announced X
// ---------------------------------------------------------------------------

fn encoded_default(observation: &Observation) -> EncodedObservation {
    encode(observation, &ObservationEncoderConfig::default()).expect("v5 encodes")
}

fn object_id(s: &Scenario, card: CardId) -> i32 {
    s.game().state.cards[card].id.0 as i32
}

/// The encoded row of `card` on either side of `observation`, if listed.
fn listed_card_row(observation: &Observation, card: i32) -> Option<Vec<f32>> {
    let encoded = encoded_default(observation);
    for (cards, rows) in [
        (&observation.agent_cards, &encoded.agent_cards),
        (&observation.opponent_cards, &encoded.opponent_cards),
    ] {
        if let Some(slot) = cards.iter().position(|c| c.id == card) {
            return Some(rows[slot * CARD_DIM..(slot + 1) * CARD_DIM].to_vec());
        }
    }
    None
}

fn permission_columns(row: &[f32]) -> &[f32] {
    &row[PERMISSION..PERMISSION + 12]
}

/// Airbend's permission is public: both players see who may cast the exiled
/// card and for how much, with "viewer may cast" set only for its owner.
#[test]
fn airbend_permission_reaches_both_viewers_tensors() {
    let mut s = five_color_setup(&["Airbending Lesson", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    let id = object_id(&s, ogre_card);
    cast(&mut s, "Airbending Lesson");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);

    let airbender = Observation::for_player(s.game(), PlayerId(0));
    let listed = airbender
        .opponent_cards
        .iter()
        .find(|c| c.id == id)
        .expect("the exiled Ogre is public");
    let permission = listed.cast_permission.as_ref().expect("a permission");
    assert_eq!(permission.player_id, airbender.opponent.id);
    assert_eq!(permission.cost.as_ref().map(|c| c.mana_value), Some(2));
    assert!(airbender.to_json().contains("\"cast_permission\""));

    let row = listed_card_row(&airbender, id).unwrap();
    assert_eq!(
        permission_columns(&row),
        &[1.0, 0.0, 1.0, 0.2, 0., 0., 0., 0., 0., 0., 0., 0.]
    );
    let owner = listed_card_row(&Observation::for_player(s.game(), PlayerId(1)), id).unwrap();
    assert_eq!(owner[PERMISSION + 1], 1.0, "the owner may cast it");
}

/// Avatar's Wrath's lock is on the restricted player's row for both viewers,
/// and the permissions it overrides stay visible.
#[test]
fn hand_only_lock_reaches_both_viewers_tensors() {
    let mut s = five_color_setup(&["Avatar's Wrath", "Gray Ogre", "Craw Wurm"]);
    let spared = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let wurm_id = object_id(&s, card_of(&s, wurm));
    let baseline = encoded_default(&Observation::for_player(s.game(), PlayerId(0)));
    assert_eq!(baseline.opponent_player[PLAYER_HAND_ONLY_INDEX], 0.0);
    cast(&mut s, "Avatar's Wrath");
    assert!(s.choose_target(Target::Permanent(spared)));
    settle(&mut s);

    let caster = Observation::for_player(s.game(), PlayerId(0));
    assert_eq!(caster.opponent.hand_only_until, Some(caster.agent.id));
    assert_eq!(
        encoded_default(&caster).opponent_player[PLAYER_HAND_ONLY_INDEX],
        1.0
    );
    assert_eq!(
        encoded_default(&caster).agent_player[PLAYER_HAND_ONLY_INDEX],
        0.0
    );
    let locked = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(
        encoded_default(&locked).agent_player[PLAYER_HAND_ONLY_INDEX],
        1.0
    );
    assert_eq!(listed_card_row(&locked, wurm_id).unwrap()[PERMISSION], 1.0);
}

/// Restricted mana is split by what it may pay for: Jasmine Dragon Tea
/// Shop's green counts under Ally only.
#[test]
fn restricted_mana_is_encoded_by_spending_subtype() {
    let mut s = five_color_setup(&["Jasmine Dragon Tea Shop"]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    s.set_player_mana_pool(0, "");
    activate(&mut s, shop, 0);
    choose_mode(&mut s, 4);

    let observation = Observation::for_player(s.game(), PlayerId(0));
    let classes = &observation.agent.restricted_mana;
    assert_eq!(classes.len(), 1);
    assert_eq!(classes[0].spells, ["Ally"]);
    assert_eq!(classes[0].abilities, ["Ally"]);
    assert_eq!(classes[0].mana, [0, 0, 0, 0, 1, 0]);
    let player = encoded_default(&observation).agent_player;
    let by_subtype = &player[PLAYER_RESTRICTED_INDEX..PLAYER_HAND_ONLY_INDEX];
    let mut expected = [0.0; 18];
    expected[12 + 4] = 0.1; // Ally, green
    assert_eq!(by_subtype, expected);
}

/// A spell on the stack shows its announced X.
#[test]
fn announced_x_reaches_the_stack_card_row() {
    let mut s = five_color_setup(&["United Front"]);
    s.set_player_mana_pool(0, "3WW");
    let front = cast(&mut s, "United Front");
    choose_mode(&mut s, 3);
    let id = object_id(&s, front);
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let card = observation.agent_cards.iter().find(|c| c.id == id).unwrap();
    assert_eq!(card.x, Some(3));
    let row = listed_card_row(&observation, id).unwrap();
    assert_eq!(&row[CARD_X_INDEX..CARD_X_INDEX + 2], &[1.0, 0.3]);
}

// --- Viewer safety -------------------------------------------------------

/// A foretold card and its permission reach only its owner: the owner's
/// tensor marks it foretold and not yet castable; the opponent's tensor and
/// JSON hold no trace of it beyond the exile count.
#[test]
fn foretold_card_and_permission_reach_only_the_owner() {
    let mut s = five_color_setup(&["Sozin's Comet"]);
    let comet = library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    s.set_player_mana_pool(0, "2");
    let opponent_before = encoded_default(&Observation::for_player(s.game(), PlayerId(1)));
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { card, .. } if *card == comet),
    );
    let id = object_id(&s, comet);

    let owner = Observation::for_player(s.game(), PlayerId(0));
    let row = listed_card_row(&owner, id).expect("the owner sees it");
    assert_eq!(row[PERMISSION], 1.0);
    assert_eq!(row[PERMISSION + 6], 1.0, "castable only on a later turn");
    assert_eq!(row[PERMISSION + 11], 1.0, "foretold");

    let opponent = Observation::for_player(s.game(), PlayerId(1));
    assert!(listed_card_row(&opponent, id).is_none());
    let json = opponent.to_json();
    assert!(!json.contains("Sozin") && !json.contains("foretold") && !json.contains("face_down"));
    let after = encoded_default(&opponent);
    for rows in [&after.agent_cards, &after.opponent_cards] {
        assert!(rows
            .chunks(CARD_DIM)
            .all(|row| row[PERMISSION + 11] == 0.0 && row[PERMISSION] == 0.0));
    }
    assert_eq!(after.opponent_cards, opponent_before.opponent_cards);

    // On its owner's next turn the later-turn term clears.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let row = listed_card_row(&Observation::for_player(s.game(), PlayerId(0)), id).unwrap();
    assert_eq!(row[PERMISSION + 6], 0.0);
}

// ---------------------------------------------------------------------------
// Continuous-effect records
// ---------------------------------------------------------------------------

#[test]
fn each_modification_reaches_an_effect_fact_row_with_its_duration_and_order() {
    for (on_player, modification) in modification_witnesses() {
        let (mut s, ogre) = ogre_state();
        let affected = if on_player {
            Affected::Player(PlayerId(1))
        } else {
            on_permanent(&s, ogre)
        };
        let index = if on_player {
            1 // the opponent's player row, from player 0's view
        } else {
            object_index(&s, 0, ogre)
        };
        let record = resolved(&modification);
        s.game_mut().apply_continuous(
            None,
            PlayerId(0),
            [affected],
            record.clone(),
            EffectDuration::Indefinite,
        );
        let (name, _) = modification_seen(&modification);
        let row = fact_row(&facts_about(&s, 0, index), FactKind::Effect, name_value(name));
        assert_eq!(
            row[FACT_DURATION_INDEX],
            duration_code(EffectDuration::Indefinite),
            "{name}"
        );
        assert!(row[FACT_ORDER_INDEX] > 0.0, "{name}: the record is ordered");
        // The opponent sees every record too.
        let opponent_index = if on_player {
            0
        } else {
            object_index(&s, 1, ogre)
        };
        fact_row(
            &facts_about(&s, 1, opponent_index),
            FactKind::Effect,
            name_value(name),
        );
    }
}

#[test]
fn a_later_record_has_a_later_order_than_an_earlier_one() {
    let (mut s, ogre) = ogre_state();
    let affected = on_permanent(&s, ogre);
    apply(&mut s, affected, Modification::LoseAllAbilities);
    apply(
        &mut s,
        affected,
        Modification::AddKeywords(Keywords {
            flying: true,
            ..Default::default()
        }),
    );
    let rows = facts_about(&s, 0, object_index(&s, 0, ogre));
    let order = |name: &str| fact_row(&rows, FactKind::Effect, name_value(name))[FACT_ORDER_INDEX];
    assert!(order("Modification::AddKeywords") > order("Modification::LoseAllAbilities"));
    assert!(
        permanent_value(&s, 0, ogre, PERMANENT_TIMESTAMP_INDEX)
            < order("Modification::LoseAllAbilities")
    );
    // CR 613.7: the later grant survives the earlier loss.
    assert!(managym::agent::observation::Observation::for_player(s.game(), PlayerId(0))
        .opponent_permanents[0]
        .keywords
        .flying);
}
