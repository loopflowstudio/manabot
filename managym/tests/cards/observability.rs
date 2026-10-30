//! Every mechanic reaches the policy twice: match state through observation
//! tensor columns (`agent::observation_encoder`), card rules through learning
//! tokens (`cards::tokens`). The guards below destructure or match
//! exhaustively, with no wildcard, so a new keyword, permanent or player
//! status, `Effect`, `Ability` or `TriggerCondition` variant does not compile
//! until it names where the policy sees it. Directed tests then put a game in
//! that state and check the named column or token changes, and viewer-safety
//! tests check hidden cards change neither.

//!
//! The observation-row guards destructure each viewer row type
//! (`CardData`, `PlayerData`, `PermanentData`, `KeywordData`) without `..`, so
//! a new viewer fact fails to compile until it names the v5 tensor columns
//! that encode it, or says why it needs none; directed tests put real games
//! into each cast-permission and restricted-mana state, and hidden facts (an
//! opponent's foretold card) reach neither the tensor nor the JSON.

use std::ops::Range;

use crate::support::{
    game::{deck, five_color_setup, Scenario},
    inspect::{card_of, library_card},
    play::{activate, cast, choose, choose_mode, settle},
};
use managym::{
    agent::{
        action::Action,
        observation::{
            CardData, CardTypeData, CastPermissionData, KeywordData, Observation, PermanentData,
            PlayerData,
        },
        observation_encoder::{
            encode, EncodedObservation, ObservationEncoderConfig, CARD_DIM, CARD_PERMISSION_INDEX,
            CARD_X_INDEX, PERMANENT_DIM, PERMANENT_EXHAUSTED_INDEX, PERMANENT_FIREBENDING_INDEX,
            PERMANENT_ONCE_EACH_TURN_INDEX, PLAYER_DIM, PLAYER_HAND_ONLY_INDEX,
            PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX, PLAYER_RESTRICTED_INDEX,
        },
    },
    cards::content_pack::default_content_pack,
    cards::tokens::{definition_tokens, name_value, TokenKind},
    flow::turn::StepKind,
    state::{
        ability::{
            Ability, Effect, EffectValue, LinkedChooser, LinkedGrant, PaymentCost, StaticCondition,
            TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
        },
        card::{ActivatedAbilityDefinition, CardDefinition, Keywords},
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::{Mana, ManaCost},
        permanent::Permanent,
        player::Player,
        predicate::CardPredicate,
        zone::ZoneType,
    },
    Game, PlayerConfig,
};

// ---------------------------------------------------------------------------
// Where each piece of state is observed.

/// How the policy sees one state field.
#[derive(Clone, Copy, Debug)]
enum Seen {
    /// Permanent tensor column.
    Permanent(usize),
    /// Player tensor columns `start..end`.
    Player(usize, usize),
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

fn permanent_fields(permanent: &Permanent) -> [(&'static str, Seen); 20] {
    let Permanent {
        id: _,
        card: _,
        controller: _,
        tapped: _,
        summoning_sick: _,
        damage: _,
        deathtouch_damage: _,
        temp_power: _,
        temp_toughness: _,
        plus1_counters: _,
        cant_be_blocked_this_turn: _,
        animated: _,
        temp_keywords: _,
        temp_subtypes: _,
        exile_if_dies_this_turn: _,
        attacking: _,
        was_cast: _,
        firebending_this_turn: _,
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
        ("temp_power", Seen::Permanent(6)),
        ("temp_toughness", Seen::Permanent(7)),
        ("plus1_counters", Seen::Permanent(4)),
        ("cant_be_blocked_this_turn", Seen::Permanent(5)),
        ("animated", Seen::Permanent(8)),
        (
            "temp_keywords",
            Seen::Via("keyword_columns: effective keywords"),
        ),
        ("temp_subtypes", Seen::Permanent(26)),
        ("exile_if_dies_this_turn", Seen::Permanent(23)),
        ("attacking", Seen::Permanent(27)),
        (
            "was_cast",
            Seen::Transient("read only by entering-the-battlefield triggers as it enters"),
        ),
        (
            "firebending_this_turn",
            Seen::Permanent(PERMANENT_FIREBENDING_INDEX),
        ),
        ("exhausted", Seen::Permanent(PERMANENT_EXHAUSTED_INDEX)),
        (
            "activated_this_turn",
            Seen::Permanent(PERMANENT_ONCE_EACH_TURN_INDEX),
        ),
    ]
}

fn player_fields(player: &Player) -> [(&'static str, Seen); 12] {
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
        no_maximum_hand_size: _,
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
        (
            "no_maximum_hand_size",
            Seen::Player(PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX, PLAYER_DIM),
        ),
    ]
}

/// The token every `Effect` variant contributes. No wildcard: a new variant
/// must be named here (and given a witness below).
fn effect_token(effect: &Effect) -> &'static str {
    match effect {
        Effect::CreateTokenCopy { .. } => "Effect::CreateTokenCopy",
        Effect::Airbend { .. } => "Effect::Airbend",
        Effect::AirbendAll { .. } => "Effect::AirbendAll",
        Effect::OpponentsCastOnlyFromHand => "Effect::OpponentsCastOnlyFromHand",
        Effect::ExileTopCardPlayThisTurn => "Effect::ExileTopCardPlayThisTurn",
        Effect::GrantFirebendingUntilEot { .. } => "Effect::GrantFirebendingUntilEot",
        Effect::ExileLinked { .. } => "Effect::ExileLinked",
        Effect::ExileUntilNonlandCastFree { .. } => "Effect::ExileUntilNonlandCastFree",
        Effect::CastFromHandFree { .. } => "Effect::CastFromHandFree",
        Effect::NoMaximumHandSize => "Effect::NoMaximumHandSize",
        Effect::ShuffleGraveyardIntoLibrary => "Effect::ShuffleGraveyardIntoLibrary",
        Effect::OwnerPutsIntoLibrary { .. } => "Effect::OwnerPutsIntoLibrary",
        Effect::RevealHandDiscardChosen { .. } => "Effect::RevealHandDiscardChosen",
        Effect::SearchLandsOpponentSplits { .. } => "Effect::SearchLandsOpponentSplits",
        Effect::PlaceTargetInLibrary { .. } => "Effect::PlaceTargetInLibrary",
        Effect::SacrificeSource { .. } => "Effect::SacrificeSource",
        Effect::ExileSourceSpell => "Effect::ExileSourceSpell",
        Effect::BuffTriggeringPermanentValue { .. } => "Effect::BuffTriggeringPermanentValue",
        Effect::GrantKeywordsToSource { .. } => "Effect::GrantKeywordsToSource",
        Effect::ReturnSourceToHand => "Effect::ReturnSourceToHand",
        Effect::TapTarget { .. } => "Effect::TapTarget",
        Effect::ModifyControlled { .. } => "Effect::ModifyControlled",
        Effect::OptionalPayment { .. } => "Effect::OptionalPayment",
        Effect::Sacrifice { .. } => "Effect::Sacrifice",
        Effect::DealDamageToController { .. } => "Effect::DealDamageToController",
        Effect::DrainLife { .. } => "Effect::DrainLife",
        Effect::CantBeBlockedTarget { .. } => "Effect::CantBeBlockedTarget",
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
        Effect::BuffTargetValue { .. } => "Effect::BuffTargetValue",
        Effect::ReturnToHandDrawIfControlled { .. } => "Effect::ReturnToHandDrawIfControlled",
        Effect::ModifyTriggeringPermanent { .. } => "Effect::ModifyTriggeringPermanent",
        Effect::Discard => "Effect::Discard",
        Effect::DealDamageValue { .. } => "Effect::DealDamageValue",
        Effect::PutCountersValue { .. } => "Effect::PutCountersValue",
        Effect::AddSubtypeToTarget { .. } => "Effect::AddSubtypeToTarget",
        Effect::ExileIfDiesThisTurn { .. } => "Effect::ExileIfDiesThisTurn",
        Effect::IfCondition { .. } => "Effect::IfCondition",
        Effect::ReturnToHand { .. } => "Effect::ReturnToHand",
        Effect::DealDamage { .. } => "Effect::DealDamage",
        Effect::CounterSpell { .. } => "Effect::CounterSpell",
        Effect::ModifyUntilEot { .. } => "Effect::ModifyUntilEot",
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
        Effect::CantBeBlockedThisTurnSource => "Effect::CantBeBlockedThisTurnSource",
        Effect::GainLife { .. } => "Effect::GainLife",
        Effect::OnNthResolutionThisTurn { .. } => "Effect::OnNthResolutionThisTurn",
        Effect::Scry { .. } => "Effect::Scry",
        Effect::Surveil { .. } => "Effect::Surveil",
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
        Effect::BuffTarget { .. } => "Effect::BuffTarget",
        Effect::UntapTarget { .. } => "Effect::UntapTarget",
        Effect::GrantKeywordsToTarget { .. } => "Effect::GrantKeywordsToTarget",
        Effect::IfTargetMatches { .. } => "Effect::IfTargetMatches",
        Effect::ForEachTarget { .. } => "Effect::ForEachTarget",
        Effect::PutCountersOnEachMatching { .. } => "Effect::PutCountersOnEachMatching",
    }
}

/// Number of `Effect` variants named in [`effect_token`].
const EFFECT_VARIANTS: usize = 84;

/// One value of every `Effect` variant.
fn effect_witnesses() -> Vec<Effect> {
    let creature = CardPredicate::creature;
    let target = || TargetSpec::Creature;
    let value = || EffectValue::CreaturesWithCounters;
    let draw = || vec![Effect::DrawCards { count: 1 }];
    vec![
        Effect::CreateTokenCopy { nonlegendary: true },
        Effect::Airbend { target: target() },
        Effect::AirbendAll {
            predicate: creature(),
        },
        Effect::OpponentsCastOnlyFromHand,
        Effect::ExileTopCardPlayThisTurn,
        Effect::GrantFirebendingUntilEot { amount: 1 },
        Effect::ExileLinked {
            chooser: LinkedChooser::TargetPlayer,
            zone: ZoneType::Graveyard,
            predicate: creature(),
            up_to: false,
            grant: LinkedGrant::WaterbendManaValue,
            target: TargetSpec::Player,
        },
        Effect::ExileUntilNonlandCastFree { below: value() },
        Effect::CastFromHandFree {
            predicate: creature(),
        },
        Effect::NoMaximumHandSize,
        Effect::ShuffleGraveyardIntoLibrary,
        Effect::OwnerPutsIntoLibrary { target: target() },
        Effect::RevealHandDiscardChosen {
            target: TargetSpec::Player,
            predicate: creature(),
        },
        Effect::SearchLandsOpponentSplits {
            count: 4,
            opponent_picks: 2,
        },
        Effect::PlaceTargetInLibrary {
            second_from_top: true,
        },
        Effect::SacrificeSource { then: draw() },
        Effect::ExileSourceSpell,
        Effect::BuffTriggeringPermanentValue {
            power: value(),
            toughness: value(),
        },
        Effect::GrantKeywordsToSource {
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
        },
        Effect::ReturnSourceToHand,
        Effect::TapTarget { target: target() },
        Effect::ModifyControlled {
            predicate: creature(),
            power: 1,
            toughness: 1,
        },
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
        Effect::CantBeBlockedTarget { target: target() },
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
        Effect::BuffTargetValue {
            power: value(),
            toughness: 0,
            target: target(),
        },
        Effect::ReturnToHandDrawIfControlled { target: target() },
        Effect::ModifyTriggeringPermanent {
            counters: 1,
            keywords: Keywords::default(),
        },
        Effect::Discard,
        Effect::DealDamageValue {
            amount: value(),
            target: target(),
        },
        Effect::PutCountersValue {
            count: value(),
            target: target(),
        },
        Effect::AddSubtypeToTarget {
            target: target(),
            subtype: "Avatar".into(),
        },
        Effect::ExileIfDiesThisTurn { target: target() },
        Effect::IfCondition {
            condition: StaticCondition::YourTurn,
            effects: draw(),
        },
        Effect::ReturnToHand { target: target() },
        Effect::DealDamage {
            amount: 1,
            target: target(),
        },
        Effect::CounterSpell {
            target: TargetSpec::Spell,
        },
        Effect::ModifyUntilEot {
            power_delta: 1,
            toughness_delta: 1,
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
        },
        Effect::PutCountersOnSource { count: 1 },
        Effect::PutCounters {
            count: 1,
            target: target(),
        },
        Effect::TapSource,
        Effect::UntapSource,
        Effect::CantBeBlockedThisTurnSource,
        Effect::GainLife { amount: 1 },
        Effect::OnNthResolutionThisTurn {
            n: 2,
            effect: Box::new(Effect::DrawCards { count: 1 }),
        },
        Effect::Scry { count: 1 },
        Effect::Surveil { count: 1 },
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
        Effect::BuffTarget {
            power: 1,
            toughness: 1,
            target: target(),
        },
        Effect::UntapTarget { target: target() },
        Effect::GrantKeywordsToTarget {
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            target: target(),
        },
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
        TriggerCondition::AttacksAlone { .. } => "TriggerCondition::AttacksAlone",
        TriggerCondition::BecomesTapped { .. } => "TriggerCondition::BecomesTapped",
        TriggerCondition::TappedForMana { .. } => "TriggerCondition::TappedForMana",
        TriggerCondition::BecomesTargeted { .. } => "TriggerCondition::BecomesTargeted",
        TriggerCondition::BeginningOfYourUpkeep => "TriggerCondition::BeginningOfYourUpkeep",
        TriggerCondition::BeginningOfYourCombat => "TriggerCondition::BeginningOfYourCombat",
        TriggerCondition::YouDrawNthCardThisTurn { .. } => {
            "TriggerCondition::YouDrawNthCardThisTurn"
        }
        TriggerCondition::ActiveIf { .. } => "TriggerCondition::ActiveIf",
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
        TriggerCondition::AttacksAlone { subject: this() },
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
            Seen::Identity => {}
        }
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
fn sorcery_timing_reaches_the_activated_ability_tokens() {
    let definition = CardDefinition {
        name: "Witness".into(),
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            effects: vec![Effect::Surveil { count: 1 }],
            ..Default::default()
        }],
        ..Default::default()
    };
    let tokens = tokens_of(&definition);
    let at = tokens
        .iter()
        .position(|token| *token == (TokenKind::Field, name_value("sorcery_timing")))
        .expect("sorcery_timing token");
    assert_eq!(tokens[at + 1], (TokenKind::Boolean, 1));
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
        "cant_be_blocked_this_turn" => permanent.cant_be_blocked_this_turn = true,
        "animated" => permanent.animated = true,
        "temp_subtypes" => permanent.temp_subtypes = vec!["Avatar".into()],
        "exile_if_dies_this_turn" => permanent.exile_if_dies_this_turn = true,
        "attacking" => permanent.attacking = true,
        "firebending_this_turn" => permanent.firebending_this_turn = 2,
        "exhausted" => permanent.exhausted = vec![0],
        "activated_this_turn" => permanent.activated_this_turn = vec![0],
        other => panic!("no directed mutation for {other}"),
    };
    for (field, seen) in fields {
        let Seen::Permanent(column) = seen else {
            continue;
        };
        if matches!(field, "card" | "controller") {
            continue; // Identity of the row itself; covered by the encoder's own tests.
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
// (d) Observation rows: every viewer fact names its v5 tensor columns, and
// cast permissions and restricted mana reach exactly the viewers they should.

/// How the policy input carries one observation field.
#[derive(Debug)]
enum Encoding {
    /// These columns of the field's row.
    Columns(Range<usize>),
    /// Carried outside the row's own columns, as described.
    Elsewhere(&'static str),
}
use Encoding::{Columns, Elsewhere};

const PERMISSION: usize = CARD_PERMISSION_INDEX;

fn keyword_data_columns(keywords: &KeywordData, card: bool) -> Vec<(&'static str, Encoding)> {
    let KeywordData {
        flying: _,
        reach: _,
        haste: _,
        flash: _,
        vigilance: _,
        trample: _,
        first_strike: _,
        double_strike: _,
        deathtouch: _,
        lifelink: _,
        defender: _,
        menace: _,
        hexproof: _,
        indestructible: _,
    } = keywords;
    let at = |card_column: usize, permanent_column: usize| {
        let column = if card { card_column } else { permanent_column };
        Columns(column..column + 1)
    };
    vec![
        ("flying", at(17, 10)),
        ("reach", at(18, 11)),
        ("haste", at(19, 12)),
        ("flash", at(28, 13)),
        ("vigilance", at(20, 14)),
        ("trample", at(21, 15)),
        ("first_strike", at(22, 16)),
        ("double_strike", at(23, 17)),
        ("deathtouch", at(24, 18)),
        ("lifelink", at(25, 19)),
        ("defender", at(26, 20)),
        ("menace", at(27, 21)),
        ("hexproof", at(36, 22)),
        ("indestructible", at(38, 25)),
    ]
}

fn card_columns(card: &CardData) -> Vec<(&'static str, Encoding)> {
    let CardData {
        zone: _,
        owner_id: _,
        id: _,
        registry_key: _,
        name: _,
        text_box: _,
        power: _,
        toughness: _,
        is_token: _,
        is_legendary: _,
        is_ally: _,
        is_lesson: _,
        ward_cost: _,
        kicker_cost: _,
        card_types,
        keywords,
        mana_cost: _,
        face_down: _,
        cast_permission,
        x: _,
    } = card;
    let CardTypeData {
        is_castable: _,
        is_permanent: _,
        is_non_land_permanent: _,
        is_non_creature_permanent: _,
        is_spell: _,
        is_creature: _,
        is_land: _,
        is_planeswalker: _,
        is_enchantment: _,
        is_artifact: _,
        is_kindred: _,
        is_battle: _,
    } = card_types;
    if let Some(CastPermissionData {
        player_id: _,
        cost: _,
        this_turn: _,
        own_turn_only: _,
        after_turn: _,
        source_permanent_id: _,
        flash: _,
        any_mana: _,
        waterbend: _,
        foretold: _,
    }) = cast_permission
    {}
    let mut columns = vec![
        ("zone", Columns(0..7)),
        (
            "owner_id",
            Elsewhere("rows split by owner; is_mine column 7"),
        ),
        ("id", Elsewhere("object index for action focus")),
        ("registry_key", Elsewhere("semantic_cards definition key")),
        ("name", Elsewhere("named by registry_key")),
        ("text_box", Elsewhere("named by registry_key")),
        ("power", Columns(8..9)),
        ("toughness", Columns(9..10)),
        ("mana_cost", Columns(10..11)),
        ("is_land", Columns(11..12)),
        ("is_creature", Columns(12..13)),
        ("is_artifact", Columns(13..14)),
        ("is_enchantment", Columns(14..15)),
        ("is_planeswalker", Columns(15..16)),
        ("is_battle", Columns(16..17)),
        (
            "is_castable, is_permanent, is_non_land_permanent, \
             is_non_creature_permanent, is_spell, is_kindred",
            Elsewhere("derived from the type flags and registry_key"),
        ),
        ("is_token", Columns(29..30)),
        ("is_ally", Columns(30..31)),
        ("is_lesson", Columns(31..32)),
        ("ward_cost", Columns(32..34)),
        ("kicker_cost", Columns(34..36)),
        ("is_legendary", Columns(39..40)),
        ("face_down", Columns(PERMISSION + 11..PERMISSION + 12)),
        (
            "cast_permission.player_id",
            Columns(PERMISSION..PERMISSION + 2),
        ),
        (
            "cast_permission.cost",
            Columns(PERMISSION + 2..PERMISSION + 4),
        ),
        (
            "cast_permission.this_turn",
            Columns(PERMISSION + 4..PERMISSION + 5),
        ),
        (
            "cast_permission.own_turn_only",
            Columns(PERMISSION + 5..PERMISSION + 6),
        ),
        (
            "cast_permission.after_turn",
            Columns(PERMISSION + 6..PERMISSION + 7),
        ),
        (
            "cast_permission.source_permanent_id",
            Columns(PERMISSION + 7..PERMISSION + 8),
        ),
        (
            "cast_permission.flash",
            Columns(PERMISSION + 8..PERMISSION + 9),
        ),
        (
            "cast_permission.any_mana",
            Columns(PERMISSION + 9..PERMISSION + 10),
        ),
        (
            "cast_permission.waterbend",
            Columns(PERMISSION + 10..PERMISSION + 11),
        ),
        (
            "cast_permission.foretold",
            Columns(PERMISSION + 11..PERMISSION + 12),
        ),
        ("x", Columns(CARD_X_INDEX..CARD_X_INDEX + 2)),
    ];
    columns.extend(keyword_data_columns(keywords, true));
    columns
}

fn player_columns(player: &PlayerData) -> Vec<(&'static str, Encoding)> {
    let PlayerData {
        player_index: _,
        id: _,
        is_agent: _,
        is_active: _,
        life: _,
        zone_counts: _,
        graveyard_lessons: _,
        combat_mana: _,
        mana: _,
        lesson_mana: _,
        restricted_mana: _,
        hand_only_until: _,
        no_maximum_hand_size: _,
        known_hand: _,
        sideboard_counts: _,
        remaining_sideboard_counts: _,
    } = player;
    vec![
        (
            "player_index",
            Elsewhere("agent_player and opponent_player rows"),
        ),
        ("id", Elsewhere("object index for action focus")),
        (
            "is_agent",
            Elsewhere("agent_player and opponent_player rows"),
        ),
        ("is_active", Columns(1..2)),
        ("life", Columns(0..1)),
        ("zone_counts", Columns(2..9)),
        ("graveyard_lessons", Columns(26..27)),
        ("combat_mana", Columns(27..28)),
        ("mana", Columns(28..34)),
        ("lesson_mana", Columns(34..40)),
        (
            "restricted_mana",
            Columns(PLAYER_RESTRICTED_INDEX..PLAYER_HAND_ONLY_INDEX),
        ),
        (
            "hand_only_until",
            Columns(PLAYER_HAND_ONLY_INDEX..PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX),
        ),
        (
            "no_maximum_hand_size",
            Columns(PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX..PLAYER_DIM),
        ),
        ("known_hand", Elsewhere("known_hand tensor")),
        (
            "sideboard_counts, remaining_sideboard_counts",
            Elsewhere("agent sideboard rows in agent_cards"),
        ),
    ]
}

fn permanent_columns(permanent: &PermanentData) -> Vec<(&'static str, Encoding)> {
    let PermanentData {
        id: _,
        registry_key: _,
        controller_id: _,
        tapped: _,
        damage: _,
        is_summoning_sick: _,
        plus1_counters: _,
        cant_be_blocked_this_turn: _,
        power: _,
        toughness: _,
        is_animated: _,
        has_exile_link: _,
        exile_if_dies_this_turn: _,
        attacking: _,
        exhausted_abilities: _,
        once_each_turn_used: _,
        granted_firebending: _,
        temp_subtypes: _,
        keywords,
    } = permanent;
    let mut columns = vec![
        ("id", Elsewhere("object index for action focus")),
        ("registry_key", Columns(24..25)),
        ("controller_id", Columns(0..1)),
        ("tapped", Columns(1..2)),
        ("damage", Columns(2..3)),
        ("is_summoning_sick", Columns(3..4)),
        ("plus1_counters", Columns(4..5)),
        ("cant_be_blocked_this_turn", Columns(5..6)),
        ("power", Columns(6..7)),
        ("toughness", Columns(7..8)),
        ("is_animated", Columns(8..9)),
        ("has_exile_link", Columns(9..10)),
        ("exile_if_dies_this_turn", Columns(23..24)),
        ("attacking", Columns(27..28)),
        (
            "exhausted_abilities",
            Columns(PERMANENT_EXHAUSTED_INDEX..PERMANENT_EXHAUSTED_INDEX + 1),
        ),
        (
            "once_each_turn_used",
            Columns(PERMANENT_ONCE_EACH_TURN_INDEX..PERMANENT_ONCE_EACH_TURN_INDEX + 1),
        ),
        (
            "granted_firebending",
            Columns(PERMANENT_FIREBENDING_INDEX..PERMANENT_FIREBENDING_INDEX + 1),
        ),
        // Avatar is the only temporary subtype the current pool grants.
        ("temp_subtypes", Columns(26..27)),
    ];
    columns.extend(keyword_data_columns(keywords, false));
    columns
}

/// Observations of a game holding every row type, from both seats.
fn rich_observations() -> Vec<Observation> {
    let mut s = five_color_setup(&["Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = library_card(&s, 0, "Gray Ogre");
    s.game_mut().move_card(card, ZoneType::Hand);
    [PlayerId(0), PlayerId(1)]
        .map(|viewer| Observation::for_player(s.game(), viewer))
        .into()
}

#[test]
fn every_row_field_names_columns_inside_its_v5_row() {
    for observation in rich_observations() {
        let card = observation.agent_cards.first().expect("a card");
        let permanent = observation.agent_permanents.first().expect("a permanent");
        // Card and permanent rows end in their validity flag; player rows
        // carry validity in a separate tensor.
        for (columns, encoded_width) in [
            (card_columns(card), CARD_DIM - 1),
            (player_columns(&observation.agent), PLAYER_DIM),
            (permanent_columns(permanent), PERMANENT_DIM - 1),
        ] {
            for (field, encoding) in columns {
                match encoding {
                    Columns(range) => assert!(
                        !range.is_empty() && range.end <= encoded_width,
                        "{field}: {range:?} must sit within the first {encoded_width} columns"
                    ),
                    Elsewhere(how) => assert!(!how.is_empty(), "{field} must say where"),
                }
            }
        }
    }
}

// --- Directed mechanics --------------------------------------------------

fn encoded(observation: &Observation) -> EncodedObservation {
    encode(observation, &ObservationEncoderConfig::default()).expect("v5 encodes")
}

fn object_id(s: &Scenario, card: CardId) -> i32 {
    s.game().state.cards[card].id.0 as i32
}

/// The encoded row of `card` on either side of `observation`, if listed.
fn listed_card_row(observation: &Observation, card: i32) -> Option<Vec<f32>> {
    let encoded = encoded(observation);
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
    let baseline = encoded(&Observation::for_player(s.game(), PlayerId(0)));
    assert_eq!(baseline.opponent_player[PLAYER_HAND_ONLY_INDEX], 0.0);
    cast(&mut s, "Avatar's Wrath");
    assert!(s.choose_target(Target::Permanent(spared)));
    settle(&mut s);

    let caster = Observation::for_player(s.game(), PlayerId(0));
    assert_eq!(caster.opponent.hand_only_until, Some(caster.agent.id));
    assert_eq!(
        encoded(&caster).opponent_player[PLAYER_HAND_ONLY_INDEX],
        1.0
    );
    assert_eq!(encoded(&caster).agent_player[PLAYER_HAND_ONLY_INDEX], 0.0);
    let locked = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(encoded(&locked).agent_player[PLAYER_HAND_ONLY_INDEX], 1.0);
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
    let player = encoded(&observation).agent_player;
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
    let opponent_before = encoded(&Observation::for_player(s.game(), PlayerId(1)));
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
    let after = encoded(&opponent);
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
