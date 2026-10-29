//! Every mechanic reaches the policy twice: match state through observation
//! tensor columns (`agent::observation_encoder`), card rules through learning
//! tokens (`semantic::tokens`). The guards below destructure or match
//! exhaustively, with no wildcard, so a new keyword, permanent or player
//! status, `Effect`, `Ability` or `TriggerCondition` variant does not compile
//! until it names where the policy sees it. Directed tests then put a game in
//! that state and check the named column or token changes, and viewer-safety
//! tests check hidden cards change neither.

use std::collections::BTreeMap;

use managym::{
    agent::{
        observation::Observation,
        observation_encoder::{encode, EncodedObservation, ObservationEncoderConfig},
        observation_encoder::{CARD_DIM, PERMANENT_DIM, PLAYER_DIM},
    },
    cards::content_pack::default_content_pack,
    semantic::tokens::{definition_tokens, name_value, TokenKind},
    state::{
        ability::{
            Ability, Effect, EffectValue, PaymentCost, StaticCondition, TargetRequirement,
            TargetSpec, TriggerCondition, TriggerSubject,
        },
        card::{CardDefinition, Keywords},
        game_object::{PermanentId, PlayerId},
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

fn permanent_fields(permanent: &Permanent) -> [(&'static str, Seen); 17] {
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

/// The token every `Effect` variant contributes. No wildcard: a new variant
/// must be named here (and given a witness below).
fn effect_token(effect: &Effect) -> &'static str {
    match effect {
        Effect::CreateNonlegendaryCopy => "Effect::CreateNonlegendaryCopy",
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
const EFFECT_VARIANTS: usize = 63;

/// One value of every `Effect` variant.
fn effect_witnesses() -> Vec<Effect> {
    let creature = CardPredicate::creature;
    let target = || TargetSpec::Creature;
    let value = || EffectValue::CreaturesWithCounters;
    let draw = || vec![Effect::DrawCards { count: 1 }];
    vec![
        Effect::CreateNonlegendaryCopy,
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
        },
        Effect::MillCards { count: 2 },
        Effect::MillAndSelect {
            count: 2,
            predicate: creature(),
            if_declined: draw(),
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
        },
        Effect::OptionalDiscardDraw { count: 1 },
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
        Effect::LookAndSelect {
            reveal: false,
            look: 3,
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
            count: 1,
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

fn deck(cards: &[(&str, usize)]) -> BTreeMap<String, usize> {
    cards
        .iter()
        .map(|(name, count)| (name.to_string(), *count))
        .collect()
}

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
