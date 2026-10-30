//! Every mechanic reaches the policy twice: match state through observation
//! tensor columns (`agent::observation_encoder`), card rules through learning
//! tokens (`cards::tokens`). The guards below destructure or match
//! exhaustively, with no wildcard, so a new keyword, permanent or player
//! status, `Effect`, `Ability`, `TriggerCondition`, `StaticAbility` or
//! `AdditionalCost` variant does not compile until it names where the policy
//! sees it. Directed tests then put a game in
//! that state and check the named column or token changes, and viewer-safety
//! tests check hidden cards change neither.

use crate::support::{
    game::{
        clear_summoning_sickness, deck, five_color_setup, put_onto_battlefield, ur_lessons_setup,
    },
    inspect::card_of,
    play::resolve_top,
};
use managym::{
    agent::{
        action::ActionSpaceKind,
        observation::Observation,
        observation_encoder::{encode, EncodedObservation, ObservationEncoderConfig},
        observation_encoder::{CARD_DIM, PERMANENT_DIM, PLAYER_DIM},
    },
    cards::content_pack::default_content_pack,
    cards::tokens::{definition_tokens, learning_catalog, name_value, TokenKind},
    flow::turn::StepKind,
    state::{
        ability::{
            Ability, Effect, EffectValue, PaymentCost, StaticCondition, TargetRequirement,
            TargetSpec, TriggerCondition, TriggerSubject,
        },
        card::{
            AdditionalCost, BlockRestriction, CardDefinition, Keywords, PowerCda, SacrificeCost,
            SpellCostReduction, StaticAbility, StaticPtBuff, StaticScope,
        },
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
    /// Card tensor column.
    Card(usize),
    /// Player tensor columns `start..end`.
    Player(usize, usize),
    /// Observed through another field named here.
    Via(&'static str),
    /// Changes which actions are legal (the action tensor), as described.
    Actions(&'static str),
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

/// One static ability or additional cost: its learning token, where its
/// effect is seen beyond that token, and the real card whose directed test in
/// `tests/cards/<file>.rs` runs it.
struct Witnessed {
    token: &'static str,
    seen: Seen,
    card: &'static str,
    file: &'static str,
    test: &'static str,
}

const fn witnessed(
    token: &'static str,
    seen: Seen,
    card: &'static str,
    file: &'static str,
    test: &'static str,
) -> Witnessed {
    Witnessed {
        token,
        seen,
        card,
        file,
        test,
    }
}

/// No wildcard: a new `StaticAbility` must be named here.
fn static_seen(ability: &StaticAbility) -> Witnessed {
    match ability {
        StaticAbility::GraveyardSpellsHaveFlashback => witnessed(
            "StaticAbility::GraveyardSpellsHaveFlashback",
            Seen::Actions("graveyard spells are offered as casts"),
            "Iroh, Grand Lotus",
            "tla",
            "flashback_is_exiled_when_countered_after_iroh_leaves",
        ),
        StaticAbility::GrantsAnyColorManaAbility(_) => witnessed(
            "StaticAbility::GrantsAnyColorManaAbility",
            Seen::Actions("granted mana abilities make more costs payable"),
            "Great Divide Guide",
            "tla",
            "guide_grants_tap_mana_with_sickness_and_removes_it_on_departure",
        ),
        StaticAbility::CantBeBlockedIf(_) => witnessed(
            "StaticAbility::CantBeBlockedIf",
            Seen::Actions("declare-blockers offers omit the attacker"),
            "June, Bounty Hunter",
            "tla",
            "june_is_unblockable_after_two_draws",
        ),
        StaticAbility::PlayersCantGainLife => witnessed(
            "StaticAbility::PlayersCantGainLife",
            Seen::Via("life column: prevented gains leave it unchanged"),
            "Giant Cindermaw",
            "blb",
            "savor_makes_food_with_a_real_tap_cost_and_cindermaw_prevents_its_life_gain",
        ),
        StaticAbility::CantAttackOrBlockUnlessYouControlAnother(_) => witnessed(
            "StaticAbility::CantAttackOrBlockUnlessYouControlAnother",
            Seen::Actions("attack and block offers omit the creature"),
            "Tiger-Dillo",
            "tla",
            "tiger_dillo_needs_another_four_power_creature_to_be_an_attacker",
        ),
        StaticAbility::BlockRestriction(_) => witnessed(
            "StaticAbility::BlockRestriction",
            Seen::Actions("block offers omit forbidden pairs"),
            "Spirit",
            "tla",
            "spirit_tokens_cannot_block_or_be_blocked_by_nonspirits",
        ),
        StaticAbility::SpellsCostLess(_) => witnessed(
            "StaticAbility::SpellsCostLess",
            Seen::Actions("casts become affordable at the reduced cost"),
            "Gran-Gran",
            "tla",
            "gran_gran_reduction_threshold_noncreature_and_generic_floor",
        ),
        StaticAbility::CostsLessPerPermanent(_) => witnessed(
            "StaticAbility::CostsLessPerPermanent",
            Seen::Actions("the cast becomes affordable at the reduced cost"),
            "Allies at Last",
            "tla",
            "allies_at_last_affinity_and_two_attackers",
        ),
        StaticAbility::NoMaximumHandSize => witnessed(
            "StaticAbility::NoMaximumHandSize",
            Seen::Player(3, 4),
            "Proft's Eidetic Memory",
            "mkm",
            "proft_hand_limit_only_while_controlled_and_legend_rule_for_both_card_types",
        ),
        StaticAbility::PtBuff(_) => witnessed(
            "StaticAbility::PtBuff",
            Seen::Permanent(6),
            "First-Time Flyer",
            "tla",
            "first_time_flyer_conditional_static_recomputes_on_read",
        ),
        StaticAbility::CharacteristicPower(_) => witnessed(
            "StaticAbility::CharacteristicPower",
            Seen::Permanent(6),
            "Suki, Kyoshi Warrior",
            "tla",
            "suki_power_tracks_creature_count_and_attack_makes_a_token",
        ),
        StaticAbility::Ward(_) => witnessed(
            "StaticAbility::Ward",
            Seen::Card(33),
            "Waterfall Aerialist",
            "stx",
            "ward_asks_spell_controller_and_counters_on_decline",
        ),
    }
}

fn block_restriction_token(restriction: &BlockRestriction) -> &'static str {
    match restriction {
        BlockRestriction::CantBlock(_) => "BlockRestriction::CantBlock",
        BlockRestriction::CantBeBlockedBy(_) => "BlockRestriction::CantBeBlockedBy",
    }
}

/// No wildcard: a new `AdditionalCost` must be named here.
fn cost_seen(cost: &AdditionalCost) -> Witnessed {
    match cost {
        AdditionalCost::Sacrifice(_) => witnessed(
            "AdditionalCost::Sacrifice",
            Seen::Actions("the cast is offered only with a sacrifice available"),
            "Deadly Precision",
            "tla",
            "deadly_precision_sacrifices_artifact_before_destroying_a_different_creature",
        ),
        AdditionalCost::Kicker(_) => witnessed(
            "AdditionalCost::Kicker",
            Seen::Card(35),
            "Firebending Lesson",
            "tla",
            "firebending_lesson_kicked_deals_five",
        ),
    }
}

/// One witness per `StaticAbility` variant and `BlockRestriction` side.
fn static_witnesses() -> Vec<StaticAbility> {
    let creature = CardPredicate::creature;
    vec![
        StaticAbility::GraveyardSpellsHaveFlashback,
        StaticAbility::GrantsAnyColorManaAbility(creature()),
        StaticAbility::CantBeBlockedIf(StaticCondition::YourTurn),
        StaticAbility::PlayersCantGainLife,
        StaticAbility::CantAttackOrBlockUnlessYouControlAnother(creature()),
        StaticAbility::BlockRestriction(BlockRestriction::CantBlock(creature())),
        StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(creature())),
        StaticAbility::SpellsCostLess(SpellCostReduction {
            first_on_own_turn: false,
            predicate: CardPredicate::default(),
            condition: StaticCondition::Always,
            generic: 1,
        }),
        StaticAbility::CostsLessPerPermanent(creature()),
        StaticAbility::NoMaximumHandSize,
        StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::This,
            condition: None,
            power: 2,
            toughness: 0,
        }),
        StaticAbility::CharacteristicPower(PowerCda::CreaturesYouControl),
        StaticAbility::Ward(ManaCost::parse("{2}")),
    ]
}

fn cost_witnesses() -> Vec<AdditionalCost> {
    vec![
        AdditionalCost::Sacrifice(SacrificeCost {
            predicate: CardPredicate::creature(),
            another: false,
            or_mana: None,
        }),
        AdditionalCost::Kicker(ManaCost::parse("{4}")),
    ]
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
            Seen::Card(column) => assert!(column < CARD_DIM, "{field}"),
            Seen::Via(place) | Seen::Actions(place) | Seen::Transient(place) => {
                assert!(!place.is_empty(), "{field}")
            }
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

// ---------------------------------------------------------------------------
// Static abilities and additional costs.

/// The definition of `name` in the default pack.
fn card_definition(name: &str) -> CardDefinition {
    let pack = default_content_pack();
    let id = pack.definition_id(name).unwrap_or_else(|| panic!("{name}"));
    pack.definition(id).unwrap().clone()
}

/// The source of `fn name(` through the next top-level `fn` in `source`.
fn function_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let start = source.find(&format!("\nfn {name}("))? + 1;
    let end = source[start..]
        .find("\nfn ")
        .map_or(source.len(), |offset| start + offset);
    Some(&source[start..end])
}

/// Every variant is witnessed by a real card that prints it, and that card's
/// named directed test exists in its card test file and names the card.
#[test]
fn every_static_and_cost_variant_names_a_real_card_and_its_behavior_test() {
    let statics = static_witnesses().into_iter().map(|ability| {
        let row = static_seen(&ability);
        let printed = card_definition(row.card)
            .statics
            .iter()
            .any(|candidate| std::mem::discriminant(candidate) == std::mem::discriminant(&ability));
        (row, printed)
    });
    let costs = cost_witnesses().into_iter().map(|cost| {
        let row = cost_seen(&cost);
        let printed = card_definition(row.card)
            .additional_costs
            .iter()
            .any(|candidate| std::mem::discriminant(candidate) == std::mem::discriminant(&cost));
        (row, printed)
    });
    for (row, printed) in statics.chain(costs) {
        assert!(printed, "{} does not print {}", row.card, row.token);
        let path = format!("{}/tests/cards/{}.rs", env!("CARGO_MANIFEST_DIR"), row.file);
        let source = std::fs::read_to_string(&path).unwrap();
        let body = function_body(&source, row.test)
            .unwrap_or_else(|| panic!("{path} has no test {}", row.test));
        // The test names the card itself or through a setup helper it calls.
        let card = format!("{:?}", row.card);
        let runs_card = body.contains(&card)
            || source
                .match_indices("\nfn ")
                .filter_map(|(at, _)| source[at + 4..].split('(').next())
                .any(|helper| {
                    body.contains(&format!("{helper}("))
                        && function_body(&source, helper).is_some_and(|it| it.contains(&card))
                });
        assert!(runs_card, "{} does not run {}", row.test, row.card);
        match row.seen {
            Seen::Permanent(column) => assert!(column < PERMANENT_DIM - 1, "{}", row.token),
            Seen::Card(column) => assert!(column < CARD_DIM, "{}", row.token),
            Seen::Player(start, end) => assert!(start < end && end <= PLAYER_DIM, "{}", row.token),
            Seen::Via(place) | Seen::Actions(place) | Seen::Transient(place) => {
                assert!(!place.is_empty(), "{}", row.token)
            }
            Seen::Identity => panic!("{} is rules, not identity", row.token),
        }
    }
}

/// Each variant's token is in the real card's tokens, and removing the
/// ability from the card removes the token and changes the card's tokens.
#[test]
fn every_static_and_cost_variant_reaches_the_learning_tokens() {
    let variant = |name: &str| (TokenKind::Variant, name_value(name));
    for ability in static_witnesses() {
        let row = static_seen(&ability);
        let card = card_definition(row.card);
        let mut names = vec![row.token];
        if let StaticAbility::BlockRestriction(restriction) = &ability {
            names.push(block_restriction_token(restriction));
        }
        let tokens = tokens_of(&card);
        let mut stripped = card.clone();
        stripped.statics.retain(|candidate| {
            std::mem::discriminant(candidate) != std::mem::discriminant(&ability)
        });
        let without = tokens_of(&stripped);
        assert_ne!(tokens, without, "{}", row.card);
        for name in names {
            assert!(tokens.contains(&variant(name)), "{} lacks {name}", row.card);
            assert!(!without.contains(&variant(name)), "{name}");
        }
        // The token also appears for a bare definition with only this ability.
        let bare = CardDefinition {
            name: "Witness".into(),
            statics: vec![ability.clone()],
            ..Default::default()
        };
        assert!(
            tokens_of(&bare).contains(&variant(row.token)),
            "{ability:?}"
        );
    }
    for cost in cost_witnesses() {
        let row = cost_seen(&cost);
        let card = card_definition(row.card);
        let tokens = tokens_of(&card);
        let mut stripped = card.clone();
        stripped
            .additional_costs
            .retain(|candidate| std::mem::discriminant(candidate) != std::mem::discriminant(&cost));
        let without = tokens_of(&stripped);
        assert_ne!(tokens, without, "{}", row.card);
        assert!(tokens.contains(&variant(row.token)), "{}", row.card);
        assert!(!without.contains(&variant(row.token)), "{}", row.token);
    }
}

/// Every variant whose row names a tensor column moves that column when the
/// ogre gains it: ward and kicker set card columns; a P/T buff and
/// characteristic power move the permanent's power column.
#[test]
fn tensor_visible_statics_and_costs_move_their_named_columns() {
    let (base_game, ogre) = ogre_game();
    let (base_obs, base) = observe(&base_game, 0);
    let card = base_game.state.permanents[ogre].as_ref().unwrap().card;
    let edits = static_witnesses()
        .into_iter()
        .map(|ability| {
            let row = static_seen(&ability);
            let edit: Box<dyn Fn(&mut CardDefinition)> =
                Box::new(move |definition| definition.statics.push(ability.clone()));
            (row, edit)
        })
        .chain(cost_witnesses().into_iter().map(|cost| {
            let row = cost_seen(&cost);
            let edit: Box<dyn Fn(&mut CardDefinition)> =
                Box::new(move |definition| definition.additional_costs.push(cost.clone()));
            (row, edit)
        }));
    let mut checked = 0;
    for (row, edit) in edits {
        let mut game = base_game.clone();
        edit(&mut game.state.cards[card]);
        let (obs, after) = observe(&game, 0);
        let moved = match row.seen {
            Seen::Card(column) => changed(
                card_row(&base_game, &base_obs, &base, ogre),
                card_row(&game, &obs, &after, ogre),
            )
            .contains(&column),
            Seen::Permanent(column) => {
                changed(permanent_row(&base), permanent_row(&after)).contains(&column)
            }
            _ => continue,
        };
        assert!(moved, "{} did not move its column", row.token);
        checked += 1;
    }
    assert_eq!(checked, 4);
}

/// With no maximum hand size, the hand-count column keeps its size through
/// cleanup; without it, the controller discards to seven.
#[test]
fn no_maximum_hand_size_keeps_the_hand_count_column_through_cleanup() {
    let Seen::Player(column, _) = static_seen(&StaticAbility::NoMaximumHandSize).seen else {
        panic!("hand size names a player column");
    };
    let hand_column = |has_static: bool| {
        let mut s = ur_lessons_setup();
        let proft = put_onto_battlefield(&mut s, 0, "Proft's Eidetic Memory");
        s.pass_priority();
        resolve_top(&mut s);
        if !has_static {
            let card = card_of(&s, proft);
            s.game_mut().move_card(card, ZoneType::Hand);
        }
        s.game_mut().draw_cards(PlayerId(0), 5);
        s.advance_to_active_step(0, StepKind::End);
        resolve_top(&mut s);
        while s.action_space().kind == ActionSpaceKind::Discard {
            s.step_action(0);
        }
        s.advance_to_active_step(1, StepKind::Upkeep);
        observe(s.game(), 0).1.agent_player[column]
    };
    let kept = hand_column(true);
    let discarded = hand_column(false);
    assert_eq!(discarded, 7.0 / 60.0);
    assert!(kept > discarded, "{kept} vs {discarded}");
}

/// June's conditional unblockability removes the blocks against it from the
/// defending player's action tensor.
#[test]
fn cant_be_blocked_if_removes_blocks_from_the_action_tensor() {
    let valid_actions = |draws: usize| {
        let mut s = five_color_setup(&["June, Bounty Hunter", "Callous Inspector"]);
        let june = s.force_permanent_on_battlefield(0, "June, Bounty Hunter");
        s.force_permanent_on_battlefield(1, "Callous Inspector");
        clear_summoning_sickness(&mut s, june);
        s.game_mut().draw_cards(PlayerId(0), draws);
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        s.declare_attack();
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        let (_, encoded) = observe(s.game(), 1);
        encoded.actions_valid.iter().sum::<f32>()
    };
    assert!(valid_actions(2) < valid_actions(0));
}

// ---------------------------------------------------------------------------
// Tokens.

/// Defined tokens whose producing card is not defined yet, with that producer.
/// When the producer lands, its reference makes the token leave this list.
const TOKENS_AWAITING_PRODUCER: [(&str, &str); 5] = [
    ("token.ballistic_boulder", "Fire Navy Trebuchet"),
    ("token.bear", "The Earth King"),
    ("token.dragon", "The Legend of Roku"),
    ("token.monk", "Crescent Island Temple"),
    ("token.soldier", "Fire Nation Attacks"),
];

/// Every `token.*` definition is a definition reference from a card that
/// creates it (or awaits a named producer), and a created token's observation
/// carries its characteristics.
#[test]
fn every_token_is_referenced_by_its_creator_and_observed_with_its_characteristics() {
    let pack = default_content_pack();
    let catalog = learning_catalog(&pack).unwrap();
    let tokens = managym::cardsets::catalog()
        .iter()
        .filter(|definition| definition.key.starts_with("token."))
        .collect::<Vec<_>>();
    assert!(!tokens.is_empty());
    for token in tokens {
        assert!(token.is_token, "{}", token.key);
        let row = pack.definition_id(&token.name).unwrap().0 as i32;
        let referencing = catalog
            .definition_ref_source_tokens
            .iter()
            .zip(&catalog.definition_ref_target_rows)
            .filter(|(_, &target)| target == row)
            .count();
        let awaiting = TOKENS_AWAITING_PRODUCER
            .iter()
            .any(|(key, _)| *key == token.key);
        assert_eq!(
            referencing > 0,
            !awaiting,
            "{}: {referencing} creating references",
            token.key
        );

        let (mut game, _) = ogre_game();
        game.create_token(&token.name, PlayerId(0), false);
        let (observation, encoded) = observe(&game, 0);
        let card = observation
            .agent_cards
            .iter()
            .position(|card| card.name == token.name && card.is_token)
            .unwrap_or_else(|| panic!("{} is not observed", token.key));
        let card_row = &encoded.agent_cards[card * CARD_DIM..(card + 1) * CARD_DIM];
        assert_eq!(card_row[29], 1.0, "{} is_token column", token.key);
        assert_eq!(card_row[8], token.power.unwrap_or(0) as f32 / 10.0);
        assert_eq!(card_row[9], token.toughness.unwrap_or(0) as f32 / 10.0);
        assert_eq!(card_row[12], f32::from(token.types.is_creature()));
        assert_eq!(card_row[13], f32::from(token.types.is_artifact()));
        assert_eq!(card_row[17], f32::from(token.keywords.flying));
    }
}
