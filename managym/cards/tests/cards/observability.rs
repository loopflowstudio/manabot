//! Every mechanic reaches the policy three ways: match state through the
//! observation tensor (`agent::observation_encoder`), card
//! rules through learning tokens (`cards::tokens`), and both through the
//! LLM text (`manabot/semantic/text_observation.py`, guarded in Python). The
//! guards destructure or match exhaustively, with no wildcard, so a new
//! keyword, counter kind, `Observation`, `Permanent` or `Player` field, or
//! `Effect`, `Ability`, `TriggerCondition`, `StaticAbility` or `AdditionalCost`
//! variant does not compile until it
//! names where the policy sees it; every listed column is checked against the
//! row widths and every row column must be claimed. Directed tests put a game
//! into each state and check that the named column or token changes against
//! an otherwise equal baseline.
//! Viewer-safety tests check hidden cards change neither.

use crate::support::game::move_card;
use crate::support::inspect::{current_object, zone_of};
use crate::support::{
    game::{
        clear_summoning_sickness, deck, five_color_setup, five_color_unfunded_setup, move_to_hand,
        put_onto_battlefield, until_end_of_turn, ur_lessons_setup, Scenario,
    },
    inspect::{card_of, library_card},
    play::{activate, cast, choose, choose_mode, pass_until, resolve_top, settle, spend_colorless},
};
use managym::agent::observation_layout::{
    action, card, fact, permanent, player, relation, turn_state,
};
use managym::state::game_object::ObjectKind;
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        continuation::{
            delayed_event_code, ClauseTarget, ContinuationFrame, ManaOutput, VisibleReference,
            RELATION_DIM,
        },
        observation::{
            ActionOption, ActionSpaceData, CardData, CardTypeData, CastPermissionData,
            ContinuousEffectData, CopyData, EventData, Observation, PermanentData, PlayerData,
            PooledManaClassData, RestrictedManaData, SideboardCardData, StackObjectData,
            StackTargetData, TurnData, KEYWORD_COUNT,
        },
        observation_encoder::{
            encode, EncodedObservation, FactKind, ObservationEncoderConfig, CARD_DIM, FACT_DIM,
            PERMANENT_DIM, PLAYER_DIM, TURN_STATE_DIM,
        },
        shared_effects::{EffectMembership, EffectObject, SharedEffectData},
    },
    flow::{
        trigger::DelayedTriggerEvent,
        turn::{PlayerTurnHistory, StepKind, TurnState},
    },
    state::{
        effect::{Affected, EffectDuration, Modification},
        game_object::{ObjectId, PermanentId, PlayerId, Target},
        mana::Color,
        permanent::CounterKind,
        zone::ZoneType,
    },
};
use managym::{
    cards::content_pack::default_content_pack,
    cards::tokens::{catalog_tokens, definition_tokens, name_value, TokenKind},
    state::{
        ability::{
            Ability, Actor, AttackSize, Condition, DelayedEvent, DelayedExpiry, DelayedFrequency,
            DistinctSearchEnd, Effect, EffectValue, Except, KeywordAction, LinkedChooser,
            LinkedExile, LinkedGrant, ObjectSelector, PaymentCost, Per, PlayWindow, PlayerSelector,
            Scope, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
        },
        card::{
            ActivatedAbilityDefinition, AdditionalCost, BlockRestriction, CardType, Keyword,
            Keywords, ObjectDefinition, PowerCda, PrintedCharacteristics, Restriction,
            SacrificeCost, SetCharacteristics, SpellCostReduction, StaticAbility,
            StaticKeywordGrant, StaticLandType, StaticPtBuff, StaticRestriction, StaticScope,
            StaticTriggeredGrant, StaticTypeGrant, TargetCostReduction, TriggeredManaAbility,
            TriggeredManaOutput,
        },
        mana::{Mana, ManaCost},
        permanent::Permanent,
        player::Player,
        predicate::CardPredicate,
    },
    Game, PlayerConfig,
};
use std::collections::BTreeSet;
use std::sync::Arc;

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
    /// Fact rows of this kind (`FACT_*`).
    Fact(FactKind),
    /// Observed through another field named here.
    Via(&'static str),
    /// Changes which actions are legal (the action tensor), as described.
    Actions(&'static str),
    /// Never visible at a decision point, with the reason.
    Transient(&'static str),
    /// Object identity or bookkeeping rather than game state.
    Identity,
}

/// Keyword → (card column, permanent column). Every keyword is evergreen
/// today, so each has columns; the first non-evergreen keyword names its fact
/// row here instead of columns. The token is the `Keyword::<name>` variant in
/// the definition's keyword list, present only when the keyword is.
fn keyword_columns(keyword: Keyword) -> (usize, usize) {
    match keyword {
        Keyword::Flying => (card::FLYING, permanent::FLYING),
        Keyword::Reach => (card::REACH, permanent::REACH),
        Keyword::Haste => (card::HASTE, permanent::HASTE),
        Keyword::Flash => (card::FLASH, permanent::FLASH),
        Keyword::Vigilance => (card::VIGILANCE, permanent::VIGILANCE),
        Keyword::Trample => (card::TRAMPLE, permanent::TRAMPLE),
        Keyword::FirstStrike => (card::FIRST_STRIKE, permanent::FIRST_STRIKE),
        Keyword::DoubleStrike => (card::DOUBLE_STRIKE, permanent::DOUBLE_STRIKE),
        Keyword::Deathtouch => (card::DEATHTOUCH, permanent::DEATHTOUCH),
        Keyword::Lifelink => (card::LIFELINK, permanent::LIFELINK),
        Keyword::Defender => (card::DEFENDER, permanent::DEFENDER),
        Keyword::Menace => (card::MENACE, permanent::MENACE),
        Keyword::Hexproof => (card::HEXPROOF, permanent::HEXPROOF),
        Keyword::Indestructible => (card::INDESTRUCTIBLE, permanent::INDESTRUCTIBLE),
    }
}

fn permanent_fields(permanent: &Permanent) -> [(&'static str, Seen); 19] {
    let Permanent {
        transforms: _,
        id: _,
        object: _,
        controller: _,
        tapped: _,
        summoning_sick: _,
        damage: _,
        deathtouch_damage: _,
        dealt_damage_this_turn: _,
        timestamp: _,
        counters: _,
        attacking: _,
        was_cast: _,
        cast_x: _,
        cast_spent: _,
        attached_to: _,
        exhausted: _,
        activated_this_turn: _,
        keyword_counter_timestamps: _,
    } = permanent;
    [
        ("id", Seen::Identity),
        ("card", Seen::Permanent(permanent::DEFINITION)),
        ("controller", Seen::Permanent(permanent::IS_MINE)),
        ("transforms", Seen::Via("Transforms fact rows")),
        ("tapped", Seen::Permanent(permanent::TAPPED)),
        ("summoning_sick", Seen::Permanent(permanent::SUMMONING_SICK)),
        ("damage", Seen::Permanent(permanent::DAMAGE)),
        (
            "deathtouch_damage",
            Seen::Transient("state-based actions destroy the creature before priority"),
        ),
        (
            "dealt_damage_this_turn",
            Seen::Permanent(permanent::DEALT_DAMAGE_THIS_TURN),
        ),
        ("timestamp", Seen::Permanent(permanent::TIMESTAMP)),
        ("counters", Seen::Fact(FactKind::Counter)),
        (
            "keyword_counter_timestamps",
            Seen::Fact(FactKind::KeywordCounterTimestamp),
        ),
        ("attacking", Seen::Permanent(permanent::ATTACKING)),
        (
            "was_cast",
            Seen::Transient("read only by entering-the-battlefield triggers as it enters"),
        ),
        (
            "cast_x",
            Seen::Transient("copied into its own entering trigger's X as it fires"),
        ),
        (
            "cast_spent",
            Seen::Transient("copied into its own entering trigger's mana spent as it fires"),
        ),
        ("attached_to", Seen::Permanent(permanent::ATTACHED_TO)),
        ("exhausted", Seen::Fact(FactKind::UsedAbility)),
        ("activated_this_turn", Seen::Fact(FactKind::UsedAbility)),
    ]
}

fn player_fields(player: &Player) -> [(&'static str, Seen); 12] {
    let Player {
        id: _,
        index: _,
        deck: _,
        sideboard: _,
        disclosure: _,
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
        (
            "deck",
            Seen::Via("decklists tensor (main and unseen counts)"),
        ),
        (
            "sideboard",
            Seen::Via("decklists tensor (sideboard counts)"),
        ),
        (
            "disclosure",
            Seen::Via("decklists tensor rows, all zero for a closed opponent"),
        ),
        ("known_hand", Seen::Via("known_hand tensor")),
        ("name", Seen::Identity),
        ("life", Seen::Player(player::LIFE, player::IS_ACTIVE)),
        (
            "drew_when_empty",
            Seen::Transient("the player loses at the next state-based check"),
        ),
        ("alive", Seen::Transient("the game ends")),
        (
            "mana_pool",
            Seen::Player(player::MANA_WHITE, player::RESTRICTED_LESSON_WHITE),
        ),
        (
            "combat_mana_pool",
            Seen::Player(player::COMBAT_MANA, player::MANA_WHITE),
        ),
    ]
}

/// Where each `Modification` shows besides its fact rows: the derived
/// column that already applies it. No wildcard: a new variant must be named
/// here and given a witness in [`modification_witnesses`].
fn modification_seen<N>(modification: &Modification<N>) -> (&'static str, Seen) {
    match modification {
        Modification::SetCreatureTypes(_) => (
            "Modification::SetCreatureTypes",
            Seen::Fact(FactKind::AddedSubtype),
        ),
        Modification::AddTypes(_) => ("Modification::AddTypes", Seen::Card(card::POWER)),
        Modification::SetTypes { .. } => ("Modification::SetTypes", Seen::Card(card::POWER)),
        Modification::SetColors(_) => (
            "Modification::SetColors",
            Seen::Fact(FactKind::EffectiveColor),
        ),
        Modification::AddColors(_) => (
            "Modification::AddColors",
            Seen::Fact(FactKind::EffectiveColor),
        ),
        Modification::GrantActivated(_) => (
            "Modification::GrantActivated",
            Seen::Fact(FactKind::EffectProgram),
        ),
        Modification::GrantTriggered(_) => (
            "Modification::GrantTriggered",
            Seen::Fact(FactKind::EffectProgram),
        ),
        Modification::GrantStatic(_) => (
            "Modification::GrantStatic",
            Seen::Fact(FactKind::EffectProgram),
        ),
        Modification::LoseNonManaAbilities => (
            "Modification::LoseNonManaAbilities",
            Seen::Permanent(permanent::ABILITIES_LOST),
        ),
        Modification::SwitchPt => ("Modification::SwitchPt", Seen::Permanent(permanent::POWER)),
        Modification::AddSubtypes(_) => (
            "Modification::AddSubtypes",
            Seen::Fact(FactKind::AddedSubtype),
        ),
        Modification::BecomeCreature => (
            "Modification::BecomeCreature",
            Seen::Permanent(permanent::ANIMATED),
        ),
        Modification::Nonlegendary => {
            ("Modification::Nonlegendary", Seen::Card(card::IS_LEGENDARY))
        }
        Modification::AddKeywords(_) => (
            "Modification::AddKeywords",
            Seen::Via("keyword_columns: effective keywords"),
        ),
        Modification::LoseAllAbilities => (
            "Modification::LoseAllAbilities",
            Seen::Permanent(permanent::ABILITIES_LOST),
        ),
        Modification::Firebending(_) => (
            "Modification::Firebending",
            Seen::Permanent(permanent::GRANTED_FIREBENDING),
        ),
        Modification::SetBasePt { .. } => {
            ("Modification::SetBasePt", Seen::Permanent(permanent::POWER))
        }
        Modification::ModifyPt { .. } => {
            ("Modification::ModifyPt", Seen::Permanent(permanent::POWER))
        }
        Modification::CantBeBlocked => (
            "Modification::CantBeBlocked",
            Seen::Permanent(permanent::CANT_BE_BLOCKED_THIS_TURN),
        ),
        Modification::CantBlock => (
            "Modification::CantBlock",
            Seen::Via("the action tensor omits its blocks"),
        ),
        Modification::DoesntUntap => (
            "Modification::DoesntUntap",
            Seen::Via("its effect fact row and the tapped column after the untap step"),
        ),
        Modification::CantActivateAbilities => (
            "Modification::CantActivateAbilities",
            Seen::Via("its effect fact row; the action tensor omits its activations"),
        ),
        Modification::MustBeBlockedByAllAble => (
            "Modification::MustBeBlockedByAllAble",
            Seen::Via("its effect fact row; the action tensor offers only its blocks"),
        ),
        Modification::LoseKeywords(_) => (
            "Modification::LoseKeywords",
            Seen::Via("keyword_columns: effective keywords"),
        ),
        Modification::ExileIfWouldDie => (
            "Modification::ExileIfWouldDie",
            Seen::Permanent(permanent::EXILE_IF_DIES_THIS_TURN),
        ),
        Modification::ExileIfWouldLeave => (
            "Modification::ExileIfWouldLeave",
            Seen::Via("its shared-effect row (modification id)"),
        ),
        Modification::KeepUnspentMana(_) => (
            "Modification::KeepUnspentMana",
            Seen::Player(player::KEEPS_RED_MANA, player::KEEPS_RED_MANA + 1),
        ),
        Modification::ControlChanged { .. } => (
            "Modification::ControlChanged",
            Seen::Via("the permanent's row moves to its new controller's permanents"),
        ),
        Modification::ModeChosen { .. } => (
            "Modification::ModeChosen",
            Seen::Actions("a chosen mode is no longer offered"),
        ),
        Modification::HasAbilitiesOfSource => (
            "Modification::HasAbilitiesOfSource",
            Seen::Via("its shared-effect row"),
        ),
        Modification::NextCreatureCastFlash => (
            "Modification::NextCreatureCastFlash",
            Seen::Via("its shared-effect row"),
        ),
        Modification::SkipsCombatPhasesNextTurn => (
            "Modification::SkipsCombatPhasesNextTurn",
            Seen::Via("its shared-effect row"),
        ),
        Modification::LifeTotalCantChange => (
            "Modification::LifeTotalCantChange",
            Seen::Via("its shared-effect row; the action tensor omits life payments"),
        ),
        Modification::ProtectionFromEverything => (
            "Modification::ProtectionFromEverything",
            Seen::Via("its shared-effect row; the action tensor omits targets naming the player"),
        ),
        Modification::PhasedOut => (
            "Modification::PhasedOut",
            Seen::Via(
                "its shared-effect row on the permanent's kept row; the action tensor omits it",
            ),
        ),
        Modification::NoMaximumHandSize => (
            "Modification::NoMaximumHandSize",
            Seen::Player(
                player::NO_MAXIMUM_HAND_SIZE,
                player::NO_MAXIMUM_HAND_SIZE + 1,
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
        permanent(Modification::SetCreatureTypes(vec!["Bear".into()])),
        permanent(Modification::AddTypes(vec![CardType::Artifact])),
        permanent(Modification::SetTypes {
            types: vec![CardType::Land],
            subtypes: vec![],
        }),
        permanent(Modification::SetColors([Color::Red].into_iter().collect())),
        permanent(Modification::AddColors([Color::Blue].into_iter().collect())),
        permanent(Modification::GrantActivated(Box::default())),
        permanent(Modification::GrantTriggered(Box::new(Ability::Triggered {
            condition: TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
            intervening_if: None,
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }))),
        permanent(Modification::GrantStatic(Box::new(
            StaticAbility::NoMaximumHandSize,
        ))),
        permanent(Modification::SwitchPt),
        permanent(Modification::LoseNonManaAbilities),
        permanent(Modification::AddSubtypes(vec!["Ally".into()])),
        permanent(Modification::BecomeCreature),
        permanent(Modification::AddKeywords(Keywords::of(&[Keyword::Flying]))),
        permanent(Modification::Nonlegendary),
        permanent(Modification::LoseAllAbilities),
        permanent(Modification::Firebending(2)),
        permanent(Modification::base_pt(Some(4), None)),
        permanent(Modification::pump(2, 1)),
        permanent(Modification::CantBeBlocked),
        permanent(Modification::CantBlock),
        permanent(Modification::DoesntUntap),
        permanent(Modification::CantActivateAbilities),
        permanent(Modification::MustBeBlockedByAllAble),
        permanent(Modification::LoseKeywords(Keywords::of(&[
            Keyword::Hexproof,
        ]))),
        permanent(Modification::ExileIfWouldDie),
        permanent(Modification::ExileIfWouldLeave),
        (true, Modification::KeepUnspentMana(Color::Red)),
        (true, Modification::NoMaximumHandSize),
        (true, Modification::NextCreatureCastFlash),
        (true, Modification::SkipsCombatPhasesNextTurn),
        (true, Modification::LifeTotalCantChange),
        (true, Modification::ProtectionFromEverything),
        permanent(Modification::PhasedOut),
        permanent(Modification::ControlChanged {
            previous: PlayerId(1),
        }),
        permanent(Modification::ModeChosen { mode: 1 }),
        permanent(Modification::HasAbilitiesOfSource),
    ]
}

const MODIFICATION_VARIANTS: usize = 36;

/// The fact-row duration code of each `EffectDuration`; no wildcard.
fn duration_code(duration: EffectDuration) -> f32 {
    match duration {
        EffectDuration::EndOfTurn => 1.0,
        EffectDuration::Indefinite => 2.0,
        EffectDuration::WhileYouControlSource => 3.0,
        EffectDuration::UntilYourNextTurn => 4.0,
    }
}

/// The token each `ObjectSelector` contributes; no wildcard.
fn selector_token(selector: &ObjectSelector) -> &'static str {
    match selector {
        ObjectSelector::This => "ObjectSelector::This",
        ObjectSelector::Target(_) => "ObjectSelector::Target",
        ObjectSelector::Targets(_) => "ObjectSelector::Targets",
        ObjectSelector::Triggering => "ObjectSelector::Triggering",
        ObjectSelector::Attached => "ObjectSelector::Attached",
        ObjectSelector::CreatedThisWay => "ObjectSelector::CreatedThisWay",
        ObjectSelector::Remembered => "ObjectSelector::Remembered",
        ObjectSelector::Grantor => "ObjectSelector::Grantor",
        ObjectSelector::Moved => "ObjectSelector::Moved",
        ObjectSelector::CardsThisWay => "ObjectSelector::CardsThisWay",
        ObjectSelector::Selected => "ObjectSelector::Selected",
        ObjectSelector::Zone(_) => "ObjectSelector::Zone",
        ObjectSelector::TargetPlayersZone { .. } => "ObjectSelector::TargetPlayersZone",
        ObjectSelector::You => "ObjectSelector::You",
        ObjectSelector::Chosen { .. } => "ObjectSelector::Chosen",
        ObjectSelector::Each { .. } => "ObjectSelector::Each",
    }
}

fn selector_witnesses() -> [ObjectSelector; 16] {
    [
        ObjectSelector::This,
        ObjectSelector::Target(TargetSpec::Creature),
        ObjectSelector::Targets(TargetSpec::Creature),
        ObjectSelector::Triggering,
        ObjectSelector::Attached,
        ObjectSelector::CreatedThisWay,
        ObjectSelector::Remembered,
        ObjectSelector::Grantor,
        ObjectSelector::Moved,
        ObjectSelector::CardsThisWay,
        ObjectSelector::Selected,
        ObjectSelector::Zone(ZoneType::Hand),
        ObjectSelector::TargetPlayersZone {
            spec: Box::new(TargetSpec::Opponent),
            zone: ZoneType::Library,
        },
        ObjectSelector::You,
        ObjectSelector::another_chosen(CardPredicate::creature()),
        ObjectSelector::Each {
            of: Scope::YouControl,
            predicate: CardPredicate::creature(),
            except: None,
            max_mana_value: Some(Box::new(EffectValue::X)),
        },
    ]
}

/// The token each `Scope` of an `Each` selector contributes; no wildcard.
fn scope_token(scope: Scope) -> &'static str {
    match scope {
        Scope::YouControl => "Scope::YouControl",
        Scope::OpponentsControl => "Scope::OpponentsControl",
        Scope::Battlefield => "Scope::Battlefield",
        Scope::TargetPlayerControls => "Scope::TargetPlayerControls",
    }
}

/// The token each `Except` contributes; no wildcard.
fn except_token(except: Except) -> &'static str {
    match except {
        Except::This => "Except::This",
        Except::Target => "Except::Target",
    }
}

/// The token each `PlayerSelector` contributes; no wildcard.
fn player_selector_token(player: &PlayerSelector) -> &'static str {
    match player {
        PlayerSelector::You => "PlayerSelector::You",
        PlayerSelector::EachOpponent => "PlayerSelector::EachOpponent",
        PlayerSelector::TriggeringPlayer => "PlayerSelector::TriggeringPlayer",
        PlayerSelector::TriggeringPlayersOpponents => "PlayerSelector::TriggeringPlayersOpponents",
        PlayerSelector::Target(_) => "PlayerSelector::Target",
        PlayerSelector::TargetController => "PlayerSelector::TargetController",
        PlayerSelector::Owner => "PlayerSelector::Owner",
    }
}

fn player_selector_witnesses() -> [PlayerSelector; 7] {
    [
        PlayerSelector::You,
        PlayerSelector::EachOpponent,
        PlayerSelector::TriggeringPlayer,
        PlayerSelector::TriggeringPlayersOpponents,
        PlayerSelector::Target(Box::new(TargetSpec::Player)),
        PlayerSelector::TargetController,
        PlayerSelector::Owner,
    ]
}

/// Turn-scoped state. Per-player turn history reaches each player's row.
fn turn_fields(turn: &TurnState) -> [(&'static str, Seen); 24] {
    let TurnState {
        player_controls: _,
        active_player: _,
        turn_number: _,
        lands_played: _,
        spells_cast: _,
        history,
        graveyard_from_battlefield: _,
        ability_triggers_this_turn: _,
        ability_resolutions_this_turn: _,
        extra_turns: _,
        resume_with: _,
        additional_combats: _,
        combat_attackers: _,
        current_phase: _,
        current_step: _,
        step_initialized: _,
        turn_based_actions_complete: _,
        cleanup_priority: _,
    } = turn;
    let PlayerTurnHistory {
        attacked: _,
        permanents_sacrificed: _,
        land_entered: _,
        cards_drawn: _,
        bent: _,
        life_lost: _,
    } = history[0];
    [
        ("player_controls", Seen::Via("v5 relations kind 13: controller, affected player, duration, active/pending, creation order")),
        ("active_player", Seen::Player(player::IS_ACTIVE, player::LIBRARY_COUNT)),
        (
            "turn_number",
            Seen::Player(player::TURN, player::TURN + 1),
        ),
        (
            "lands_played",
            Seen::Via("the land-play action is offered only while one remains"),
        ),
        (
            "spells_cast",
            Seen::Via("input v10 player_turn_state column 1 (PlayerData.spells_cast)"),
        ),
        (
            "history",
            Seen::Via("each player's history columns, listed next"),
        ),
        (
            "extra_turns",
            Seen::Player(player::EXTRA_TURNS, player::EXTRA_TURNS + 1),
        ),
        (
            "additional_combats",
            Seen::Player(
                player::ADDITIONAL_COMBATS,
                player::ADDITIONAL_COMBATS + 1,
            ),
        ),
        (
            "combat_attackers",
            Seen::Actions("attack offers omit creatures the additional combat excludes"),
        ),
        (
            "resume_with",
            Seen::Via("the player after the interrupted turn: the extra-turn columns and the active player"),
        ),
        (
            "history.cards_drawn",
            Seen::Player(player::CARDS_DRAWN, player::CARDS_DRAWN + 1),
        ),
        (
            "history.bent",
            Seen::Player(player::BENT_WATER, player::BENT_WATER + 4),
        ),
        (
            "history.life_lost",
            Seen::Via("input v10 player_turn_state column 0; v9 refuses it once a visible card reads it"),
        ),
        (
            "graveyard_from_battlefield",
            Seen::Via("input v10 card_turn_state of each graveyard card row (CardData.from_battlefield_this_turn)"),
        ),
        (
            "history.attacked",
            Seen::Player(player::ATTACKED, player::ATTACKED + 1),
        ),
        (
            "history.permanents_sacrificed",
            Seen::Player(player::PERMANENTS_SACRIFICED, player::PERMANENTS_SACRIFICED + 1),
        ),
        (
            "history.land_entered",
            Seen::Player(player::LAND_ENTERED, player::LAND_ENTERED + 1),
        ),
        (
            "ability_triggers_this_turn",
            Seen::Via("used once-each-turn limits on permanent rows"),
        ),
        (
            "ability_resolutions_this_turn",
            Seen::Transient("read only by the resolving ability's own Nth-resolution check"),
        ),
        ("current_phase", Seen::Player(player::PHASE_BEGINNING, player::STEP_UNTAP)),
        ("current_step", Seen::Player(player::STEP_UNTAP, player::GRAVEYARD_LESSONS)),
        ("step_initialized", Seen::Identity),
        ("turn_based_actions_complete", Seen::Identity),
        ("cleanup_priority", Seen::Identity),
    ]
}

/// The token every `Effect` variant contributes. No wildcard: a new variant
/// must be named here (and given a witness below).
fn effect_token(effect: &Effect) -> &'static str {
    match effect {
        Effect::CreateTokenCopy { .. } => "Effect::CreateTokenCopy",
        Effect::TransformSource => "Effect::TransformSource",
        Effect::ControlPlayer { .. } => "Effect::ControlPlayer",
        Effect::TakeExtraTurn => "Effect::TakeExtraTurn",
        Effect::EndTheTurn => "Effect::EndTheTurn",
        Effect::AdditionalCombatPhase { .. } => "Effect::AdditionalCombatPhase",
        Effect::EachPlayerChoosesToExile { .. } => "Effect::EachPlayerChoosesToExile",
        Effect::NameAndExileFromTargetPlayer { .. } => "Effect::NameAndExileFromTargetPlayer",
        Effect::ExileNamed { .. } => "Effect::ExileNamed",
        Effect::OptionalManaPaymentX { .. } => "Effect::OptionalManaPaymentX",
        Effect::MayExileTriggeringWithSource => "Effect::MayExileTriggeringWithSource",
        Effect::ChooseExiledWithSource { .. } => "Effect::ChooseExiledWithSource",
        Effect::HaveAbilitiesOf { .. } => "Effect::HaveAbilitiesOf",
        Effect::ApplyContinuous { .. } => "Effect::ApplyContinuous",
        Effect::OptionalPayment { .. } => "Effect::OptionalPayment",
        Effect::Sacrifice { .. } => "Effect::Sacrifice",
        Effect::DealDamageToPlayer { .. } => "Effect::DealDamageToPlayer",
        Effect::DrainLife { .. } => "Effect::DrainLife",
        Effect::MillCards { .. } => "Effect::MillCards",
        Effect::SelectCards { .. } => "Effect::SelectCards",
        Effect::TargetClause { .. } => "Effect::TargetClause",
        Effect::SelectTargetRequirement { .. } => "Effect::SelectTargetRequirement",
        Effect::Destroy { .. } => "Effect::Destroy",
        Effect::Discard { .. } => "Effect::Discard",
        Effect::If { .. } => "Effect::If",
        Effect::DealDamage { .. } => "Effect::DealDamage",
        Effect::CounterSpell { .. } => "Effect::CounterSpell",
        Effect::DrawCards { .. } => "Effect::DrawCards",
        Effect::LoseLife { .. } => "Effect::LoseLife",
        Effect::CreateToken { .. } => "Effect::CreateToken",
        Effect::PutCounters { .. } => "Effect::PutCounters",
        Effect::PutCountersFrom { .. } => "Effect::PutCountersFrom",
        Effect::Move { .. } => "Effect::Move",
        Effect::Tap { .. } => "Effect::Tap",
        Effect::Untap { .. } => "Effect::Untap",
        Effect::RemoveAllCounters { .. } => "Effect::RemoveAllCounters",
        Effect::GainLife { .. } => "Effect::GainLife",
        Effect::OnNthResolutionThisTurn { .. } => "Effect::OnNthResolutionThisTurn",
        Effect::Scry { .. } => "Effect::Scry",
        Effect::Surveil { .. } => "Effect::Surveil",
        Effect::LookAndSelect { .. } => "Effect::LookAndSelect",
        Effect::PutTopCardsInHand { .. } => "Effect::PutTopCardsInHand",
        Effect::Learn => "Effect::Learn",
        Effect::Modal { .. } => "Effect::Modal",
        Effect::CounterUnlessPays { .. } => "Effect::CounterUnlessPays",
        Effect::TargetCreaturesDealPowerDamageToLastTarget => {
            "Effect::TargetCreaturesDealPowerDamageToLastTarget"
        }
        Effect::Earthbend { .. } => "Effect::Earthbend",
        Effect::AddMana { .. } => "Effect::AddMana",
        Effect::ForEachTarget { .. } => "Effect::ForEachTarget",
        Effect::Airbend { .. } => "Effect::Airbend",
        Effect::OpponentsCastOnlyFromHand => "Effect::OpponentsCastOnlyFromHand",
        Effect::ExileTopCardsToPlay { .. } => "Effect::ExileTopCardsToPlay",
        Effect::ExileLinked { .. } => "Effect::ExileLinked",
        Effect::ExileUntilNonlandCastFree { .. } => "Effect::ExileUntilNonlandCastFree",
        Effect::Cascade => "Effect::Cascade",
        Effect::CumulativeUpkeep { .. } => "Effect::CumulativeUpkeep",
        Effect::CastFromHandFree { .. } => "Effect::CastFromHandFree",
        Effect::ShuffleLibrary => "Effect::ShuffleLibrary",
        Effect::ShuffleGraveyardIntoLibrary => "Effect::ShuffleGraveyardIntoLibrary",
        Effect::OwnerPutsIntoLibrary { .. } => "Effect::OwnerPutsIntoLibrary",
        Effect::RevealHandDiscardChosen { .. } => "Effect::RevealHandDiscardChosen",
        Effect::SearchDistinctNames { .. } => "Effect::SearchDistinctNames",
        Effect::PlaceTargetInLibrary { .. } => "Effect::PlaceTargetInLibrary",
        Effect::MayOnceEachTurn { .. } => "Effect::MayOnceEachTurn",
        Effect::CastLibraryTopFree => "Effect::CastLibraryTopFree",
        Effect::WhenYouDo { .. } => "Effect::WhenYouDo",
        Effect::TriggerModes { .. } => "Effect::TriggerModes",
        Effect::AttachSourceTo { .. } => "Effect::AttachSourceTo",
        Effect::DistributeCountersToTappedOpponentCreatures { .. } => {
            "Effect::DistributeCountersToTappedOpponentCreatures"
        }
        Effect::CreateDelayedTrigger { .. } => "Effect::CreateDelayedTrigger",
        Effect::ForPlayer { .. } => "Effect::ForPlayer",
        Effect::RestoreController(_) => "Effect::RestoreController",
        Effect::EachPlayerKeepsCreatures { .. } => "Effect::EachPlayerKeepsCreatures",
        Effect::GainControl { .. } => "Effect::GainControl",
        Effect::CopyTriggeringObject => "Effect::CopyTriggeringObject",
        Effect::CopyTriggeringSpellForEach { .. } => "Effect::CopyTriggeringSpellForEach",
        Effect::ChangeTarget { .. } => "Effect::ChangeTarget",
        Effect::CounterAbility { .. } => "Effect::CounterAbility",
        Effect::CastExiledWithSource { .. } => "Effect::CastExiledWithSource",
        Effect::AddOneColorManaValue { .. } => "Effect::AddOneColorManaValue",
        Effect::FightLastTarget => "Effect::FightLastTarget",
        Effect::RemoveCounters { .. } => "Effect::RemoveCounters",
    }
}

/// Number of `Effect` variants named in [`effect_token`].
const EFFECT_VARIANTS: usize = 82;

/// One value of every `Effect` variant.
fn effect_witnesses() -> Vec<Effect> {
    let creature = CardPredicate::creature;
    let target = || TargetSpec::Creature;
    let value = || EffectValue::CreaturesWithCounters;
    let draw = || {
        vec![Effect::DrawCards {
            count: EffectValue::Fixed(1),
        }]
    };
    vec![
        Effect::CreateTokenCopy {
            of: ObjectSelector::Target(TargetSpec::Creature),
            exceptions: vec![Modification::Nonlegendary],
        },
        Effect::until_end_of_turn(ObjectSelector::This, Modification::pump(1, 1)),
        Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne, 1),
        Effect::PutCountersFrom {
            from: ObjectSelector::Triggering,
            on: ObjectSelector::This,
        },
        Effect::may_sacrifice(CardPredicate::creature(), true),
        Effect::exile_linked(
            ObjectSelector::Target(target()),
            LinkedExile::UntilSourceLeaves,
        ),
        Effect::tap(ObjectSelector::Attached),
        Effect::untap(ObjectSelector::This),
        Effect::RemoveAllCounters {
            from: ObjectSelector::Attached,
        },
        Effect::RemoveCounters {
            from: ObjectSelector::Target(target()),
            up_to: 2,
            chosen: None,
        },
        Effect::OptionalPayment {
            cost: PaymentCost::Waterbend { generic: 2 },
            if_paid: draw(),
            if_declined: vec![Effect::discard()],
        },
        Effect::DealDamageToPlayer {
            player: PlayerSelector::You,
            amount: EffectValue::Fixed(1),
        },
        Effect::MillCards {
            count: EffectValue::Fixed(2),
            target: None,
        },
        Effect::SelectCards {
            from: ObjectSelector::CardsThisWay,
            predicate: creature(),
            min: 0,
            max: EffectValue::Fixed(2),
            effects: draw(),
            if_declined: draw(),
        },
        Effect::DrainLife { amount: value() },
        Effect::TargetClause {
            requirement: TargetRequirement::one(target()),
            effects: draw(),
        },
        Effect::SelectTargetRequirement { index: 0 },
        Effect::destroy(ObjectSelector::Target(target())),
        Effect::Discard {
            what: ObjectSelector::Target(TargetSpec::Opponent),
            count: 2,
        },
        Effect::If {
            condition: Condition::YourTurn,
            then: draw(),
            otherwise: vec![],
        },
        Effect::deal_damage(value(), ObjectSelector::Target(target())),
        Effect::CounterSpell {
            what: ObjectSelector::Target(TargetSpec::Spell),
            exile: true,
        },
        Effect::DrawCards { count: value() },
        Effect::LoseLife {
            player: PlayerSelector::TriggeringPlayer,
            amount: EffectValue::Fixed(2),
        },
        Effect::create_tokens("token.clue", value()),
        Effect::gain_life(value()),
        Effect::OnNthResolutionThisTurn {
            n: 2,
            effect: Box::new(Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }),
        },
        Effect::Scry { count: 1 },
        Effect::Surveil { count: 1 },
        Effect::LookAndSelect {
            to_battlefield: true,
            random_order: true,
            reveal: false,
            look: EffectValue::Fixed(3),
            min_select: 0,
            max_select: 1,
            predicate: creature(),
        },
        Effect::PutTopCardsInHand { count: 1 },
        Effect::Learn,
        Effect::Modal {
            modes: vec![draw(), vec![Effect::discard()]],
        },
        Effect::CounterUnlessPays {
            cost: PaymentCost::Mana(ManaCost::parse("{2}")),
        },
        Effect::TargetCreaturesDealPowerDamageToLastTarget,
        Effect::Earthbend {
            count: EffectValue::Fixed(1),
            what: ObjectSelector::Target(TargetSpec::LandYouControl),
        },
        Effect::AddMana {
            mana: Mana::parse("R"),
            amount: EffectValue::ExcessDamage,
            until_end_of_combat: true,
        },
        Effect::ForEachTarget { effects: draw() },
        Effect::Airbend {
            what: ObjectSelector::Each {
                of: Scope::Battlefield,
                predicate: CardPredicate::creature(),
                except: Some(Except::Target),
                max_mana_value: None,
            },
        },
        Effect::OpponentsCastOnlyFromHand,
        Effect::ExileTopCardsToPlay {
            count: 3,
            until: PlayWindow::ThroughYourNextTurn,
        },
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
        Effect::Cascade,
        Effect::CumulativeUpkeep {
            each: ManaCost::parse("{1}"),
        },
        Effect::CastFromHandFree {
            predicate: CardPredicate {
                not_card_types: vec![CardType::Creature, CardType::Land],
                ..Default::default()
            },
        },
        Effect::ShuffleLibrary,
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
        Effect::SearchDistinctNames {
            predicate: CardPredicate::subtype("Aura"),
            count: 3,
            end: DistinctSearchEnd::Hand,
        },
        Effect::PlaceTargetInLibrary {
            second_from_top: true,
        },
        Effect::MayOnceEachTurn { effects: draw() },
        Effect::CastLibraryTopFree,
        Effect::WhenYouDo { effects: draw() },
        Effect::TriggerModes {
            up_to: None,
            unchosen: false,
            modes: vec![draw(), vec![Effect::Scry { count: 1 }]],
        },
        Effect::AttachSourceTo {
            what: ObjectSelector::Target(TargetSpec::CreatureYouControl),
        },
        Effect::DistributeCountersToTappedOpponentCreatures {
            kind: CounterKind::Stun,
            count: 3,
        },
        Effect::CreateDelayedTrigger {
            remember: None,
            event: DelayedEvent::CreatureDies,
            frequency: DelayedFrequency::EachMatch,
            expiry: DelayedExpiry::ThisTurn,
            effects: vec![Effect::gain_life(1)],
        },
        Effect::ForPlayer {
            player: PlayerSelector::TargetController,
            effects: vec![Effect::discard()],
        },
        Effect::RestoreController(PlayerId(0)),
        Effect::EachPlayerKeepsCreatures { max_total_power: 4 },
        Effect::TransformSource,
        Effect::ControlPlayer {
            what: ObjectSelector::Target(TargetSpec::Opponent),
            duration: managym::state::ability::PlayerControlDuration::NextCombat,
        },
        Effect::TakeExtraTurn,
        Effect::EndTheTurn,
        Effect::AdditionalCombatPhase {
            attackers: Some(CardPredicate::creature()),
        },
        Effect::EachPlayerChoosesToExile {
            predicate: CardPredicate::creature(),
        },
        Effect::NameAndExileFromTargetPlayer { count: 4 },
        Effect::ExileNamed {
            name: "Gray Ogre".into(),
            count: 4,
        },
        Effect::OptionalManaPaymentX {
            if_paid: vec![Effect::discard()],
        },
        Effect::MayExileTriggeringWithSource,
        Effect::ChooseExiledWithSource {
            predicate: CardPredicate::creature(),
        },
        Effect::HaveAbilitiesOf {
            definition: managym::state::card::CardDefId(0),
        },
        Effect::GainControl {
            player: PlayerSelector::You,
            what: ObjectSelector::Target(TargetSpec::Creature),
            duration: EffectDuration::EndOfTurn,
        },
        Effect::CopyTriggeringObject,
        Effect::CopyTriggeringSpellForEach {
            each: ObjectSelector::each_other_you_control(CardPredicate::creature()),
        },
        Effect::CastExiledWithSource {
            predicate: CardPredicate::subtype("Ally"),
        },
        Effect::ChangeTarget {
            target: TargetSpec::SpellOrAbilityWithSingleTarget,
        },
        Effect::CounterAbility {
            what: ObjectSelector::Target(TargetSpec::Ability),
        },
        Effect::AddOneColorManaValue {
            amount: EffectValue::GreatestSharedCreatureType,
        },
        Effect::FightLastTarget,
    ]
}

/// The token every triggered-ability condition contributes.
fn trigger_token(condition: &TriggerCondition) -> &'static str {
    match condition {
        TriggerCondition::WasCast { .. } => "TriggerCondition::WasCast",
        TriggerCondition::Sacrificed { .. } => "TriggerCondition::Sacrificed",
        TriggerCondition::OnceEachTurn { .. } => "TriggerCondition::OnceEachTurn",
        TriggerCondition::YouCastThis => "TriggerCondition::YouCastThis",
        TriggerCondition::NoCounters { .. } => "TriggerCondition::NoCounters",
        TriggerCondition::SpellTargetsOnlyThis { .. } => "TriggerCondition::SpellTargetsOnlyThis",
        TriggerCondition::FromZone { .. } => "TriggerCondition::FromZone",
        TriggerCondition::EntersTheBattlefield { .. } => "TriggerCondition::EntersTheBattlefield",
        TriggerCondition::Dies { .. } => "TriggerCondition::Dies",
        TriggerCondition::Attacks { .. } => "TriggerCondition::Attacks",
        TriggerCondition::BecomesTapped { .. } => "TriggerCondition::BecomesTapped",
        TriggerCondition::TappedForMana { .. } => "TriggerCondition::TappedForMana",
        TriggerCondition::BecomesTargeted { .. } => "TriggerCondition::BecomesTargeted",
        TriggerCondition::PutsCounters { .. } => "TriggerCondition::PutsCounters",
        TriggerCondition::LeavesTheBattlefield { .. } => "TriggerCondition::LeavesTheBattlefield",
        TriggerCondition::ActiveIf { .. } => "TriggerCondition::ActiveIf",
        TriggerCondition::InGraveyard { .. } => "TriggerCondition::InGraveyard",
        TriggerCondition::YourAttackingCreatureTriggers => {
            "TriggerCondition::YourAttackingCreatureTriggers"
        }
        TriggerCondition::Chapter { .. } => "TriggerCondition::Chapter",
        TriggerCondition::Casts { .. } => "TriggerCondition::Casts",
        TriggerCondition::Draws { .. } => "TriggerCondition::Draws",
        TriggerCondition::BeginningOf { .. } => "TriggerCondition::BeginningOf",
        TriggerCondition::Performs { .. } => "TriggerCondition::Performs",
        TriggerCondition::PutIntoGraveyard { .. } => "TriggerCondition::PutIntoGraveyard",
        TriggerCondition::DealsCombatDamage { .. } => "TriggerCondition::DealsCombatDamage",
        TriggerCondition::WhileSourceAttacking { .. } => "TriggerCondition::WhileSourceAttacking",
    }
}

fn trigger_witnesses() -> Vec<TriggerCondition> {
    let this = || TriggerSubject::This;
    let enters = || Box::new(TriggerCondition::EntersTheBattlefield { subject: this() });
    vec![
        TriggerCondition::WasCast {
            condition: enters(),
        },
        TriggerCondition::Casts {
            who: Actor::Opponent,
            predicate: CardPredicate::creature(),
            nth: Some(2),
        },
        TriggerCondition::Draws {
            who: Actor::AnyPlayer,
            nth: None,
        },
        TriggerCondition::beginning_of(TurnStep::Upkeep, Actor::You),
        TriggerCondition::Performs {
            who: Actor::Opponent,
            any_of: vec![KeywordAction::Search],
        },
        TriggerCondition::PutIntoGraveyard {
            whose: Actor::Opponent,
            predicate: CardPredicate::default(),
        },
        TriggerCondition::DealsCombatDamage {
            subject: TriggerSubject::any(CardPredicate::creature()),
            to: Actor::You,
            per: Per::Batch,
        },
        TriggerCondition::Sacrificed {
            subject: TriggerSubject::another_you_control(CardPredicate::creature()),
        },
        TriggerCondition::OnceEachTurn {
            condition: enters(),
        },
        TriggerCondition::EntersTheBattlefield { subject: this() },
        TriggerCondition::Dies { subject: this() },
        TriggerCondition::attacks(this(), Per::Object),
        TriggerCondition::BecomesTapped { subject: this() },
        TriggerCondition::TappedForMana { subject: this() },
        TriggerCondition::BecomesTargeted { subject: this() },
        TriggerCondition::PutsCounters {
            who: Actor::You,
            kind: CounterKind::PlusOnePlusOne,
            on: CardPredicate::creature(),
        },
        TriggerCondition::LeavesTheBattlefield {
            subject: TriggerSubject::This,
        },
        TriggerCondition::ActiveIf {
            active_if: Condition::YourTurn,
            condition: enters(),
        },
        TriggerCondition::SpellTargetsOnlyThis {
            predicate: CardPredicate::creature(),
        },
        TriggerCondition::FromZone {
            zone: ZoneType::Exile,
            condition: enters(),
        },
        TriggerCondition::InGraveyard {
            condition: Box::new(TriggerCondition::Draws {
                who: Actor::You,
                nth: Some(2),
            }),
        },
        TriggerCondition::YourAttackingCreatureTriggers,
        TriggerCondition::Chapter { chapter: 2 },
        TriggerCondition::YouCastThis,
        TriggerCondition::NoCounters {
            kind: CounterKind::Ice,
        },
        TriggerCondition::WhileSourceAttacking {
            condition: Box::new(TriggerCondition::casts(
                Actor::You,
                CardPredicate::default(),
            )),
        },
    ]
}

fn ability_token(ability: &Ability) -> &'static str {
    match ability {
        Ability::Triggered { .. } => "Ability::Triggered",
    }
}

/// The token every `Condition` variant contributes. No wildcard.
fn condition_token(condition: &Condition) -> &'static str {
    match condition {
        Condition::YourTurn => "Condition::YourTurn",
        Condition::Always => "Condition::Always",
        Condition::CardsDrawnAtLeast { .. } => "Condition::CardsDrawnAtLeast",
        Condition::GraveyardAtLeast { .. } => "Condition::GraveyardAtLeast",
        Condition::AnyGraveyardAtLeast { .. } => "Condition::AnyGraveyardAtLeast",
        Condition::ControlsAtLeast { .. } => "Condition::ControlsAtLeast",
        Condition::ControlsUnownedAtLeast { .. } => "Condition::ControlsUnownedAtLeast",
        Condition::CardsInHandAtLeast { .. } => "Condition::CardsInHandAtLeast",
        Condition::UnspentManaAtLeast { .. } => "Condition::UnspentManaAtLeast",
        Condition::AttackedThisTurn => "Condition::AttackedThisTurn",
        Condition::BentAllFourThisTurn => "Condition::BentAllFourThisTurn",
        Condition::LandEnteredThisTurn => "Condition::LandEnteredThisTurn",
        Condition::SacrificedThisTurnAtLeast { .. } => "Condition::SacrificedThisTurnAtLeast",
        Condition::OpponentLostLifeThisTurnAtLeast { .. } => {
            "Condition::OpponentLostLifeThisTurnAtLeast"
        }
        Condition::AnyOf(_) => "Condition::AnyOf",
        Condition::Kicked => "Condition::Kicked",
        Condition::TargetMatches(_) => "Condition::TargetMatches",
        Condition::TargetDealtDamageThisTurn => "Condition::TargetDealtDamageThisTurn",
        Condition::SourceCountersAtLeast { .. } => "Condition::SourceCountersAtLeast",
        Condition::SourceMatches(_) => "Condition::SourceMatches",
        Condition::SourceTapped => "Condition::SourceTapped",
        Condition::TargetCardExiled(_) => "Condition::TargetCardExiled",
        Condition::TriggeringHadCounters => "Condition::TriggeringHadCounters",
        Condition::CardsThisWayMatch(_) => "Condition::CardsThisWayMatch",
        Condition::TriggeringSpellMatches(_) => "Condition::TriggeringSpellMatches",
        Condition::Moved => "Condition::Moved",
        Condition::TargetControlledByYou => "Condition::TargetControlledByYou",
    }
}

/// One value of every `Condition` variant.
fn condition_witnesses() -> Vec<Condition> {
    vec![
        Condition::YourTurn,
        Condition::Always,
        Condition::CardsDrawnAtLeast { count: 2 },
        Condition::GraveyardAtLeast {
            count: 1,
            predicate: CardPredicate::creature(),
        },
        Condition::AnyGraveyardAtLeast { count: 20 },
        Condition::ControlsAtLeast {
            count: 1,
            predicate: CardPredicate::creature(),
        },
        Condition::ControlsUnownedAtLeast { count: 3 },
        Condition::CardsInHandAtLeast { count: 4 },
        Condition::UnspentManaAtLeast { count: 3 },
        Condition::AttackedThisTurn,
        Condition::LandEnteredThisTurn,
        Condition::BentAllFourThisTurn,
        Condition::SacrificedThisTurnAtLeast { count: 2 },
        Condition::OpponentLostLifeThisTurnAtLeast { amount: 2 },
        Condition::AnyOf(vec![Condition::YourTurn]),
        Condition::Kicked,
        Condition::TargetMatches(CardPredicate::creature()),
        Condition::TargetDealtDamageThisTurn,
        Condition::SourceCountersAtLeast {
            kind: CounterKind::Quest,
            count: 4,
        },
        Condition::SourceTapped,
        Condition::SourceMatches(CardPredicate::subtype("Scout")),
        Condition::TargetCardExiled(CardPredicate::creature()),
        Condition::TriggeringHadCounters,
        Condition::CardsThisWayMatch(CardPredicate::creature()),
        Condition::TriggeringSpellMatches(CardPredicate::subtype("Lesson")),
        Condition::Moved,
        Condition::TargetControlledByYou,
    ]
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
        StaticAbility::Continuous(_) => witnessed(
            "StaticAbility::Continuous",
            Seen::Permanent(permanent::POWER),
            "Beorn the Fierce",
            "hob",
            "beorn_adds_bear_and_keeps_trample_after_kellan_replaces_types",
        ),
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
            "token.spirit",
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
        StaticAbility::OpponentsCastOnlyFromHand => witnessed(
            "StaticAbility::OpponentsCastOnlyFromHand",
            Seen::Actions("opponents' casts from outside their hands are not offered"),
            "Drannith Magistrate",
            "iko",
            "drannith_magistrate_stops_opponents_casting_from_the_graveyard",
        ),
        StaticAbility::LegendRuleDoesntApply => witnessed(
            "StaticAbility::LegendRuleDoesntApply",
            Seen::Actions("no legend-rule choice is asked of its controller"),
            "Sakashima of a Thousand Faces",
            "cmr",
            "sakashima_copies_a_legend_you_control_and_both_stay",
        ),
        StaticAbility::AdditionalLandPlayEachPlayer => witnessed(
            "StaticAbility::AdditionalLandPlayEachPlayer",
            Seen::Actions("a second land play is offered"),
            "Rites of Flourishing",
            "fut",
            "rites_of_flourishing_draws_for_each_player_and_grants_a_land_play",
        ),
        StaticAbility::ActivatedAbilitiesCostLess { .. } => witnessed(
            "StaticAbility::ActivatedAbilitiesCostLess",
            Seen::Actions("the activation becomes affordable at the reduced cost"),
            "Training Grounds",
            "roe",
            "training_grounds_reduces_creature_activations_to_no_less_than_one_mana",
        ),
        StaticAbility::CantBeCountered => witnessed(
            "StaticAbility::CantBeCountered",
            Seen::Via("the spell's learning tokens; a counter leaves it on the stack"),
            "Rending Volley",
            "dtk",
            "rending_volley_targets_white_or_blue_creatures_and_cannot_be_countered",
        ),
        StaticAbility::CostsLessPerPermanentOnBattlefield(_) => witnessed(
            "StaticAbility::CostsLessPerPermanentOnBattlefield",
            Seen::Actions("the cast becomes affordable at the reduced cost"),
            "Blasphemous Act",
            "isd",
            "blasphemous_act_costs_one_less_per_creature_on_the_battlefield",
        ),
        StaticAbility::CostsLessPerGraveyardCard(_) => witnessed(
            "StaticAbility::CostsLessPerGraveyardCard",
            Seen::Actions("the cast becomes affordable at the reduced cost"),
            "Serpent of the Pass",
            "tla",
            "serpent_of_the_pass_is_cheaper_per_graveyard_spell_and_flashes_with_three_lessons",
        ),
        StaticAbility::CostsLessByGreatestPower(_) => witnessed(
            "StaticAbility::CostsLessByGreatestPower",
            Seen::Actions("the cast becomes affordable at the reduced cost"),
            "The Great Henge",
            "eld",
            "the_great_henge_costs_less_by_greatest_power_and_rewards_nontoken_creatures",
        ),
        StaticAbility::FlashIf(_) => witnessed(
            "StaticAbility::FlashIf",
            Seen::Actions("the cast is offered at instant timing"),
            "Serpent of the Pass",
            "tla",
            "serpent_of_the_pass_is_cheaper_per_graveyard_spell_and_flashes_with_three_lessons",
        ),
        StaticAbility::Flashback(_) => witnessed(
            "StaticAbility::Flashback",
            Seen::Actions("the graveyard card is offered as a cast"),
            "Rockalanche",
            "tla",
            "rockalanche_earthbends_by_forests_you_control_and_flashes_back",
        ),
        StaticAbility::Foretell(_) => witnessed(
            "StaticAbility::Foretell",
            Seen::Actions("foretelling is offered from hand; then the foretold permission columns"),
            "Sozin's Comet",
            "tla",
            "sozins_comet_foretells_face_down_and_casts_on_a_later_turn",
        ),
        StaticAbility::NoMaximumHandSize => witnessed(
            "StaticAbility::NoMaximumHandSize",
            Seen::Player(player::HAND_COUNT, player::BATTLEFIELD_COUNT),
            "Proft's Eidetic Memory",
            "mkm",
            "proft_hand_limit_only_while_controlled_and_legend_rule_for_both_card_types",
        ),
        StaticAbility::PtBuff(_) => witnessed(
            "StaticAbility::PtBuff",
            Seen::Permanent(permanent::POWER),
            "First-Time Flyer",
            "tla",
            "first_time_flyer_conditional_static_recomputes_on_read",
        ),
        StaticAbility::CharacteristicPower(_) => witnessed(
            "StaticAbility::CharacteristicPower",
            Seen::Permanent(permanent::POWER),
            "Suki, Kyoshi Warrior",
            "tla",
            "suki_power_tracks_creature_count_and_attack_makes_a_token",
        ),
        StaticAbility::Ward(_) => witnessed(
            "StaticAbility::Ward",
            Seen::Card(card::WARD_COST),
            "Waterfall Aerialist",
            "stx",
            "ward_asks_spell_controller_and_counters_on_decline",
        ),
        StaticAbility::LookAtLibraryTop => witnessed(
            "StaticAbility::LookAtLibraryTop",
            Seen::Via("the controller's library top is a visible CardData row"),
            "Hakoda, Selfless Commander",
            "tla",
            "hakoda_reveals_the_top_card_to_you_and_casts_allies_from_it",
        ),
        StaticAbility::CastFromLibraryTop(_) => witnessed(
            "StaticAbility::CastFromLibraryTop",
            Seen::Actions("the library's top card is offered as a cast"),
            "Hakoda, Selfless Commander",
            "tla",
            "hakoda_reveals_the_top_card_to_you_and_casts_allies_from_it",
        ),
        StaticAbility::AttachedHas(_) => witnessed(
            "StaticAbility::AttachedHas",
            Seen::Actions("the attached creature is offered the granted activation"),
            "Trusty Boomerang",
            "tla",
            "trusty_boomerang_grants_its_creature_a_tap_ability_that_returns_it",
        ),
        StaticAbility::AttachedSetsCharacteristics(_) => witnessed(
            "StaticAbility::AttachedSetsCharacteristics",
            Seen::Via(
                "the enchanted creature's characteristics-set, P/T and lost-abilities columns",
            ),
            "Honest Work",
            "tla",
            "honest_work_turns_the_creature_into_a_humble_merchant",
        ),
        StaticAbility::CostsLessIfTargets(_) => witnessed(
            "StaticAbility::CostsLessIfTargets",
            Seen::Actions("matching targets stay offered at the reduced cost"),
            "Swampsnare Trap",
            "tla",
            "swampsnare_trap_costs_less_onto_a_flier_and_shrinks_its_creature",
        ),
        StaticAbility::KeywordGrant(_) => witnessed(
            "StaticAbility::KeywordGrant",
            Seen::Via("the granted permanents' keyword columns"),
            "Badgermole",
            "tla",
            "badgermole_earthbends_and_grants_trample_to_countered_creatures",
        ),
        StaticAbility::Restriction(_) => witnessed(
            "StaticAbility::Restriction",
            Seen::Actions("attack, block and untap offers omit what it forbids"),
            "Flopsie, Bumi's Buddy",
            "tla",
            "flopsie_counters_your_team_and_limits_big_attackers_to_one_blocker",
        ),
        StaticAbility::SpellsCostLessMana(_) => witnessed(
            "StaticAbility::SpellsCostLessMana",
            Seen::Actions("casts are offered at the reduced cost"),
            "Aang, Master of Elements",
            "tla",
            "avatar_aang_draws_on_each_bending_and_transforms_after_all_four",
        ),
        StaticAbility::UntapsDuringOtherUntapSteps => witnessed(
            "StaticAbility::UntapsDuringOtherUntapSteps",
            Seen::Via("the permanent's tapped column after another player's untap step"),
            "Bender's Waterskin",
            "tla",
            "benders_waterskin_untaps_during_the_other_players_untap_step",
        ),
        StaticAbility::TypeGrant(_) => witnessed(
            "StaticAbility::TypeGrant",
            Seen::Via("the granted permanents' card type columns"),
            "Toph, the First Metalbender",
            "tla",
            "toph_makes_your_nontoken_artifacts_lands_and_earthbends_at_your_end_step",
        ),
        StaticAbility::AttacksDespiteDefender(_) => witnessed(
            "StaticAbility::AttacksDespiteDefender",
            Seen::Actions("the defender is offered as an attacker while it holds"),
            "Platypus-Bear",
            "tla",
            "platypus_bear_mills_two_and_attacks_once_a_lesson_is_in_your_graveyard",
        ),
        StaticAbility::TriggersAdditionalTime(_) => witnessed(
            "StaticAbility::TriggersAdditionalTime",
            Seen::Via("each additional occurrence's stack row"),
            "Katara, the Fearless",
            "tla",
            "katara_the_fearless_makes_ally_triggers_trigger_an_additional_time",
        ),
        StaticAbility::EntersTapped(_) => witnessed(
            "StaticAbility::EntersTapped",
            Seen::Via("the entering permanent's tapped column"),
            "Zhao, the Moon Slayer",
            "tla",
            "zhao_taps_nonbasic_lands_and_makes_them_mountains_once_he_conquers",
        ),
        StaticAbility::SetsLandType(_) => witnessed(
            "StaticAbility::SetsLandType",
            Seen::Via("the land's abilities-lost column and added-subtype fact row"),
            "Zhao, the Moon Slayer",
            "tla",
            "zhao_taps_nonbasic_lands_and_makes_them_mountains_once_he_conquers",
        ),
        StaticAbility::TriggeredGrant(_) => witnessed(
            "StaticAbility::TriggeredGrant",
            Seen::Via("the granted ability's stack rows and their effects"),
            "Sokka, Tenacious Tactician",
            "tla",
            "sokka_tenacious_tactician_gives_allies_menace_and_prowess_and_makes_allies",
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
            Seen::Card(card::KICKER_COST),
            "Firebending Lesson",
            "tla",
            "firebending_lesson_kicked_deals_five",
        ),
        AdditionalCost::Waterbend(_) => witnessed(
            "AdditionalCost::Waterbend",
            Seen::Card(card::KICKER_COST),
            "Spirit Water Revival",
            "tla",
            "spirit_water_revival_waterbends_its_additional_cost",
        ),
        AdditionalCost::PayLifeOrMana { .. } => witnessed(
            "AdditionalCost::PayLifeOrMana",
            Seen::Card(card::KICKER_COST),
            "Redirect Lightning",
            "tla",
            "redirect_lightning_moves_a_spell_to_another_target",
        ),
        AdditionalCost::ExileFromHandAlternative { .. } => witnessed(
            "AdditionalCost::ExileFromHandAlternative",
            Seen::Actions("the alternative is offered only while another matching card is in hand outside your turn, then its exile choices"),
            "Force of Negation",
            "mh1",
            "force_of_negation_offers_its_alternative_only_outside_your_turn",
        ),
        AdditionalCost::Replicate(_) => witnessed(
            "AdditionalCost::Replicate",
            Seen::Actions("the announcement offers each payable replicate count; the stack spell's announced X shows it"),
            "Shattering Spree",
            "gpt",
            "shattering_spree_replicated_twice_destroys_three_artifacts",
        ),
        AdditionalCost::RequiredWaterbend(_) => witnessed(
            "AdditionalCost::RequiredWaterbend",
            Seen::Actions("the cast is offered only while the waterbend total is payable"),
            "Crashing Wave",
            "tla",
            "crashing_wave_waterbends_x_taps_up_to_x_and_distributes_three_stun_counters",
        ),
    }
}

/// One witness per `StaticAbility` variant and `BlockRestriction` side.
fn static_witnesses() -> Vec<StaticAbility> {
    let creature = CardPredicate::creature;
    vec![
        StaticAbility::Continuous(managym::state::card::StaticContinuous {
            scope: StaticScope::This,
            condition: None,
            characteristic_defining: false,
            modifications: vec![Modification::ModifyPt {
                power: managym::state::card::LayerValue::Fixed(2),
                toughness: managym::state::card::LayerValue::Fixed(2),
            }],
        }),
        StaticAbility::GraveyardSpellsHaveFlashback,
        StaticAbility::GrantsAnyColorManaAbility(creature()),
        StaticAbility::CantBeBlockedIf(Condition::YourTurn),
        StaticAbility::PlayersCantGainLife,
        StaticAbility::CantAttackOrBlockUnlessYouControlAnother(creature()),
        StaticAbility::BlockRestriction(BlockRestriction::CantBlock(creature())),
        StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(creature())),
        StaticAbility::SpellsCostLess(SpellCostReduction {
            first_on_own_turn: false,
            predicate: CardPredicate::default(),
            condition: Condition::Always,
            generic: 1,
        }),
        StaticAbility::CostsLessPerPermanent(creature()),
        StaticAbility::CostsLessPerPermanentOnBattlefield(creature()),
        StaticAbility::CantBeCountered,
        StaticAbility::CostsLessPerGraveyardCard(creature()),
        StaticAbility::CostsLessByGreatestPower(creature()),
        StaticAbility::FlashIf(Condition::YourTurn),
        StaticAbility::Flashback(ManaCost::parse("{2}")),
        StaticAbility::Foretell(ManaCost::parse("{2}")),
        StaticAbility::NoMaximumHandSize,
        StaticAbility::AdditionalLandPlayEachPlayer,
        StaticAbility::LegendRuleDoesntApply,
        StaticAbility::ActivatedAbilitiesCostLess {
            predicate: creature(),
            generic: 2,
        },
        StaticAbility::OpponentsCastOnlyFromHand,
        StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::This,
            condition: None,
            power: 2,
            toughness: 0,
            per_creature_type_max: None,
            per_color_among: None,
            per_graveyard_matching: None,
        }),
        StaticAbility::CharacteristicPower(PowerCda::CreaturesYouControl),
        StaticAbility::Ward(PaymentCost::Waterbend { generic: 4 }),
        StaticAbility::LookAtLibraryTop,
        StaticAbility::CastFromLibraryTop(creature()),
        StaticAbility::AttachedHas(ActivatedAbilityDefinition::default()),
        StaticAbility::AttachedSetsCharacteristics(SetCharacteristics {
            named: "Witness".into(),
            creature_types: vec!["Citizen".into()],
            power: 1,
            toughness: 1,
        }),
        StaticAbility::CostsLessIfTargets(TargetCostReduction {
            predicate: creature(),
            generic: 1,
        }),
        StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::This,
            condition: None,
            keywords: Keywords::of(&[Keyword::Trample]),
        }),
        StaticAbility::Restriction(StaticRestriction {
            scope: StaticScope::This,
            kind: Restriction::DoesntUntap,
            unless: None,
        }),
        StaticAbility::UntapsDuringOtherUntapSteps,
        StaticAbility::SpellsCostLessMana(ManaCost::parse("{W}{U}{B}{R}{G}")),
        StaticAbility::TypeGrant(StaticTypeGrant {
            scope: StaticScope::This,
            types: vec![CardType::Creature],
            subtypes: Vec::new(),
            condition: Some(Condition::SourceCountersAtLeast {
                kind: CounterKind::Fire,
                count: 3,
            }),
        }),
        StaticAbility::AttacksDespiteDefender(Condition::YourTurn),
        StaticAbility::TriggersAdditionalTime(CardPredicate::subtype("Ally")),
        StaticAbility::EntersTapped(CardPredicate::creature()),
        StaticAbility::SetsLandType(StaticLandType {
            predicate: CardPredicate::creature(),
            subtype: "Mountain".into(),
            condition: Condition::SourceCountersAtLeast {
                kind: CounterKind::Conqueror,
                count: 1,
            },
        }),
        StaticAbility::TriggeredGrant(StaticTriggeredGrant {
            scope: StaticScope::This,
            ability: Box::new(Ability::Triggered {
                condition: TriggerCondition::Performs {
                    who: Actor::You,
                    any_of: vec![KeywordAction::Scry],
                },
                intervening_if: None,
                effects: vec![Effect::discard()],
            }),
        }),
    ]
}

fn cost_witnesses() -> Vec<AdditionalCost> {
    vec![
        AdditionalCost::Sacrifice(Box::new(SacrificeCost {
            predicate: CardPredicate::creature(),
            another: false,
            or_mana: None,
        })),
        AdditionalCost::Kicker(ManaCost::parse("{4}")),
        AdditionalCost::Waterbend(ManaCost::parse("{6}")),
        AdditionalCost::RequiredWaterbend(ManaCost::parse("{X}")),
        AdditionalCost::Replicate(ManaCost::parse("{R}")),
        AdditionalCost::PayLifeOrMana {
            life: 5,
            mana: ManaCost::parse("{2}"),
        },
        AdditionalCost::ExileFromHandAlternative {
            predicate: CardPredicate::default(),
            not_your_turn: true,
        },
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
    let row = card_index(game, observation, permanent);
    &encoded.agent_cards[row * CARD_DIM..(row + 1) * CARD_DIM]
}

fn card_index(game: &Game, observation: &Observation, permanent: PermanentId) -> usize {
    let object = game.state.permanents[permanent].as_ref().unwrap().object;
    let id = game.state.objects[object].id.0 as i32;
    observation
        .agent_cards
        .iter()
        .position(|object| object.id == id)
        .unwrap()
}

fn changed(before: &[f32], after: &[f32]) -> Vec<usize> {
    (0..before.len())
        .filter(|&column| before[column] != after[column])
        .collect()
}

fn tokens_of(definition: &ObjectDefinition) -> Vec<(TokenKind, i32)> {
    definition_tokens(definition, default_content_pack()).unwrap()
}

fn spell(effects: Vec<Effect>) -> ObjectDefinition {
    ObjectDefinition::face(PrintedCharacteristics {
        name: "Witness".into(),
        spell_effects: effects,
        ..Default::default()
    })
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
            intervening_if: None,
            condition: condition.clone(),
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        };
        let definition = ObjectDefinition::face(PrintedCharacteristics {
            name: "Witness".into(),
            abilities: vec![ability.clone()],
            ..Default::default()
        });
        let tokens = tokens_of(&definition);
        for name in [trigger_token(&condition), ability_token(&ability)] {
            assert!(
                tokens.contains(&(TokenKind::Variant, name_value(name))),
                "{name}"
            );
        }
    }
}

/// A triggered mana ability's tapping actor and each output reach the
/// learning tokens.
#[test]
fn every_triggered_mana_output_and_actor_reaches_the_learning_tokens() {
    fn output_token(output: &TriggeredManaOutput) -> &'static str {
        match output {
            TriggeredManaOutput::Fixed(_) => "TriggeredManaOutput::Fixed",
            TriggeredManaOutput::OneProducedType => "TriggeredManaOutput::OneProducedType",
        }
    }
    for (who, actor) in [
        (Actor::You, "Actor::You"),
        (Actor::AnyPlayer, "Actor::AnyPlayer"),
    ] {
        for output in [
            TriggeredManaOutput::Fixed(Mana::parse("G")),
            TriggeredManaOutput::OneProducedType,
        ] {
            let definition = ObjectDefinition::face(PrintedCharacteristics {
                name: "Witness".into(),
                triggered_mana_abilities: vec![TriggeredManaAbility {
                    who,
                    predicate: CardPredicate::default(),
                    output: output.clone(),
                }],
                ..Default::default()
            });
            let tokens = tokens_of(&definition);
            for name in [actor, output_token(&output)] {
                assert!(
                    tokens.contains(&(TokenKind::Variant, name_value(name))),
                    "{name} in {output:?}"
                );
            }
        }
    }
}

/// Every trigger actor, step, keyword action and batching choice is a
/// parameter of one event and reaches the learning tokens.
#[test]
fn every_trigger_event_parameter_reaches_the_learning_tokens() {
    let steps = [
        TurnStep::Upkeep,
        TurnStep::Draw,
        TurnStep::FirstMain,
        TurnStep::Combat,
        TurnStep::End,
    ];
    let mut actions = vec![
        KeywordAction::Search,
        KeywordAction::Scry,
        KeywordAction::Surveil,
    ];
    actions.extend(KeywordAction::BENDING);
    let mut conditions = Vec::new();
    let mut expected = Vec::new();
    for who in [Actor::You, Actor::Opponent, Actor::AnyPlayer] {
        conditions.push(TriggerCondition::Draws { who, nth: None });
        expected.push(format!("Actor::{who:?}"));
    }
    for step in steps {
        conditions.push(TriggerCondition::BeginningOf {
            step,
            whose: Actor::You,
        });
        expected.push(format!("TurnStep::{step:?}"));
    }
    conditions.push(TriggerCondition::Performs {
        who: Actor::You,
        any_of: actions.clone(),
    });
    expected.extend(
        actions
            .iter()
            .map(|action| format!("KeywordAction::{action:?}")),
    );
    for per in [Per::Object, Per::Batch] {
        conditions.push(TriggerCondition::DealsCombatDamage {
            subject: TriggerSubject::This,
            to: Actor::AnyPlayer,
            per,
        });
        expected.push(format!("Per::{per:?}"));
    }
    for attackers in [AttackSize::Any, AttackSize::Alone, AttackSize::WithAnother] {
        conditions.push(TriggerCondition::Attacks {
            subject: TriggerSubject::you_control(CardPredicate::creature()),
            per: Per::Batch,
            attackers,
        });
        expected.push(format!("AttackSize::{attackers:?}"));
    }
    let definition = ObjectDefinition::face(PrintedCharacteristics {
        name: "Witness".into(),
        abilities: conditions
            .into_iter()
            .map(|condition| Ability::Triggered {
                intervening_if: None,
                condition,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
            })
            .collect(),
        ..Default::default()
    });
    let tokens = tokens_of(&definition);
    for name in expected {
        assert!(
            tokens.contains(&(TokenKind::Variant, name_value(&name))),
            "{name}"
        );
    }

    // An object actor's controller and "another" are parameters of one
    // subject: each combination is its own token stream.
    let mut subjects: Vec<Vec<(TokenKind, i32)>> = Vec::new();
    for controlled_by in [Actor::You, Actor::Opponent, Actor::AnyPlayer] {
        for another in [false, true] {
            let tokens = tokens_of(&ObjectDefinition::face(PrintedCharacteristics {
                name: "Witness".into(),
                abilities: vec![Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::Dies {
                        subject: TriggerSubject::matching(
                            controlled_by,
                            CardPredicate::creature(),
                            another,
                        ),
                    },
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                }],
                ..Default::default()
            }));
            assert!(
                tokens.contains(&(
                    TokenKind::Variant,
                    name_value(&format!("Actor::{controlled_by:?}"))
                )),
                "{controlled_by:?}"
            );
            assert!(!subjects.contains(&tokens), "{controlled_by:?} {another}");
            subjects.push(tokens);
        }
    }
}

#[test]
fn every_condition_reaches_the_learning_tokens() {
    let witnesses = condition_witnesses();
    let names: BTreeSet<_> = witnesses.iter().map(condition_token).collect();
    assert_eq!(names.len(), witnesses.len(), "one witness per variant");
    for condition in witnesses {
        let tokens = tokens_of(&spell(vec![Effect::If {
            condition: condition.clone(),
            then: vec![Effect::discard()],
            otherwise: vec![],
        }]));
        let expected = (TokenKind::Variant, name_value(condition_token(&condition)));
        assert!(tokens.contains(&expected), "{condition:?}");
    }
}

#[test]
fn an_intervening_if_reaches_the_learning_tokens() {
    let definition = ObjectDefinition::face(PrintedCharacteristics {
        name: "Witness".into(),
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::beginning_of(TurnStep::End, Actor::You),
            intervening_if: Some(Condition::AttackedThisTurn),
            effects: vec![Effect::discard()],
        }],
        ..Default::default()
    });
    let expected = (
        TokenKind::Variant,
        name_value(condition_token(&Condition::AttackedThisTurn)),
    );
    assert!(tokens_of(&definition).contains(&expected));
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
                .chain(turn_fields(&game.state.turn))
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
            // Fact rows of a kind that states a permanent's field.
            Seen::Fact(kind) => assert!(fact_field(kind).starts_with("PermanentData."), "{field}"),
            Seen::Identity => {}
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
    assert_eq!(
        names.len(),
        MODIFICATION_VARIANTS,
        "one witness per variant"
    );
    for (on_player, modification) in witnesses {
        let affected = if on_player {
            ObjectSelector::You
        } else {
            ObjectSelector::This
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
    for duration in [
        EffectDuration::EndOfTurn,
        EffectDuration::Indefinite,
        EffectDuration::UntilYourNextTurn,
    ] {
        let effect = Effect::ApplyContinuous {
            affected: ObjectSelector::This,
            modification: Modification::CantBeBlocked,
            duration,
        };
        let name = format!("EffectDuration::{duration:?}");
        assert!(tokens_of(&spell(vec![effect])).contains(&(TokenKind::Variant, name_value(&name))));
        assert!(duration_code(duration) > 0.0);
    }
    let scopes = [
        Scope::YouControl,
        Scope::OpponentsControl,
        Scope::Battlefield,
        Scope::TargetPlayerControls,
    ];
    let excepts = [None, Some(Except::This), Some(Except::Target)];
    for (scope, except) in scopes.into_iter().zip(excepts.into_iter().cycle()) {
        let affected = ObjectSelector::Each {
            of: scope,
            predicate: CardPredicate::creature(),
            except,
            max_mana_value: None,
        };
        let effect = Effect::ApplyContinuous {
            affected,
            modification: Modification::CantBeBlocked,
            duration: EffectDuration::EndOfTurn,
        };
        let tokens = tokens_of(&spell(vec![effect]));
        let name = scope_token(scope);
        assert!(
            tokens.contains(&(TokenKind::Variant, name_value(name))),
            "{name}"
        );
        if let Some(except) = except {
            let name = except_token(except);
            assert!(
                tokens.contains(&(TokenKind::Variant, name_value(name))),
                "{name}"
            );
        }
    }
}

/// Every selector a verb can act through, every player a verb can name and
/// both exile links reach the tokens, and two verbs that differ only in
/// their selector are different token sequences of one shape.
#[test]
fn every_object_and_player_selector_reaches_the_learning_tokens() {
    for selector in selector_witnesses() {
        let name = selector_token(&selector);
        let tokens = tokens_of(&spell(vec![Effect::tap(selector)]));
        assert!(
            tokens.contains(&(TokenKind::Variant, name_value(name))),
            "{name}"
        );
    }
    for player in player_selector_witnesses() {
        let name = player_selector_token(&player);
        let effect = Effect::Move {
            library_end: None,
            what: ObjectSelector::Moved,
            to: ZoneType::Battlefield,
            tapped: false,
            transformed: false,
            under: Some(player),
            with_counter: None,
            linked: None,
        };
        assert!(
            tokens_of(&spell(vec![effect])).contains(&(TokenKind::Variant, name_value(name))),
            "{name}"
        );
    }
    for linked in [LinkedExile::WithSource, LinkedExile::UntilSourceLeaves] {
        let effect = Effect::exile_linked(ObjectSelector::This, linked);
        let name = format!("LinkedExile::{linked:?}");
        assert!(tokens_of(&spell(vec![effect])).contains(&(TokenKind::Variant, name_value(&name))));
    }
    let this = tokens_of(&spell(vec![Effect::exile(ObjectSelector::This)]));
    let attached = tokens_of(&spell(vec![Effect::exile(ObjectSelector::Attached)]));
    assert_ne!(this, attached);
    assert_eq!(this.len(), attached.len(), "the verb's shape is shared");
}

// ---------------------------------------------------------------------------
// (b) Directed: each mechanic moves its named column or token.

#[test]
fn printed_and_granted_keywords_set_their_card_and_permanent_columns_and_tokens() {
    let (base_game, ogre) = ogre_game();
    let (base_obs, base) = observe(&base_game, 0);
    let bare = tokens_of(&ObjectDefinition::face(PrintedCharacteristics {
        name: "Witness".into(),
        ..Default::default()
    }));
    for keyword in Keyword::ALL {
        let name = keyword.name();
        let (card_column, permanent_column) = keyword_columns(keyword);
        let keywords = Keywords::of(&[keyword]);

        let mut granted = base_game.clone();
        let affected = granted.affected_permanent(ogre).unwrap();
        granted.apply_continuous(
            None,
            PlayerId(0),
            [affected],
            Modification::AddKeywords(keywords),
            EffectDuration::EndOfTurn,
        );
        let (_, after) = observe(&granted, 0);
        assert_eq!(
            changed(permanent_row(&base), permanent_row(&after)),
            [permanent_column],
            "{name}"
        );

        let mut printed = base_game.clone();
        let object = printed.state.permanents[ogre].as_ref().unwrap().object;
        printed.state.objects[object].keywords = keywords;
        let (obs, after) = observe(&printed, 0);
        assert!(
            changed(
                card_row(&base_game, &base_obs, &base, ogre),
                card_row(&printed, &obs, &after, ogre)
            )
            .contains(&card_column),
            "{name}"
        );

        let definition = ObjectDefinition::face(PrintedCharacteristics {
            name: "Witness".into(),
            keywords,
            ..Default::default()
        });
        let keyword_token = (TokenKind::Variant, name_value(&format!("Keyword::{name}")));
        let tokens = tokens_of(&definition);
        let at = tokens
            .iter()
            .position(|token| *token == (TokenKind::Field, name_value("keywords")))
            .unwrap();
        assert_eq!(
            &tokens[at + 1..at + 4],
            [
                (TokenKind::Structure, name_value("[")),
                keyword_token,
                (TokenKind::Structure, name_value("/[")),
            ],
            "{name}"
        );
        assert!(!bare.contains(&keyword_token), "{name}");
    }
}

#[test]
fn sorcery_timing_reaches_the_activated_ability_tokens() {
    let definition = ObjectDefinition::face(PrintedCharacteristics {
        name: "Witness".into(),
        activated_abilities: vec![ActivatedAbilityDefinition {
            sorcery_timing: true,
            effects: vec![Effect::Surveil { count: 1 }],
            ..Default::default()
        }],
        ..Default::default()
    });
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
        "dealt_damage_this_turn" => permanent.dealt_damage_this_turn = true,
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
        if matches!(field, "attached_to" | "timestamp") {
            // Need a real attachment, or other timestamps to be ranked
            // among; the directed v5 tests below reach these states.
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

#[test]
fn turn_history_moves_each_players_named_column() {
    let (base_game, _) = ogre_game();
    let (_, base) = observe(&base_game, 0);
    for (field, seen) in turn_fields(&base_game.state.turn) {
        let Seen::Player(start, end) = seen else {
            continue;
        };
        if start < player::KEEPS_RED_MANA && field != "turn_number" {
            continue; // Phase, step and the active player: the turn structure itself.
        }
        let mut game = base_game.clone();
        let turn = &mut game.state.turn;
        let history = &mut turn.history[0];
        match field {
            "turn_number" => turn.turn_number += 1,
            "history.cards_drawn" => history.cards_drawn += 1,
            "history.bent" => history.bent = 0b0101,
            "history.attacked" => history.attacked = true,
            "history.permanents_sacrificed" => history.permanents_sacrificed = 2,
            "history.land_entered" => history.land_entered = true,
            "extra_turns" => turn.extra_turns.push(PlayerId(0)),
            "additional_combats" => turn
                .additional_combats
                .push(managym::flow::turn::AdditionalCombat { attackers: None }),
            other => panic!("no directed mutation for {other}"),
        }
        let (_, after) = observe(&game, 0);
        let moved = changed(&base.agent_player, &after.agent_player);
        assert!(
            moved.iter().any(|column| (start..end).contains(column)),
            "{field} moved {moved:?}, expected {start}..{end}"
        );
        if field != "turn_number" {
            // Player 0's history is not the opponent's.
            assert!(
                changed(&base.opponent_player, &after.opponent_player).is_empty(),
                "{field}"
            );
        }
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
            .zone_objects(ZoneType::Hand, PlayerId(1))
            .iter()
            .map(|object| game.state.objects[*object].name.clone())
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
        let tokens = definition_tokens(definition, pack).unwrap();
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

/// The definition of the card named `name`, or of the token keyed `name`
/// (`token.*`), in the default pack.
fn card_definition(name: &str) -> ObjectDefinition {
    let pack = default_content_pack();
    let id = if name.starts_with("token.") {
        pack.token_id(name)
    } else {
        pack.definition_id(name)
    }
    .unwrap_or_else(|| panic!("{name}"));
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
            Seen::Fact(kind) => assert!(!fact_field(kind).is_empty(), "{}", row.token),
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
        let printed = Arc::make_mut(&mut stripped.printed);
        printed.statics.retain(|candidate| {
            std::mem::discriminant(candidate) != std::mem::discriminant(&ability)
        });
        // A copy exception granting the same static (Sakashima) goes too.
        for effect in &mut printed.replacement_effects {
            if let managym::state::card::ReplacementEffect::EntersAsCopy { exceptions, .. } = effect
            {
                exceptions.retain(|exception| {
                    !matches!(exception, Modification::GrantStatic(granted)
                        if std::mem::discriminant(&**granted) == std::mem::discriminant(&ability))
                });
            }
        }
        let without = tokens_of(&stripped);
        assert_ne!(tokens, without, "{}", row.card);
        for name in names {
            assert!(tokens.contains(&variant(name)), "{} lacks {name}", row.card);
            assert!(!without.contains(&variant(name)), "{name}");
        }
        // The token also appears for a bare definition with only this ability.
        let bare = ObjectDefinition::face(PrintedCharacteristics {
            name: "Witness".into(),
            statics: vec![ability.clone()],
            ..Default::default()
        });
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
        Arc::make_mut(&mut stripped.printed)
            .additional_costs
            .retain(|candidate| std::mem::discriminant(candidate) != std::mem::discriminant(&cost));
        let without = tokens_of(&stripped);
        assert_ne!(tokens, without, "{}", row.card);
        assert!(tokens.contains(&variant(row.token)), "{}", row.card);
        assert!(!without.contains(&variant(row.token)), "{}", row.token);
    }
}

/// Every variant whose row names a tensor column moves that column when the
/// ogre gains it: ward, kicker, optional waterbend and pay-life-or-mana set card
/// columns; a P/T buff and
/// characteristic power move the permanent's power column.
#[test]
fn tensor_visible_statics_and_costs_move_their_named_columns() {
    let (base_game, ogre) = ogre_game();
    let (base_obs, base) = observe(&base_game, 0);
    let object = base_game.state.permanents[ogre].as_ref().unwrap().object;
    let edits = static_witnesses()
        .into_iter()
        .map(|ability| {
            let row = static_seen(&ability);
            let edit: Box<dyn Fn(&mut PrintedCharacteristics)> =
                Box::new(move |definition| definition.statics.push(ability.clone()));
            (row, edit)
        })
        .chain(cost_witnesses().into_iter().map(|cost| {
            let row = cost_seen(&cost);
            let edit: Box<dyn Fn(&mut PrintedCharacteristics)> =
                Box::new(move |definition| definition.additional_costs.push(cost.clone()));
            (row, edit)
        }));
    let mut checked = 0;
    for (row, edit) in edits {
        let mut game = base_game.clone();
        edit(&mut game.state.objects[object]);
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
    assert_eq!(checked, 7);
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
            let object = card_of(&s, proft);
            move_card(&mut s, object, ZoneType::Hand);
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

/// Every `token.*` definition is a definition reference from a card that
/// creates it, and a created token's observation carries its characteristics.
#[test]
fn every_token_is_referenced_by_its_creator_and_observed_with_its_characteristics() {
    let pack = default_content_pack();
    let catalog = catalog_tokens(pack).unwrap();
    let tokens = managym_cards::token_catalog();
    assert!(!tokens.is_empty());
    for definition in tokens {
        let row = pack.token_id(definition.key).unwrap();
        let token = pack.definition(row).unwrap();
        assert!(token.is_token, "{}", token.key);
        let row = row.0 as i32;
        let referencing = catalog
            .definition_ref_source_tokens
            .iter()
            .zip(&catalog.definition_ref_target_rows)
            .filter(|(_, &target)| target == row)
            .count();
        assert!(
            referencing > 0,
            "{}: {referencing} creating references",
            token.key
        );

        let (mut game, _) = ogre_game();
        game.create_token(token.key, PlayerId(0), false);
        let (observation, encoded) = observe(&game, 0);
        let object = observation
            .agent_cards
            .iter()
            .position(|object| object.name == token.name && object.is_token)
            .unwrap_or_else(|| panic!("{} is not observed", token.key));
        let card_row = &encoded.agent_cards[object * CARD_DIM..(object + 1) * CARD_DIM];
        assert_eq!(card_row[29], 1.0, "{} is_token column", token.key);
        assert_eq!(card_row[8], token.power.unwrap_or(0) as f32 / 10.0);
        assert_eq!(card_row[9], token.toughness.unwrap_or(0) as f32 / 10.0);
        assert_eq!(card_row[12], f32::from(token.types.is_creature()));
        assert_eq!(card_row[13], f32::from(token.types.is_artifact()));
        assert_eq!(card_row[17], f32::from(token.keywords.has(Keyword::Flying)));
    }
}

// ---------------------------------------------------------------------------
// Policy input: every `Observation` field.

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
    Facts,
    /// Input v10 `player_turn_state`: one row per seat.
    PlayerTurnState,
    /// Input v10 `card_turn_state`: one value per card row.
    CardTurnState,
}

impl Buffer {
    /// Row width.
    fn width(self) -> usize {
        match self {
            Buffer::Player => PLAYER_DIM,
            Buffer::Card => CARD_DIM,
            Buffer::Permanent => PERMANENT_DIM,
            Buffer::Relation => RELATION_DIM,
            Buffer::Action => managym::agent::observation_encoder::ACTION_DIM,
            Buffer::Facts => FACT_DIM,
            Buffer::PlayerTurnState => TURN_STATE_DIM,
            Buffer::CardTurnState => 1,
        }
    }

    /// The row's validity column; player validity is a buffer of its own.
    fn validity(self) -> Option<usize> {
        match self {
            Buffer::Player | Buffer::PlayerTurnState | Buffer::CardTurnState => None,
            Buffer::Relation => Some(relation::VALID),
            Buffer::Facts => Some(fact::VALID),
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
    /// A further ordered occurrence of a fact kind another field owns,
    /// separated by an ordinal reset.
    FactOccurrence(FactKind),
    /// Not a decision input, for the stated reason.
    Excluded(&'static str),
}

/// The field each fact kind states; no wildcard, so a new kind must name it.
fn fact_field(kind: FactKind) -> &'static str {
    match kind {
        FactKind::ContinuationProgram => "ContinuationFrame.program",
        FactKind::EffectProgram => "ContinuousEffectData.program",
        FactKind::EffectiveColor => "PermanentData.colors",
        FactKind::KeywordCounterTimestamp => "PermanentData.keyword_counter_timestamps",
        FactKind::Counter => "PermanentData.counters",
        FactKind::AddedSubtype => "PermanentData.temp_subtypes",
        FactKind::SetName => "PermanentData.name",
        FactKind::SetCreatureType => "PermanentData.creature_types",
        FactKind::UsedAbility => "PermanentData.used_abilities",
        FactKind::Copy => "PermanentData.copy",
        FactKind::CopyException => "CopyData.exceptions",
        FactKind::Transforms => "PermanentData.transforms",
        FactKind::CombatRestriction => "ContinuationFrame.attackers",
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

const HISTORY: &str = "zone and incarnation metadata belong to the public history rows \
    (policy_history_version 1); the v0 event tensor is frozen";

fn guard() -> Vec<(&'static str, Vec<Encoded>)> {
    use Buffer::*;
    use Encoded::*;
    let mut fields = encoding!(Observation {
        shared_effects => [Rows("v7 effects, effect_details, effect_members")],
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
    fields.extend(encoding!(SharedEffectData {
        effect => [Rows("v7 effects columns 0..10, 11 and effect_details")],
        controller_id => [Rows("v7 effects column 10: exact player join")],
        source => [Rows("v7 effects column 12: exact source join")],
        members => [Rows("v7 effects columns 15/16 index effect_members")],
    }));
    fields.extend(encoding!(EffectObject {
        object_id => [Identity, Rows("v7 controller/source/member joins")],
        incarnation => [Identity, Rows("native exact-reference admission, never a numeric identity feature")],
    }));
    fields.extend(encoding!(EffectMembership {
        target => [Rows("v7 effect_members column 0: exact affected object")],
        order => [Rows("v7 effect_members column 1: reversible native record order")],
    }));
    fields.extend(encoding!(TurnData {
        turn_number => [column(Player, player::TURN)],
        phase => [Columns(Player, player::PHASE_BEGINNING, 5)],
        step => [Columns(Player, player::STEP_UNTAP, 12)],
        active_player_id => [column(Player, player::IS_ACTIVE)],
        agent_player_id => [Side],
    }));
    fields.extend(encoding!(PlayerData {
        player_index => [column(Player, player::SEAT)],
        id => [Identity],
        is_agent => [Side],
        is_active => [column(Player, player::IS_ACTIVE)],
        life => [column(Player, player::LIFE)],
        zone_counts => [Columns(Player, player::LIBRARY_COUNT, 7)],
        graveyard_lessons => [column(Player, player::GRAVEYARD_LESSONS)],
        combat_mana => [column(Player, player::COMBAT_MANA)],
        mana => [Columns(Player, player::MANA_WHITE, 6)],
        lesson_mana => [Columns(Player, player::LESSON_MANA_WHITE, 6)],
        restricted_mana => [Rows("restricted mana below")],
        pooled_mana_classes => [Rows("pooled mana classes below")],
        hand_only_until => [column(Player, player::HAND_ONLY)],
        known_hand => [Rows("known_hand")],
        main_counts => [Rows("decklists")],
        unseen_counts => [Rows("decklists")],
        sideboard_counts => [Rows("decklists")],
        remaining_sideboard_counts => [Rows("decklists")],
        keeps_red_mana => [column(Player, player::KEEPS_RED_MANA)],
        no_maximum_hand_size => [column(Player, player::NO_MAXIMUM_HAND_SIZE)],
        effects => [Rows("shared-effect rows, ContinuousEffectData below")],
        turn_history => [Rows("turn history below")],
        spells_cast => [column(PlayerTurnState, turn_state::SPELLS_CAST)],
        extra_turns => [column(Player, player::EXTRA_TURNS)],
        additional_combats => [column(Player, player::ADDITIONAL_COMBATS)],
    }));
    fields.extend(encoding!(PlayerTurnHistory {
        attacked => [column(Player, player::ATTACKED)],
        permanents_sacrificed => [column(Player, player::PERMANENTS_SACRIFICED)],
        land_entered => [column(Player, player::LAND_ENTERED)],
        cards_drawn => [column(Player, player::CARDS_DRAWN)],
        bent => [Columns(Player, player::BENT_WATER, 4)],
        life_lost => [column(PlayerTurnState, turn_state::LIFE_LOST)],
    }));
    fields.extend(encoding!(CardData {
        zone => [Columns(Card, card::ZONE_LIBRARY, 7)],
        owner_id => [column(Card, card::IS_MINE), Side],
        id => [Identity],
        registry_key => [Rows("semantic_cards")],
        name => [Definition],
        text_box => [Definition],
        power => [column(Card, card::POWER)],
        toughness => [column(Card, card::TOUGHNESS)],
        is_token => [column(Card, card::IS_TOKEN)],
        is_legendary => [column(Card, card::IS_LEGENDARY)],
        is_ally => [column(Card, card::IS_ALLY)],
        is_lesson => [column(Card, card::IS_LESSON)],
        ward_cost => [Columns(Card, card::HAS_WARD, 2)],
        kicker_cost => [Columns(Card, card::HAS_KICKER, 2)],
        card_types => [Rows("card types below")],
        keywords => [Rows("keywords below")],
        mana_cost => [column(Card, card::MANA_VALUE), Definition],
        // Face-down cards are foretold, and only their owner sees them.
        face_down => [column(Card, card::PERMISSION_FORETOLD)],
        cast_permission => [Rows("cast permission below")],
        x => [Columns(Card, card::HAS_X, 2)],
        from_battlefield_this_turn => [column(CardTurnState, 0)],
    }));
    fields.extend(encoding!(CastPermissionData {
        player_id => [Columns(Card, card::PERMISSION, 2)],
        cost => [Columns(Card, card::PERMISSION_HAS_COST, 2)],
        this_turn => [column(Card, card::PERMISSION_THIS_TURN)],
        own_turn_only => [column(Card, card::PERMISSION_OWN_TURN_ONLY)],
        after_turn => [column(Card, card::PERMISSION_NOT_YET)],
        source_permanent_id => [column(Card, card::PERMISSION_HAS_SOURCE)],
        flash => [column(Card, card::PERMISSION_FLASH)],
        any_mana => [column(Card, card::PERMISSION_ANY_MANA)],
        waterbend => [column(Card, card::PERMISSION_WATERBEND)],
        foretold => [column(Card, card::PERMISSION_FORETOLD)],
        through_next_turn => [column(Card, card::PERMISSION_THROUGH_NEXT_TURN)],
    }));
    fields.extend(encoding!(PooledManaClassData {
        combat => [FactOccurrence(FactKind::EffectProgram)],
        restriction => [FactOccurrence(FactKind::EffectProgram)],
        effect => [FactOccurrence(FactKind::EffectProgram)],
        mana => [
            FactOccurrence(FactKind::EffectProgram),
            Columns(Player, player::MANA_WHITE, 6),
        ],
        program => [FactOccurrence(FactKind::EffectProgram)],
    }));
    fields.extend(encoding!(RestrictedManaData {
        spells => [Columns(Player, player::RESTRICTED_LESSON_WHITE, 18)],
        abilities => [Columns(Player, player::RESTRICTED_LESSON_WHITE, 18)],
        except_nonartifact_spells => [Columns(PlayerTurnState, turn_state::NONARTIFACT_MANA_WHITE, 6)],
        mana => [
            Columns(Player, player::RESTRICTED_LESSON_WHITE, 18),
            Columns(Player, player::LESSON_MANA_WHITE, 6),
            Columns(PlayerTurnState, turn_state::NONARTIFACT_MANA_WHITE, 6),
        ],
    }));
    fields.extend(encoding!(CardTypeData {
        is_castable => [Definition],
        is_permanent => [Definition],
        is_non_land_permanent => [Definition],
        is_non_creature_permanent => [Definition],
        is_spell => [Definition],
        is_creature => [column(Card, card::IS_CREATURE)],
        is_land => [column(Card, card::IS_LAND)],
        is_planeswalker => [column(Card, card::IS_PLANESWALKER)],
        is_enchantment => [column(Card, card::IS_ENCHANTMENT)],
        is_artifact => [column(Card, card::IS_ARTIFACT)],
        is_kindred => [Definition],
        is_battle => [column(Card, card::IS_BATTLE)],
    }));
    fields.extend(encoding!(SideboardCardData {
        candidate_id => [Identity, column(Card, card::OUTSIDE)],
        owner_id => [column(Card, card::IS_MINE)],
        registry_key => [Rows("semantic_cards")],
        name => [Definition],
        text_box => [Definition],
    }));
    fields.extend(encoding!(PermanentData {
        colors => [Fact(FactKind::EffectiveColor)],
        keyword_counter_timestamps => [Fact(FactKind::KeywordCounterTimestamp)],
        transforms => [Fact(FactKind::Transforms)],
        id => [Identity],
        registry_key => [Join(Permanent, permanent::DEFINITION)],
        controller_id => [column(Permanent, permanent::IS_MINE), Side],
        tapped => [column(Permanent, permanent::TAPPED)],
        damage => [column(Permanent, permanent::DAMAGE)],
        is_summoning_sick => [column(Permanent, permanent::SUMMONING_SICK)],
        plus1_counters => [column(Permanent, permanent::PLUS_ONE_COUNTERS), Rows("Counter fact rows")],
        cant_be_blocked_this_turn => [column(Permanent, permanent::CANT_BE_BLOCKED_THIS_TURN)],
        power => [column(Permanent, permanent::POWER)],
        toughness => [column(Permanent, permanent::TOUGHNESS)],
        is_animated => [column(Permanent, permanent::ANIMATED)],
        has_exile_link => [column(Permanent, permanent::HAS_EXILE_LINK)],
        exile_if_dies_this_turn => [column(Permanent, permanent::EXILE_IF_DIES_THIS_TURN)],
        attacking => [column(Permanent, permanent::ATTACKING)],
        exhausted_abilities => [column(Permanent, permanent::EXHAUSTED_ABILITIES)],
        once_each_turn_used => [column(Permanent, permanent::ONCE_EACH_TURN_USED)],
        granted_firebending => [column(Permanent, permanent::GRANTED_FIREBENDING)],
        temp_subtypes => [
            column(Permanent, permanent::ADDED_AVATAR),
            Fact(FactKind::AddedSubtype),
            Columns(Facts, fact::ID_HIGH, 2),
        ],
        keywords => [Rows("keywords below")],
        counters => [
            Fact(FactKind::Counter),
            Columns(Facts, fact::ID_HIGH, 2),
            column(Facts, fact::VALUE),
        ],
        attached_to => [Join(Permanent, permanent::ATTACHED_TO)],
        name => [Fact(FactKind::SetName)],
        creature_types => [Fact(FactKind::SetCreatureType)],
        is_granted_land => [column(Permanent, permanent::GRANTED_LAND)],
        blocking => [Join(Permanent, permanent::BLOCKING)],
        abilities_lost => [column(Permanent, permanent::ABILITIES_LOST)],
        dealt_damage_this_turn => [column(Permanent, permanent::DEALT_DAMAGE_THIS_TURN)],
        used_abilities => [Fact(FactKind::UsedAbility)],
        effects => [Rows("shared-effect rows, ContinuousEffectData below")],
        timestamp => [column(Permanent, permanent::TIMESTAMP)],
        copy => [Fact(FactKind::Copy)],
    }));
    fields.extend(encoding!(CopyData {
        registry_key => [Join(Facts, fact::SOURCE_DEFINITION)],
        name => [Definition],
        exceptions => [Fact(FactKind::CopyException)],
    }));
    fields.extend(encoding!(ContinuousEffectData {
        program => [Fact(FactKind::EffectProgram)],
        description => [Definition],
        modification => [
            Rows("effects columns 0-1; a copy exception's fact id"),
            Columns(Facts, fact::ID_HIGH, 2),
        ],
        power => [column(Facts, fact::VALUE)],
        toughness => [column(Facts, fact::VALUE + 1)],
        amount => [column(Facts, fact::VALUE)],
        keywords => [Columns(Facts, fact::DETAIL_HIGH, 2)],
        subtypes => [Columns(Facts, fact::DETAIL_HIGH, 2)],
        color => [Columns(Facts, fact::DETAIL_HIGH, 2)],
        duration => [column(Facts, fact::DURATION)],
        source_registry_key => [Join(Facts, fact::SOURCE_DEFINITION)],
        source_name => [Definition],
        timestamp => [column(Facts, fact::TIMESTAMP)],
    }));
    fields.extend(encoding!(StackObjectData {
        description => [Definition],
        stack_object_id => [Rows("relations")],
        kind => [column(Relation, relation::FRAME_KIND)],
        controller_id => [column(Relation, relation::CONTROLLER)],
        source_card_registry_key => [Join(Relation, relation::SOURCE_DEFINITION)],
        source_permanent_id => [Join(Relation, relation::SOURCE)],
        chosen_mode => [column(Relation, relation::MODE)],
        ability_index => [column(Relation, relation::ABILITY)],
        targets => [Rows("relations")],
        x => [column(Relation, relation::X)],
        granted_keywords => [Columns(Relation, relation::FLYING, KEYWORD_COUNT)],
    }));
    fields.extend(encoding!(StackTargetData {
        kind => [column(Relation, relation::REFERENCE_PLAYER), Join(Relation, relation::REFERENCE)],
        player_id => [column(Relation, relation::REFERENCE_PLAYER)],
        permanent_id => [Join(Relation, relation::REFERENCE)],
        stack_object_id => [Join(Relation, relation::REFERENCE)],
        card_id => [Join(Relation, relation::REFERENCE)],
    }));
    fields.extend(encoding!(ContinuationFrame {
        effect_execution => [FactOccurrence(FactKind::ContinuationProgram)],
        effect_execution_references => [Rows("v8 effect execution reference slots, relation kind 10")],
        effect_execution_description => [Definition],
        effect_execution_summary => [Definition],
        program => [Fact(FactKind::ContinuationProgram)],
        program_description => [Definition],
        remembered => [Rows("v5 captured object reference relations, kind 8")],
        inherited_from => [Rows("v5 inherited-ability definition relation, kind 7")],
        attackers => [Fact(FactKind::CombatRestriction)],
        source_transforms => [Rows("Transforms fact rows, id = frame index + 1")],
        kind => [column(Relation, relation::FRAME_KIND)],
        decision => [column(Relation, relation::DECISION)],
        parent => [column(Relation, relation::PARENT)],
        controller => [column(Relation, relation::CONTROLLER)],
        source => [Join(Relation, relation::SOURCE_DEFINITION), Join(Relation, relation::SOURCE)],
        mode => [column(Relation, relation::MODE)],
        ability => [column(Relation, relation::ABILITY)],
        kicked => [column(Relation, relation::KICKED)],
        active_clause => [column(Relation, relation::ACTIVE_CLAUSE)],
        targets => [column(Relation, relation::KIND), column(Relation, relation::ORDINAL)],
        context => [column(Relation, relation::KIND)],
        sacrifice => [column(Relation, relation::KIND)],
        cost => [column(Relation, relation::HAS_COST), Columns(Relation, relation::MANA_WHITE, 7)],
        hybrid => [Columns(Relation, relation::HYBRID_FIRST, 2)],
        lesson => [column(Relation, relation::RESTRICTED)],
        waterbend => [column(Relation, relation::WATERBEND)],
        can_decline => [column(Relation, relation::CAN_DECLINE)],
        mana_outputs => [column(Relation, relation::MODE), Columns(Relation, relation::MANA_WHITE, 6), column(Relation, relation::RESTRICTED)],
        resolutions_this_turn => [column(Relation, relation::RESOLUTIONS_THIS_TURN)],
        x => [column(Relation, relation::X)],
        granted_keywords => [Columns(Relation, relation::FLYING, KEYWORD_COUNT)],
        this_turn => [column(Relation, relation::THIS_TURN)],
        delayed_frequency => [Rows("v5/v6 delayed frequency relation kind 9, code in column 28")],
    }));
    fields.extend(encoding!(ClauseTarget {
        reference => [Rows("VisibleReference below")],
        clause => [column(Relation, relation::CLAUSE)],
        legal_at_resolution_start => [column(Relation, relation::LEGAL_AT_RESOLUTION_START)],
    }));
    fields.extend(encoding!(VisibleReference {
        object => [Join(Relation, relation::REFERENCE)],
        definition => [Join(Relation, relation::REFERENCE_DEFINITION)],
        player => [column(Relation, relation::REFERENCE_PLAYER)],
    }));
    fields.extend(encoding!(ManaOutput {
        mode => [column(Relation, relation::MODE)],
        mana => [Columns(Relation, relation::MANA_WHITE, 6)],
        lesson => [Columns(Relation, relation::MANA_WHITE, 6), column(Relation, relation::RESTRICTED)],
    }));
    fields.extend(encoding!(ActionSpaceData {
        action_space_type => [Excluded("implied by the offered action types")],
        actions => [Rows("actions")],
        focus => [Rows("action_focus")],
    }));
    fields.extend(encoding!(ActionOption {
        description => [Excluded("text for the selectors encoded beside it")],
        action_type => [Columns(Action, action::PLAY_LAND, 17)],
        focus => [Rows("action_focus")],
        declared => [column(Action, action::DECLARED)],
        mode_index => [column(Action, action::MODE)],
        name_definition => [Rows("name_candidates columns 0-1 (v9)")],
        name_index => [Rows("name_candidates columns 0-1 (v9)")],
        name_group => [Rows("name_candidates columns 2-3 (v9)")],
        ability_index => [column(Action, action::ABILITY)],
        continuation_index => [column(Action, action::CONTINUATION)],
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
const FACT_STRUCTURE: [usize; 2] = [fact::OBJECT, fact::KIND];

/// Structural relation columns: row kind, frame index and reference ordinal.
const RELATION_STRUCTURE: [usize; 3] = [relation::KIND, relation::FRAME, relation::ORDINAL];

#[test]
fn every_observed_field_names_its_encoding_and_every_column_is_claimed() {
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
                Encoded::FactOccurrence(kind) => {
                    assert_ne!(fact_field(kind), field, "{kind:?} is this field's own");
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
                    "{field}: {buffer:?} column {index} is not a feature column"
                );
                claimed.insert((buffer, index));
            }
        }
    }
    for keyword in Keyword::ALL {
        let (object, permanent) = keyword_columns(keyword);
        for (buffer, index) in [(Buffer::Card, object), (Buffer::Permanent, permanent)] {
            assert!(
                index < buffer.width() && Some(index) != buffer.validity(),
                "{keyword:?}: {buffer:?} {index}"
            );
            claimed.insert((buffer, index));
        }
    }
    for offset in 0..KEYWORD_COUNT {
        claimed.insert((Buffer::Relation, relation::FLYING + offset));
    }
    assert_eq!(
        fact_kinds,
        FactKind::ALL.iter().map(|kind| *kind as u8).collect(),
        "every fact kind is claimed by the field it states"
    );
    for index in FACT_STRUCTURE {
        claimed.insert((Buffer::Facts, index));
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
        Buffer::Facts,
        Buffer::PlayerTurnState,
        Buffer::CardTurnState,
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
// Where a card row's values come from on the battlefield.

/// Where a `CardData` field reads while its card is on the battlefield (Jack
/// Heart, 2026-11-02). A layered field reads `Game::characteristics` there
/// and the printed card plus its CDAs elsewhere; a printed field never states a
/// characteristic continuous effects change. The pattern is exhaustive, so a
/// new field does not compile until it is classified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CardSource {
    Layered,
    Printed(&'static str),
}

macro_rules! sources {
    ($ty:ident { $($field:ident => $source:expr),* $(,)? }) => {{
        #[allow(dead_code)]
        fn exhaustive(value: &$ty) {
            let $ty { $($field: _),* } = value;
        }
        vec![$((stringify!($field), $source)),*]
    }};
}

fn card_sources() -> Vec<(&'static str, CardSource)> {
    use CardSource::*;
    sources!(CardData {
        zone => Printed("where the card is"),
        owner_id => Printed("identity"),
        id => Printed("identity"),
        registry_key => Printed("identity: the definition index"),
        name => Layered,
        text_box => Printed("the printed text names the definition; lost abilities are \
            PermanentData.abilities_lost"),
        power => Layered,
        toughness => Layered,
        is_token => Printed("token status is not a characteristic (CR 111.1)"),
        is_legendary => Layered,
        is_ally => Layered,
        is_lesson => Layered,
        ward_cost => Layered,
        kicker_cost => Printed("a cost paid when casting"),
        card_types => Layered,
        keywords => Layered,
        mana_cost => Printed("a cost paid when casting; a copy carries the copied cost"),
        face_down => Printed("viewer state"),
        cast_permission => Printed("a permission to play the card from exile"),
        x => Printed("announced when cast"),
        from_battlefield_this_turn => Printed("turn history of the graveyard object"),
    })
}

/// The card row of `permanent` as the viewer's JSON observation states it.
fn card_json(s: &Scenario, viewer: usize, permanent: PermanentId) -> serde_json::Value {
    let id = s.game().state.objects[card_of(s, permanent)].id.0 as i64;
    let observation: serde_json::Value =
        serde_json::from_str(&Observation::for_player(s.game(), PlayerId(viewer)).to_json())
            .unwrap();
    ["agent_cards", "opponent_cards"]
        .into_iter()
        .flat_map(|rows| observation[rows].as_array().unwrap().clone())
        .find(|object| object["id"].as_i64() == Some(id))
        .expect("card row")
}

/// Every layered card field moves under some continuous effect while its
/// card is on the battlefield, and no printed field moves: records change
/// Gran-Gran's supertype, P/T, keywords, subtypes and types, Dragonfly
/// Swarm loses its ward, and Honest Work renames the Drake.
#[test]
fn layered_card_fields_read_characteristics_on_the_battlefield() {
    let mut s = five_color_setup(&["Gran-Gran", "Dragonfly Swarm", "Honest Work", "Wind Drake"]);
    let gran = s.force_permanent_on_battlefield(0, "Gran-Gran");
    let swarm = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    let witnesses = [gran, swarm, drake];
    let before = witnesses.map(|id| card_json(&s, 0, id));
    let gran_card = s.game().state.permanents[gran].as_ref().unwrap().object;
    s.game_mut().state.objects[gran_card]
        .statics
        .push(StaticAbility::TypeGrant(StaticTypeGrant {
            scope: StaticScope::This,
            types: vec![CardType::Artifact],
            subtypes: Vec::new(),
            condition: None,
        }));
    for modification in [
        Modification::Nonlegendary,
        Modification::ModifyPt {
            power: 1,
            toughness: 1,
        },
        Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
        Modification::AddSubtypes(vec!["Lesson".into()]),
        Modification::BecomeCreature,
    ] {
        until_end_of_turn(&mut s, gran, modification);
    }
    until_end_of_turn(&mut s, swarm, Modification::LoseAllAbilities);
    until_end_of_turn(
        &mut s,
        swarm,
        Modification::AddSubtypes(vec!["Ally".into()]),
    );
    cast(&mut s, "Honest Work");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let after = witnesses.map(|id| card_json(&s, 0, id));

    let mut moved = BTreeSet::new();
    for (before, after) in before.iter().zip(&after) {
        for (field, value) in before.as_object().unwrap() {
            if after[field] != *value {
                moved.insert(field.as_str());
            }
        }
    }
    for (field, source) in card_sources() {
        match source {
            CardSource::Layered => assert!(
                moved.contains(field),
                "{field} is layered, so one of these effects must move it"
            ),
            CardSource::Printed(_) => {
                assert!(!moved.contains(field), "{field} is printed by definition")
            }
        }
    }
    let swarm_row = &after[1];
    assert_eq!(
        swarm_row["ward_cost"], 0,
        "ward is an ability the record removed"
    );
    assert_eq!(swarm_row["is_ally"], true);
    let drake_row = &after[2];
    assert_eq!(drake_row["name"], "Humble Merchant");
    assert_eq!(
        (drake_row["power"].clone(), drake_row["toughness"].clone()),
        (1.into(), 1.into())
    );
}

/// Off the battlefield the row is the printed card: the same records leave a
/// card's row unchanged once it is in the graveyard.
#[test]
fn card_rows_off_the_battlefield_read_the_printed_card() {
    let mut s = five_color_setup(&["Gran-Gran"]);
    let gran = s.force_permanent_on_battlefield(0, "Gran-Gran");
    let object = card_of(&s, gran);
    assert!(s.game().state.objects[object].is_legendary());
    until_end_of_turn(&mut s, gran, Modification::Nonlegendary);
    assert_eq!(card_json(&s, 0, gran)["is_legendary"], false);
    let object = move_card(&mut s, object, ZoneType::Graveyard);
    let id = s.game().state.objects[object].id.0 as i32;
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let row = observation
        .agent_cards
        .iter()
        .find(|row| row.id == id)
        .unwrap();
    assert_eq!(row.zone, ZoneType::Graveyard);
    assert!(
        row.is_legendary,
        "printed again once it has left the battlefield"
    );
}

/// A `Nonlegendary` record outside a copy (CR 205.4): the legend rule
/// spares the permanent and card column 39 clears, both from
/// `Characteristics::is_legendary`.
#[test]
fn a_nonlegendary_record_clears_the_legend_rule_and_card_column_39_together() {
    let mut s = ur_lessons_setup();
    let first = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    let (observation, encoded) = observe(s.game(), 0);
    assert_eq!(card_row(s.game(), &observation, &encoded, first)[39], 1.0);
    until_end_of_turn(&mut s, first, Modification::Nonlegendary);
    assert!(!s.game().characteristics(first).is_legendary);
    let (observation, encoded) = observe(s.game(), 0);
    assert_eq!(card_row(s.game(), &observation, &encoded, first)[39], 0.0);
    let second = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    s.pass_priority();
    assert_ne!(s.action_space().kind, ActionSpaceKind::LegendRule);
    for id in [first, second] {
        assert_eq!(zone_of(&s, card_of(&s, id)), Some(ZoneType::Battlefield));
    }
}

// ---------------------------------------------------------------------------
// Directed states
// ---------------------------------------------------------------------------

fn encoded(s: &Scenario, viewer: usize) -> EncodedObservation {
    let observation = Observation::for_player(s.game(), PlayerId(viewer));
    encode(&observation, &ObservationEncoderConfig::default()).expect("encodes")
}

/// `viewer`'s encoding and the index of its shared-effect header row for the
/// modification `name`.
fn effect_row(s: &Scenario, viewer: usize, name: &str) -> (EncodedObservation, usize) {
    let encoded = encoded(s, viewer);
    let id = id_columns(name_value(name));
    let row = encoded
        .effects
        .as_chunks::<18>()
        .0
        .iter()
        .position(|row| row[17] == 1.0 && row[..2] == id)
        .unwrap_or_else(|| panic!("no shared effect {name}"));
    (encoded, row)
}

/// The flattened object, relation, action and event buffers; fact and
/// shared-effect rows are compared separately.
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
    let encoded = encoded(s, viewer);
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

/// Two states that differ exactly at `column` of `id`.
fn assert_column_alone_sees(
    baseline: &Scenario,
    changed: &Scenario,
    viewer: usize,
    id: PermanentId,
    column: usize,
) {
    assert_column_sees(baseline, changed, viewer, id, column, 1);
}

/// As [`assert_column_alone_sees`], when the change also shows in other columns.
fn assert_column_sees(
    baseline: &Scenario,
    changed: &Scenario,
    viewer: usize,
    id: PermanentId,
    column: usize,
    differences: usize,
) {
    let (before, after) = (
        permanent_value(baseline, viewer, id, column),
        permanent_value(changed, viewer, id, column),
    );
    assert_ne!(before, after, "column {column} misses the change");
    let differing = flat(&encoded(baseline, viewer))
        .iter()
        .zip(flat(&encoded(changed, viewer)))
        .filter(|(a, b)| **a != *b)
        .count();
    assert_eq!(differing, differences, "columns that differ");
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

/// `viewer`'s fact rows about the object at `object_index`.
fn facts_about(s: &Scenario, viewer: usize, object_index: usize) -> Vec<Vec<f32>> {
    encoded(s, viewer)
        .facts
        .as_chunks::<FACT_DIM>()
        .0
        .iter()
        .filter(|row| row[fact::VALID] == 1.0 && row[fact::OBJECT] == (object_index + 1) as f32)
        .map(|row| row.to_vec())
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
            row[fact::KIND] == kind as u8 as f32
                && row[fact::ID_HIGH..fact::ID_HIGH + 2] == id_columns(id)
        })
        .unwrap_or_else(|| panic!("no {kind:?} fact {id}: {rows:?}"))
        .clone()
}

/// Two states only the fact rows tell apart: every object, relation, action
/// and event buffer is equal.
fn assert_facts_alone_see(baseline: &Scenario, changed: &Scenario, viewer: usize) {
    assert_eq!(
        flat(&encoded(baseline, viewer)),
        flat(&encoded(changed, viewer)),
        "a column tells the states apart"
    );
    assert_ne!(
        encoded(baseline, viewer).facts,
        encoded(changed, viewer).facts,
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
        if let Some(keyword) = kind.keyword() {
            // CR 122.1b: the counter grants its keyword, which the keyword
            // columns see; grant it in both states so only the counter differs.
            let grant = Modification::AddKeywords(Keywords::of(&[keyword]));
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
        assert_eq!(row[fact::VALUE], 0.1, "{kind:?}");
        if matches!(
            kind,
            CounterKind::PlusOnePlusOne | CounterKind::MinusOneMinusOne
        ) {
            continue; // Columns see these counters and the power and toughness they change.
        }
        assert_facts_alone_see(&baseline, &changed, 0);
    }
}

#[test]
fn a_stunned_tapped_creature_differs_from_an_unstunned_one() {
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
        .permanent_object(if on_first { first } else { second });
    s.game_mut().state.permanents[boomerang]
        .as_mut()
        .unwrap()
        .attached_to = host;
    (s, boomerang, [first, second])
}

#[test]
fn equipment_on_one_creature_differs_from_equipment_on_another() {
    let (on_first, boomerang, [first, second]) = equipped(true);
    let (on_second, ..) = equipped(false);
    assert_eq!(
        permanent_value(&on_first, 0, boomerang, permanent::ATTACHED_TO),
        (object_index(&on_first, 0, first) + 1) as f32
    );
    assert_eq!(
        permanent_value(&on_second, 0, boomerang, permanent::ATTACHED_TO),
        (object_index(&on_second, 0, second) + 1) as f32
    );
    assert_column_alone_sees(&on_first, &on_second, 0, boomerang, permanent::ATTACHED_TO);
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
fn stack_x_one_and_three_differ() {
    let one = united_front_with_x(1);
    let three = united_front_with_x(3);
    let x = |s: &Scenario| encoded(s, 0).relations[relation::X];
    assert_eq!((x(&one), x(&three)), (1.0, 3.0), "the stack frame is row 0");
    // The opponent sees the announced X of a spell on the stack too.
    let opponent_x = |s: &Scenario| encoded(s, 1).relations[relation::X];
    assert_eq!((opponent_x(&one), opponent_x(&three)), (1.0, 3.0));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TophState {
    Absent,
    Present,
    Abilityless,
}

fn tile_beside(toph: TophState) -> (Scenario, PermanentId) {
    let mut s = five_color_unfunded_setup(&["Toph, the First Metalbender", "White Lotus Tile"]);
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    if toph != TophState::Absent {
        let permanent = s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
        if toph == TophState::Abilityless {
            let affected = on_permanent(&s, permanent);
            apply(&mut s, affected, Modification::LoseAllAbilities);
        }
    }
    (s, tile)
}

/// A granted land type reaches the card row's land column (layer 4 types)
/// in every version; v5 also marks the permanent as granted rather than
/// printed. Removing Toph's abilities in layer 6 cannot undo the type effect
/// that already applied in layer 4 (CR 613.6).
#[test]
fn a_toph_granted_land_shows_in_its_card_row_and_the_granted_column() {
    let (plain, tile) = tile_beside(TophState::Absent);
    let (granted, _) = tile_beside(TophState::Present);
    let (abilityless, _) = tile_beside(TophState::Abilityless);
    let land = |s: &Scenario| {
        let observation = Observation::for_player(s.game(), PlayerId(0));
        card_row(s.game(), &observation, &encoded(s, 0), tile)[11]
    };
    let granted_land = |s: &Scenario| permanent_value(s, 0, tile, permanent::GRANTED_LAND);
    assert_eq!(
        (land(&plain), land(&granted), land(&abilityless)),
        (0.0, 1.0, 1.0)
    );
    assert_eq!(
        (
            granted_land(&plain),
            granted_land(&granted),
            granted_land(&abilityless),
        ),
        (0.0, 1.0, 1.0)
    );
}

#[test]
fn granted_firebending_reaches_its_permanent_column_and_an_effect_row() {
    let (baseline, ogre) = ogre_state();
    let (mut changed, _) = ogre_state();
    let affected = on_permanent(&changed, ogre);
    apply(&mut changed, affected, Modification::Firebending(4));
    assert_eq!(
        permanent_value(&changed, 0, ogre, permanent::GRANTED_FIREBENDING),
        0.4
    );
    let (encoded, row) = effect_row(&changed, 0, "Modification::Firebending");
    assert!(encoded.effects[18 * row + 2..18 * row + 5].contains(&0.4));
    assert_column_alone_sees(&baseline, &changed, 0, ogre, permanent::GRANTED_FIREBENDING);
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
        let (encoded, row) = effect_row(&changed, 0, "Modification::AddSubtypes");
        let header = &encoded.effects[18 * row..18 * (row + 1)];
        assert_eq!(header[14], 1.0, "one detail: the subtype");
        let detail = 3 * header[13] as usize;
        assert_eq!(encoded.effect_details[detail], 2.0);
        assert_eq!(
            encoded.effect_details[detail + 1..detail + 3],
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
                    .contains(&row[fact::KIND])
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
        permanent_value(&s, 0, work, permanent::ATTACHED_TO),
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
        let encoded = encode(&changed, &ObservationEncoderConfig::default()).unwrap();
        let relation = &encoded.relations[relation::FLYING..][..KEYWORD_COUNT];
        assert_eq!(relation.iter().position(|v| *v == 1.0), Some(position));
        assert_eq!(relation.iter().sum::<f32>(), 1.0, "{position}");
    }
}

#[test]
fn lo_and_li_grant_lifelink_to_a_lesson_on_the_stack() {
    let lifelink = relation::FLYING + 9;
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
        encoded(&s, 1).relations
    };
    let (granted, plain) = (lesson(true), lesson(false));
    assert_eq!(granted[3], 1.0, "the top frame is the spell");
    assert_eq!((granted[lifelink], plain[lifelink]), (1.0, 0.0));
}

/// Every waiting delayed trigger's event → its frame's `decision` code and
/// whether its watched object or remembered context is a reference row.
fn delayed_event_seen(event: &DelayedTriggerEvent) -> (u8, bool) {
    match event {
        DelayedTriggerEvent::Leaves {
            or_exiled: false, ..
        } => (1, true),
        DelayedTriggerEvent::Leaves {
            or_exiled: true, ..
        } => (2, true),
        DelayedTriggerEvent::CreatureDies => (3, false),
        DelayedTriggerEvent::EndStep { your_turn } => (if *your_turn { 7 } else { 4 }, false),
        DelayedTriggerEvent::YouCastSpell(_) => (5, false),
        DelayedTriggerEvent::Upkeep => (6, false),
    }
}

#[test]
fn every_delayed_trigger_event_has_its_own_relation_code() {
    let watched = managym::state::zone::ZoneManager::default().allocate();
    let events = [
        DelayedTriggerEvent::Leaves {
            watched,
            or_exiled: false,
        },
        DelayedTriggerEvent::Leaves {
            watched,
            or_exiled: true,
        },
        DelayedTriggerEvent::CreatureDies,
        DelayedTriggerEvent::EndStep { your_turn: false },
        DelayedTriggerEvent::EndStep { your_turn: true },
        DelayedTriggerEvent::YouCastSpell(Box::new(CardPredicate::subtype("Lesson"))),
        DelayedTriggerEvent::Upkeep,
    ];
    let codes: BTreeSet<u8> = events.iter().map(delayed_event_code).collect();
    assert_eq!(codes.len(), events.len());
    for event in &events {
        assert_eq!(delayed_event_seen(event).0, delayed_event_code(event));
    }
}

/// Fatal Fissure's waiting trigger is a public relation row for both
/// viewers — its source, controller, event, watched creature and "this turn"
/// — until it fires or ends at cleanup.
#[test]
fn a_waiting_delayed_trigger_is_observed_until_it_fires_or_expires() {
    let waiting = |s: &Scenario, viewer: usize| -> Option<(f32, f32, f32, f32)> {
        let encoded = encoded(s, viewer);
        let rows = encoded.relations.as_chunks::<RELATION_DIM>().0;
        let header = rows
            .iter()
            .position(|row| row[0] == 1.0 && row[3] == 10.0)?;
        let context = rows
            .iter()
            .find(|row| row[0] == 3.0 && row[3] == 10.0)
            .map_or(0.0, |row| row[24]);
        let row = rows[header];
        Some((row[4], row[28], row[relation::THIS_TURN], context))
    };
    let setup = || {
        let mut s = five_color_setup(&["Fatal Fissure", "Lightning Strike", "Gray Ogre"]);
        s.force_permanent_on_battlefield(0, "Forest");
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        assert_eq!(waiting(&s, 0), None);
        cast(&mut s, "Fatal Fissure");
        assert!(s.choose_target(Target::Permanent(ogre)));
        settle(&mut s);
        (s, ogre)
    };
    let (mut s, ogre) = setup();
    let fissure = s.game().state.delayed_triggers[0].source_object;
    for viewer in [0, 1] {
        let (controller, event, this_turn, watched) = waiting(&s, viewer).expect("visible");
        assert_eq!(
            (controller, event, this_turn),
            (if viewer == 0 { 1.0 } else { 2.0 }, 1.0, 1.0)
        );
        assert!(watched > 0.0, "the watched creature is joined");
        let observation = encoded(&s, viewer);
        let remembered = observation
            .relations
            .as_chunks::<RELATION_DIM>()
            .0
            .iter()
            .find(|row| row[0] == 8.0 && row[3] == 10.0)
            .expect("captured subject row");
        assert_eq!(
            remembered[24], watched,
            "capture and event watch are distinct relations"
        );
        assert!(definitions_seen(&s, viewer).contains(&current_object(&s, fissure).definition_id.0));
    }
    let frame = Observation::for_player(s.game(), PlayerId(1))
        .continuation
        .into_iter()
        .find(|frame| frame.kind == 10)
        .unwrap();
    assert_eq!(
        frame.context.and_then(|c| c.object),
        s.game().state.permanents[ogre]
            .as_ref()
            .map(|permanent| permanent.id.0 as i32),
        "the watched permanent"
    );
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(waiting(&s, 0), None, "ended at cleanup");

    let (mut s, ogre) = setup();
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(ogre)));
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert_eq!(waiting(&s, 0), None, "fired");
}

#[test]
fn turn_number_and_seat_reach_the_player_rows() {
    let mut s = five_color_unfunded_setup(&[]);
    let turn = |s: &Scenario, viewer: usize| {
        let encoded = encoded(s, viewer);
        (
            encoded.agent_player[player::TURN],
            encoded.agent_player[player::SEAT],
            encoded.opponent_player[player::SEAT],
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
    let encoded = encoded(s, viewer);
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
        .as_chunks::<RELATION_DIM>()
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
    let comet = move_card(&mut s, comet, ZoneType::Hand);
    let front = move_to_hand(&mut s, "United Front");
    s.set_player_mana_pool(0, "3");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { object, .. } if *object == comet),
    );
    assert_eq!(zone_of(&s, comet), Some(ZoneType::Exile));
    let hidden = [comet, front].map(|object| current_object(&s, object).definition_id.0);

    let owner = encoded(&s, 0);
    let face_down = owner
        .agent_cards
        .as_chunks::<CARD_DIM>()
        .0
        .iter()
        .filter(|row| row[card::PERMISSION_FORETOLD] == 1.0)
        .count();
    assert_eq!(face_down, 1, "its owner sees the foretold card");
    for definition in hidden {
        assert!(definitions_seen(&s, 0).contains(&definition));
        assert!(
            !definitions_seen(&s, 1).contains(&definition),
            "definition {definition} leaks to the opponent"
        );
    }
    let opponent = encoded(&s, 1);
    assert!(opponent
        .opponent_cards
        .as_chunks::<CARD_DIM>()
        .0
        .iter()
        .all(|row| row[card::PERMISSION_FORETOLD] == 0.0));
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
fn blocking_one_attacker_differs_from_blocking_another() {
    let (mut on_first, blocker, [first, second]) = combat_state();
    let (mut on_second, ..) = combat_state();
    declare(&mut on_first, &[first, second], Some((first, blocker)));
    declare(&mut on_second, &[first, second], Some((second, blocker)));
    assert_eq!(
        permanent_value(&on_first, 0, blocker, permanent::BLOCKING),
        (object_index(&on_first, 0, first) + 1) as f32
    );
    assert_column_alone_sees(&on_first, &on_second, 0, blocker, permanent::BLOCKING);
}

#[test]
fn a_creature_that_lost_its_abilities_differs() {
    let (baseline, ogre) = ogre_state();
    let (mut changed, _) = ogre_state();
    let affected = on_permanent(&changed, ogre);
    apply(&mut changed, affected, Modification::LoseAllAbilities);
    assert_column_alone_sees(&baseline, &changed, 0, ogre, permanent::ABILITIES_LOST);
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
fn a_spent_exhaust_ability_differs_from_an_unused_one() {
    let (unused, mai_id) = mai(false);
    let (spent, _) = mai(true);
    let rows = facts_about(&spent, 0, object_index(&spent, 0, mai_id));
    // Mai's only activated ability is exhaust: index 0.
    fact_row(&rows, FactKind::UsedAbility, 0);
    assert_eq!(
        permanent_value(&spent, 0, mai_id, permanent::EXHAUSTED_ABILITIES),
        1.0,
        "the exhausted count moves with it"
    );
    assert_column_alone_sees(&unused, &spent, 0, mai_id, permanent::EXHAUSTED_ABILITIES);
}

#[test]
fn kept_red_mana_and_no_hand_size_reach_the_player_rows() {
    for column in [player::KEEPS_RED_MANA, player::NO_MAXIMUM_HAND_SIZE] {
        let (baseline, _) = ogre_state();
        let (mut changed, _) = ogre_state();
        let modification = if column == player::KEEPS_RED_MANA {
            Modification::KeepUnspentMana(Color::Red)
        } else {
            Modification::NoMaximumHandSize
        };
        apply(&mut changed, Affected::Player(PlayerId(1)), modification);
        let (before, after) = (encoded(&baseline, 0), encoded(&changed, 0));
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

#[test]
fn sacrifices_this_turn_reach_the_player_row() {
    let (baseline, _) = ogre_state();
    let (mut changed, _) = ogre_state();
    // Sacrifice-count bookkeeping itself is covered by Phoenix Fleet
    // Airship's tests; this checks only the encoding.
    changed.game_mut().state.turn.history[1].permanents_sacrificed = 1;
    let (before, after) = (encoded(&baseline, 0), encoded(&changed, 0));
    assert_eq!(
        (
            before.opponent_player[player::PERMANENTS_SACRIFICED],
            after.opponent_player[player::PERMANENTS_SACRIFICED]
        ),
        (0.0, 0.1)
    );
}

// ---------------------------------------------------------------------------
// Cast permissions, restricted mana and announced X
// ---------------------------------------------------------------------------

fn encoded_default(observation: &Observation) -> EncodedObservation {
    encode(observation, &ObservationEncoderConfig::default()).expect("v5 encodes")
}

fn object_id(s: &Scenario, object: ObjectId) -> i32 {
    current_object(s, object).id.0 as i32
}

/// The encoded row of `card` on either side of `observation`, if listed.
fn listed_card_row(observation: &Observation, object: i32) -> Option<Vec<f32>> {
    let encoded = encoded_default(observation);
    for (objects, rows) in [
        (&observation.agent_cards, &encoded.agent_cards),
        (&observation.opponent_cards, &encoded.opponent_cards),
    ] {
        if let Some(slot) = objects.iter().position(|c| c.id == object) {
            return Some(rows[slot * CARD_DIM..(slot + 1) * CARD_DIM].to_vec());
        }
    }
    None
}

fn permission_columns(row: &[f32]) -> &[f32] {
    &row[card::PERMISSION..card::PERMISSION_THROUGH_NEXT_TURN]
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
    assert_eq!(
        owner[card::PERMISSION_IS_MINE],
        1.0,
        "the owner may cast it"
    );
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
    assert_eq!(baseline.opponent_player[player::HAND_ONLY], 0.0);
    cast(&mut s, "Avatar's Wrath");
    assert!(s.choose_target(Target::Permanent(spared)));
    settle(&mut s);

    let caster = Observation::for_player(s.game(), PlayerId(0));
    assert_eq!(caster.opponent.hand_only_until, Some(caster.agent.id));
    assert_eq!(
        encoded_default(&caster).opponent_player[player::HAND_ONLY],
        1.0
    );
    assert_eq!(
        encoded_default(&caster).agent_player[player::HAND_ONLY],
        0.0
    );
    let locked = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(
        encoded_default(&locked).agent_player[player::HAND_ONLY],
        1.0
    );
    assert_eq!(
        listed_card_row(&locked, wurm_id).unwrap()[card::PERMISSION],
        1.0
    );
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
    let by_subtype = &player[player::RESTRICTED_LESSON_WHITE..player::HAND_ONLY];
    let mut expected = [0.0; 18];
    expected[12 + 4] = 0.1; // Ally, green
    assert_eq!(by_subtype, expected);
}

/// Animal Attendant's mana counts as ordinary green and states its spend
/// effect as a program on its player, in catalog tokens, for both viewers.
#[test]
fn pooled_mana_classes_are_programs_on_their_player() {
    let mut s = five_color_setup(&["Animal Attendant"]);
    let attendant = s.force_permanent_on_battlefield(0, "Animal Attendant");
    clear_summoning_sickness(&mut s, attendant);
    s.set_player_mana_pool(0, "");
    activate(&mut s, attendant, 0);
    choose_mode(&mut s, 4);

    for viewer in [0, 1] {
        let observation = Observation::for_player(s.game(), PlayerId(viewer));
        let owner = if viewer == 0 {
            &observation.agent
        } else {
            &observation.opponent
        };
        assert_eq!(owner.mana, [0, 0, 0, 0, 1, 0]);
        assert_eq!(owner.lesson_mana, [0; 6]);
        let classes = &owner.pooled_mana_classes;
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].mana, [0, 0, 0, 0, 1, 0]);
        assert!(classes[0]
            .effect
            .as_ref()
            .is_some_and(|effect| effect.contains("non-Human creature spell")));
        assert_eq!(
            managym::agent::observation_encoder::unrepresentable_state(&observation),
            None
        );

        let encoded = encoded_default(&observation);
        let pooled = id_columns(name_value(managym::agent::observation_encoder::POOLED_MANA));
        let rows: Vec<&[f32]> = encoded
            .facts
            .chunks(FACT_DIM)
            .filter(|row| {
                row[fact::KIND] == FactKind::EffectProgram as u8 as f32
                    && row[fact::ID_HIGH..fact::ID_HIGH + 2] == pooled
            })
            .collect();
        assert_eq!(rows.len(), classes[0].program.len());
        let owner_row = if viewer == 0 { 1.0 } else { 2.0 };
        for (ordinal, (row, &(kind, value))) in rows.iter().zip(&classes[0].program).enumerate() {
            assert_eq!(row[fact::OBJECT], owner_row);
            assert_eq!(row[fact::VALUE], f32::from(kind));
            assert_eq!(row[fact::VALUE + 1], ordinal as f32);
            assert_eq!(
                row[fact::DETAIL_HIGH..fact::DETAIL_HIGH + 2],
                id_columns(value)
            );
        }
        // Every program token is one the Attendant's own definition spells.
        let definition = managym_cards::catalog()
            .iter()
            .find(|definition| definition.name == "Animal Attendant")
            .unwrap();
        let tokens = tokens_of(&ObjectDefinition::card(definition));
        let literal = [TokenKind::Integer, TokenKind::Boolean, TokenKind::Null].map(|k| k as u16);
        for &(kind, value) in &classes[0].program {
            assert!(
                literal.contains(&kind)
                    || tokens.iter().any(|&(k, v)| k as u16 == kind && v == value),
                "program token ({kind}, {value}) is outside the definition's vocabulary"
            );
        }
    }
}

/// The mana spent to cast a spell that counts it (converge) is an execution
/// occurrence of its stack frame for both viewers, in registered tokens; a
/// spell no rule reads the receipt of has none.
#[test]
fn cast_receipt_reaches_the_stack_frame_only_when_a_rule_reads_it() {
    let mut s = five_color_setup(&["Uncle's Musings", "Kyoshi Warrior Guard"]);
    s.set_player_mana_pool(0, "WUGG");
    cast(&mut s, "Uncle's Musings");
    // The policy's token vocabulary: every definition's tokens and the
    // registered runtime names.
    let vocabulary: std::collections::HashSet<(u16, i32)> =
        managym::agent::record_vocabulary::record_names()
            .into_iter()
            .filter_map(|name| name.token)
            .chain(
                managym_cards::catalog()
                    .iter()
                    .flat_map(|definition| tokens_of(&ObjectDefinition::card(definition)))
                    .map(|(kind, value)| (kind as u16, value)),
            )
            .collect();
    let literal = [TokenKind::Integer, TokenKind::Boolean, TokenKind::Null].map(|k| k as u16);
    let marker = (TokenKind::Text as u16, name_value("mana spent"));
    for viewer in [0, 1] {
        let observation = Observation::for_player(s.game(), PlayerId(viewer));
        let frame = &observation.continuation[0];
        assert_eq!(frame.kind, 1);
        assert!(frame.effect_execution.contains(&marker));
        for token in &frame.effect_execution {
            assert!(
                literal.contains(&token.0) || vocabulary.contains(token),
                "unregistered execution token {token:?}"
            );
        }
        let rows = encoded_default(&observation).facts;
        let programs = rows
            .chunks(FACT_DIM)
            .filter(|row| row[fact::KIND] == FactKind::ContinuationProgram as u8 as f32)
            .count();
        assert_eq!(programs, frame.effect_execution.len());
    }

    let mut s = five_color_setup(&["Kyoshi Warrior Guard"]);
    s.set_player_mana_pool(0, "1W");
    cast(&mut s, "Kyoshi Warrior Guard");
    let observation = Observation::for_player(s.game(), PlayerId(0));
    assert_eq!(observation.continuation[0].kind, 1);
    assert!(observation.continuation[0].effect_execution.is_empty());
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
    let object = observation.agent_cards.iter().find(|c| c.id == id).unwrap();
    assert_eq!(object.x, Some(3));
    let row = listed_card_row(&observation, id).unwrap();
    assert_eq!(&row[card::HAS_X..card::HAS_X + 2], &[1.0, 0.3]);
}

// --- Viewer safety -------------------------------------------------------

/// A foretold card and its permission reach only its owner: the owner's
/// tensor marks it foretold and not yet castable; the opponent's tensor and
/// JSON hold no trace of it beyond the exile count.
#[test]
fn foretold_card_and_permission_reach_only_the_owner() {
    let mut s = five_color_setup(&["Sozin's Comet"]);
    let comet = library_card(&s, 0, "Sozin's Comet");
    let comet = move_card(&mut s, comet, ZoneType::Hand);
    s.set_player_mana_pool(0, "2");
    let opponent_before = encoded_default(&Observation::for_player(s.game(), PlayerId(1)));
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { object, .. } if *object == comet),
    );
    let id = object_id(&s, comet);

    let owner = Observation::for_player(s.game(), PlayerId(0));
    let row = listed_card_row(&owner, id).expect("the owner sees it");
    assert_eq!(row[card::PERMISSION], 1.0);
    assert_eq!(
        row[card::PERMISSION_NOT_YET],
        1.0,
        "castable only on a later turn"
    );
    assert_eq!(row[card::PERMISSION_FORETOLD], 1.0, "foretold");

    let opponent = Observation::for_player(s.game(), PlayerId(1));
    assert!(listed_card_row(&opponent, id).is_none());
    let json = opponent.to_json();
    assert!(!json.contains("Sozin") && !json.contains("foretold") && !json.contains("face_down"));
    let after = encoded_default(&opponent);
    for rows in [&after.agent_cards, &after.opponent_cards] {
        assert!(rows
            .chunks(CARD_DIM)
            .all(|row| row[card::PERMISSION_FORETOLD] == 0.0 && row[card::PERMISSION] == 0.0));
    }
    assert_eq!(after.opponent_cards, opponent_before.opponent_cards);

    // On its owner's next turn the later-turn term clears.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let row = listed_card_row(&Observation::for_player(s.game(), PlayerId(0)), id).unwrap();
    assert_eq!(row[card::PERMISSION_NOT_YET], 0.0);
}

// ---------------------------------------------------------------------------
// Continuous-effect records
// ---------------------------------------------------------------------------

#[test]
fn each_modification_reaches_a_shared_effect_row_with_its_duration_and_order() {
    for (on_player, modification) in modification_witnesses() {
        let (mut s, ogre) = ogre_state();
        let affected = if on_player {
            Affected::Player(PlayerId(1))
        } else {
            on_permanent(&s, ogre)
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
        // The opponent sees every record too.
        for viewer in [0, 1] {
            let (encoded, row) = effect_row(&s, viewer, name);
            let header = &encoded.effects[18 * row..18 * (row + 1)];
            assert_eq!(
                header[8],
                duration_code(EffectDuration::Indefinite),
                "{name}"
            );
            assert!(header[9] > 0.0, "{name}: the record is ordered");
            assert_eq!(header[16], 1.0, "{name}: one affected object");
        }
    }
}

#[test]
fn a_token_copy_reaches_copy_and_exception_fact_rows_for_both_viewers() {
    let mut s = five_color_setup(&["Ran and Shaw", "Waterbending Lesson"]);
    for _ in 0..3 {
        let lesson = library_card(&s, 0, "Waterbending Lesson");
        move_card(&mut s, lesson, ZoneType::Graveyard);
    }
    cast(&mut s, "Ran and Shaw");
    settle(&mut s);
    let dragons = s.battlefield_permanents_named(0, "Ran and Shaw");
    let is_tok =
        |id: PermanentId| s.game().state.objects[card_of(&s, id)].kind() == ObjectKind::Token;
    let token = *dragons.iter().find(|id| is_tok(**id)).unwrap();
    let source = *dragons.iter().find(|id| !is_tok(**id)).unwrap();
    let key = s.game().state.objects[card_of(&s, source)].registry_key.0 as f32;
    for viewer in [0, 1] {
        let rows = facts_about(&s, viewer, object_index(&s, viewer, token));
        let copy = fact_row(&rows, FactKind::Copy, name_value("CopyEffect"));
        assert_eq!(
            copy[fact::SOURCE_DEFINITION],
            key + 1.0,
            "joined to the copied definition"
        );
        assert!(
            copy[fact::TIMESTAMP] > 0.0,
            "ordered by the copy's timestamp"
        );
        let exception = fact_row(
            &rows,
            FactKind::CopyException,
            name_value("Modification::Nonlegendary"),
        );
        assert_eq!(exception[fact::SOURCE_DEFINITION], key + 1.0);
        let original = facts_about(&s, viewer, object_index(&s, viewer, source));
        assert!(
            original
                .iter()
                .all(|row| row[fact::KIND] != FactKind::Copy as u8 as f32),
            "the original is not a copy"
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
        Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
    );
    let order = |name: &str| {
        let (encoded, row) = effect_row(&s, 0, name);
        encoded.effects[18 * row + 9]
    };
    assert!(order("Modification::AddKeywords") > order("Modification::LoseAllAbilities"));
    assert!(
        permanent_value(&s, 0, ogre, permanent::TIMESTAMP)
            < order("Modification::LoseAllAbilities")
    );
    // CR 613.7: the later grant survives the earlier loss.
    assert!(
        managym::agent::observation::Observation::for_player(s.game(), PlayerId(0))
            .opponent_permanents[0]
            .keywords
            .flying
    );
}

#[test]
fn transforms_and_scheduled_combat_restrictions_reach_policy_facts() {
    let mut s = five_color_setup(&["Aang, Swift Savior"]);
    let aang = s.force_permanent_on_battlefield(0, "Aang, Swift Savior");
    let baseline = encoded(&s, 0);
    s.game_mut().transform_permanent(aang);
    s.game_mut().transform_permanent(aang);
    let flipped = encoded(&s, 0);
    assert_ne!(baseline.facts, flipped.facts, "a round trip is visible");
    s.game_mut()
        .state
        .turn
        .additional_combats
        .push(managym::flow::turn::AdditionalCombat {
            attackers: Some(CardPredicate::creature()),
        });
    let scheduled = encoded(&s, 0);
    assert_ne!(flipped.facts, scheduled.facts);
    assert_ne!(flipped.relations, scheduled.relations);
    let before = Observation::for_player(s.game(), PlayerId(0));
    s.game_mut().state.turn.extra_turns = vec![PlayerId(0), PlayerId(1)];
    let order = encoded(&s, 0).relations;
    s.game_mut().state.turn.extra_turns.reverse();
    assert_ne!(
        order,
        encoded(&s, 0).relations,
        "same turn counts, different order"
    );
    assert!(before
        .continuation
        .iter()
        .any(|f| f.kind == 12 && f.attackers.is_some()));
}

#[test]
fn off_battlefield_cda_power_reaches_card_tensor_and_learning_tokens() {
    let mut s = five_color_setup(&["Suki, Kyoshi Warrior", "Gray Ogre"]);
    s.force_card_in_hand(0, "Suki, Kyoshi Warrior");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let suki = observation
        .agent_cards
        .iter()
        .find(|c| c.name == "Suki, Kyoshi Warrior" && c.zone == ZoneType::Hand)
        .unwrap();
    let row = listed_card_row(&observation, suki.id).unwrap();
    assert_eq!(row[8], 0.2);
    let tokens = tokens_of(&card_definition("Suki, Kyoshi Warrior"));
    assert!(tokens.contains(&(
        TokenKind::Variant,
        name_value("PowerCda::CreaturesYouControl")
    )));
}

#[test]
fn library_order_and_any_target_are_distinct_learning_rules() {
    let effect = |random_order| Effect::LookAndSelect {
        random_order,
        to_battlefield: false,
        reveal: false,
        look: EffectValue::Fixed(3),
        min_select: 1,
        max_select: 1,
        predicate: CardPredicate::default(),
    };
    assert_ne!(
        tokens_of(&spell(vec![effect(false)])),
        tokens_of(&spell(vec![effect(true)]))
    );
    let tokens = tokens_of(&card_definition("Lightning Strike"));
    assert!(tokens.contains(&(TokenKind::Variant, name_value("TargetSpec::AnyTarget"))));
}

/// Secret of Bloodbending's scheduled/active authority is public state;
/// the paid and unpaid durations have distinct rule tokens and tensor rows.
#[test]
fn secret_of_bloodbending_control_reaches_tokens_and_native_policy_relations() {
    use managym::flow::turn::PlayerControl;
    use managym::state::ability::PlayerControlDuration;
    let mut s = five_color_setup(&["Secret of Bloodbending"]);
    let baseline = encoded(&s, 0);
    s.game_mut().state.turn.player_controls.push(PlayerControl {
        player: PlayerId(1),
        controller: PlayerId(0),
        duration: PlayerControlDuration::NextCombat,
        active: false,
    });
    let pending = encoded(&s, 0);
    assert_ne!(pending.relations, baseline.relations);
    s.game_mut().state.turn.player_controls[0].duration = PlayerControlDuration::NextTurn;
    let whole_turn = encoded(&s, 0);
    assert_ne!(pending.relations, whole_turn.relations);
    s.game_mut().state.turn.player_controls[0].active = true;
    let active = encoded(&s, 0);
    assert_ne!(active.relations, whole_turn.relations);
    let headers: Vec<_> = active
        .relations
        .as_chunks::<RELATION_DIM>()
        .0
        .iter()
        .filter(|r| r[0] == 1.0 && r[3] == 13.0)
        .collect();
    assert_eq!(headers.len(), 1);
    assert_eq!(
        (headers[0][4], headers[0][7], headers[0][28]),
        (1.0, 2.0, 2.0)
    );
    assert!(active
        .relations
        .as_chunks::<RELATION_DIM>()
        .0
        .iter()
        .any(|r| r[0] == 3.0 && r[3] == 13.0 && r[25] == 2.0));
    let tokens = tokens_of(&card_definition("Secret of Bloodbending"));
    for variant in [
        "Effect::ControlPlayer",
        "PlayerControlDuration::NextCombat",
        "PlayerControlDuration::NextTurn",
    ] {
        assert!(
            tokens.contains(&(TokenKind::Variant, name_value(variant))),
            "{variant}"
        );
    }
}

#[test]
fn delayed_lifetime_reaches_tokens_and_relation_rows_independently() {
    for frequency in [DelayedFrequency::Once, DelayedFrequency::EachMatch] {
        for expiry in [DelayedExpiry::ThisTurn, DelayedExpiry::Never] {
            let tokens = tokens_of(&spell(vec![Effect::CreateDelayedTrigger {
                event: DelayedEvent::Upkeep,
                frequency,
                expiry,
                remember: None,
                effects: vec![Effect::gain_life(1)],
            }]));
            assert!(tokens.contains(&(
                TokenKind::Variant,
                name_value(match frequency {
                    DelayedFrequency::Once => "DelayedFrequency::Once",
                    DelayedFrequency::EachMatch => "DelayedFrequency::EachMatch",
                })
            )));
            assert!(tokens.contains(&(
                TokenKind::Variant,
                name_value(match expiry {
                    DelayedExpiry::ThisTurn => "DelayedExpiry::ThisTurn",
                    DelayedExpiry::Never => "DelayedExpiry::Never",
                })
            )));
            let code = if frequency == DelayedFrequency::Once {
                1
            } else {
                2
            };
            let frame = ContinuationFrame {
                kind: 10,
                decision: 6,
                delayed_frequency: Some(code),
                this_turn: expiry == DelayedExpiry::ThisTurn,
                ..Default::default()
            };
            let mut rows = vec![0.0; 64 * RELATION_DIM];
            managym::agent::continuation::encode_into(&[frame], 0, &Default::default(), &mut rows)
                .unwrap();
            assert_eq!(rows[28], 6.0, "event code stays on header");
            assert_eq!(rows[RELATION_DIM], 9.0);
            assert_eq!(rows[RELATION_DIM + 28], f32::from(code));
            assert_eq!(
                rows[RELATION_DIM + relation::THIS_TURN],
                f32::from(expiry == DelayedExpiry::ThisTurn)
            );
        }
    }
}

/// Replacement definitions reach the same catalog as spell programs; these
/// two cards differ in both causal applicability and transformation, not text.
#[test]
fn counter_and_creation_replacements_reach_learning_tokens() {
    let doubling = tokens_of(&card_definition("Doubling Season"));
    let caradora = tokens_of(&card_definition("Caradora, Heart of Alacria"));
    for variant in [
        "ReplacementEffect::ModifyCounters",
        "ReplacementEffect::MultiplyTokens",
    ] {
        assert!(doubling.contains(&(TokenKind::Variant, name_value(variant))));
    }
    assert!(caradora.contains(&(
        TokenKind::Variant,
        name_value("ReplacementEffect::ModifyCounters")
    )));
    assert!(caradora.contains(&(TokenKind::Text, name_value("Vehicle"))));
    let mut definition = card_definition("Doubling Season");
    let managym::state::card::ReplacementEffect::ModifyCounters { effects_only, .. } =
        Arc::make_mut(&mut definition.printed)
            .replacement_effects
            .get_mut(1)
            .unwrap()
    else {
        panic!("counter rule")
    };
    *effects_only = false;
    assert_ne!(doubling, tokens_of(&definition));
}

#[test]
fn merchant_counted_tap_cost_reaches_learning_tokens() {
    let definition = managym_cards::catalog()
        .iter()
        .find(|object| object.name == "The Cabbage Merchant")
        .unwrap();
    let tokens = tokens_of(&ObjectDefinition::card(definition));
    let cost = tokens
        .iter()
        .position(|token| *token == (TokenKind::Structure, name_value("TapCost")))
        .unwrap();
    let count = tokens[cost..]
        .iter()
        .position(|token| *token == (TokenKind::Field, name_value("count")))
        .unwrap()
        + cost;
    assert_eq!(tokens[count + 1], (TokenKind::Integer, 2));
    assert!(tokens[cost..count].contains(&(TokenKind::Text, name_value("Food"))));
}

#[test]
fn input_v10_carries_turn_state_that_v9_refuses() {
    use crate::support::{
        game::force_new_card_in_hand,
        inspect::card_of,
        play::{cast, settle},
    };
    use managym::agent::observation_encoder::ObservationEncodeError;
    let v9 = ObservationEncoderConfig::default();
    let v10 = ObservationEncoderConfig {
        turn_state: true,
        ..ObservationEncoderConfig::default()
    };
    let refusal = |s: &Scenario| match encode(&Observation::for_player(s.game(), PlayerId(0)), &v9)
    {
        Err(ObservationEncodeError::RequiresInputV10(state)) => Some(state),
        Ok(_) => None,
        Err(error) => panic!("{error}"),
    };
    let encoded = |s: &Scenario| {
        encode(&Observation::for_player(s.game(), PlayerId(0)), &v10).expect("v10 encodes")
    };

    // Life lost is refused only once a visible card reads it.
    let mut s = five_color_setup(&["Bloodchief Ascension", "Lightning Strike"]);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(refusal(&s), None);
    s.force_permanent_on_battlefield(0, "Bloodchief Ascension");
    assert_eq!(refusal(&s), Some("life lost this turn"));
    let row = encoded(&s).player_turn_state;
    assert_eq!(row[turn_state::LIFE_LOST], 0.0);
    assert_eq!(row[TURN_STATE_DIM + turn_state::LIFE_LOST], 3.0 / 20.0);
    // Spells cast this turn, per seat; v9 does not refuse it.
    assert_eq!(row[turn_state::SPELLS_CAST], 0.1);
    assert_eq!(row[TURN_STATE_DIM + turn_state::SPELLS_CAST], 0.0);

    // A graveyard card put there from the battlefield, read by Brought Back
    // in the viewer's hand.
    let mut s = five_color_setup(&["Brought Back", "Gray Ogre", "Lightning Strike"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(refusal(&s), None);
    force_new_card_in_hand(&mut s, 0, "Brought Back");
    assert_eq!(
        refusal(&s),
        Some("a card put into a graveyard from the battlefield this turn")
    );
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let slot = observation
        .agent_cards
        .iter()
        .position(|object| object.id == current_object(&s, ogre_card).id.0 as i32)
        .expect("graveyard row");
    let objects = encoded(&s).card_turn_state;
    assert_eq!(objects[slot], 1.0);
    assert_eq!(objects.iter().sum::<f32>(), 1.0);

    // Powerstone mana in a pool (its production is exercised by the
    // castability test in `tests/cards/bro.rs`).
    let mut s = five_color_setup(&["Gray Ogre"]);
    assert_eq!(refusal(&s), None);
    s.game_mut().state.players[0].mana_pool = managym::state::mana::Mana::restricted_to(
        managym::state::mana::ManaRestriction::ExceptNonartifactSpells,
        [0, 0, 0, 0, 0, 1],
    );
    assert_eq!(
        refusal(&s),
        Some("mana that can't be spent to cast nonartifact spells")
    );
    let row = encoded(&s).player_turn_state;
    assert_eq!(row[turn_state::NONARTIFACT_MANA_COLORLESS], 0.1);
}

#[test]
fn release_and_lita_tokens_preserve_identity_and_expose_their_rules() {
    let pack = default_content_pack();
    let plain = pack
        .definition(pack.token_id("token.plain_spirit").unwrap())
        .unwrap();
    let tla = pack
        .definition(pack.token_id("token.spirit").unwrap())
        .unwrap();
    assert_ne!(tokens_of(plain), tokens_of(tla));
    let zeppelin = pack
        .definition(pack.token_id("token.zeppelin").unwrap())
        .unwrap();
    assert!(tokens_of(zeppelin).contains(&(TokenKind::Variant, name_value("Keyword::flying"))));
    assert!(tokens_of(zeppelin).contains(&(TokenKind::Field, name_value("crew"))));

    let release = pack
        .definition(pack.definition_id("Release to Memory").unwrap())
        .unwrap();
    let rules = tokens_of(release);
    for variant in [
        "EffectValue::Objects",
        "ObjectSelector::Moved",
        "CardType::Creature",
    ] {
        assert!(
            rules.contains(&(TokenKind::Variant, name_value(variant))),
            "{variant}"
        );
    }
    assert!(rules.contains(&(TokenKind::Field, name_value("predicate"))));

    let (mut game, _) = ogre_game();
    let plain_id = game.create_token(plain.key, PlayerId(0), false).unwrap();
    let tla_id = game.create_token(tla.key, PlayerId(0), false).unwrap();
    let vehicle = game.create_token(zeppelin.key, PlayerId(0), false).unwrap();
    let (observation, encoded) = observe(&game, 0);
    let plain_row = card_row(&game, &observation, &encoded, plain_id);
    let tla_row = card_row(&game, &observation, &encoded, tla_id);
    // Shared printed names still bind distinct catalog rows and programs.
    assert_ne!(
        encoded.semantic_cards[card_index(&game, &observation, plain_id)],
        encoded.semantic_cards[card_index(&game, &observation, tla_id)]
    );
    assert_eq!(plain_row[card::FLYING], 0.0);
    assert_eq!(tla_row[card::FLYING], 0.0);
    let vehicle_row = card_row(&game, &observation, &encoded, vehicle);
    assert_eq!(vehicle_row[card::FLYING], 1.0);
    assert_eq!(vehicle_row[card::IS_CREATURE], 0.0);
    assert_eq!(vehicle_row[card::IS_ARTIFACT], 1.0);
    assert_eq!(vehicle_row[card::POWER], 0.5);
    assert_eq!(vehicle_row[card::TOUGHNESS], 0.5);
}

#[test]
fn pooled_mana_expiry_preserves_totals_beyond_one_class_capacity() {
    let mut s = five_color_setup(&["Animal Attendant"]);
    s.set_player_mana_pool(0, "255");
    s.game_mut().state.players[0].combat_mana_pool = managym::state::mana::Mana::parse("1");
    for viewer in [0, 1] {
        let observation = Observation::for_player(s.game(), PlayerId(viewer));
        let player = if viewer == 0 {
            &observation.agent
        } else {
            &observation.opponent
        };
        assert_eq!(player.mana[5], 256);
        assert_eq!(player.pooled_mana_classes.len(), 1);
        assert!(player.pooled_mana_classes[0].combat);
        assert_eq!(player.pooled_mana_classes[0].mana[5], 1);
        let class = &player.pooled_mana_classes[0];
        assert_eq!(class.effect, None);
        assert_eq!(class.restriction, None);
        let serialized = serde_json::to_value(class).unwrap();
        assert!(serialized.get("effect").is_none());
        assert!(serialized.get("restriction").is_none());
        assert_eq!(
            serde_json::from_value::<managym::agent::observation::PooledManaClassData>(serialized)
                .unwrap(),
            *class
        );
        encoded_default(&observation);
    }
}
