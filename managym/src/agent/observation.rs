use crate::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::{
        event::{EventEntity, EventSubject, GameEvent, ObjectEventRef},
        game::Game,
        turn::{PhaseKind, PlayerTurnHistory, StepKind},
    },
    state::{
        effect::{Affected, ContinuousEffect, Modification},
        game_object::{CardId, ObjectId, PlayerId, Target},
        mana::{ManaCost, ManaUse, SPEND_SUBTYPES},
        permanent::{CounterKind, Permanent},
        stack_object::StackObject,
        zone::ZoneType,
    },
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnData {
    pub turn_number: u32,
    #[serde(deserialize_with = "super::observation_json::phase")]
    pub phase: PhaseKind,
    #[serde(deserialize_with = "super::observation_json::step")]
    pub step: StepKind,
    pub active_player_id: i32,
    pub agent_player_id: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerData {
    pub player_index: i32,
    pub id: i32,
    pub is_agent: bool,
    pub is_active: bool,
    pub life: i32,
    pub zone_counts: [i32; 7],
    /// Lesson cards in this player's graveyard.
    pub graveyard_lessons: i32,
    /// Total until-end-of-combat mana in this player's pool (firebending).
    pub combat_mana: i32,
    /// Public pooled mana by W/U/B/R/G/C; restricted mana remains separate.
    pub mana: [i32; 6],
    /// Spend-restricted mana by color, whatever its restriction (CR 106.6).
    /// The name predates restrictions other than Herbalist's Lesson-only mana;
    /// `restricted_mana` splits this total by restriction.
    pub lesson_mana: [i32; 6],
    /// Restricted mana by restriction class, in class order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restricted_mana: Vec<RestrictedManaData>,
    /// While set, this player can cast spells only from their hand until the
    /// named player's next turn begins (Avatar's Wrath). It overrides every
    /// card's `cast_permission`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand_only_until: Option<i32>,
    /// No maximum hand size for the rest of the game (Spirit Water Revival).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_maximum_hand_size: bool,
    pub known_hand: BTreeMap<u32, u32>,
    pub sideboard_counts: BTreeMap<u32, u32>,
    pub remaining_sideboard_counts: BTreeMap<u32, u32>,
    /// Unspent red mana stays until end of turn (The Last Agni Kai).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub keeps_red_mana: bool,
    /// What this player has done this turn (public: every event in it was).
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub turn_history: PlayerTurnHistory,
    /// Continuous effects on this player, in timestamp order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<ContinuousEffectData>,
}

/// Mana that may be spent only on the named purposes (CR 106.6): spells with
/// one of `spells`' subtypes, or abilities of sources with one of
/// `abilities`' subtypes. Amounts are W/U/B/R/G/C.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RestrictedManaData {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spells: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub abilities: Vec<String>,
    pub mana: [i32; 6],
}

impl RestrictedManaData {
    fn new(restriction: ManaUse, amounts: [u8; 6]) -> Self {
        let names = |mask: u8| {
            SPEND_SUBTYPES
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & 1 << bit != 0)
                .map(|(_, name)| name.to_string())
                .collect()
        };
        Self {
            spells: names(restriction.spells),
            abilities: names(restriction.abilities),
            mana: amounts.map(i32::from),
        }
    }

    /// Whether this mana may pay for a spell or an ability source with `subtype`.
    pub fn admits(&self, subtype: &str) -> bool {
        self.spells
            .iter()
            .chain(&self.abilities)
            .any(|name| name == subtype)
    }
}

/// A public permission to play an exiled card (airbend, impulse draw,
/// linked exile, foretell), projected from the engine's exile grant. The
/// terms describe when it may be used; a pending offer to cast the card
/// exists only while they are all met and the player is not hand-only.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CastPermissionData {
    /// The player who may play the card.
    pub player_id: i32,
    /// The cost of casting it under this permission, with any-type mana
    /// already counted as generic; absent for a land, which is played.
    #[serde(
        default,
        deserialize_with = "super::observation_json::optional_mana_cost"
    )]
    pub cost: Option<ManaCost>,
    /// Ends in this turn's cleanup step.
    #[serde(default)]
    pub this_turn: bool,
    /// Usable only during the permitted player's turn.
    #[serde(default)]
    pub own_turn_only: bool,
    /// Usable only on a turn numbered after this one.
    #[serde(default)]
    pub after_turn: Option<u32>,
    /// Lasts while the permitted player controls this permanent.
    #[serde(default)]
    pub source_permanent_id: Option<i32>,
    #[serde(default)]
    pub flash: bool,
    #[serde(default)]
    pub any_mana: bool,
    /// The cost's generic may be paid by waterbending.
    #[serde(default)]
    pub waterbend: bool,
    /// Foretold: face down, so only its owner sees the card or this permission.
    #[serde(default)]
    pub foretold: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardTypeData {
    pub is_castable: bool,
    pub is_permanent: bool,
    pub is_non_land_permanent: bool,
    pub is_non_creature_permanent: bool,
    pub is_spell: bool,
    pub is_creature: bool,
    pub is_land: bool,
    pub is_planeswalker: bool,
    pub is_enchantment: bool,
    pub is_artifact: bool,
    pub is_kindred: bool,
    pub is_battle: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CardData {
    #[serde(deserialize_with = "super::observation_json::zone")]
    pub zone: ZoneType,
    pub owner_id: i32,
    pub id: i32,
    pub registry_key: i32,
    pub name: String,
    pub text_box: String,
    pub power: i32,
    pub toughness: i32,
    pub is_token: bool,
    pub is_legendary: bool,
    /// Subtype tags the current pool cares about.
    pub is_ally: bool,
    pub is_lesson: bool,
    /// Ward cost (mana value; 0 = no ward).
    pub ward_cost: i32,
    /// Kicker cost (mana value; 0 = no kicker).
    pub kicker_cost: i32,
    pub card_types: CardTypeData,
    pub keywords: KeywordData,
    #[serde(deserialize_with = "super::observation_json::mana_cost")]
    pub mana_cost: ManaCost,
    /// Foretold and face down in exile; listed only for its owner.
    #[serde(default)]
    pub face_down: bool,
    /// The permission to play this card from exile, when it has one.
    #[serde(default)]
    pub cast_permission: Option<CastPermissionData>,
    /// The value of X announced for this spell on the stack (CR 107.3),
    /// when its mana cost has X.
    #[serde(default)]
    pub x: Option<i32>,
}

/// An owned outside-game copy. The candidate is private to this viewer and
/// binds action focus; it is not an in-game object reference or a zone.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SideboardCardData {
    pub candidate_id: i32,
    pub owner_id: i32,
    pub registry_key: i32,
    pub name: String,
    pub text_box: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeywordData {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    pub flash: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
    pub hexproof: bool,
    pub indestructible: bool,
}

/// Number of `KeywordData` flags; tensor keyword blocks follow field order.
pub const KEYWORD_COUNT: usize = 14;

impl KeywordData {
    /// Every flag in field order, for fixed-width policy columns.
    pub fn flags(&self) -> [bool; KEYWORD_COUNT] {
        let Self {
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
        } = *self;
        [
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
        ]
    }

    pub fn from_keywords(keywords: &crate::state::card::Keywords) -> Self {
        Self {
            flying: keywords.flying,
            reach: keywords.reach,
            haste: keywords.haste,
            flash: keywords.flash,
            vigilance: keywords.vigilance,
            trample: keywords.trample,
            first_strike: keywords.first_strike,
            double_strike: keywords.double_strike,
            deathtouch: keywords.deathtouch,
            lifelink: keywords.lifelink,
            defender: keywords.defender,
            menace: keywords.menace,
            hexproof: keywords.hexproof,
            indestructible: keywords.indestructible,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StackObjectKindData {
    Spell = 0,
    ActivatedAbility = 1,
    TriggeredAbility = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StackTargetKindData {
    Player = 0,
    Permanent = 1,
    StackObject = 2,
    Card = 3,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackTargetData {
    #[serde(deserialize_with = "super::observation_json::stack_target")]
    pub kind: StackTargetKindData,
    pub player_id: Option<i32>,
    pub permanent_id: Option<i32>,
    pub stack_object_id: Option<i32>,
    pub card_id: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StackObjectData {
    /// Derived ability text; absent on spells and historical observations.
    #[serde(default)]
    pub description: Option<String>,
    pub stack_object_id: i32,
    #[serde(deserialize_with = "super::observation_json::stack_kind")]
    pub kind: StackObjectKindData,
    pub controller_id: i32,
    pub source_card_registry_key: i32,
    pub source_permanent_id: Option<i32>,
    pub chosen_mode: Option<usize>,
    pub ability_index: Option<i32>,
    pub targets: Vec<StackTargetData>,
    /// The announced value of X (CR 107.3); omitted when zero.
    #[serde(default)]
    pub x: i32,
    /// Keywords a static grants this spell while on the stack (a Lesson's
    /// lifelink); absent when none.
    #[serde(default)]
    pub granted_keywords: Option<KeywordData>,
}

/// One continuous-effect record (`state::effect`) affecting a player or
/// permanent, as its holder's `effects` list shows it. Every record from a
/// resolved spell or ability is public.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuousEffectData {
    /// The `Modification` variant (`ModifyPt`, `AddKeywords`, ...).
    pub modification: String,
    /// P/T amounts: the change for `ModifyPt`, the base for `SetBasePt`
    /// (absent when that base is left alone).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toughness: Option<i32>,
    /// Firebending N.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount: Option<i32>,
    /// Keywords gained (field names, `first_strike`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    /// Subtypes gained.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subtypes: Vec<String>,
    /// The mana color kept (`Red`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// The `EffectDuration` variant (`EndOfTurn`, `Indefinite`).
    pub duration: String,
    /// Definition of the spell or ability's source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_registry_key: Option<i32>,
    /// The source's printed name: public, since the spell or ability resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_name: Option<String>,
    /// Timestamp order (CR 613.7); larger is later, comparable across
    /// records and permanents in one observation.
    pub timestamp: i32,
}

impl PermanentData {
    /// Its timestamp: renewed by an attachment, otherwise its id.
    pub fn timestamp(&self) -> i32 {
        self.timestamp.unwrap_or(self.id)
    }
}

impl ContinuousEffectData {
    fn new(game: &Game, effect: &ContinuousEffect) -> Self {
        let mut data = Self {
            modification: String::new(),
            duration: format!("{:?}", effect.duration),
            source_registry_key: effect.source.map(|id| id.0 as i32),
            source_name: effect
                .source
                .and_then(|id| game.state.content.definition(id))
                .map(|definition| definition.name.clone()),
            timestamp: effect.timestamp.0 as i32,
            ..Default::default()
        };
        data.modification = match &effect.modification {
            Modification::AddSubtypes(subtypes) => {
                data.subtypes = subtypes.clone();
                "AddSubtypes"
            }
            Modification::BecomeArtifactCreature => "BecomeArtifactCreature",
            Modification::AddKeywords(keywords) => {
                data.keywords = keywords.names().into_iter().map(str::to_owned).collect();
                "AddKeywords"
            }
            Modification::LoseAllAbilities => "LoseAllAbilities",
            Modification::Firebending(amount) => {
                data.amount = Some(i32::from(*amount));
                "Firebending"
            }
            Modification::SetBasePt { power, toughness } => {
                (data.power, data.toughness) = (*power, *toughness);
                "SetBasePt"
            }
            Modification::ModifyPt { power, toughness } => {
                (data.power, data.toughness) = (Some(*power), Some(*toughness));
                "ModifyPt"
            }
            Modification::CantBeBlocked => "CantBeBlocked",
            Modification::ExileIfWouldDie => "ExileIfWouldDie",
            Modification::KeepUnspentMana(color) => {
                data.color = Some(format!("{color:?}"));
                "KeepUnspentMana"
            }
            Modification::NoMaximumHandSize => "NoMaximumHandSize",
        }
        .to_string();
        data
    }

    /// The records affecting `affected`, in timestamp order.
    fn list(game: &Game, affected: Option<Affected>) -> Vec<Self> {
        game.state
            .effects
            .iter()
            .filter(|effect| Some(effect.affected) == affected)
            .map(|effect| Self::new(game, effect))
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermanentData {
    pub id: i32,
    /// Definition of the backing card, in the match's bound content pack.
    pub registry_key: i32,
    pub controller_id: i32,
    pub tapped: bool,
    pub damage: i32,
    pub is_summoning_sick: bool,
    pub plus1_counters: i32,
    pub cant_be_blocked_this_turn: bool,
    /// Effective power/toughness — CDA base, static continuous effects,
    /// until-EOT deltas, and +1/+1 counters all applied.
    pub power: i32,
    pub toughness: i32,
    /// A creature only through an effect: earthbend (a land that's also a
    /// 0/0 creature), crew, or another type-changing effect.
    pub is_animated: bool,
    /// This permanent holds cards in exile until it leaves (Jailer).
    pub has_exile_link: bool,
    pub exile_if_dies_this_turn: bool,
    /// Declared as an attacker in the current combat. Omitted when false, so
    /// observations retained before this field stay canonical.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub attacking: bool,
    /// Exhaust abilities this object has activated; each can never be
    /// activated again by this object (CR 702.177a).
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub exhausted_abilities: i32,
    /// "Once each turn" abilities this object has activated this turn.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub once_each_turn_used: i32,
    /// Firebending granted until end of turn, summed over grants.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub granted_firebending: i32,
    pub temp_subtypes: Vec<String>,
    /// Effective keywords — printed keywords unioned with until-EOT grants
    /// (Yip Yip! flying, Enter the Avatar State, ...). The card entry keeps
    /// printed keywords; this is what the permanent can actually do now.
    pub keywords: KeywordData,
    /// Counters other than +1/+1 by kind (CR 122.1; `CounterKind` names in
    /// snake case), only the kinds present.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub counters: BTreeMap<String, i32>,
    /// The permanent this Aura or Equipment is attached to (CR 301.5, 303.4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<i32>,
    /// Name and creature types set by an effect (Honest Work); absent while
    /// the printed ones apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub creature_types: Option<Vec<String>>,
    /// A land only through an effect (a static land-type grant).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_granted_land: bool,
    /// The attacking creature this one blocks (CR 509.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocking: Option<i32>,
    /// It has lost all abilities (CR 613.1f), until end of turn or while an
    /// Aura sets its characteristics.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub abilities_lost: bool,
    /// Activated abilities (indices in its ability list) whose limit is
    /// spent: exhaust (CR 702.177a) or once each turn (CR 602.5b).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub used_abilities: Vec<usize>,
    /// It was dealt damage this turn, whether or not that damage is still
    /// marked.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dealt_damage_this_turn: bool,
    /// Continuous effects on this permanent, in timestamp order; the
    /// effective values above already apply them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<ContinuousEffectData>,
    /// Its timestamp (CR 613.7d/e), the order its static abilities apply in
    /// among the records; absent while it is `id`, the timestamp it entered
    /// with (an attachment renews it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EventType {
    CardMoved = 0,
    DamageDealt = 1,
    LifeChanged = 2,
    SpellCast = 3,
    SpellResolved = 4,
    SpellCountered = 5,
    AbilityTriggered = 6,
    CombatAttackersDeclared = 7,
    BlockersDeclared = 8,
    CombatDamageDealt = 9,
    PermanentsDied = 10,
    TurnStarted = 11,
    CardRevealed = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EventEntityKind {
    None = 0,
    Card = 1,
    Permanent = 2,
    Player = 3,
    Object = 4,
    Definition = 5,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventData {
    pub event_type: i32,
    pub source_kind: i32,
    pub source_id: i32,
    pub target_kind: i32,
    pub target_id: i32,
    pub amount: i32,
    pub controller_id: i32,
    /// Committed CardMoved zone metadata; -1 for other event kinds.
    /// The learning event tensor remains intentionally unchanged.
    pub from_zone: i32,
    pub to_zone: i32,
    /// Exact viewer identity metadata for object subjects; -1 otherwise.
    /// These additive fields are excluded from the fixed learning tensor.
    pub source_incarnation: i32,
    pub target_incarnation: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionOption {
    /// Engine-owned meaning for choices whose index alone is insufficient.
    pub description: Option<String>,
    #[serde(deserialize_with = "super::observation_json::action")]
    #[serde(rename = "type")]
    pub action_type: ActionType,
    pub focus: Vec<i32>,
    /// Native yes/no declaration, shared by learning and experience consumers.
    pub declared: Option<bool>,
    pub mode_index: Option<usize>,
    pub ability_index: Option<usize>,
    /// Decision-local continuation frame; occurrence IDs never enter policy features.
    pub continuation_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionSpaceData {
    #[serde(deserialize_with = "super::observation_json::space")]
    #[serde(rename = "type")]
    pub action_space_type: ActionSpaceKind,
    pub actions: Vec<ActionOption>,
    #[serde(skip)]
    pub focus: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    #[serde(skip)]
    pub continuation: Vec<super::continuation::ContinuationFrame>,
    pub game_over: bool,
    pub won: bool,
    pub turn: TurnData,
    pub action_space: ActionSpaceData,
    pub agent: PlayerData,
    pub agent_cards: Vec<CardData>,
    pub agent_sideboard: Vec<SideboardCardData>,
    pub agent_permanents: Vec<PermanentData>,
    pub opponent: PlayerData,
    pub opponent_cards: Vec<CardData>,
    pub opponent_permanents: Vec<PermanentData>,
    pub stack_objects: Vec<StackObjectData>,
    pub recent_events: Vec<EventData>,
    #[serde(skip)]
    pub policy_history: crate::agent::policy_history::HistoryWindow,
}

impl Observation {
    pub fn new<'a, I>(game: &Game, recent_events: I) -> Self
    where
        I: IntoIterator<Item = &'a GameEvent>,
    {
        Self::for_player_with_events(game, game.agent_player(), recent_events, true)
    }

    /// Build a fixed-player, current-state projection.
    ///
    /// Unlike [`Self::new`], the perspective does not follow the player who
    /// currently owns the action space. Presentation events are deliberately
    /// absent: their visibility is a separate product boundary. If another
    /// player is deciding, the prompt kind remains public but its candidates
    /// and focus are suppressed.
    pub fn for_player(game: &Game, viewer: PlayerId) -> Self {
        assert!(
            viewer.0 < game.state.players.len(),
            "viewer must name a player"
        );
        let expose_actions = game
            .current_action_space
            .as_ref()
            .and_then(|space| space.player)
            == Some(viewer);
        Self::for_player_with_events(game, viewer, std::iter::empty(), expose_actions)
    }

    fn for_player_with_events<'a, I>(
        game: &Game,
        agent_player: PlayerId,
        recent_events: I,
        expose_actions: bool,
    ) -> Self
    where
        I: IntoIterator<Item = &'a GameEvent>,
    {
        let opponent_player = PlayerId((agent_player.0 + 1) % 2);

        let game_over = game.is_game_over();
        let won = game
            .winner_index()
            .is_some_and(|winner| winner == agent_player.0);

        let turn = TurnData {
            turn_number: game.state.turn.turn_number,
            phase: game.state.turn.current_phase_kind(),
            step: game.state.turn.current_step_kind(),
            active_player_id: game.state.players[game.active_player().0].id.0 as i32,
            agent_player_id: game.state.players[agent_player.0].id.0 as i32,
        };

        let action_space = game
            .current_action_space
            .as_ref()
            .map(|space| ActionSpaceData {
                action_space_type: space.kind,
                actions: if expose_actions {
                    space
                        .actions
                        .iter()
                        .map(|action| ActionOption {
                            description: game.action_description(action),
                            continuation_index: game.trigger_action_frame(action, agent_player),
                            action_type: action.action_type(),
                            focus: Self::action_focus(game, action)
                                .into_iter()
                                .map(|id| id.0 as i32)
                                .collect(),
                            declared: match action {
                                Action::DeclareAttacker { attack, .. } => Some(*attack),
                                Action::DeclareBlocker { attacker, .. } => Some(attacker.is_some()),
                                _ => None,
                            },
                            mode_index: match action {
                                Action::ChooseMode { mode, .. } => Some(*mode),
                                _ => None,
                            },
                            ability_index: match action {
                                Action::ActivateAbility { ability_index, .. }
                                | Action::ActivateCardAbility { ability_index, .. } => {
                                    Some(*ability_index)
                                }
                                _ => None,
                            },
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                focus: if expose_actions {
                    space.focus.iter().map(|id| id.0 as i32).collect()
                } else {
                    Vec::new()
                },
            })
            .unwrap_or(ActionSpaceData {
                action_space_type: ActionSpaceKind::GameOver,
                actions: Vec::new(),
                focus: Vec::new(),
            });

        let agent = Self::player_data(game, agent_player, true);
        let opponent = Self::player_data(game, opponent_player, false);

        let mut obs = Self {
            continuation: game.continuation_frames(agent_player),
            game_over,
            won,
            turn,
            action_space,
            agent,
            agent_cards: Vec::new(),
            agent_sideboard: Vec::new(),
            agent_permanents: Vec::new(),
            opponent,
            opponent_cards: Vec::new(),
            opponent_permanents: Vec::new(),
            stack_objects: Vec::new(),
            policy_history: game.policy_history.clone(),
            recent_events: recent_events
                .into_iter()
                .flat_map(Self::event_data)
                .collect(),
        };

        obs.populate_cards(game, agent_player);
        obs.populate_permanents(game, agent_player);
        obs.populate_stack_objects(game);
        obs
    }

    fn player_data(game: &Game, player: PlayerId, is_agent: bool) -> PlayerData {
        let p = &game.state.players[player.0];
        let mut zone_counts = [0_i32; 7];
        for (zone_index, count) in zone_counts.iter_mut().enumerate() {
            *count = game
                .state
                .zones
                .size(Self::zone_from_index(zone_index), player) as i32;
        }

        let mut sideboard_counts = BTreeMap::new();
        let mut remaining_sideboard_counts = BTreeMap::new();
        for card in &p.sideboard {
            let definition = game.state.cards[*card].definition_id.0;
            *sideboard_counts.entry(definition).or_default() += 1;
            if game.state.zones.zone_of(*card).is_none() {
                *remaining_sideboard_counts.entry(definition).or_default() += 1;
            }
        }

        // Regular and combat mana together, as the player may spend it.
        let mut pool = p.mana_pool.clone();
        pool.add(&p.combat_mana_pool);
        PlayerData {
            sideboard_counts,
            remaining_sideboard_counts,
            player_index: p.index as i32,
            id: p.id.0 as i32,
            is_agent,
            is_active: game.active_player() == player,
            life: p.life,
            zone_counts,
            graveyard_lessons: game.count_graveyard_matching(
                player,
                &crate::state::predicate::CardPredicate::subtype("Lesson"),
            ) as i32,
            combat_mana: p.combat_mana_pool.total() as i32,
            mana: pool.mana.map(i32::from),
            lesson_mana: pool.restricted_colors().map(i32::from),
            restricted_mana: pool
                .restricted()
                .iter()
                .map(|(restriction, amounts)| RestrictedManaData::new(*restriction, *amounts))
                .collect(),
            hand_only_until: game
                .hand_only_until(player)
                .map(|until| game.state.players[until.0].id.0 as i32),
            no_maximum_hand_size: game.has_no_maximum_hand_size_effect(player),
            known_hand: p
                .known_hand
                .iter()
                .map(|(definition, count)| (definition.0, *count))
                .collect(),
            keeps_red_mana: game.keeps_unspent_mana(player, crate::state::mana::Color::Red),
            turn_history: game.state.turn.history[player.0],
            effects: ContinuousEffectData::list(game, Some(Affected::Player(player))),
        }
    }

    fn populate_cards(&mut self, game: &Game, agent_player: PlayerId) {
        for card in &game.state.players[agent_player.0].sideboard {
            if game.state.zones.zone_of(*card).is_none() {
                let card = &game.state.cards[*card];
                self.agent_sideboard.push(SideboardCardData {
                    candidate_id: card.id.0 as i32,
                    owner_id: self.agent.id,
                    registry_key: card.definition_id.0 as i32,
                    name: card.name.clone(),
                    text_box: card.text_box.clone(),
                });
            }
        }

        for card in game.state.zones.zone_cards(ZoneType::Hand, agent_player) {
            self.add_card(game, *card, ZoneType::Hand);
        }

        // The top of the agent's own library, when they may look at it any
        // time (CR 401.2).
        if let Some(card) = game
            .library_top_card(agent_player)
            .filter(|_| game.looks_at_library_top(agent_player))
        {
            self.add_card(game, card, ZoneType::Library);
        }

        // Library cards revealed to the agent by a pending mid-resolution
        // decision. Search candidates are derived from the current library;
        // scry/look choices retain their exact cards. Only the deciding player
        // sees them.
        if let Some(suspended) = &game.state.suspended_decision {
            if suspended.decision.player() == agent_player {
                let cards = match &suspended.decision {
                    crate::flow::decision::Decision::SearchBasicLand {
                        player,
                        subtype,
                        to_hand,
                        ..
                    } => game.basic_land_search_candidates(*player, subtype, *to_hand),
                    crate::flow::decision::Decision::SearchLibraryToHand { player, predicate } => {
                        game.library_search_candidates(*player, predicate)
                    }
                    decision @ crate::flow::decision::Decision::DiscardFromRevealedHand {
                        ..
                    } => game.revealed_hand_candidates(decision),
                    decision @ crate::flow::decision::Decision::SearchDistinctLands {
                        selected,
                        ..
                    } => {
                        let mut cards = selected.clone();
                        cards.extend(game.distinct_land_search_candidates(decision));
                        cards
                    }
                    decision => decision.revealed_cards().to_vec(),
                };
                for card in cards {
                    let zone = game.state.zones.zone_of(card).unwrap_or(ZoneType::Library);
                    self.add_card(game, card, zone);
                }
            }
        }

        for player in [agent_player, PlayerId((agent_player.0 + 1) % 2)] {
            for card in game.state.zones.zone_cards(ZoneType::Graveyard, player) {
                self.add_card(game, *card, ZoneType::Graveyard);
            }
            for card in game.state.zones.zone_cards(ZoneType::Exile, player) {
                // A foretold card is face down: only its owner sees it.
                if player == agent_player || !game.is_foretold(*card) {
                    self.add_card(game, *card, ZoneType::Exile);
                }
            }
        }

        for stack_object in game.state.stack_objects.iter().rev() {
            if let StackObject::Spell(spell) = stack_object {
                let mut card = Self::card_data(game, spell.card, ZoneType::Stack);
                if game.state.cards[spell.card]
                    .mana_cost
                    .as_ref()
                    .is_some_and(|cost| cost.x > 0)
                {
                    card.x = Some(i32::from(spell.x));
                }
                self.push_card(card, game.state.cards[spell.card].owner);
            }
        }
    }

    fn populate_permanents(&mut self, game: &Game, agent_player: PlayerId) {
        for player in [agent_player, PlayerId((agent_player.0 + 1) % 2)] {
            for card_id in game.state.zones.zone_cards(ZoneType::Battlefield, player) {
                if let Some(perm_id) = game.state.card_to_permanent[card_id] {
                    let Some(permanent) = game.state.permanents[perm_id].as_ref() else {
                        continue;
                    };
                    self.add_permanent(game, perm_id, permanent);
                }
            }
        }
    }

    fn populate_stack_objects(&mut self, game: &Game) {
        for object in game.state.stack_objects.iter().rev() {
            self.stack_objects
                .push(Self::stack_object_data(game, object));
        }
    }

    /// List a card on its owner's side; see [`Self::push_card`].
    fn add_card(
        &mut self,
        game: &Game,
        card_id: crate::state::game_object::CardId,
        zone: ZoneType,
    ) {
        let owner = game.state.cards[card_id].owner;
        self.push_card(Self::card_data(game, card_id, zone), owner);
    }

    /// List a card on `side`: its owner's, except that battlefield cards
    /// sit with their permanent's controller, in permanent order.
    fn push_card(&mut self, card_data: CardData, side: PlayerId) {
        if side.0 == self.agent.player_index as usize {
            self.agent_cards.push(card_data);
        } else {
            self.opponent_cards.push(card_data);
        }
    }

    fn add_permanent(
        &mut self,
        game: &Game,
        perm_id: crate::state::game_object::PermanentId,
        permanent: &Permanent,
    ) {
        let (power, toughness) = game.effective_pt(perm_id);
        let set_characteristics = game.set_characteristics(perm_id);
        let pdat = PermanentData {
            id: permanent.id.0 as i32,
            registry_key: game.state.cards[permanent.card].registry_key.0 as i32,
            controller_id: game.state.players[permanent.controller.0].id.0 as i32,
            tapped: permanent.tapped,
            damage: permanent.damage,
            is_summoning_sick: permanent.summoning_sick,
            plus1_counters: permanent.counter_count(CounterKind::PlusOnePlusOne),
            exile_if_dies_this_turn: game.exiled_if_it_would_die(perm_id),
            attacking: permanent.attacking,
            exhausted_abilities: permanent.exhausted.len() as i32,
            once_each_turn_used: permanent.activated_this_turn.len() as i32,
            granted_firebending: i32::from(game.granted_firebending(perm_id)),
            temp_subtypes: game
                .added_subtypes(perm_id)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            cant_be_blocked_this_turn: game.cant_be_blocked(perm_id),
            power,
            toughness,
            is_animated: game.permanent_is_animated(perm_id),
            has_exile_link: game
                .permanent_object_ref(perm_id)
                .is_some_and(|object_ref| {
                    game.state
                        .exile_links
                        .iter()
                        .any(|link| link.source == object_ref)
                }),
            keywords: KeywordData::from_keywords(&game.effective_keywords(perm_id)),
            counters: permanent
                .counters
                .iter()
                .filter(|(kind, _)| *kind != CounterKind::PlusOnePlusOne)
                .map(|(kind, count)| (kind.name().to_string(), count))
                .collect(),
            attached_to: permanent
                .attached_to
                .and_then(|object| game.object_event_ref(object))
                .map(|object| object.entity.0 as i32),
            name: set_characteristics.map(|set| set.named.clone()),
            creature_types: set_characteristics.map(|_| {
                game.creature_types(perm_id)
                    .into_iter()
                    .map(str::to_owned)
                    .collect()
            }),
            is_granted_land: game.permanent_is_land(perm_id)
                && !game.state.cards[permanent.card].types.is_land(),
            blocking: game.state.combat.as_ref().and_then(|combat| {
                combat
                    .attacker_to_blockers
                    .iter()
                    .find(|(_, blockers)| blockers.contains(&perm_id))
                    .and_then(|(attacker, _)| game.state.permanents[*attacker].as_ref())
                    .map(|attacker| attacker.id.0 as i32)
            }),
            abilities_lost: game.has_lost_abilities(perm_id),
            dealt_damage_this_turn: permanent.dealt_damage_this_turn,
            used_abilities: game
                .activated_abilities_for(perm_id)
                .iter()
                .enumerate()
                .filter(|(index, ability)| permanent.activation_used(*index, ability))
                .map(|(index, _)| index)
                .collect(),
            effects: ContinuousEffectData::list(game, game.affected_permanent(perm_id)),
            timestamp: (permanent.timestamp != permanent.id)
                .then_some(permanent.timestamp.0 as i32),
        };

        if permanent.controller.0 == self.agent.player_index as usize {
            self.agent_permanents.push(pdat);
        } else {
            self.opponent_permanents.push(pdat);
        }

        self.push_card(
            Self::card_data(game, permanent.card, ZoneType::Battlefield),
            permanent.controller,
        );
    }

    fn card_data(game: &Game, card_id: CardId, zone: ZoneType) -> CardData {
        let card = &game.state.cards[card_id];
        CardData {
            zone,
            owner_id: game.state.players[card.owner.0].id.0 as i32,
            id: card.id.0 as i32,
            registry_key: card.registry_key.0 as i32,
            name: card.name.clone(),
            text_box: card.text_box.clone(),
            power: card.power.unwrap_or(0),
            toughness: card.toughness.unwrap_or(0),
            is_token: card.is_token,
            is_legendary: card
                .supertypes
                .iter()
                .any(|s| s.eq_ignore_ascii_case("legendary")),
            // A characteristic-setting effect replaces a permanent's types.
            is_ally: match game.state.card_to_permanent[card_id] {
                Some(permanent) if zone == ZoneType::Battlefield => {
                    game.permanent_has_subtype(permanent, "Ally")
                }
                _ => card.has_subtype("Ally"),
            },
            is_lesson: card.has_subtype("Lesson"),
            ward_cost: card
                .ward()
                .map(|cost| i32::from(cost.mana_value))
                .unwrap_or(0),
            kicker_cost: card
                .optional_cost()
                .map(|cost| i32::from(cost.mana_value))
                .unwrap_or(0),
            card_types: CardTypeData {
                is_castable: card.types.is_castable(),
                is_permanent: card.types.is_permanent(),
                is_non_land_permanent: card.types.is_non_land_permanent(),
                is_non_creature_permanent: card.types.is_non_creature_permanent(),
                is_spell: card.types.is_spell(),
                is_creature: card.types.is_creature(),
                is_land: card.types.is_land(),
                is_planeswalker: card.types.is_planeswalker(),
                is_enchantment: card.types.is_enchantment(),
                is_artifact: card.types.is_artifact(),
                is_kindred: card.types.is_kindred(),
                is_battle: card.types.is_battle(),
            },
            keywords: KeywordData::from_keywords(&card.keywords),
            mana_cost: card.mana_cost.clone().unwrap_or_default(),
            face_down: zone == ZoneType::Exile && game.is_foretold(card_id),
            cast_permission: game
                .exile_grant_on(card_id)
                .map(|grant| CastPermissionData {
                    player_id: game.state.players[grant.player.0].id.0 as i32,
                    cost: game.grant_cast_cost(grant, card_id),
                    this_turn: grant.this_turn,
                    own_turn_only: grant.own_turn_only,
                    after_turn: grant.after_turn,
                    source_permanent_id: grant.source.and_then(|source| {
                        game.lookup_current_permanent(source)
                            .ok()
                            .map(|id| id.0 as i32)
                    }),
                    flash: grant.flash,
                    any_mana: grant.any_mana,
                    waterbend: grant.waterbend,
                    foretold: grant.foretold,
                }),
            x: None,
        }
    }

    fn stack_object_data(game: &Game, stack_object: &StackObject) -> StackObjectData {
        StackObjectData {
            description: match stack_object {
                StackObject::TriggeredAbility(a) => {
                    Some(super::ability_description::trigger_description(
                        &game.state.cards[a.source_card],
                        a.ability_index,
                        a.inline_effects.as_deref(),
                    ))
                }
                _ => None,
            },
            stack_object_id: stack_object.id().0 as i32,
            kind: match stack_object {
                StackObject::Spell(_) => StackObjectKindData::Spell,
                StackObject::ActivatedAbility(_) => StackObjectKindData::ActivatedAbility,
                StackObject::TriggeredAbility(_) => StackObjectKindData::TriggeredAbility,
            },
            controller_id: game.state.players[stack_object.controller().0].id.0 as i32,
            source_card_registry_key: stack_object.source_card_registry_key().0 as i32,
            source_permanent_id: match stack_object {
                StackObject::ActivatedAbility(ability) => game
                    .object_event_ref(ability.source_ref)
                    .map(|object| object.entity.0 as i32),
                _ => None,
            },
            chosen_mode: match stack_object {
                StackObject::Spell(spell) => spell.mode,
                _ => None,
            },
            ability_index: stack_object.ability_index().map(|index| index as i32),
            x: match stack_object {
                StackObject::Spell(spell) => i32::from(spell.x),
                StackObject::ActivatedAbility(ability) => i32::from(ability.x),
                StackObject::TriggeredAbility(_) => 0,
            },
            granted_keywords: match stack_object {
                StackObject::Spell(spell) => {
                    let keywords = game.spell_keywords(spell.card, spell.controller);
                    (!crate::state::is_zero(&keywords))
                        .then(|| KeywordData::from_keywords(&keywords))
                }
                _ => None,
            },
            targets: stack_object
                .targets()
                .iter()
                .filter_map(|target| Self::stack_target_data(game, *target))
                .collect(),
        }
    }

    fn stack_target_data(game: &Game, target: Target) -> Option<StackTargetData> {
        match target {
            Target::Player(player) => Some(StackTargetData {
                kind: StackTargetKindData::Player,
                player_id: Some(game.state.players[player.0].id.0 as i32),
                permanent_id: None,
                stack_object_id: None,
                card_id: None,
            }),
            Target::Permanent(permanent_id) => {
                let permanent = game.state.permanents[permanent_id].as_ref()?;
                Some(StackTargetData {
                    kind: StackTargetKindData::Permanent,
                    player_id: None,
                    permanent_id: Some(permanent.id.0 as i32),
                    stack_object_id: None,
                    card_id: None,
                })
            }
            Target::GraveyardCard(object) => Some(StackTargetData {
                kind: StackTargetKindData::Card,
                player_id: None,
                permanent_id: None,
                stack_object_id: None,
                card_id: Some(game.state.cards[CardId::from(object.entity)].id.0 as i32),
            }),
            Target::StackSpell(card_id) => {
                let stack_object_id = Self::stack_spell_object_id(game, card_id)?.0 as i32;
                Some(StackTargetData {
                    kind: StackTargetKindData::StackObject,
                    player_id: None,
                    permanent_id: None,
                    stack_object_id: Some(stack_object_id),
                    card_id: None,
                })
            }
            Target::StackAbility(id) => {
                game.find_ability_on_stack_index(id)?;
                Some(StackTargetData {
                    kind: StackTargetKindData::StackObject,
                    player_id: None,
                    permanent_id: None,
                    stack_object_id: Some(id.0 as i32),
                    card_id: None,
                })
            }
        }
    }

    fn stack_spell_object_id(game: &Game, card_id: CardId) -> Option<ObjectId> {
        game.state
            .stack_objects
            .iter()
            .find_map(|object| match object {
                StackObject::Spell(spell) if spell.card == card_id => Some(spell.id),
                _ => None,
            })
    }

    fn zone_from_index(index: usize) -> ZoneType {
        match index {
            0 => ZoneType::Library,
            1 => ZoneType::Hand,
            2 => ZoneType::Battlefield,
            3 => ZoneType::Graveyard,
            4 => ZoneType::Stack,
            5 => ZoneType::Exile,
            6 => ZoneType::Command,
            _ => unreachable!("invalid zone index: {index}"),
        }
    }

    pub(crate) fn event_data(event: &GameEvent) -> Vec<EventData> {
        use crate::flow::event::DamageTarget;
        match event {
            GameEvent::PermanentSacrificed { .. } => Vec::new(),
            GameEvent::CardRevealed { owner, definition } => {
                let mut data =
                    Self::build_event_data(EventType::CardRevealed, None, None, 0, Some(*owner));
                data.source_kind = EventEntityKind::Definition as i32;
                data.source_id = definition.0 as i32;
                vec![data]
            }
            GameEvent::CardMoved {
                card,
                from,
                to,
                controller,
                face_down,
            } => {
                let mut data = Self::card_event_data(EventType::CardMoved, *card, *controller);
                data.from_zone = from.map_or(-1, |zone| zone as i32);
                data.to_zone = *to as i32;
                // Outside cards and face-down arrivals name no public object.
                if (from.is_none() && *to == ZoneType::Hand) || *face_down {
                    data.source_kind = EventEntityKind::None as i32;
                    data.source_id = -1;
                }
                vec![data]
            }
            GameEvent::DamageDealt {
                source,
                target,
                amount,
            } => {
                let source_entity = source.map(EventEntity::Card);
                let target_entity = Some(match *target {
                    DamageTarget::Player(p) => EventEntity::Player(p),
                    DamageTarget::Permanent(p) => EventEntity::Permanent(p),
                });
                vec![Self::build_event_data(
                    EventType::DamageDealt,
                    source_entity,
                    target_entity,
                    *amount as i32,
                    None,
                )]
            }
            GameEvent::LifeChanged { player, old, new } => vec![Self::build_event_data(
                EventType::LifeChanged,
                Some(EventEntity::Player(*player)),
                Some(EventEntity::Player(*player)),
                new - old,
                None,
            )],
            GameEvent::SpellCast { card, .. } => vec![Self::card_event_data(
                EventType::SpellCast,
                *card,
                PlayerId(0),
            )],
            GameEvent::SpellResolved { card } => vec![Self::card_event_data(
                EventType::SpellResolved,
                *card,
                PlayerId(0),
            )],
            GameEvent::SpellCountered { card, .. } => vec![Self::card_event_data(
                EventType::SpellCountered,
                *card,
                PlayerId(0),
            )],
            GameEvent::AbilityTriggered {
                source_card,
                controller,
            } => vec![Self::card_event_data(
                EventType::AbilityTriggered,
                *source_card,
                *controller,
            )],
            GameEvent::CombatAttackersDeclared {
                player,
                defender,
                attackers,
            } => attackers
                .iter()
                .map(|attacker| {
                    let mut data = Self::object_event_data(
                        EventType::CombatAttackersDeclared,
                        Some(*attacker),
                        None,
                        0,
                        Some(*player),
                    );
                    data.target_kind = EventEntityKind::Player as i32;
                    data.target_id = defender.0 as i32;
                    data
                })
                .collect(),
            GameEvent::BlockersDeclared { assignments } => assignments
                .iter()
                .flat_map(|(attacker, blockers)| {
                    blockers.iter().map(|blocker| {
                        Self::object_event_data(
                            EventType::BlockersDeclared,
                            Some(*attacker),
                            Some(*blocker),
                            0,
                            None,
                        )
                    })
                })
                .collect(),
            GameEvent::CombatDamageDealt {
                source,
                target,
                amount,
            } => {
                let mut data = match target {
                    EventSubject::Object(target) => Self::object_event_data(
                        EventType::CombatDamageDealt,
                        Some(*source),
                        Some(*target),
                        *amount as i32,
                        None,
                    ),
                    EventSubject::Player(_) => Self::object_event_data(
                        EventType::CombatDamageDealt,
                        Some(*source),
                        None,
                        *amount as i32,
                        None,
                    ),
                };
                if let EventSubject::Player(player) = target {
                    data.target_kind = EventEntityKind::Player as i32;
                    data.target_id = player.0 as i32;
                }
                vec![data]
            }
            GameEvent::PermanentsDied { objects } => objects
                .iter()
                .map(|object| {
                    Self::object_event_data(
                        EventType::PermanentsDied,
                        Some(*object),
                        None,
                        objects.len() as i32,
                        None,
                    )
                })
                .collect(),
            GameEvent::TurnStarted {
                player,
                turn_number,
            } => vec![Self::build_event_data(
                EventType::TurnStarted,
                Some(EventEntity::Player(*player)),
                None,
                *turn_number as i32,
                Some(*player),
            )],
            // Internal trigger-plumbing events; the underlying state changes
            // are already visible via CardMoved / permanent state.
            GameEvent::CardDrawn { .. }
            | GameEvent::PermanentTapped { .. }
            | GameEvent::PermanentTargeted { .. }
            | GameEvent::AttackersDeclared { .. }
            | GameEvent::StepStarted { .. } => Vec::new(),
        }
    }

    fn object_event_data(
        event_type: EventType,
        source: Option<ObjectEventRef>,
        target: Option<ObjectEventRef>,
        amount: i32,
        controller: Option<PlayerId>,
    ) -> EventData {
        let mut data = Self::build_event_data(event_type, None, None, amount, controller);
        if let Some(source) = source {
            data.source_kind = EventEntityKind::Object as i32;
            data.source_id = source.entity.0 as i32;
            data.source_incarnation = source.incarnation.0 as i32;
        }
        if let Some(target) = target {
            data.target_kind = EventEntityKind::Object as i32;
            data.target_id = target.entity.0 as i32;
            data.target_incarnation = target.incarnation.0 as i32;
        }
        data
    }

    fn card_event_data(event_type: EventType, card: CardId, controller: PlayerId) -> EventData {
        Self::build_event_data(
            event_type,
            Some(EventEntity::Card(card)),
            None,
            0,
            Some(controller),
        )
    }

    fn build_event_data(
        event_type: EventType,
        source: Option<EventEntity>,
        target: Option<EventEntity>,
        amount: i32,
        controller: Option<PlayerId>,
    ) -> EventData {
        let (source_kind, source_id) = Self::event_entity(source);
        let (target_kind, target_id) = Self::event_entity(target);

        EventData {
            event_type: event_type as i32,
            source_kind,
            source_id,
            target_kind,
            target_id,
            amount,
            controller_id: controller.map_or(-1, |player| player.0 as i32),
            from_zone: -1,
            to_zone: -1,
            source_incarnation: -1,
            target_incarnation: -1,
        }
    }

    fn event_entity(entity: Option<EventEntity>) -> (i32, i32) {
        match entity {
            Some(EventEntity::Card(card)) => (EventEntityKind::Card as i32, card.0 as i32),
            Some(EventEntity::Permanent(permanent)) => {
                (EventEntityKind::Permanent as i32, permanent.0 as i32)
            }
            Some(EventEntity::Player(player)) => (EventEntityKind::Player as i32, player.0 as i32),
            None => (EventEntityKind::None as i32, -1),
        }
    }

    pub(crate) fn action_focus(game: &Game, action: &Action) -> Vec<ObjectId> {
        match action {
            Action::ChooseTrigger { trigger_id, player } => game
                .state
                .trigger_placement
                .as_ref()
                .and_then(|p| p.remaining.iter().find(|t| t.id == *trigger_id))
                .and_then(|t| t.source_ref)
                .and_then(|r| game.visible_reference(r, *player).object)
                .map(|id| vec![ObjectId(id as u32)])
                .unwrap_or_default(),
            Action::PlayLand { card, .. } | Action::CastSpell { card, .. } => {
                vec![game.state.cards[card].id]
            }
            Action::ActivateCardAbility { card, .. } | Action::Foretell { card, .. } => {
                vec![game.state.cards[card].id]
            }
            Action::ActivateAbility { permanent, .. } => game.state.permanents[*permanent]
                .as_ref()
                .map(|perm| vec![perm.id])
                .unwrap_or_default(),
            Action::PassPriority { .. } => vec![],
            Action::DeclareAttacker { permanent, .. } => game.state.permanents[permanent]
                .as_ref()
                .map(|perm| vec![perm.id])
                .unwrap_or_default(),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => {
                let mut focus = Vec::new();
                if let Some(perm) = game.state.permanents[blocker].as_ref() {
                    focus.push(perm.id);
                }
                if let Some(attacker) = attacker {
                    if let Some(perm) = game.state.permanents[attacker].as_ref() {
                        focus.push(perm.id);
                    }
                }
                focus
            }
            Action::ChooseTarget { target, .. } => match target {
                crate::state::target::Target::Player(player) => {
                    vec![game.state.players[player.0].id]
                }
                crate::state::target::Target::Permanent(permanent_id) => game.state.permanents
                    [*permanent_id]
                    .as_ref()
                    .map(|perm| vec![perm.id])
                    .unwrap_or_default(),
                crate::state::target::Target::GraveyardCard(object) => {
                    vec![game.state.cards[CardId::from(object.entity)].id]
                }
                crate::state::target::Target::StackSpell(card_id) => {
                    vec![game.state.cards[card_id].id]
                }
                crate::state::target::Target::StackAbility(id) => vec![*id],
            },
            Action::ScryCard { card, .. }
            | Action::SelectCard { card, .. }
            | Action::LearnDiscard { card, .. }
            | Action::LearnTakeLesson { card, .. } => {
                vec![game.state.cards[card].id]
            }
            Action::WaterbendTap { permanent, .. } => game.state.permanents[*permanent]
                .as_ref()
                .map(|perm| vec![perm.id])
                .unwrap_or_default(),
            // Decisions about the resolving/cast object: focus its card.
            Action::Decline { .. } | Action::PayCost { .. } | Action::ChooseMode { .. } => {
                let card = game
                    .state
                    .suspended_decision
                    .as_ref()
                    .and_then(|suspended| suspended.frame.source)
                    .or(match &game.pending_choice {
                        Some(crate::flow::game::PendingChoice::SpellMode { card, .. })
                        | Some(crate::flow::game::PendingChoice::KickerChoice { card, .. })
                        | Some(crate::flow::game::PendingChoice::AnnounceX { card, .. })
                        | Some(crate::flow::game::PendingChoice::ChooseTargets { card, .. }) => {
                            Some(*card)
                        }
                        Some(crate::flow::game::PendingChoice::ActivatedTarget {
                            source: crate::flow::game::ActivationSource::Card(card),
                            ..
                        })
                        | Some(crate::flow::game::PendingChoice::DiscardPayment {
                            source: crate::flow::game::ActivationSource::Card(card),
                            ..
                        }) => Some(*card),
                        Some(crate::flow::game::PendingChoice::ActivatedTarget {
                            source: crate::flow::game::ActivationSource::Permanent(permanent),
                            ..
                        })
                        | Some(crate::flow::game::PendingChoice::DiscardPayment {
                            source: crate::flow::game::ActivationSource::Permanent(permanent),
                            ..
                        })
                        | Some(crate::flow::game::PendingChoice::ActivationX {
                            permanent, ..
                        })
                        | Some(crate::flow::game::PendingChoice::Crew { permanent, .. }) => {
                            game.state.permanents[*permanent]
                                .as_ref()
                                .map(|perm| perm.card)
                        }
                        Some(crate::flow::game::PendingChoice::Waterbend { payment, .. }) => {
                            game.payment_source_card(payment)
                        }
                        Some(
                            crate::flow::game::PendingChoice::SacrificePayment { .. }
                            | crate::flow::game::PendingChoice::ManaPayment { .. },
                        ) => None,
                        None => None,
                    });
                card.map(|card| vec![game.state.cards[card].id])
                    .unwrap_or_default()
            }
        }
    }

    pub fn validate(&self) -> bool {
        if self.agent.id == self.opponent.id {
            return false;
        }
        if self.agent.is_agent == self.opponent.is_agent {
            return false;
        }
        for card in &self.agent_sideboard {
            if card.owner_id != self.agent.id {
                return false;
            }
        }
        // A battlefield card sits with its controller, who may not own it.
        for (cards, side) in [
            (&self.agent_cards, self.agent.id),
            (&self.opponent_cards, self.opponent.id),
        ] {
            if cards
                .iter()
                .any(|card| card.zone != ZoneType::Battlefield && card.owner_id != side)
            {
                return false;
            }
        }
        for permanent in &self.agent_permanents {
            if permanent.controller_id != self.agent.id {
                return false;
            }
        }
        for permanent in &self.opponent_permanents {
            if permanent.controller_id != self.opponent.id {
                return false;
            }
        }
        true
    }

    pub fn to_json(&self) -> String {
        fn card_json(card: &CardData) -> Value {
            let mut value = json!({
                "id": card.id,
                "registry_key": card.registry_key,
                "name": card.name,
                "text_box": card.text_box,
                "zone": card.zone as i32,
                "owner_id": card.owner_id,
                "power": card.power,
                "toughness": card.toughness,
                "is_token": card.is_token,
                "is_legendary": card.is_legendary,
                "is_ally": card.is_ally,
                "is_lesson": card.is_lesson,
                "ward_cost": card.ward_cost,
                "kicker_cost": card.kicker_cost,
                "card_types": card.card_types,
                "keywords": card.keywords,
                "mana_cost": mana_cost_json(&card.mana_cost),
            });
            if card.face_down {
                value["face_down"] = json!(true);
            }
            if let Some(permission) = &card.cast_permission {
                value["cast_permission"] = cast_permission_json(permission);
            }
            if let Some(x) = card.x {
                value["x"] = json!(x);
            }
            value
        }

        fn mana_cost_json(cost: &ManaCost) -> Value {
            let mut value = json!({
                "cost": cost.cost[..6].iter().map(|v| i32::from(*v)).collect::<Vec<_>>(),
                "mana_value": i32::from(cost.mana_value),
            });
            if !cost.hybrid.is_empty() {
                value["hybrid"] = json!(cost
                    .hybrid
                    .iter()
                    .map(|pip| [pip[0] as u8, pip[1] as u8])
                    .collect::<Vec<_>>());
            }
            value
        }

        /// Only set terms are written, so the JSON round-trips canonically.
        fn cast_permission_json(permission: &CastPermissionData) -> Value {
            let mut value = json!({ "player_id": permission.player_id });
            if let Some(cost) = &permission.cost {
                value["cost"] = mana_cost_json(cost);
            }
            for (key, set) in [
                ("this_turn", permission.this_turn),
                ("own_turn_only", permission.own_turn_only),
                ("flash", permission.flash),
                ("any_mana", permission.any_mana),
                ("waterbend", permission.waterbend),
                ("foretold", permission.foretold),
            ] {
                if set {
                    value[key] = json!(true);
                }
            }
            if let Some(turn) = permission.after_turn {
                value["after_turn"] = json!(turn);
            }
            if let Some(id) = permission.source_permanent_id {
                value["source_permanent_id"] = json!(id);
            }
            value
        }

        fn stack_target_json(target: &StackTargetData) -> Value {
            json!({
                "kind": target.kind as i32,
                "player_id": target.player_id,
                "permanent_id": target.permanent_id,
                "stack_object_id": target.stack_object_id,
                "card_id": target.card_id,
            })
        }

        fn stack_object_json(stack_object: &StackObjectData) -> Value {
            let mut value = json!({
                "description": stack_object.description,
                "stack_object_id": stack_object.stack_object_id,
                "kind": stack_object.kind as i32,
                "controller_id": stack_object.controller_id,
                "source_card_registry_key": stack_object.source_card_registry_key,
                "source_permanent_id": stack_object.source_permanent_id,
                "ability_index": stack_object.ability_index,
                "chosen_mode": stack_object.chosen_mode,
                "targets": stack_object.targets.iter().map(stack_target_json).collect::<Vec<_>>(),
            });
            if stack_object.x != 0 {
                value["x"] = json!(stack_object.x);
            }
            if let Some(keywords) = &stack_object.granted_keywords {
                value["granted_keywords"] = json!(keywords);
            }
            value
        }

        let action_json = self
            .action_space
            .actions
            .iter()
            .map(|action| {
                let mut value = json!({
                    "type": action.action_type as i32,
                    "focus": action.focus,
                    "declared": action.declared,
                    "mode_index": action.mode_index,
                    "ability_index": action.ability_index,
                    "continuation_index": action.continuation_index,
                });
                if let Some(description) = &action.description {
                    value["description"] = serde_json::Value::String(description.clone());
                }
                value
            })
            .collect::<Vec<_>>();

        json!({
            "game_over": self.game_over,
            "won": self.won,
            "turn": {
                "turn_number": self.turn.turn_number,
                "phase": self.turn.phase as i32,
                "step": self.turn.step as i32,
                "active_player_id": self.turn.active_player_id,
                "agent_player_id": self.turn.agent_player_id,
            },
            "action_space": {
                "type": self.action_space.action_space_type as i32,
                "actions": action_json,
            },
            "agent": self.agent,
            "agent_cards": self.agent_cards.iter().map(card_json).collect::<Vec<_>>(),
            "agent_sideboard": self.agent_sideboard,
            "agent_permanents": self.agent_permanents,
            "opponent": self.opponent,
            "opponent_cards": self.opponent_cards.iter().map(card_json).collect::<Vec<_>>(),
            "opponent_permanents": self.opponent_permanents,
            "stack_objects": self
                .stack_objects
                .iter()
                .map(stack_object_json)
                .collect::<Vec<_>>(),
            "recent_events": self.recent_events.iter().map(|event| {
                json!({
                    "event_type": event.event_type,
                    "source_kind": event.source_kind,
                    "source_id": event.source_id,
                    "target_kind": event.target_kind,
                    "target_id": event.target_id,
                    "amount": event.amount,
                    "controller_id": event.controller_id,
                    "from_zone": event.from_zone,
                    "to_zone": event.to_zone,
                    "source_incarnation": event.source_incarnation,
                    "target_incarnation": event.target_incarnation,
                })
            }).collect::<Vec<_>>(),
        })
        .to_string()
    }
}
