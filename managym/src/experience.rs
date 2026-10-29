//! Rust-owned wire types for the protocol-v1 Python adapter.
//!
//! These types intentionally describe the narrow experience boundary, not
//! `GameState` or the learning observation ABI. The current frame projection
//! remains the legacy hero observation while the semantic table projection is
//! developed; that transitional shape is named explicitly below.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{de, Deserialize, Deserializer, Serialize};

macro_rules! numeric_id {
    ($name:ident, $inner:ty) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Deserialize,
            Eq,
            Hash,
            JsonSchema,
            Ord,
            PartialEq,
            PartialOrd,
            Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub $inner);
    };
}

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Deserialize, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

/// A nullable wire value whose key must still be present.
///
/// Serde otherwise treats a missing ``Option`` field as ``None``, and schemars
/// marks it optional. This transparent wrapper keeps null on the wire while
/// making presence part of the generated contract.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct RequiredNullable<T>(pub Option<T>);

impl<T> RequiredNullable<T> {
    pub fn as_ref(&self) -> Option<&T> {
        self.0.as_ref()
    }
}

fn deserialize_required_nullable<'de, D, T>(
    deserializer: D,
) -> Result<RequiredNullable<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(RequiredNullable)
}

/// The only version accepted by this module and its generated schema.
#[derive(Clone, Copy, Debug, Eq, Hash, JsonSchema, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct ProtocolVersion(#[schemars(range(min = 1, max = 1))] pub u16);

impl<'de> Deserialize<'de> for ProtocolVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let version = u16::deserialize(deserializer)?;
        if version == 1 {
            Ok(Self(version))
        } else {
            Err(de::Error::custom(format!(
                "unsupported experience protocol version {version}"
            )))
        }
    }
}

numeric_id!(Revision, u64);
numeric_id!(PromptId, u64);
numeric_id!(OfferId, u32);
numeric_id!(CandidateId, u32);
numeric_id!(CandidateSourceId, u32);
numeric_id!(RoleId, u16);
numeric_id!(PlayerId, u8);
numeric_id!(StackRenderId, u64);
numeric_id!(PresentationSeq, u64);
numeric_id!(PresentationCursor, u64);
numeric_id!(PresentationGroupId, u64);
numeric_id!(ReplayCursor, u64);

string_id!(MatchId);
string_id!(CommandId);
string_id!(ContentHash);
string_id!(AssetManifestHash);
string_id!(FrameHash);
string_id!(CheckpointId);
string_id!(PaymentPlanId);

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectRenderId {
    pub entity: u32,
    pub incarnation: u32,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum SubjectRef {
    Object { id: ObjectRenderId },
    Stack { id: StackRenderId },
    Player { id: PlayerId },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OfferVerb {
    Cast,
    PlayLand,
    Activate,
    PassPriority,
    DeclareAttackers,
    DeclareBlockers,
    Choose,
    Pay,
    Special,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum CandidateValue {
    Subject { subject: SubjectRef },
    Mode { key: String },
    PaymentPlan { id: PaymentPlanId },
    Boolean { value: bool },
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub id: CandidateId,
    pub value: CandidateValue,
    pub label: String,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub help: RequiredNullable<String>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub preview: RequiredNullable<String>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSource {
    pub id: CandidateSourceId,
    pub depends_on: Vec<RoleId>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub initial: RequiredNullable<Vec<Candidate>>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum ChoiceStep {
    Select {
        role: RoleId,
        label: String,
        candidates: CandidateSource,
        min: u16,
        max: u16,
        ordered: bool,
        distinct: bool,
    },
    Number {
        role: RoleId,
        label: String,
        min: i32,
        max: i32,
    },
    Assign {
        role: RoleId,
        label: String,
        sources: CandidateSource,
        destinations: CandidateSource,
        min_per_source: u16,
        max_per_source: u16,
    },
    Order {
        role: RoleId,
        label: String,
        candidates: CandidateSource,
    },
    Payment {
        role: RoleId,
        label: String,
        plans: CandidateSource,
        allow_auto: bool,
    },
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InteractionOffer {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subjects: Vec<OfferSubjectView>,
    pub id: OfferId,
    pub actor: PlayerId,
    pub verb: OfferVerb,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub source: RequiredNullable<SubjectRef>,
    pub label: String,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub help: RequiredNullable<String>,
    pub choices: Vec<ChoiceStep>,
    pub confirm_label: String,
    /// Transitional bridge to today's positional engine action.
    pub action_type: String,
    /// Transitional direct-manipulation focus IDs for the current table.
    pub focus: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum ChoiceAnswer {
    Candidates {
        role: RoleId,
        candidates: Vec<CandidateId>,
    },
    Number {
        role: RoleId,
        value: i32,
    },
    Assignments {
        role: RoleId,
        pairs: Vec<(CandidateId, CandidateId)>,
    },
    Order {
        role: RoleId,
        candidates: Vec<CandidateId>,
    },
    Payment {
        role: RoleId,
        plan: PaymentPlanId,
    },
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Command {
    pub command_id: CommandId,
    pub match_id: MatchId,
    pub expected_revision: Revision,
    pub prompt_id: PromptId,
    pub offer_id: OfferId,
    pub answers: Vec<ChoiceAnswer>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromptView {
    pub id: PromptId,
    pub actor: PlayerId,
    pub kind: String,
    pub title: String,
    pub instruction: String,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityStatus {
    Ready,
    Thinking,
    Resolving,
    Reconnecting,
    GameOver,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyCardTypesView {
    pub is_creature: bool,
    pub is_land: bool,
    pub is_spell: bool,
    pub is_artifact: bool,
    pub is_enchantment: bool,
    pub is_planeswalker: bool,
    pub is_battle: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyCardView {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text_box: String,
    pub id: u32,
    pub registry_key: u32,
    pub name: String,
    pub zone: String,
    pub owner_id: u32,
    pub power: i32,
    pub toughness: i32,
    pub mana_value: i32,
    pub types: LegacyCardTypesView,
    /// Foretold and face down in exile; only its owner's views list it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub face_down: bool,
    /// The permission to play this card from exile, when it has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cast_permission: Option<LegacyCastPermissionView>,
    /// The value of X announced for this spell on the stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x: Option<i32>,
}

/// A public permission to play an exiled card. Only set terms are present.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyCastPermissionView {
    pub player_id: u32,
    /// Mana symbols paid to cast it, such as `{2}`; absent for a land play.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub this_turn: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub own_turn_only: bool,
    /// Usable only on a turn numbered after this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after_turn: Option<u32>,
    /// Lasts while the permitted player controls this permanent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_permanent_id: Option<u32>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub flash: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub any_mana: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub waterbend: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub foretold: bool,
}

/// Mana spendable only on spells with one of `spells`' subtypes or abilities
/// of sources with one of `abilities`' subtypes; W/U/B/R/G/C amounts.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyRestrictedManaView {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spells: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub abilities: Vec<String>,
    pub mana: [i32; 6],
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyPermanentView {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<LegacyCardTypesView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_token: Option<bool>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub exile_if_dies_this_turn: bool,
    /// Exhaust abilities this object has used (CR 702.177a).
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub exhausted_abilities: u32,
    /// "Once each turn" abilities this object has used this turn.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub once_each_turn_used: u32,
    /// Firebending granted until end of turn.
    #[serde(default, skip_serializing_if = "crate::state::is_zero")]
    pub granted_firebending: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub temp_subtypes: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text_box: String,
    pub id: u32,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub name: RequiredNullable<String>,
    pub controller_id: u32,
    pub tapped: bool,
    pub damage: i32,
    pub summoning_sick: bool,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub power: RequiredNullable<i32>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub toughness: RequiredNullable<i32>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub base_power: RequiredNullable<i32>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub base_toughness: RequiredNullable<i32>,
    pub plus1_counters: u32,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SideboardCardView {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text_box: String,
    pub candidate_id: u32,
    pub owner_id: u32,
    pub registry_key: u32,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyPlayerView {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mana: Option<[i32; 6]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lesson_mana: Option<[i32; 6]>,
    /// `lesson_mana` split by restriction class.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restricted_mana: Vec<LegacyRestrictedManaView>,
    /// Player id whose next turn ends this player's hand-only casting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand_only_until: Option<u32>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_maximum_hand_size: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combat_mana: Option<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inspected: Vec<LegacyCardView>,
    pub player_index: u8,
    pub id: u32,
    pub is_active: bool,
    pub is_agent: bool,
    pub life: i32,
    pub zone_counts: BTreeMap<String, u32>,
    pub library_count: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub known_hand: BTreeMap<String, u32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sideboard_counts: BTreeMap<String, u32>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub remaining_sideboard_counts: BTreeMap<String, u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sideboard: Vec<SideboardCardView>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand_hidden_count: Option<u32>,
    pub hand: Vec<LegacyCardView>,
    pub graveyard: Vec<LegacyCardView>,
    pub exile: Vec<LegacyCardView>,
    pub stack: Vec<LegacyCardView>,
    pub battlefield: Vec<LegacyPermanentView>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyTurnView {
    pub turn_number: u32,
    pub phase: String,
    pub step: String,
    pub active_player_id: u32,
    pub agent_player_id: u32,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OfferSubjectView {
    pub role: String,
    pub object: DecisionObjectView,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionTargetView {
    pub clause: u32,
    pub object: DecisionObjectView,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StackEntryView {
    pub id: u64,
    pub kind: String,
    pub controller: u8,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_permanent: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ability_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_mode: Option<u32>,
}

/// Read-only explanation of the current native continuation. This adds no
/// choices, forecasts, or local transaction state to the experience contract.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionContextView {
    pub recent_facts: Vec<String>,
    pub actor: u8,
    pub stage: String,
    pub instruction: String,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub source: RequiredNullable<DecisionObjectView>,
    pub targets: Vec<DecisionTargetView>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub reserved_sacrifice: RequiredNullable<DecisionObjectView>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub mana_cost: RequiredNullable<String>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub chosen_mode: RequiredNullable<u32>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionObjectView {
    pub entity_id: u32,
    pub subject: SubjectRef,
    pub display_id: u32,
    pub label: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyHeroObservation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack_objects: Option<Vec<StackEntryView>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision_context: Option<DecisionContextView>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub definition_names: BTreeMap<String, String>,
    pub game_over: bool,
    pub won: bool,
    pub turn: LegacyTurnView,
    pub agent: LegacyPlayerView,
    pub opponent: LegacyPlayerView,
}

impl LegacyHeroObservation {
    /// Cross-viewer explanations may contain public ownership and kind, but
    /// never the other player's private continuation or selected objects.
    pub fn exposes_other_private_context(&self) -> bool {
        self.decision_context.as_ref().is_some_and(|context| {
            context.actor != self.agent.player_index
                && (context.source.0.is_some()
                    || !context.targets.is_empty()
                    || context.reserved_sacrifice.0.is_some()
                    || context.mana_cost.0.is_some()
                    || context.chosen_mode.0.is_some())
        })
    }
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StopsConfig {
    pub my: Vec<String>,
    pub opponent: Vec<String>,
    pub stop_on_stack: bool,
    pub auto_pass: bool,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeckNames {
    pub hero: String,
    pub villain: String,
}

/// Exact installed presentation pack for an authored matchup.
///
/// Custom and legacy games have no pack reference, while curated games bind
/// the human-facing assets to the same manifest digest carried by the frame.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssetPackReference {
    pub id: String,
    pub version: String,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExperienceFrame {
    pub protocol: ProtocolVersion,
    pub match_id: MatchId,
    pub revision: Revision,
    pub frame_hash: FrameHash,
    pub content_hash: ContentHash,
    pub asset_manifest_hash: AssetManifestHash,
    pub status: AuthorityStatus,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub prompt: RequiredNullable<PromptView>,
    pub projection: LegacyHeroObservation,
    pub offers: Vec<InteractionOffer>,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub winner: RequiredNullable<PlayerId>,
    pub action_space: String,
    pub stops: StopsConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deck_names: Option<DeckNames>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_pack: Option<AssetPackReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_passed: Option<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationImportance {
    Ambient,
    Normal,
    Emphasized,
    Critical,
}

/// Viewer-safe semantic presentation payload shared by Rust, Python, and the
/// TypeScript table. Rules code produces these facts; clients may choose
/// timing and composition but never reconstruct them from frame differences.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields, tag = "kind", rename_all = "snake_case")]
pub enum PresentationKind {
    Cast {
        object: ObjectRenderId,
        controller: PlayerId,
        stack: StackRenderId,
    },
    Targeted {
        source: SubjectRef,
        target: SubjectRef,
    },
    Resolved {
        stack: StackRenderId,
    },
    Damage {
        #[serde(deserialize_with = "deserialize_required_nullable")]
        source: RequiredNullable<SubjectRef>,
        target: SubjectRef,
        amount: i32,
    },
    Destroyed {
        objects: Vec<ObjectRenderId>,
    },
    Died {
        objects: Vec<ObjectRenderId>,
    },
    AttackGroup {
        attackers: Vec<ObjectRenderId>,
        defender: SubjectRef,
    },
    Blocked {
        attacker: ObjectRenderId,
        blockers: Vec<ObjectRenderId>,
    },
    TurnStarted {
        player: PlayerId,
        turn_number: u32,
    },
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PresentationEvent {
    pub seq: PresentationSeq,
    pub from_revision: Revision,
    pub to_revision: Revision,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub caused_by: RequiredNullable<CommandId>,
    pub group: PresentationGroupId,
    pub importance: PresentationImportance,
    pub suggested_ms: u32,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub sound: RequiredNullable<String>,
    pub kind: PresentationKind,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CommandReceipt {
    pub command_id: CommandId,
    pub actor: PlayerId,
    pub accepted_at: Revision,
    pub resulting_revision: Revision,
    pub resulting_frame_hash: FrameHash,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryReason {
    InitialConnect,
    ExplicitResync,
    RevisionGap,
    Reconnect,
    DuplicateCommand,
    StaleCommand,
    AuthorityRestart,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryEnvelope {
    pub protocol: ProtocolVersion,
    pub engine_version: String,
    pub content_hash: ContentHash,
    pub asset_manifest_hash: AssetManifestHash,
    pub reason: RecoveryReason,
    pub frame: ExperienceFrame,
    /// Sequence address of the first event in `presentation_tail`. For an
    /// empty tail this is the authority's next presentation sequence.
    pub presentation_cursor: PresentationCursor,
    pub presentation_tail: Vec<PresentationEvent>,
    pub accepted_commands: Vec<CommandReceipt>,
    pub replay_cursor: ReplayCursor,
    #[serde(deserialize_with = "deserialize_required_nullable")]
    pub checkpoint: RequiredNullable<CheckpointId>,
}

/// Shared executable fixture: one recovery frame and the command selected
/// from that exact prompt. Keeping this root intentionally small makes the
/// current certification claim precise.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolV1ConformanceBundle {
    pub recovery: RecoveryEnvelope,
    pub command: Command,
}
