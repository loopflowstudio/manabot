//! Named columns of the policy-input encoding, shared by native writes and
//! public discovery.
//!
//! Each `columns!` module is one feature axis: its constants are the column
//! indices the encoders write (`player::LIFE`, `permanent::DEFINITION`, ...),
//! `COLUMNS` carries each column's name and value encoding, and `WIDTH` is the
//! axis width the buffers allocate. [`observation_layout`] assembles them into
//! the per-array description `managym.observation_layout()` publishes.
//! Capacities (cards, permanents, actions, events, facts) stay the caller's
//! allocation contract and are published as `None` extents.

use std::collections::BTreeMap;

use super::{
    continuation::MAX_DECISION_CODE,
    observation_encoder::{FactKind, InputLayout},
    shared_effects::{DETAIL_ROWS, EFFECT_ROWS, MEMBER_ROWS},
};

/// A permanent row's columns: input v11 writes every `permanent` column; the
/// deprecated v9/v10 rows end at `dealt_damage_this_turn`, then validity
/// (`legacy_input`).
pub fn permanent_columns(layout: InputLayout) -> Vec<Column> {
    match layout {
        InputLayout::OneToken => permanent::COLUMNS.to_vec(),
        InputLayout::Legacy => {
            let mut columns = permanent::COLUMNS[..super::legacy_input::PERMANENT_VALID].to_vec();
            columns.push(Column {
                name: "valid",
                index: super::legacy_input::PERMANENT_VALID,
                value: ValueLayout::FLAG,
            });
            columns
        }
    }
}

/// What a column's stored value is: the closed set every consumer chooses its
/// treatment from. `scale` is the divisor the encoder already applied (1 when
/// unscaled), and exists only for amounts. `size` is a category's largest
/// code; a code above it is never an error (a category may grow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    /// 0 or 1.
    Flag,
    /// A nonnegative amount, stored divided by `scale`.
    Count { scale: u16 },
    /// Floating mana in a player pool, stored divided by `scale`.
    Mana { scale: u16 },
    /// A signed amount, stored divided by `scale`.
    Number { scale: u16 },
    /// A one-based order rank; 0 absent.
    Rank,
    /// A one-based position in a zone; 0 absent.
    Position,
    /// A code from a small fixed set, at most `size`.
    Category { size: u16 },
    /// A one-based choice index (a mode, an ability, a continuation frame); 0 absent.
    Selector,
    /// A row of the padded object table.
    Reference,
    /// An identity (a card definition, a token value half); never a magnitude.
    Id,
    /// A value whose meaning depends on its row's kind (`row_kinds`).
    Tagged,
}

impl ValueKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Flag => "flag",
            Self::Count { .. } => "count",
            Self::Mana { .. } => "mana",
            Self::Number { .. } => "number",
            Self::Rank => "rank",
            Self::Position => "position",
            Self::Category { .. } => "category",
            Self::Selector => "selector",
            Self::Reference => "reference",
            Self::Id => "id",
            Self::Tagged => "tagged",
        }
    }

    pub const fn scale(self) -> Option<u16> {
        match self {
            Self::Count { scale } | Self::Mana { scale } | Self::Number { scale } => Some(scale),
            _ => None,
        }
    }

    pub const fn size(self) -> Option<u16> {
        match self {
            Self::Category { size } => Some(size),
            _ => None,
        }
    }
}

const UNSCALED: u16 = 1;

/// Stored value semantics, shared by feature columns and scalar array elements.
#[derive(Clone, Copy, Debug)]
pub struct ValueLayout {
    pub kind: ValueKind,
    pub encoding: &'static str,
    pub range: &'static str,
}

impl ValueLayout {
    const fn new(kind: ValueKind, encoding: &'static str, range: &'static str) -> Self {
        Self {
            kind,
            encoding,
            range,
        }
    }

    const FLAG: Self = Self::new(ValueKind::Flag, "Boolean: 0 false, 1 true", "0 or 1");
    const COUNT: Self = Self::new(
        ValueKind::Count { scale: UNSCALED },
        "Unscaled count",
        "nonnegative; no fixed bound",
    );
    const COUNT_TENTHS: Self = Self::new(
        ValueKind::Count { scale: 10 },
        "Value / 10",
        "nonnegative; no fixed bound",
    );
    const COUNT_SIXTIETHS: Self = Self::new(
        ValueKind::Count { scale: 60 },
        "Value / 60",
        "nonnegative; no fixed bound",
    );
    const MANA_TENTHS: Self = Self::new(
        ValueKind::Mana { scale: 10 },
        "Value / 10",
        "nonnegative; no fixed bound",
    );
    const RESTRICTED_MANA: Self = Self::new(
        ValueKind::Mana { scale: 10 },
        "All spend-restricted mana of this color / 10 (legacy field name)",
        "nonnegative; no fixed bound",
    );
    const SUBTYPE_MANA: Self = Self::new(
        ValueKind::Mana { scale: 10 },
        "Spend-restricted mana admitting this subtype / 10; subtype totals can overlap",
        "nonnegative; no fixed bound",
    );
    const NONARTIFACT_MANA: Self = Self::new(
        ValueKind::Mana { scale: 10 },
        "Mana of this color that can't be spent to cast nonartifact spells / 10",
        "nonnegative; no fixed bound",
    );
    const NUMBER_TENTHS: Self = Self::new(
        ValueKind::Number { scale: 10 },
        "Value / 10",
        "signed; no fixed bound",
    );
    const DEFINITION: Self = Self::new(
        ValueKind::Id,
        "CardDefId + 1; 0 means absent",
        "nonnegative integers; 0 absent",
    );
    const OBJECT_REFERENCE: Self = Self::new(ValueKind::Reference, "Padded object table index + 1; 0 absent. Order: agent player, opponent player, agent cards, opponent cards, agent permanents, opponent permanents, then (input v11) stack", "nonnegative integers; 0 absent");
    const TIMESTAMP: Self = Self::new(
        ValueKind::Rank,
        "One-based rank among observation timestamps",
        "nonnegative integers; 0 absent",
    );
    const SELECTOR: Self = Self::new(
        ValueKind::Selector,
        "Zero-based selector + 1; 0 absent",
        "0..16777216",
    );
    const FRAME: Self = Self::new(
        ValueKind::Selector,
        "Continuation frame index + 1; 0 absent",
        "nonnegative integers; 0 absent",
    );
    const TOKEN_HALF: Self = Self::new(ValueKind::Tagged, "One unsigned 16-bit half of a signed i32 token value; combine high << 16 | low, then interpret as i32", "0..65535");
}

#[derive(Clone, Copy, Debug)]
pub struct Column {
    pub name: &'static str,
    pub index: usize,
    pub value: ValueLayout,
}

macro_rules! columns {
    ($module:ident { $($index:literal: $key:ident => ($name:literal, $value:expr)),* $(,)? }) => {
        pub mod $module {
            #[allow(unused_imports)]
            use super::{Column, ValueKind, ValueLayout, MAX_DECISION_CODE, UNSCALED};

            $(pub const $key: usize = $index;)*
            pub const COLUMNS: &[Column] = &[
                $(Column { name: $name, index: $key, value: $value }),*
            ];
            pub const WIDTH: usize = COLUMNS.len();
        }
    };
}

columns! { player {
    0: LIFE => ("life", ValueLayout::new(ValueKind::Number { scale: 20 }, "Value / 20", "signed; no fixed bound")),
    1: IS_ACTIVE => ("is_active", ValueLayout::FLAG),
    2: LIBRARY_COUNT => ("library_count", ValueLayout::COUNT_SIXTIETHS),
    3: HAND_COUNT => ("hand_count", ValueLayout::COUNT_SIXTIETHS),
    4: BATTLEFIELD_COUNT => ("battlefield_count", ValueLayout::COUNT_SIXTIETHS),
    5: GRAVEYARD_COUNT => ("graveyard_count", ValueLayout::COUNT_SIXTIETHS),
    6: STACK_COUNT => ("stack_count", ValueLayout::COUNT_SIXTIETHS),
    7: EXILE_COUNT => ("exile_count", ValueLayout::COUNT_SIXTIETHS),
    8: COMMAND_COUNT => ("command_count", ValueLayout::COUNT_SIXTIETHS),
    9: PHASE_BEGINNING => ("phase_beginning", ValueLayout::FLAG),
    10: PHASE_PRECOMBAT_MAIN => ("phase_precombat_main", ValueLayout::FLAG),
    11: PHASE_COMBAT => ("phase_combat", ValueLayout::FLAG),
    12: PHASE_POSTCOMBAT_MAIN => ("phase_postcombat_main", ValueLayout::FLAG),
    13: PHASE_ENDING => ("phase_ending", ValueLayout::FLAG),
    14: STEP_UNTAP => ("step_untap", ValueLayout::FLAG),
    15: STEP_UPKEEP => ("step_upkeep", ValueLayout::FLAG),
    16: STEP_DRAW => ("step_draw", ValueLayout::FLAG),
    17: STEP_MAIN => ("step_main", ValueLayout::FLAG),
    18: STEP_BEGINNING_OF_COMBAT => ("step_beginning_of_combat", ValueLayout::FLAG),
    19: STEP_DECLARE_ATTACKERS => ("step_declare_attackers", ValueLayout::FLAG),
    20: STEP_DECLARE_BLOCKERS => ("step_declare_blockers", ValueLayout::FLAG),
    21: STEP_COMBAT_DAMAGE => ("step_combat_damage", ValueLayout::FLAG),
    22: STEP_END_OF_COMBAT => ("step_end_of_combat", ValueLayout::FLAG),
    23: STEP_POSTCOMBAT_MAIN => ("step_postcombat_main", ValueLayout::FLAG),
    24: STEP_END => ("step_end", ValueLayout::FLAG),
    25: STEP_CLEANUP => ("step_cleanup", ValueLayout::FLAG),
    26: GRAVEYARD_LESSONS => ("graveyard_lessons", ValueLayout::COUNT_TENTHS),
    27: COMBAT_MANA => ("combat_mana", ValueLayout::MANA_TENTHS),
    28: MANA_WHITE => ("mana_white", ValueLayout::MANA_TENTHS),
    29: MANA_BLUE => ("mana_blue", ValueLayout::MANA_TENTHS),
    30: MANA_BLACK => ("mana_black", ValueLayout::MANA_TENTHS),
    31: MANA_RED => ("mana_red", ValueLayout::MANA_TENTHS),
    32: MANA_GREEN => ("mana_green", ValueLayout::MANA_TENTHS),
    33: MANA_COLORLESS => ("mana_colorless", ValueLayout::MANA_TENTHS),
    34: LESSON_MANA_WHITE => ("lesson_mana_white", ValueLayout::RESTRICTED_MANA),
    35: LESSON_MANA_BLUE => ("lesson_mana_blue", ValueLayout::RESTRICTED_MANA),
    36: LESSON_MANA_BLACK => ("lesson_mana_black", ValueLayout::RESTRICTED_MANA),
    37: LESSON_MANA_RED => ("lesson_mana_red", ValueLayout::RESTRICTED_MANA),
    38: LESSON_MANA_GREEN => ("lesson_mana_green", ValueLayout::RESTRICTED_MANA),
    39: LESSON_MANA_COLORLESS => ("lesson_mana_colorless", ValueLayout::RESTRICTED_MANA),
    40: RESTRICTED_LESSON_WHITE => ("restricted_lesson_white", ValueLayout::SUBTYPE_MANA),
    41: RESTRICTED_LESSON_BLUE => ("restricted_lesson_blue", ValueLayout::SUBTYPE_MANA),
    42: RESTRICTED_LESSON_BLACK => ("restricted_lesson_black", ValueLayout::SUBTYPE_MANA),
    43: RESTRICTED_LESSON_RED => ("restricted_lesson_red", ValueLayout::SUBTYPE_MANA),
    44: RESTRICTED_LESSON_GREEN => ("restricted_lesson_green", ValueLayout::SUBTYPE_MANA),
    45: RESTRICTED_LESSON_COLORLESS => ("restricted_lesson_colorless", ValueLayout::SUBTYPE_MANA),
    46: RESTRICTED_SHRINE_WHITE => ("restricted_shrine_white", ValueLayout::SUBTYPE_MANA),
    47: RESTRICTED_SHRINE_BLUE => ("restricted_shrine_blue", ValueLayout::SUBTYPE_MANA),
    48: RESTRICTED_SHRINE_BLACK => ("restricted_shrine_black", ValueLayout::SUBTYPE_MANA),
    49: RESTRICTED_SHRINE_RED => ("restricted_shrine_red", ValueLayout::SUBTYPE_MANA),
    50: RESTRICTED_SHRINE_GREEN => ("restricted_shrine_green", ValueLayout::SUBTYPE_MANA),
    51: RESTRICTED_SHRINE_COLORLESS => ("restricted_shrine_colorless", ValueLayout::SUBTYPE_MANA),
    52: RESTRICTED_ALLY_WHITE => ("restricted_ally_white", ValueLayout::SUBTYPE_MANA),
    53: RESTRICTED_ALLY_BLUE => ("restricted_ally_blue", ValueLayout::SUBTYPE_MANA),
    54: RESTRICTED_ALLY_BLACK => ("restricted_ally_black", ValueLayout::SUBTYPE_MANA),
    55: RESTRICTED_ALLY_RED => ("restricted_ally_red", ValueLayout::SUBTYPE_MANA),
    56: RESTRICTED_ALLY_GREEN => ("restricted_ally_green", ValueLayout::SUBTYPE_MANA),
    57: RESTRICTED_ALLY_COLORLESS => ("restricted_ally_colorless", ValueLayout::SUBTYPE_MANA),
    58: HAND_ONLY => ("hand_only", ValueLayout::FLAG),
    59: NO_MAXIMUM_HAND_SIZE => ("no_maximum_hand_size", ValueLayout::FLAG),
    60: TURN => ("turn", ValueLayout::new(ValueKind::Count { scale: 20 }, "Value / 20", "nonnegative; no fixed bound")),
    61: SEAT => ("seat", ValueLayout::new(ValueKind::Category { size: 1 }, "Game seat: 0 on the play, 1 on the draw", "0 or 1")),
    62: KEEPS_RED_MANA => ("keeps_red_mana", ValueLayout::FLAG),
    63: ATTACKED => ("attacked", ValueLayout::FLAG),
    64: PERMANENTS_SACRIFICED => ("permanents_sacrificed", ValueLayout::COUNT_TENTHS),
    65: LAND_ENTERED => ("land_entered", ValueLayout::FLAG),
    66: CARDS_DRAWN => ("cards_drawn", ValueLayout::COUNT_TENTHS),
    67: BENT_WATER => ("bent_water", ValueLayout::FLAG),
    68: BENT_EARTH => ("bent_earth", ValueLayout::FLAG),
    69: BENT_FIRE => ("bent_fire", ValueLayout::FLAG),
    70: BENT_AIR => ("bent_air", ValueLayout::FLAG),
    71: EXTRA_TURNS => ("extra_turns", ValueLayout::COUNT),
    72: ADDITIONAL_COMBATS => ("additional_combats", ValueLayout::COUNT),
    73: DECKLISTS_OPEN => ("decklists_open", ValueLayout::FLAG),
} }

// Input v10 player turn state, one row per seat (agent first).
columns! { turn_state {
    0: LIFE_LOST => ("life_lost", ValueLayout::new(ValueKind::Count { scale: 20 }, "Life lost this turn / 20", "nonnegative; no fixed bound")),
    1: SPELLS_CAST => ("spells_cast", ValueLayout::new(ValueKind::Count { scale: 10 }, "Spells cast this turn / 10", "nonnegative; no fixed bound")),
    2: NONARTIFACT_MANA_WHITE => ("nonartifact_mana_white", ValueLayout::NONARTIFACT_MANA),
    3: NONARTIFACT_MANA_BLUE => ("nonartifact_mana_blue", ValueLayout::NONARTIFACT_MANA),
    4: NONARTIFACT_MANA_BLACK => ("nonartifact_mana_black", ValueLayout::NONARTIFACT_MANA),
    5: NONARTIFACT_MANA_RED => ("nonartifact_mana_red", ValueLayout::NONARTIFACT_MANA),
    6: NONARTIFACT_MANA_GREEN => ("nonartifact_mana_green", ValueLayout::NONARTIFACT_MANA),
    7: NONARTIFACT_MANA_COLORLESS => ("nonartifact_mana_colorless", ValueLayout::NONARTIFACT_MANA),
} }

columns! { card {
    0: ZONE_LIBRARY => ("zone_library", ValueLayout::FLAG),
    1: ZONE_HAND => ("zone_hand", ValueLayout::FLAG),
    2: ZONE_BATTLEFIELD => ("zone_battlefield", ValueLayout::FLAG),
    3: ZONE_GRAVEYARD => ("zone_graveyard", ValueLayout::FLAG),
    4: ZONE_STACK => ("zone_stack", ValueLayout::FLAG),
    5: ZONE_EXILE => ("zone_exile", ValueLayout::FLAG),
    6: ZONE_COMMAND => ("zone_command", ValueLayout::FLAG),
    7: IS_MINE => ("is_mine", ValueLayout::FLAG),
    8: POWER => ("power", ValueLayout::NUMBER_TENTHS),
    9: TOUGHNESS => ("toughness", ValueLayout::NUMBER_TENTHS),
    10: MANA_VALUE => ("mana_value", ValueLayout::COUNT_TENTHS),
    11: IS_LAND => ("is_land", ValueLayout::FLAG),
    12: IS_CREATURE => ("is_creature", ValueLayout::FLAG),
    13: IS_ARTIFACT => ("is_artifact", ValueLayout::FLAG),
    14: IS_ENCHANTMENT => ("is_enchantment", ValueLayout::FLAG),
    15: IS_PLANESWALKER => ("is_planeswalker", ValueLayout::FLAG),
    16: IS_BATTLE => ("is_battle", ValueLayout::FLAG),
    17: FLYING => ("flying", ValueLayout::FLAG),
    18: REACH => ("reach", ValueLayout::FLAG),
    19: HASTE => ("haste", ValueLayout::FLAG),
    20: VIGILANCE => ("vigilance", ValueLayout::FLAG),
    21: TRAMPLE => ("trample", ValueLayout::FLAG),
    22: FIRST_STRIKE => ("first_strike", ValueLayout::FLAG),
    23: DOUBLE_STRIKE => ("double_strike", ValueLayout::FLAG),
    24: DEATHTOUCH => ("deathtouch", ValueLayout::FLAG),
    25: LIFELINK => ("lifelink", ValueLayout::FLAG),
    26: DEFENDER => ("defender", ValueLayout::FLAG),
    27: MENACE => ("menace", ValueLayout::FLAG),
    28: FLASH => ("flash", ValueLayout::FLAG),
    29: IS_TOKEN => ("is_token", ValueLayout::FLAG),
    30: IS_ALLY => ("is_ally", ValueLayout::FLAG),
    31: IS_LESSON => ("is_lesson", ValueLayout::FLAG),
    32: HAS_WARD => ("has_ward", ValueLayout::FLAG),
    33: WARD_COST => ("ward_cost", ValueLayout::COUNT_TENTHS),
    34: HAS_KICKER => ("has_kicker", ValueLayout::FLAG),
    35: KICKER_COST => ("kicker_cost", ValueLayout::COUNT_TENTHS),
    36: HEXPROOF => ("hexproof", ValueLayout::FLAG),
    37: OUTSIDE => ("outside", ValueLayout::FLAG),
    38: INDESTRUCTIBLE => ("indestructible", ValueLayout::FLAG),
    39: IS_LEGENDARY => ("is_legendary", ValueLayout::FLAG),
    40: PERMISSION => ("permission", ValueLayout::FLAG),
    41: PERMISSION_IS_MINE => ("permission_is_mine", ValueLayout::FLAG),
    42: PERMISSION_HAS_COST => ("permission_has_cost", ValueLayout::FLAG),
    43: PERMISSION_MANA_VALUE => ("permission_mana_value", ValueLayout::COUNT_TENTHS),
    44: PERMISSION_THIS_TURN => ("permission_this_turn", ValueLayout::FLAG),
    45: PERMISSION_OWN_TURN_ONLY => ("permission_own_turn_only", ValueLayout::FLAG),
    46: PERMISSION_NOT_YET => ("permission_not_yet", ValueLayout::FLAG),
    47: PERMISSION_HAS_SOURCE => ("permission_has_source", ValueLayout::FLAG),
    48: PERMISSION_FLASH => ("permission_flash", ValueLayout::FLAG),
    49: PERMISSION_ANY_MANA => ("permission_any_mana", ValueLayout::FLAG),
    50: PERMISSION_WATERBEND => ("permission_waterbend", ValueLayout::FLAG),
    51: PERMISSION_FORETOLD => ("permission_foretold", ValueLayout::FLAG),
    52: PERMISSION_THROUGH_NEXT_TURN => ("permission_through_next_turn", ValueLayout::FLAG),
    53: HAS_X => ("has_x", ValueLayout::FLAG),
    54: X => ("x", ValueLayout::COUNT_TENTHS),
    55: VALID => ("valid", ValueLayout::FLAG),
} }

columns! { permanent {
    0: IS_MINE => ("is_mine", ValueLayout::FLAG),
    1: TAPPED => ("tapped", ValueLayout::FLAG),
    2: DAMAGE => ("damage", ValueLayout::COUNT_TENTHS),
    3: SUMMONING_SICK => ("summoning_sick", ValueLayout::FLAG),
    4: PLUS_ONE_COUNTERS => ("plus_one_counters", ValueLayout::COUNT_TENTHS),
    5: CANT_BE_BLOCKED_THIS_TURN => ("cant_be_blocked_this_turn", ValueLayout::FLAG),
    6: POWER => ("power", ValueLayout::NUMBER_TENTHS),
    7: TOUGHNESS => ("toughness", ValueLayout::NUMBER_TENTHS),
    8: ANIMATED => ("animated", ValueLayout::FLAG),
    9: HAS_EXILE_LINK => ("has_exile_link", ValueLayout::FLAG),
    10: FLYING => ("flying", ValueLayout::FLAG),
    11: REACH => ("reach", ValueLayout::FLAG),
    12: HASTE => ("haste", ValueLayout::FLAG),
    13: FLASH => ("flash", ValueLayout::FLAG),
    14: VIGILANCE => ("vigilance", ValueLayout::FLAG),
    15: TRAMPLE => ("trample", ValueLayout::FLAG),
    16: FIRST_STRIKE => ("first_strike", ValueLayout::FLAG),
    17: DOUBLE_STRIKE => ("double_strike", ValueLayout::FLAG),
    18: DEATHTOUCH => ("deathtouch", ValueLayout::FLAG),
    19: LIFELINK => ("lifelink", ValueLayout::FLAG),
    20: DEFENDER => ("defender", ValueLayout::FLAG),
    21: MENACE => ("menace", ValueLayout::FLAG),
    22: HEXPROOF => ("hexproof", ValueLayout::FLAG),
    23: EXILE_IF_DIES_THIS_TURN => ("exile_if_dies_this_turn", ValueLayout::FLAG),
    24: DEFINITION => ("definition", ValueLayout::DEFINITION),
    25: INDESTRUCTIBLE => ("indestructible", ValueLayout::FLAG),
    26: ADDED_AVATAR => ("added_avatar", ValueLayout::FLAG),
    27: ATTACKING => ("attacking", ValueLayout::FLAG),
    28: EXHAUSTED_ABILITIES => ("exhausted_abilities", ValueLayout::COUNT),
    29: ONCE_EACH_TURN_USED => ("once_each_turn_used", ValueLayout::COUNT),
    30: GRANTED_FIREBENDING => ("granted_firebending", ValueLayout::COUNT_TENTHS),
    31: ATTACHED_TO => ("attached_to", ValueLayout::OBJECT_REFERENCE),
    32: GRANTED_LAND => ("granted_land", ValueLayout::FLAG),
    33: BLOCKING => ("blocking", ValueLayout::OBJECT_REFERENCE),
    34: ABILITIES_LOST => ("abilities_lost", ValueLayout::FLAG),
    35: TIMESTAMP => ("timestamp", ValueLayout::TIMESTAMP),
    36: DEALT_DAMAGE_THIS_TURN => ("dealt_damage_this_turn", ValueLayout::FLAG),
    37: MANA_VALUE => ("mana_value", ValueLayout::COUNT_TENTHS),
    38: IS_LAND => ("is_land", ValueLayout::FLAG),
    39: IS_CREATURE => ("is_creature", ValueLayout::FLAG),
    40: IS_ARTIFACT => ("is_artifact", ValueLayout::FLAG),
    41: IS_ENCHANTMENT => ("is_enchantment", ValueLayout::FLAG),
    42: IS_PLANESWALKER => ("is_planeswalker", ValueLayout::FLAG),
    43: IS_BATTLE => ("is_battle", ValueLayout::FLAG),
    44: IS_LEGENDARY => ("is_legendary", ValueLayout::FLAG),
    45: IS_TOKEN => ("is_token", ValueLayout::FLAG),
    46: IS_ALLY => ("is_ally", ValueLayout::FLAG),
    47: IS_LESSON => ("is_lesson", ValueLayout::FLAG),
    48: HAS_WARD => ("has_ward", ValueLayout::FLAG),
    49: WARD_COST => ("ward_cost", ValueLayout::COUNT_TENTHS),
    50: VALID => ("valid", ValueLayout::FLAG),
    51: ENTERED_THIS_TURN => ("entered_this_turn", ValueLayout::FLAG),
} }

// One row per spell or ability on the stack, top first (input v11). The
// spell's characteristics read its card record; an ability's definition is
// its source's.
columns! { stack {
    0: IS_MINE => ("is_mine", ValueLayout::FLAG),
    1: SPELL => ("spell", ValueLayout::FLAG),
    2: ACTIVATED_ABILITY => ("activated_ability", ValueLayout::FLAG),
    3: TRIGGERED_ABILITY => ("triggered_ability", ValueLayout::FLAG),
    4: DEFINITION => ("definition", ValueLayout::DEFINITION),
    5: SOURCE => ("source", ValueLayout::OBJECT_REFERENCE),
    6: ABILITY => ("ability", ValueLayout::SELECTOR),
    7: MODE => ("mode", ValueLayout::SELECTOR),
    8: HAS_X => ("has_x", ValueLayout::FLAG),
    9: X => ("x", ValueLayout::COUNT_TENTHS),
    10: KICKED => ("kicked", ValueLayout::FLAG),
    11: IS_COPY => ("is_copy", ValueLayout::FLAG),
    12: MANA_VALUE => ("mana_value", ValueLayout::COUNT_TENTHS),
    13: POWER => ("power", ValueLayout::NUMBER_TENTHS),
    14: TOUGHNESS => ("toughness", ValueLayout::NUMBER_TENTHS),
    15: IS_LAND => ("is_land", ValueLayout::FLAG),
    16: IS_CREATURE => ("is_creature", ValueLayout::FLAG),
    17: IS_ARTIFACT => ("is_artifact", ValueLayout::FLAG),
    18: IS_ENCHANTMENT => ("is_enchantment", ValueLayout::FLAG),
    19: IS_PLANESWALKER => ("is_planeswalker", ValueLayout::FLAG),
    20: IS_BATTLE => ("is_battle", ValueLayout::FLAG),
    21: IS_LEGENDARY => ("is_legendary", ValueLayout::FLAG),
    22: FLYING => ("flying", ValueLayout::FLAG),
    23: REACH => ("reach", ValueLayout::FLAG),
    24: HASTE => ("haste", ValueLayout::FLAG),
    25: FLASH => ("flash", ValueLayout::FLAG),
    26: VIGILANCE => ("vigilance", ValueLayout::FLAG),
    27: TRAMPLE => ("trample", ValueLayout::FLAG),
    28: FIRST_STRIKE => ("first_strike", ValueLayout::FLAG),
    29: DOUBLE_STRIKE => ("double_strike", ValueLayout::FLAG),
    30: DEATHTOUCH => ("deathtouch", ValueLayout::FLAG),
    31: LIFELINK => ("lifelink", ValueLayout::FLAG),
    32: DEFENDER => ("defender", ValueLayout::FLAG),
    33: MENACE => ("menace", ValueLayout::FLAG),
    34: HEXPROOF => ("hexproof", ValueLayout::FLAG),
    35: INDESTRUCTIBLE => ("indestructible", ValueLayout::FLAG),
    36: VALID => ("valid", ValueLayout::FLAG),
    37: EFFECTIVE_MANA_VALUE => ("effective_mana_value", ValueLayout::COUNT_TENTHS),
    38: FROM_PREPARED => ("from_prepared", ValueLayout::FLAG),
    39: FLASHBACK => ("flashback", ValueLayout::FLAG),
} }

columns! { action {
    0: PLAY_LAND => ("play_land", ValueLayout::FLAG),
    1: CAST_SPELL => ("cast_spell", ValueLayout::FLAG),
    2: PASS_PRIORITY => ("pass_priority", ValueLayout::FLAG),
    3: DECLARE_ATTACKER => ("declare_attacker", ValueLayout::FLAG),
    4: DECLARE_BLOCKER => ("declare_blocker", ValueLayout::FLAG),
    5: CHOOSE_TARGET => ("choose_target", ValueLayout::FLAG),
    6: ACTIVATE_ABILITY => ("activate_ability", ValueLayout::FLAG),
    7: SCRY_KEEP => ("scry_keep", ValueLayout::FLAG),
    8: SCRY_BOTTOM => ("scry_bottom", ValueLayout::FLAG),
    9: SELECT_CARD => ("select_card", ValueLayout::FLAG),
    10: DECLINE_CHOICE => ("decline_choice", ValueLayout::FLAG),
    11: PAY_COST => ("pay_cost", ValueLayout::FLAG),
    12: CHOOSE_MODE => ("choose_mode", ValueLayout::FLAG),
    13: TAP_FOR_COST => ("tap_for_cost", ValueLayout::FLAG),
    14: LEARN_TAKE_LESSON => ("learn_take_lesson", ValueLayout::FLAG),
    15: LEARN_DISCARD => ("learn_discard", ValueLayout::FLAG),
    16: CHOOSE_TRIGGER => ("choose_trigger", ValueLayout::FLAG),
    17: DECLARED => ("declared", ValueLayout::FLAG),
    18: MODE => ("mode", ValueLayout::SELECTOR),
    19: ABILITY => ("ability", ValueLayout::SELECTOR),
    20: CONTINUATION => ("continuation", ValueLayout::new(ValueKind::Reference, "Continuation frame (relation row) index + 1; 0 absent", "nonnegative integers; 0 absent")),
    21: VALID => ("valid", ValueLayout::FLAG),
    // Input v11 only. Saved 22-column prefixes keep their meaning.
    22: PAYS_DIRECTLY => ("pays_directly", ValueLayout::FLAG),
    // The offer announces a number (X, a replicate count, a payment's X):
    // `mode` keeps its index, and these name it as the stack row does.
    23: HAS_X => ("has_x", ValueLayout::FLAG),
    24: X => ("x", ValueLayout::COUNT_TENTHS),
    25: HAS_ROUTE_COST => ("has_route_cost", ValueLayout::FLAG),
    26: ROUTE_MANA_VALUE => ("route_mana_value", ValueLayout::COUNT_TENTHS),
    27: QUOTA_TYPE => ("quota_type", ValueLayout::new(ValueKind::Category {size:9}, "Permanent type spent by this play (CardType + 1)", "0 absent; 1..9")),
    28: SPELL_PROFILE => ("spell_profile", ValueLayout::new(ValueKind::Category {size:4}, "Printed spell this offer casts (spell_profile_code): 1 Adventure, 2 Omen, 3 back face", "0 front face or no spell; 1..3")),
    29: PLAY_ROUTE => ("play_route", ValueLayout::new(ValueKind::Category {size:10}, "Permission this play uses (play_route_code): 1 hand, 2 free from hand, 3 library top, 4 printed flashback, 5 granted flashback, 6 grant, 7 graveyard type, 8 exile counter, 9 printed graveyard", "0 ordinary hand play or not a play; 1..9")),
} }

pub fn action_columns(layout: InputLayout) -> &'static [Column] {
    match layout {
        InputLayout::OneToken => action::COLUMNS,
        InputLayout::Legacy => &action::COLUMNS[..action::PAYS_DIRECTLY],
    }
}

columns! { name {
    0: NAME => ("name", ValueLayout::new(ValueKind::Reference, "naming_catalog row + 1; 0 absent", "nonnegative integers; 0 absent")),
    1: DEFINITION => ("definition", ValueLayout::DEFINITION),
    2: GROUP_FIRST => ("group_first", ValueLayout::new(ValueKind::Reference, "Inclusive naming_catalog row + 1; 0 absent", "nonnegative integers; 0 absent")),
    3: GROUP_LAST => ("group_last", ValueLayout::new(ValueKind::Reference, "Inclusive naming_catalog row + 1; 0 absent", "nonnegative integers; 0 absent")),
} }

columns! { known_library {
    0: DEFINITION => ("definition", ValueLayout::DEFINITION),
    1: POSITION => ("position", ValueLayout::new(ValueKind::Position, "Library position from top + 1; 0 absent", "nonnegative integers")),
} }

columns! { known_hand {
    0: DEFINITION => ("definition", ValueLayout::DEFINITION),
    1: COUNT => ("count", ValueLayout::COUNT),
} }

columns! { decklist {
    0: DEFINITION => ("definition", ValueLayout::DEFINITION),
    1: MAIN => ("main", ValueLayout::COUNT),
    2: UNSEEN => ("unseen", ValueLayout::COUNT),
    3: SIDEBOARD => ("sideboard", ValueLayout::COUNT),
    4: REMAINING_SIDEBOARD => ("remaining_sideboard", ValueLayout::COUNT),
} }

columns! { event {
    0: KIND => ("kind", ValueLayout::new(ValueKind::Category { size: 6 }, "EventType: 0 card moved, 1 damage dealt, 2 life changed, 3 spell cast, 4 spell resolved, 5 spell countered, 6 ability triggered", "0..6")),
    1: SOURCE_KIND => ("source_kind", ValueLayout::new(ValueKind::Category { size: 5 }, "EventEntityKind: 0 none, 1 card, 2 permanent, 3 player, 4 object, 5 definition; namespace for raw id", "0..5")),
    2: SOURCE_ID => ("source_id", ValueLayout::new(ValueKind::Id, "Raw event entity id in the accompanying kind namespace; -1 absent; not a padded object-table index", "integers >= -1")),
    3: TARGET_KIND => ("target_kind", ValueLayout::new(ValueKind::Category { size: 5 }, "EventEntityKind: 0 none, 1 card, 2 permanent, 3 player, 4 object, 5 definition; namespace for raw id", "0..5")),
    4: TARGET_ID => ("target_id", ValueLayout::new(ValueKind::Id, "Raw event entity id in the accompanying kind namespace; -1 absent; not a padded object-table index", "integers >= -1")),
    5: AMOUNT => ("amount", ValueLayout::new(ValueKind::Number { scale: UNSCALED }, "Unscaled signed value", "signed; no fixed bound")),
    6: CONTROLLER => ("controller", ValueLayout::new(ValueKind::Id, "Raw player seat; -1 absent", "-1, 0 or 1")),
} }

columns! { decision_record {
    0: ORDINAL_HIGH => ("ordinal_high", ValueLayout::COUNT),
    1: ORDINAL_LOW => ("ordinal_low", ValueLayout::COUNT),
    2: COUNT_HIGH => ("count_high", ValueLayout::COUNT),
    3: COUNT_LOW => ("count_low", ValueLayout::COUNT),
    4: TURN_HIGH => ("turn_high", ValueLayout::COUNT),
    5: TURN_LOW => ("turn_low", ValueLayout::COUNT),
    6: FIRST_HIGH => ("first_high", ValueLayout::COUNT),
    7: FIRST_LOW => ("first_low", ValueLayout::COUNT),
    8: LAST_HIGH => ("last_high", ValueLayout::COUNT),
    9: LAST_LOW => ("last_low", ValueLayout::COUNT),
    10: PHASE => ("phase", ValueLayout::COUNT),
    11: STEP => ("step", ValueLayout::COUNT),
    12: KIND_HIGH => ("kind_high", ValueLayout::COUNT),
    13: KIND_LOW => ("kind_low", ValueLayout::COUNT),
    14: MEMBER_OFFSET => ("member_offset", ValueLayout::COUNT),
    15: MANA_COUNT => ("mana_count", ValueLayout::COUNT),
    16: RESPONSE_COUNT => ("response_count", ValueLayout::COUNT),
    17: CHOICE_COUNT => ("choice_count", ValueLayout::COUNT),
    18: CHOOSER_ROLE => ("chooser_role", ValueLayout::COUNT),
    19: VALID => ("valid", ValueLayout::COUNT),
    20: LIFE_VIEWER_HIGH => ("life_viewer_high", ValueLayout::COUNT),
    21: LIFE_VIEWER_LOW => ("life_viewer_low", ValueLayout::COUNT),
    22: LIFE_OPPONENT_HIGH => ("life_opponent_high", ValueLayout::COUNT),
    23: LIFE_OPPONENT_LOW => ("life_opponent_low", ValueLayout::COUNT),
    24: HAND_VIEWER_HIGH => ("hand_viewer_high", ValueLayout::COUNT),
    25: HAND_VIEWER_LOW => ("hand_viewer_low", ValueLayout::COUNT),
    26: HAND_OPPONENT_HIGH => ("hand_opponent_high", ValueLayout::COUNT),
    27: HAND_OPPONENT_LOW => ("hand_opponent_low", ValueLayout::COUNT),
    28: BOARD_VIEWER_TOTAL_HIGH => ("board_viewer_total_high", ValueLayout::COUNT),
    29: BOARD_VIEWER_TOTAL_LOW => ("board_viewer_total_low", ValueLayout::COUNT),
    30: BOARD_VIEWER_TAPPED_HIGH => ("board_viewer_tapped_high", ValueLayout::COUNT),
    31: BOARD_VIEWER_TAPPED_LOW => ("board_viewer_tapped_low", ValueLayout::COUNT),
    32: BOARD_VIEWER_LAND_HIGH => ("board_viewer_land_high", ValueLayout::COUNT),
    33: BOARD_VIEWER_LAND_LOW => ("board_viewer_land_low", ValueLayout::COUNT),
    34: BOARD_VIEWER_CREATURE_HIGH => ("board_viewer_creature_high", ValueLayout::COUNT),
    35: BOARD_VIEWER_CREATURE_LOW => ("board_viewer_creature_low", ValueLayout::COUNT),
    36: BOARD_VIEWER_ARTIFACT_HIGH => ("board_viewer_artifact_high", ValueLayout::COUNT),
    37: BOARD_VIEWER_ARTIFACT_LOW => ("board_viewer_artifact_low", ValueLayout::COUNT),
    38: BOARD_VIEWER_ENCHANTMENT_HIGH => ("board_viewer_enchantment_high", ValueLayout::COUNT),
    39: BOARD_VIEWER_ENCHANTMENT_LOW => ("board_viewer_enchantment_low", ValueLayout::COUNT),
    40: BOARD_VIEWER_PLANESWALKER_HIGH => ("board_viewer_planeswalker_high", ValueLayout::COUNT),
    41: BOARD_VIEWER_PLANESWALKER_LOW => ("board_viewer_planeswalker_low", ValueLayout::COUNT),
    42: BOARD_VIEWER_BATTLE_HIGH => ("board_viewer_battle_high", ValueLayout::COUNT),
    43: BOARD_VIEWER_BATTLE_LOW => ("board_viewer_battle_low", ValueLayout::COUNT),
    44: BOARD_OPPONENT_TOTAL_HIGH => ("board_opponent_total_high", ValueLayout::COUNT),
    45: BOARD_OPPONENT_TOTAL_LOW => ("board_opponent_total_low", ValueLayout::COUNT),
    46: BOARD_OPPONENT_TAPPED_HIGH => ("board_opponent_tapped_high", ValueLayout::COUNT),
    47: BOARD_OPPONENT_TAPPED_LOW => ("board_opponent_tapped_low", ValueLayout::COUNT),
    48: BOARD_OPPONENT_LAND_HIGH => ("board_opponent_land_high", ValueLayout::COUNT),
    49: BOARD_OPPONENT_LAND_LOW => ("board_opponent_land_low", ValueLayout::COUNT),
    50: BOARD_OPPONENT_CREATURE_HIGH => ("board_opponent_creature_high", ValueLayout::COUNT),
    51: BOARD_OPPONENT_CREATURE_LOW => ("board_opponent_creature_low", ValueLayout::COUNT),
    52: BOARD_OPPONENT_ARTIFACT_HIGH => ("board_opponent_artifact_high", ValueLayout::COUNT),
    53: BOARD_OPPONENT_ARTIFACT_LOW => ("board_opponent_artifact_low", ValueLayout::COUNT),
    54: BOARD_OPPONENT_ENCHANTMENT_HIGH => ("board_opponent_enchantment_high", ValueLayout::COUNT),
    55: BOARD_OPPONENT_ENCHANTMENT_LOW => ("board_opponent_enchantment_low", ValueLayout::COUNT),
    56: BOARD_OPPONENT_PLANESWALKER_HIGH => ("board_opponent_planeswalker_high", ValueLayout::COUNT),
    57: BOARD_OPPONENT_PLANESWALKER_LOW => ("board_opponent_planeswalker_low", ValueLayout::COUNT),
    58: BOARD_OPPONENT_BATTLE_HIGH => ("board_opponent_battle_high", ValueLayout::COUNT),
    59: BOARD_OPPONENT_BATTLE_LOW => ("board_opponent_battle_low", ValueLayout::COUNT),
    60: CHOICE_KIND_PRESENT => ("choice_kind_present", ValueLayout::COUNT),
    61: CHOICE_KIND_HIGH => ("choice_kind_high", ValueLayout::COUNT),
    62: CHOICE_KIND_LOW => ("choice_kind_low", ValueLayout::COUNT),
    63: MODE_PRESENT => ("mode_present", ValueLayout::COUNT),
    64: MODE_HIGH => ("mode_high", ValueLayout::COUNT),
    65: MODE_LOW => ("mode_low", ValueLayout::COUNT),
    66: ABILITY_PRESENT => ("ability_present", ValueLayout::COUNT),
    67: ABILITY_HIGH => ("ability_high", ValueLayout::COUNT),
    68: ABILITY_LOW => ("ability_low", ValueLayout::COUNT),
    69: NAME_PRESENT => ("name_present", ValueLayout::COUNT),
    70: NAME_HIGH => ("name_high", ValueLayout::COUNT),
    71: NAME_LOW => ("name_low", ValueLayout::COUNT),
    72: DECLARED_PRESENT => ("declared_present", ValueLayout::COUNT),
    73: DECLARED => ("declared", ValueLayout::COUNT),
} }

columns! { decision_reference {
    0: ROLE => ("role", ValueLayout::COUNT),
    1: DEFINITION => ("definition", ValueLayout::COUNT),
    2: POWER_HIGH => ("power_high", ValueLayout::COUNT),
    3: POWER_LOW => ("power_low", ValueLayout::COUNT),
    4: TOUGHNESS_HIGH => ("toughness_high", ValueLayout::COUNT),
    5: TOUGHNESS_LOW => ("toughness_low", ValueLayout::COUNT),
    6: TAPPED => ("tapped", ValueLayout::COUNT),
    7: READY => ("ready", ValueLayout::COUNT),
    8: ATTACKING => ("attacking", ValueLayout::COUNT),
    9: BLOCKING => ("blocking", ValueLayout::COUNT),
    10: DETAIL_OFFSET => ("detail_offset", ValueLayout::COUNT),
    11: DETAIL_COUNT => ("detail_count", ValueLayout::COUNT),
    12: VALID => ("valid", ValueLayout::COUNT),
    13: DEFINITION_ONLY => ("definition_only", ValueLayout::FLAG),
} }

columns! { decision_reference_detail {
    0: FIELD => ("field", ValueLayout::COUNT),
    1: TOKEN_KIND => ("token_kind", ValueLayout::COUNT),
    2: VALUE_HIGH => ("value_high", ValueLayout::COUNT),
    3: VALUE_LOW => ("value_low", ValueLayout::COUNT),
    4: ORDINAL => ("ordinal", ValueLayout::COUNT),
    5: VALID => ("valid", ValueLayout::COUNT),
} }

columns! { decision_member {
    0: REFERENCE => ("reference", ValueLayout::COUNT),
    1: ORDINAL => ("ordinal", ValueLayout::COUNT),
} }

columns! { history {
    0: KIND => ("kind", ValueLayout::new(ValueKind::Category { size: 5 }, "HistoryKind: 0 arrival, 1 damage, 2 life, 3 cast, 4 resolved, 5 countered", "0..5")),
    1: AMOUNT => ("amount", ValueLayout::new(ValueKind::Number { scale: UNSCALED }, "Unscaled signed value", "signed; no fixed bound")),
    2: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    3: SOURCE_OWNER => ("source_owner", ValueLayout::new(ValueKind::Category { size: 2 }, "0 unknown, 1 agent, 2 opponent", "0..2")),
    4: SOURCE_ZONE => ("source_zone", ValueLayout::new(ValueKind::Category { size: 7 }, "ZoneType + 1; 0 unknown", "0..7")),
    5: SOURCE_CONTEXT => ("source_context", ValueLayout::new(ValueKind::Reference, "Zero-based padded object-table index; -1 unavailable", "integers >= -1")),
    6: SOURCE_STATUS => ("source_status", ValueLayout::new(ValueKind::Category { size: 2 }, "0 unavailable, 1 current object join, 2 definition only", "0..2")),
    7: TARGET_DEFINITION => ("target_definition", ValueLayout::DEFINITION),
    8: TARGET_OWNER => ("target_owner", ValueLayout::new(ValueKind::Category { size: 2 }, "0 unknown, 1 agent, 2 opponent", "0..2")),
    9: TARGET_ZONE => ("target_zone", ValueLayout::new(ValueKind::Category { size: 7 }, "ZoneType + 1; 0 unknown", "0..7")),
    10: TARGET_CONTEXT => ("target_context", ValueLayout::new(ValueKind::Reference, "Zero-based padded object-table index; -1 unavailable", "integers >= -1")),
    11: TARGET_STATUS => ("target_status", ValueLayout::new(ValueKind::Category { size: 2 }, "0 unavailable, 1 current object join, 2 definition only", "0..2")),
} }

columns! { fact {
    0: OBJECT => ("object", ValueLayout::OBJECT_REFERENCE),
    1: KIND => ("kind", ValueLayout::new(ValueKind::Category { size: 14 }, "FactKind; see row_kinds", "1..14 excluding 6; 0 padding")),
    2: ID_HIGH => ("id_high", ValueLayout::TOKEN_HALF),
    3: ID_LOW => ("id_low", ValueLayout::TOKEN_HALF),
    4: DETAIL_HIGH => ("detail_high", ValueLayout::TOKEN_HALF),
    5: DETAIL_LOW => ("detail_low", ValueLayout::TOKEN_HALF),
    6: VALUE => ("value", ValueLayout::new(ValueKind::Tagged, "Kind-dependent payload; see row_kinds. Counts and effect power/toughness / 10; program kinds/ordinals and transforms unscaled", "signed; no fixed bound")),
    7: VALUE_2 => ("value_2", ValueLayout::new(ValueKind::Tagged, "Kind-dependent payload; see row_kinds. Counts and effect power/toughness / 10; program kinds/ordinals and transforms unscaled", "signed; no fixed bound")),
    8: DURATION => ("duration", ValueLayout::new(ValueKind::Category { size: 4 }, "0 absent, 1 end of turn, 2 indefinite, 3 while source controlled, 4 until your next turn", "0..4")),
    9: TIMESTAMP => ("timestamp", ValueLayout::TIMESTAMP),
    10: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    11: VALID => ("valid", ValueLayout::FLAG),
} }

columns! { effect {
    0: MODIFICATION_HIGH => ("modification_high", ValueLayout::new(ValueKind::Id, "One unsigned 16-bit half of a signed i32 token value; combine high << 16 | low, then interpret as i32", "0..65535")),
    1: MODIFICATION_LOW => ("modification_low", ValueLayout::new(ValueKind::Id, "One unsigned 16-bit half of a signed i32 token value; combine high << 16 | low, then interpret as i32", "0..65535")),
    2: POWER => ("power", ValueLayout::NUMBER_TENTHS),
    3: TOUGHNESS => ("toughness", ValueLayout::NUMBER_TENTHS),
    4: AMOUNT => ("amount", ValueLayout::NUMBER_TENTHS),
    5: HAS_POWER => ("has_power", ValueLayout::FLAG),
    6: HAS_TOUGHNESS => ("has_toughness", ValueLayout::FLAG),
    7: HAS_AMOUNT => ("has_amount", ValueLayout::FLAG),
    8: DURATION => ("duration", ValueLayout::new(ValueKind::Category { size: 4 }, "1 end of turn, 2 indefinite, 3 while source controlled, 4 until your next turn", "1..4; 0 padding")),
    9: TIMESTAMP => ("timestamp", ValueLayout::TIMESTAMP),
    10: CONTROLLER => ("controller", ValueLayout::OBJECT_REFERENCE),
    11: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    12: SOURCE => ("source", ValueLayout::OBJECT_REFERENCE),
    13: DETAIL_OFFSET => ("detail_offset", ValueLayout::new(ValueKind::Reference, "Zero-based start in effect_details; count determines presence", "0..4096")),
    14: DETAIL_COUNT => ("detail_count", ValueLayout::COUNT),
    15: MEMBER_OFFSET => ("member_offset", ValueLayout::new(ValueKind::Reference, "Zero-based start in effect_members; count determines presence", "0..2048")),
    16: MEMBER_COUNT => ("member_count", ValueLayout::COUNT),
    17: VALID => ("valid", ValueLayout::FLAG),
} }

columns! { detail {
    0: KIND => ("kind", ValueLayout::new(ValueKind::Category { size: 19 }, "1 keyword, 2 subtype, 3 color; 10 + catalog TokenKind (1..9)", "0 padding; 1..3 or 11..19")),
    1: VALUE_HIGH => ("value_high", ValueLayout::TOKEN_HALF),
    2: VALUE_LOW => ("value_low", ValueLayout::TOKEN_HALF),
} }

columns! { member {
    0: OBJECT => ("object", ValueLayout::OBJECT_REFERENCE),
    1: ORDER => ("order", ValueLayout::new(ValueKind::Rank, "Original EffectRecords position + 1; duplicate edges retained", "1..2048; 0 padding")),
} }

columns! { relation {
    0: KIND => ("kind", ValueLayout::new(ValueKind::Category { size: 10 }, "Relation row kind; see row_kinds", "1..10; 0 padding")),
    1: FRAME => ("frame", ValueLayout::FRAME),
    2: PARENT => ("parent", ValueLayout::FRAME),
    3: FRAME_KIND => ("frame_kind", ValueLayout::new(ValueKind::Category { size: 15 }, "1 spell, 2 activated, 3 triggered, 4 announcement, 5 resolution, 6 obligation, 7 waiting trigger, 8 placing trigger, 9 placement cursor, 10 delayed trigger, 11 scheduled turn, 12 combat, 13 control, 14 delegation, 15 opening hands", "1..15; 0 padding")),
    4: CONTROLLER => ("controller", ValueLayout::new(ValueKind::Category { size: 2 }, "1 agent, 2 opponent", "0..2")),
    5: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    6: SOURCE => ("source", ValueLayout::OBJECT_REFERENCE),
    7: MODE => ("mode", ValueLayout::SELECTOR),
    8: ABILITY => ("ability", ValueLayout::SELECTOR),
    9: KICKED => ("kicked", ValueLayout::FLAG),
    10: ACTIVE_CLAUSE => ("active_clause", ValueLayout::SELECTOR),
    11: HAS_COST => ("has_cost", ValueLayout::FLAG),
    12: MANA_WHITE => ("mana_white", ValueLayout::COUNT),
    13: MANA_BLUE => ("mana_blue", ValueLayout::COUNT),
    14: MANA_BLACK => ("mana_black", ValueLayout::COUNT),
    15: MANA_RED => ("mana_red", ValueLayout::COUNT),
    16: MANA_GREEN => ("mana_green", ValueLayout::COUNT),
    17: MANA_COLORLESS => ("mana_colorless", ValueLayout::COUNT),
    18: MANA_GENERIC => ("mana_generic", ValueLayout::COUNT),
    19: RESTRICTED => ("restricted", ValueLayout::FLAG),
    20: WATERBEND => ("waterbend", ValueLayout::FLAG),
    21: CAN_DECLINE => ("can_decline", ValueLayout::FLAG),
    22: CLAUSE => ("clause", ValueLayout::SELECTOR),
    23: REFERENCE_DEFINITION => ("reference_definition", ValueLayout::DEFINITION),
    24: REFERENCE => ("reference", ValueLayout::OBJECT_REFERENCE),
    25: REFERENCE_PLAYER => ("reference_player", ValueLayout::new(ValueKind::Category { size: 2 }, "0 absent, 1 agent, 2 opponent", "0..2")),
    26: HYBRID_FIRST => ("hybrid_first", ValueLayout::new(ValueKind::Category { size: 6 }, "Color index + 1; 0 absent", "0..6")),
    27: HYBRID_SECOND => ("hybrid_second", ValueLayout::new(ValueKind::Category { size: 6 }, "Color index + 1; 0 absent", "0..6")),
    28: DECISION => ("decision", ValueLayout::new(ValueKind::Category { size: MAX_DECISION_CODE as u16 }, "Frame decision code. Kind-9 rows: delayed frequency 1 once / 2 each match", "nonnegative integer codes")),
    29: ORDINAL => ("ordinal", ValueLayout::new(ValueKind::Selector, "Reference ordinal + 1; kind 10 is effect-execution reference slot", "nonnegative integers; 0 absent")),
    30: RESOLUTIONS_THIS_TURN => ("resolutions_this_turn", ValueLayout::COUNT),
    31: VALID => ("valid", ValueLayout::FLAG),
    32: LEGAL_AT_RESOLUTION_START => ("legal_at_resolution_start", ValueLayout::new(ValueKind::Category { size: 2 }, "0 unknown, 1 legal, 2 illegal", "0..2")),
    33: X => ("x", ValueLayout::COUNT),
    34: FLYING => ("flying", ValueLayout::FLAG),
    35: REACH => ("reach", ValueLayout::FLAG),
    36: HASTE => ("haste", ValueLayout::FLAG),
    37: FLASH => ("flash", ValueLayout::FLAG),
    38: VIGILANCE => ("vigilance", ValueLayout::FLAG),
    39: TRAMPLE => ("trample", ValueLayout::FLAG),
    40: FIRST_STRIKE => ("first_strike", ValueLayout::FLAG),
    41: DOUBLE_STRIKE => ("double_strike", ValueLayout::FLAG),
    42: DEATHTOUCH => ("deathtouch", ValueLayout::FLAG),
    43: LIFELINK => ("lifelink", ValueLayout::FLAG),
    44: DEFENDER => ("defender", ValueLayout::FLAG),
    45: MENACE => ("menace", ValueLayout::FLAG),
    46: HEXPROOF => ("hexproof", ValueLayout::FLAG),
    47: INDESTRUCTIBLE => ("indestructible", ValueLayout::FLAG),
    48: THIS_TURN => ("this_turn", ValueLayout::FLAG),
    49: SPELL_PROFILE => ("spell_profile", ValueLayout::new(ValueKind::Category {size:4}, "Printed spell an announcement frame casts (spell_profile_code): 1 Adventure, 2 Omen, 3 back face", "0 front face or no spell; 1..3")),
} }

#[derive(Clone, Debug)]
pub struct ArrayLayout {
    pub axes: Vec<&'static str>,
    /// Fixed extents; None is a capacity chosen by the caller.
    pub shape: Vec<Option<usize>>,
    pub dtype: &'static str,
    pub columns: Vec<Column>,
    /// Masks, semantic_cards and action_focus have no feature axis.
    pub element: Option<ValueLayout>,
    pub row_kinds: BTreeMap<u8, &'static str>,
    /// Each row kind's name, for consumers that read kinds by name.
    pub row_kind_names: BTreeMap<&'static str, u8>,
}

impl ArrayLayout {
    fn features(axes: &[&'static str], shape: &[Option<usize>], columns: &[Column]) -> Self {
        let mut axes = axes.to_vec();
        axes.push("feature");
        let mut shape = shape.to_vec();
        shape.push(Some(columns.len()));
        Self {
            axes,
            shape,
            dtype: "float32",
            columns: columns.to_vec(),
            element: None,
            row_kinds: BTreeMap::new(),
            row_kind_names: BTreeMap::new(),
        }
    }

    fn elements(
        axes: &[&'static str],
        shape: &[Option<usize>],
        dtype: &'static str,
        element: ValueLayout,
    ) -> Self {
        Self {
            axes: axes.to_vec(),
            shape: shape.to_vec(),
            dtype,
            columns: Vec::new(),
            element: Some(element),
            row_kinds: BTreeMap::new(),
            row_kind_names: BTreeMap::new(),
        }
    }
}

/// A fact kind's name and what its row holds.
fn fact_row_kind(kind: FactKind) -> (&'static str, &'static str) {
    match kind {
        FactKind::Counter => ("counter", "id: CounterKind token; value: count / 10"),
        FactKind::AddedSubtype => ("added_subtype", "id: added subtype token; value: 1"),
        FactKind::SetName => ("set_name", "id: set-name token; value: 1"),
        FactKind::SetCreatureType => ("set_creature_type", "id: creature-type token; value: 1"),
        FactKind::UsedAbility => ("used_ability", "id: zero-based activated ability index; value: 1"),
        FactKind::Copy => ("copy", "id: CopyEffect token; source_definition: copied definition; timestamp: copy rank; duration: 2"),
        FactKind::CopyException => ("copy_exception", "Copy exception: id: Modification token; detail: keyword/subtype/color token or 0; value: power or amount / 10; value_2: toughness / 10; duration, timestamp and source_definition identify record provenance"),
        FactKind::Transforms => ("transforms", "id: 0 for permanent, frame index + 1 for captured source; value: unscaled transform count"),
        FactKind::CombatRestriction => ("combat_restriction", "id: frame index + 1; detail: attacker-predicate token"),
        FactKind::EffectProgram => ("effect_program", "Rules program: id: Modification token (a copy exception, with duration, timestamp and source_definition as its provenance), PooledMana (a player's pooled mana with a spend effect or combat expiry, one program per class and expiry, no provenance) or Ward (a permanent's ward price that is not a mana value, no provenance); detail: signed i32 catalog token value; value: TokenKind; value_2: zero-based ordinal"),
        FactKind::EffectiveColor => ("effective_color", "id: Color token (Colorless for empty colors); value: 1"),
        FactKind::KeywordCounterTimestamp => ("keyword_counter_timestamp", "id: CounterKind token; timestamp: counter timestamp rank"),
        FactKind::ContinuationProgram => ("continuation_program", "object: controller; id: frame index + 1; detail: signed i32 catalog token value; value: TokenKind; value_2: zero-based ordinal. Source program then effect execution are separate occurrences, each resetting ordinal to 0"),
    }
}

/// Optional v11 memory arrays; both capacities belong to the caller.
pub fn decision_memory_layout() -> BTreeMap<String, ArrayLayout> {
    BTreeMap::from([
        (
            "decision_records".into(),
            ArrayLayout::features(&["opponent_record"], &[None], decision_record::COLUMNS),
        ),
        (
            "decision_references".into(),
            ArrayLayout::features(
                &["opponent_reference"],
                &[None],
                decision_reference::COLUMNS,
            ),
        ),
        (
            "decision_reference_details".into(),
            ArrayLayout::features(
                &["opponent_reference_detail"],
                &[None],
                decision_reference_detail::COLUMNS,
            ),
        ),
        (
            "decision_members".into(),
            ArrayLayout::features(&["opponent_member"], &[None], decision_member::COLUMNS),
        ),
    ])
}

/// Resolve each variable extent once, at the capacities the encoder will use.
pub fn configured_arrays(
    config: &super::observation_encoder::ObservationEncoderConfig,
) -> Result<BTreeMap<String, ArrayLayout>, String> {
    let memory = [
        config.max_decision_records,
        config.max_decision_references,
        config.max_decision_reference_details,
        config.max_decision_members,
    ];
    if memory.contains(&0) && memory.iter().any(|&n| n != 0) {
        return Err(String::from(
            "decision memory requires all companion capacities",
        ));
    }
    if config.policy_history && config.max_events > crate::agent::policy_history::HISTORY_LIMIT {
        return Err(String::from("public history v1 requires 1..32 rows"));
    }
    let mut arrays = observation_layout(config.policy_history, config.turn_state, config.layout);
    if config.max_decision_records > 0 {
        if config.layout == InputLayout::Legacy {
            return Err(String::from("decision memory requires policy input v11"));
        }
        arrays.extend(decision_memory_layout());
    }
    for array in arrays.values_mut() {
        for (extent, axis) in array.shape.iter_mut().zip(&array.axes) {
            if extent.is_some() {
                continue;
            }
            let n = match *axis {
                "card" | "definition" => config.max_cards_per_player,
                "permanent" => config.max_permanents_per_player,
                "stack" => config.max_stack_objects,
                "action" => config.max_actions,
                "focus" => config.max_focus_objects,
                "event" => config.max_events,
                "fact" => config.max_facts,
                "opponent_record" => config.max_decision_records,
                "opponent_reference" => config.max_decision_references,
                "opponent_reference_detail" => config.max_decision_reference_details,
                "opponent_member" => config.max_decision_members,
                _ => unreachable!("unknown configurable axis {axis}"),
            };
            if n == 0 {
                return Err(format!("{axis} capacity must be positive"));
            }
            *extent = Some(n);
        }
    }
    Ok(arrays)
}

/// Every array `encode` writes, keyed by its buffer name, without allocating
/// game buffers. `policy_history` selects the public history rows of `events`;
/// `turn_state` adds the turn-state buffers, as
/// `ObservationEncoderConfig::turn_state` does; `layout` selects input v11's
/// one-token rows or the deprecated v9/v10 rows.
pub fn observation_layout(
    policy_history: bool,
    turn_state: bool,
    layout_kind: InputLayout,
) -> BTreeMap<String, ArrayLayout> {
    let permanent_columns = permanent_columns(layout_kind);
    let mut layout = BTreeMap::new();
    for side in ["agent", "opponent"] {
        for (suffix, axis, capacity, columns) in [
            ("player", "player", Some(1), player::COLUMNS),
            ("cards", "card", None, card::COLUMNS),
            ("permanents", "permanent", None, &permanent_columns[..]),
        ] {
            layout.insert(
                format!("{side}_{suffix}"),
                ArrayLayout::features(&[axis], &[capacity], columns),
            );
            layout.insert(
                format!("{side}_{suffix}_valid"),
                ArrayLayout::elements(&[axis], &[capacity], "float32", ValueLayout::FLAG),
            );
        }
    }
    if layout_kind == InputLayout::OneToken {
        layout.insert(
            "stack".into(),
            ArrayLayout::features(&["stack"], &[None], stack::COLUMNS),
        );
        layout.insert(
            "stack_valid".into(),
            ArrayLayout::elements(&["stack"], &[None], "float32", ValueLayout::FLAG),
        );
    }
    layout.insert(
        "actions".into(),
        ArrayLayout::features(&["action"], &[None], action_columns(layout_kind)),
    );
    layout.insert(
        "events".into(),
        ArrayLayout::features(
            &["event"],
            &[None],
            if policy_history {
                history::COLUMNS
            } else {
                event::COLUMNS
            },
        ),
    );
    for (key, axis) in [("actions_valid", "action"), ("events_valid", "event")] {
        layout.insert(
            key.into(),
            ArrayLayout::elements(&[axis], &[None], "float32", ValueLayout::FLAG),
        );
    }
    layout.insert(
        "semantic_cards".into(),
        ArrayLayout::elements(
            &["viewer_relative_seat", "card"],
            &[Some(2), None],
            "float32",
            ValueLayout::DEFINITION,
        ),
    );
    if layout_kind == InputLayout::OneToken {
        layout.insert(
            "known_library".into(),
            ArrayLayout::features(
                &["viewer_relative_seat", "known_position"],
                &[Some(2), Some(crate::state::knowledge::KNOWN_LIBRARY_ROWS)],
                known_library::COLUMNS,
            ),
        );
    }
    layout.insert(
        "known_hand".into(),
        ArrayLayout::features(
            &["viewer_relative_seat", "definition"],
            &[Some(2), None],
            known_hand::COLUMNS,
        ),
    );
    layout.insert("action_focus".into(), ArrayLayout::elements(&["action", "focus"], &[None, None], "int32", ValueLayout {
        kind: ValueKind::Reference, encoding: "Zero-based padded object-table index; -1 absent. Order: agent player, opponent player, agent cards, opponent cards, agent permanents, opponent permanents, then (input v11) stack", range: "integers >= -1",
    }));
    let mut relations = ArrayLayout::features(
        &["relation"],
        &[Some(super::continuation::RELATION_ROWS)],
        relation::COLUMNS,
    );
    relations.row_kinds = BTreeMap::from([
        (1, "Frame header; frame_kind describes the unfinished work; other rows repeat this header"),
        (2, "Target: clause, ordinal, reference_definition/reference/reference_player and legal_at_resolution_start"),
        (3, "Event context: reference_definition/reference/reference_player"),
        (4, "Sacrifice: reference_definition/reference/reference_player"),
        (5, "Hybrid payment: hybrid_first/hybrid_second are alternative colors"),
        (6, "Mana output: mode, mana_white..mana_colorless and restricted flag; generic retains header cost"),
        (7, "Inherited ability lender: reference_definition/reference/reference_player"),
        (8, "Remembered subject: reference_definition/reference/reference_player and ordinal"),
        (9, "Delayed frequency: decision is 1 once or 2 each match; this_turn is expiry"),
        (10, "Effect execution reference: ordinal is the one-based slot in the continuation program; reference_definition/reference/reference_player identify its visible subject"),
    ]);
    layout.insert("relations".into(), relations);
    layout.insert(
        "decklists".into(),
        ArrayLayout::features(
            &["viewer_relative_seat", "definition"],
            &[Some(2), None],
            decklist::COLUMNS,
        ),
    );
    let mut facts = ArrayLayout::features(&["fact"], &[None], fact::COLUMNS);
    facts.row_kinds = FactKind::ALL
        .into_iter()
        .map(|kind| (kind as u8, fact_row_kind(kind).1))
        .collect();
    facts.row_kind_names = FactKind::ALL
        .into_iter()
        .map(|kind| (fact_row_kind(kind).0, kind as u8))
        .collect();
    layout.insert("facts".into(), facts);
    layout.insert(
        "effects".into(),
        ArrayLayout::features(&["effect"], &[Some(EFFECT_ROWS)], effect::COLUMNS),
    );
    let mut details = ArrayLayout::features(&["detail"], &[Some(DETAIL_ROWS)], detail::COLUMNS);
    details.row_kinds = BTreeMap::from([
        (1, "Keyword token in value_high/value_low"),
        (2, "Subtype token in value_high/value_low"),
        (3, "Color token in value_high/value_low"),
    ]);
    details.row_kind_names = BTreeMap::from([("keyword", 1), ("subtype", 2), ("color", 3)]);
    for (name, token) in super::shared_effects::PROGRAM_DETAIL_KINDS {
        let kind = super::shared_effects::program_detail_kind(token as u16) as u8;
        details.row_kind_names.insert(name, kind);
        details.row_kinds.insert(kind, "Catalog token: TokenKind = kind - 10; exact signed i32 value in value_high/value_low; row order is program order");
    }
    layout.insert("effect_details".into(), details);
    layout.insert(
        "effect_members".into(),
        ArrayLayout::features(&["membership"], &[Some(MEMBER_ROWS)], member::COLUMNS),
    );
    layout.insert(
        "name_candidates".into(),
        ArrayLayout::features(&["action"], &[None], name::COLUMNS),
    );
    if turn_state {
        layout.insert(
            "player_turn_state".into(),
            ArrayLayout::features(&["viewer_relative_seat"], &[Some(2)], turn_state::COLUMNS),
        );
        layout.insert(
            "card_turn_state".into(),
            ArrayLayout::elements(
                &["viewer_relative_seat", "card"],
                &[Some(2), None],
                "float32",
                ValueLayout::new(
                    ValueKind::Flag,
                    "1 for a graveyard card put there from the battlefield this turn; rows align with semantic_cards",
                    "0 or 1",
                ),
            ),
        );
    }
    layout
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{
        observation::KEYWORD_COUNT,
        observation_encoder::{ObservationEncoderConfig, FACT_DIM},
    };

    #[test]
    fn named_layout_matches_every_encoder_width() {
        for (policy_history, turn_state, layout_kind) in [
            (false, false, InputLayout::Legacy),
            (true, false, InputLayout::Legacy),
            (true, true, InputLayout::Legacy),
            (false, true, InputLayout::OneToken),
            (true, true, InputLayout::OneToken),
        ] {
            let config = ObservationEncoderConfig {
                policy_history,
                turn_state,
                layout: layout_kind,
                ..Default::default()
            };
            let layout = observation_layout(policy_history, turn_state, layout_kind);
            assert_eq!(layout.contains_key("player_turn_state"), turn_state);
            assert_eq!(layout.contains_key("card_turn_state"), turn_state);
            let one_token = layout_kind == InputLayout::OneToken;
            assert_eq!(layout.contains_key("stack"), one_token);
            assert_eq!(layout.contains_key("stack_valid"), one_token);
            for array in layout.values() {
                assert_eq!(array.axes.len(), array.shape.len());
                let names: std::collections::BTreeSet<_> =
                    array.columns.iter().map(|c| c.name).collect();
                assert_eq!(names.len(), array.columns.len());
                for (index, column) in array.columns.iter().enumerate() {
                    assert_eq!(column.index, index, "{}", column.name);
                    assert!(!column.value.encoding.is_empty() && !column.value.range.is_empty());
                }
            }
            // Numeric ABI: changing a width is a layout change that checkpoints see.
            for (key, width, abi) in [
                ("agent_player", player::WIDTH, 74),
                ("agent_cards", card::WIDTH, 56),
                (
                    "agent_permanents",
                    config.permanent_dim(),
                    if one_token { 52 } else { 38 },
                ),
                (
                    "actions",
                    config.action_dim(),
                    if one_token { 30 } else { 22 },
                ),
                (
                    "events",
                    config.event_dim(),
                    if policy_history { 12 } else { 7 },
                ),
                ("relations", relation::WIDTH, 50),
                ("facts", FACT_DIM, 12),
                ("effects", effect::WIDTH, 18),
                ("effect_details", detail::WIDTH, 3),
                ("effect_members", member::WIDTH, 2),
                ("name_candidates", name::WIDTH, 4),
                ("decklists", decklist::WIDTH, 5),
                ("known_hand", known_hand::WIDTH, 2),
            ] {
                assert_eq!(layout[key].columns.len(), width, "{key}");
                assert_eq!(width, abi, "{key}");
            }
            if one_token {
                assert_eq!(layout["stack"].columns.len(), stack::WIDTH);
                assert_eq!(stack::WIDTH, 40);
                assert_eq!(config.stack_len(), config.max_stack_objects * stack::WIDTH);
                assert_eq!(stack::VALID - stack::FLYING, KEYWORD_COUNT);
            }
            assert_eq!(
                config.cards_len(),
                config.max_cards_per_player * layout["agent_cards"].columns.len()
            );
            if turn_state {
                assert_eq!(
                    config.player_turn_state_len(),
                    2 * layout["player_turn_state"].columns.len()
                );
            }
            assert_eq!(
                config.relations_len(),
                super::super::continuation::RELATION_ROWS * relation::WIDTH
            );
            assert_eq!(relation::THIS_TURN - relation::FLYING, KEYWORD_COUNT);
            assert_eq!(layout["agent_permanents"].columns[24].name, "definition");
            assert_eq!(
                layout["agent_permanents"].columns[if one_token {
                    permanent::VALID
                } else {
                    super::super::legacy_input::PERMANENT_VALID
                }]
                .name,
                "valid"
            );
            for key in ["agent_cards", "facts", "effects"] {
                assert_eq!(layout[key].columns.last().unwrap().name, "valid", "{key}");
            }
            assert_eq!(layout["actions"].columns[action::VALID].name, "valid");
        }
    }
}
