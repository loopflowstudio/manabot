//! Named policy-array columns shared by native writes and public discovery.
//! Frozen prefixes retain their indices; legacy card/permanent validity replaces
//! the first appended column. Capacities remain the caller's allocation contract.

use std::collections::BTreeMap;

/// Stored value semantics, shared by feature columns and scalar array elements.
#[derive(Clone, Copy, Debug)]
pub struct ValueLayout {
    pub kind: &'static str,
    pub encoding: &'static str,
    pub range: &'static str,
}

impl ValueLayout {
    const fn new(kind: &'static str, encoding: &'static str, range: &'static str) -> Self {
        Self {
            kind,
            encoding,
            range,
        }
    }

    const FLAG: Self = Self::new("flag", "Boolean: 0 false, 1 true", "0 or 1");
    const COUNT: Self = Self::new("count", "Unscaled count", "nonnegative; no fixed bound");
    const COUNT_TENTHS: Self = Self::new("count", "Value / 10", "nonnegative; no fixed bound");
    const COUNT_SIXTIETHS: Self = Self::new("count", "Value / 60", "nonnegative; no fixed bound");
    const RESTRICTED_MANA: Self = Self::new(
        "count",
        "All spend-restricted mana of this color / 10 (legacy field name)",
        "nonnegative; no fixed bound",
    );
    const SUBTYPE_MANA: Self = Self::new(
        "count",
        "Spend-restricted mana admitting this subtype / 10; subtype totals can overlap",
        "nonnegative; no fixed bound",
    );
    const NUMBER_TENTHS: Self = Self::new("number", "Value / 10", "signed; no fixed bound");
    const DEFINITION: Self = Self::new(
        "id",
        "CardDefId + 1; 0 means absent",
        "nonnegative integers; 0 absent",
    );
    const OBJECT_REFERENCE: Self = Self::new("reference", "Padded object table index + 1; 0 absent. Order: agent player, opponent player, agent cards, opponent cards, agent permanents, opponent permanents", "nonnegative integers; 0 absent");
    const TIMESTAMP: Self = Self::new("rank", "One-based rank among observation timestamps; model divides by 128 regardless of fact capacity", "nonnegative integers; 0 absent");
    const SELECTOR: Self = Self::new(
        "reference",
        "Zero-based selector + 1; 0 absent",
        "0..16777216",
    );
    const TOKEN_HALF: Self = Self::new("tagged", "One unsigned 16-bit half of a signed i32 token value; combine high << 16 | low, then interpret as i32", "0..65535");
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
            use super::{Column, ValueLayout};

            $(pub const $key: usize = $index;)*
            pub const COLUMNS: &[Column] = &[
                $(Column { name: $name, index: $key, value: $value }),*
            ];
            pub const WIDTH: usize = COLUMNS.len();
        }
    };
}

columns! { player {
    0: LIFE => ("life", ValueLayout::new("number", "Value / 20", "signed; no fixed bound")),
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
    27: COMBAT_MANA => ("combat_mana", ValueLayout::COUNT_TENTHS),
    28: MANA_WHITE => ("mana_white", ValueLayout::COUNT_TENTHS),
    29: MANA_BLUE => ("mana_blue", ValueLayout::COUNT_TENTHS),
    30: MANA_BLACK => ("mana_black", ValueLayout::COUNT_TENTHS),
    31: MANA_RED => ("mana_red", ValueLayout::COUNT_TENTHS),
    32: MANA_GREEN => ("mana_green", ValueLayout::COUNT_TENTHS),
    33: MANA_COLORLESS => ("mana_colorless", ValueLayout::COUNT_TENTHS),
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
    60: TURN => ("turn", ValueLayout::new("count", "Value / 20", "nonnegative; no fixed bound")),
    61: SEAT => ("seat", ValueLayout::new("category", "Game seat: 0 on the play, 1 on the draw", "0 or 1")),
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
    37: VALID => ("valid", ValueLayout::FLAG),
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
    20: CONTINUATION => ("continuation", ValueLayout::SELECTOR),
    21: VALID => ("valid", ValueLayout::FLAG),
} }

columns! { name {
    0: NAME => ("name", ValueLayout::new("reference", "naming_catalog row + 1; 0 absent", "nonnegative integers; 0 absent")),
    1: DEFINITION => ("definition", ValueLayout::DEFINITION),
    2: GROUP_FIRST => ("group_first", ValueLayout::new("reference", "Inclusive naming_catalog row + 1; 0 absent", "nonnegative integers; 0 absent")),
    3: GROUP_LAST => ("group_last", ValueLayout::new("reference", "Inclusive naming_catalog row + 1; 0 absent", "nonnegative integers; 0 absent")),
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
    0: KIND => ("kind", ValueLayout::new("category", "EventType: 0 card moved, 1 damage dealt, 2 life changed, 3 spell cast, 4 spell resolved, 5 spell countered, 6 ability triggered", "0..6")),
    1: SOURCE_KIND => ("source_kind", ValueLayout::new("category", "EventEntityKind: 0 none, 1 card, 2 permanent, 3 player, 4 object, 5 definition; namespace for raw id", "0..5")),
    2: SOURCE_ID => ("source_id", ValueLayout::new("id", "Raw event entity id in the accompanying kind namespace; -1 absent; not a padded object-table index", "integers >= -1")),
    3: TARGET_KIND => ("target_kind", ValueLayout::new("category", "EventEntityKind: 0 none, 1 card, 2 permanent, 3 player, 4 object, 5 definition; namespace for raw id", "0..5")),
    4: TARGET_ID => ("target_id", ValueLayout::new("id", "Raw event entity id in the accompanying kind namespace; -1 absent; not a padded object-table index", "integers >= -1")),
    5: AMOUNT => ("amount", ValueLayout::new("number", "Unscaled signed value", "signed; no fixed bound")),
    6: CONTROLLER => ("controller", ValueLayout::new("id", "Raw player seat; -1 absent", "-1, 0 or 1")),
} }

columns! { history {
    0: KIND => ("kind", ValueLayout::new("category", "HistoryKind: 0 arrival, 1 damage, 2 life, 3 cast, 4 resolved, 5 countered", "0..5")),
    1: AMOUNT => ("amount", ValueLayout::new("number", "Unscaled signed value", "signed; no fixed bound")),
    2: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    3: SOURCE_OWNER => ("source_owner", ValueLayout::new("category", "0 unknown, 1 agent, 2 opponent", "0..2")),
    4: SOURCE_ZONE => ("source_zone", ValueLayout::new("category", "ZoneType + 1; 0 unknown", "0..7")),
    5: SOURCE_CONTEXT => ("source_context", ValueLayout::new("reference", "Zero-based padded object-table index; -1 unavailable", "integers >= -1")),
    6: SOURCE_STATUS => ("source_status", ValueLayout::new("category", "0 unavailable, 1 current object join, 2 definition only", "0..2")),
    7: TARGET_DEFINITION => ("target_definition", ValueLayout::DEFINITION),
    8: TARGET_OWNER => ("target_owner", ValueLayout::new("category", "0 unknown, 1 agent, 2 opponent", "0..2")),
    9: TARGET_ZONE => ("target_zone", ValueLayout::new("category", "ZoneType + 1; 0 unknown", "0..7")),
    10: TARGET_CONTEXT => ("target_context", ValueLayout::new("reference", "Zero-based padded object-table index; -1 unavailable", "integers >= -1")),
    11: TARGET_STATUS => ("target_status", ValueLayout::new("category", "0 unavailable, 1 current object join, 2 definition only", "0..2")),
} }

columns! { fact {
    0: OBJECT => ("object", ValueLayout::OBJECT_REFERENCE),
    1: KIND => ("kind", ValueLayout::new("category", "FactKind; see row_kinds", "1..14; 0 padding")),
    2: ID_HIGH => ("id_high", ValueLayout::TOKEN_HALF),
    3: ID_LOW => ("id_low", ValueLayout::TOKEN_HALF),
    4: DETAIL_HIGH => ("detail_high", ValueLayout::TOKEN_HALF),
    5: DETAIL_LOW => ("detail_low", ValueLayout::TOKEN_HALF),
    6: VALUE => ("value", ValueLayout::new("tagged", "Kind-dependent payload; see row_kinds. Counts and effect power/toughness / 10; program kinds/ordinals and transforms unscaled", "signed; no fixed bound")),
    7: VALUE_2 => ("value_2", ValueLayout::new("tagged", "Kind-dependent payload; see row_kinds. Counts and effect power/toughness / 10; program kinds/ordinals and transforms unscaled", "signed; no fixed bound")),
    8: DURATION => ("duration", ValueLayout::new("category", "0 absent, 1 end of turn, 2 indefinite, 3 while source controlled", "0..3")),
    9: TIMESTAMP => ("timestamp", ValueLayout::TIMESTAMP),
    10: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    11: VALID => ("valid", ValueLayout::FLAG),
} }

columns! { effect {
    0: MODIFICATION_HIGH => ("modification_high", ValueLayout::new("id", "One unsigned 16-bit half of a signed i32 token value; combine high << 16 | low, then interpret as i32", "0..65535")),
    1: MODIFICATION_LOW => ("modification_low", ValueLayout::new("id", "One unsigned 16-bit half of a signed i32 token value; combine high << 16 | low, then interpret as i32", "0..65535")),
    2: POWER => ("power", ValueLayout::NUMBER_TENTHS),
    3: TOUGHNESS => ("toughness", ValueLayout::NUMBER_TENTHS),
    4: AMOUNT => ("amount", ValueLayout::NUMBER_TENTHS),
    5: HAS_POWER => ("has_power", ValueLayout::FLAG),
    6: HAS_TOUGHNESS => ("has_toughness", ValueLayout::FLAG),
    7: HAS_AMOUNT => ("has_amount", ValueLayout::FLAG),
    8: DURATION => ("duration", ValueLayout::new("category", "1 end of turn, 2 indefinite, 3 while source controlled", "1..3; 0 padding")),
    9: TIMESTAMP => ("timestamp", ValueLayout::TIMESTAMP),
    10: CONTROLLER => ("controller", ValueLayout::OBJECT_REFERENCE),
    11: SOURCE_DEFINITION => ("source_definition", ValueLayout::DEFINITION),
    12: SOURCE => ("source", ValueLayout::OBJECT_REFERENCE),
    13: DETAIL_OFFSET => ("detail_offset", ValueLayout::new("reference", "Zero-based start in effect_details; count determines presence", "0..4096")),
    14: DETAIL_COUNT => ("detail_count", ValueLayout::COUNT),
    15: MEMBER_OFFSET => ("member_offset", ValueLayout::new("reference", "Zero-based start in effect_members; count determines presence", "0..2048")),
    16: MEMBER_COUNT => ("member_count", ValueLayout::COUNT),
    17: VALID => ("valid", ValueLayout::FLAG),
} }

columns! { detail {
    0: KIND => ("kind", ValueLayout::new("category", "1 keyword, 2 subtype, 3 color; v8+: 10 + learning TokenKind (1..9)", "0 padding; 1..3 or 11..19")),
    1: VALUE_HIGH => ("value_high", ValueLayout::TOKEN_HALF),
    2: VALUE_LOW => ("value_low", ValueLayout::TOKEN_HALF),
} }

columns! { member {
    0: OBJECT => ("object", ValueLayout::OBJECT_REFERENCE),
    1: ORDER => ("order", ValueLayout::new("rank", "Original EffectRecords position + 1; duplicate edges retained", "1..2048; 0 padding")),
} }

columns! { relation {
    0: KIND => ("kind", ValueLayout::new("category", "Relation row kind; see row_kinds", "1..10; 0 padding")),
    1: FRAME => ("frame", ValueLayout::new("reference", "Continuation frame index + 1; 0 absent", "nonnegative integers; 0 absent")),
    2: PARENT => ("parent", ValueLayout::new("reference", "Continuation frame index + 1; 0 absent", "nonnegative integers; 0 absent")),
    3: FRAME_KIND => ("frame_kind", ValueLayout::new("category", "1 spell, 2 activated, 3 triggered, 4 announcement, 5 resolution, 6 obligation, 7 waiting trigger, 8 placing trigger, 9 placement cursor; v5+: 10 delayed trigger, 11 scheduled turn, 12 combat, 13 control, 14 delegation", "1..14; 0 padding")),
    4: CONTROLLER => ("controller", ValueLayout::new("category", "1 agent, 2 opponent", "0..2")),
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
    25: REFERENCE_PLAYER => ("reference_player", ValueLayout::new("category", "0 absent, 1 agent, 2 opponent", "0..2")),
    26: HYBRID_FIRST => ("hybrid_first", ValueLayout::new("category", "Color index + 1; 0 absent", "0..6")),
    27: HYBRID_SECOND => ("hybrid_second", ValueLayout::new("category", "Color index + 1; 0 absent", "0..6")),
    28: DECISION => ("decision", ValueLayout::new("category", "Frame decision code; v1-v7 retain legacy selection codes 5/6/7, v8+ uses 28. Kind-9 rows: delayed frequency 1 once / 2 each match", "nonnegative integer codes")),
    29: ORDINAL => ("ordinal", ValueLayout::new("reference", "Reference ordinal + 1; kind 10 is effect-execution reference slot", "nonnegative integers; 0 absent")),
    30: RESOLUTIONS_THIS_TURN => ("resolutions_this_turn", ValueLayout::COUNT),
    31: VALID => ("valid", ValueLayout::FLAG),
    32: LEGAL_AT_RESOLUTION_START => ("legal_at_resolution_start", ValueLayout::new("category", "0 unknown, 1 legal, 2 illegal", "0..2")),
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
} }

/// The installed engine's complete vocabulary, resolved once for a new consumer.
/// This does not change implicit defaults or saved observation configurations.
pub const RECOMMENDED_POLICY_INPUT_VERSION: u8 = 9;

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
        }
    }
}

/// Versioned layout without allocating game buffers. Integers outside the supported
/// versions are rejected rather than interpreted as a future layout.
pub fn observation_layout(
    version: u8,
    history: u8,
) -> Result<BTreeMap<String, ArrayLayout>, &'static str> {
    if !(1..=RECOMMENDED_POLICY_INPUT_VERSION).contains(&version) {
        return Err("unsupported policy input version; expected 1..9");
    }
    if history > 1 {
        return Err("unsupported policy history version; expected 0 or 1");
    }
    let mut layout = BTreeMap::new();
    let player_columns = if version >= 5 {
        player::COLUMNS
    } else {
        &player::COLUMNS[..player::RESTRICTED_LESSON_WHITE]
    };
    let legacy = |columns: &[Column], end: usize| {
        let mut result = columns[..end].to_vec();
        result.push(Column {
            index: end,
            ..*columns.last().expect("validity column")
        });
        result
    };
    let card_columns = if version >= 5 {
        card::COLUMNS.to_vec()
    } else {
        legacy(card::COLUMNS, card::PERMISSION)
    };
    let permanent_columns = if version >= 5 {
        permanent::COLUMNS.to_vec()
    } else {
        legacy(permanent::COLUMNS, permanent::EXHAUSTED_ABILITIES)
    };
    for side in ["agent", "opponent"] {
        for (suffix, axis, capacity, columns) in [
            ("player", "player", Some(1), player_columns),
            ("cards", "card", None, card_columns.as_slice()),
            (
                "permanents",
                "permanent",
                None,
                permanent_columns.as_slice(),
            ),
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
    layout.insert(
        "actions".into(),
        ArrayLayout::features(&["action"], &[None], action::COLUMNS),
    );
    layout.insert(
        "events".into(),
        ArrayLayout::features(
            &["event"],
            &[None],
            if history == 1 {
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
    layout.insert(
        "known_hand".into(),
        ArrayLayout::features(
            &["viewer_relative_seat", "definition"],
            &[Some(2), None],
            known_hand::COLUMNS,
        ),
    );
    layout.insert("action_focus".into(), ArrayLayout::elements(&["action", "focus"], &[None, None], "int32", ValueLayout {
        kind: "reference", encoding: "Zero-based padded object-table index; -1 absent. Order: agent player, opponent player, agent cards, opponent cards, agent permanents, opponent permanents", range: "integers >= -1",
    }));
    if version >= 2 {
        let width = match version {
            2 | 3 => relation::LEGAL_AT_RESOLUTION_START,
            4 => relation::X,
            _ => relation::WIDTH,
        };
        let mut relations = ArrayLayout::features(
            &["relation"],
            &[Some(super::continuation::RELATION_ROWS)],
            &relation::COLUMNS[..width],
        );
        relations.row_kinds = BTreeMap::from([
            (1, "Frame header; frame_kind describes the unfinished work; other rows repeat this header"),
            (2, "Target: clause, ordinal, reference_definition/reference/reference_player; v4+ retains legal_at_resolution_start"),
            (3, "Event context: reference_definition/reference/reference_player"),
            (4, "Sacrifice: reference_definition/reference/reference_player"),
            (5, "Hybrid payment: hybrid_first/hybrid_second are alternative colors"),
            (6, "Mana output: mode, mana_white..mana_colorless and restricted flag; generic retains header cost"),
        ]);
        if version >= 5 {
            relations.row_kinds.extend([
                (7, "Inherited ability lender: reference_definition/reference/reference_player"),
                (8, "Remembered subject: reference_definition/reference/reference_player and ordinal"),
                (9, "Delayed frequency: decision is 1 once or 2 each match; this_turn is expiry"),
            ]);
        }
        if version >= 8 {
            relations.row_kinds.insert(10, "Effect execution reference: ordinal is the one-based slot in the continuation program; reference_definition/reference/reference_player identify its visible subject");
        }
        layout.insert("relations".into(), relations);
    }
    if version >= 5 {
        layout.insert(
            "decklists".into(),
            ArrayLayout::features(
                &["viewer_relative_seat", "definition"],
                &[Some(2), None],
                decklist::COLUMNS,
            ),
        );
        let mut facts = ArrayLayout::features(
            &["fact"],
            &[if version == 5 {
                Some(super::observation_encoder::V5_FACT_ROWS)
            } else {
                None
            }],
            fact::COLUMNS,
        );
        use super::observation_encoder::FactKind;
        for kind in FactKind::ALL {
            if version < 8 && kind as u8 >= FactKind::EffectProgram as u8
                || version >= 7 && kind == FactKind::Effect
            {
                continue;
            }
            facts.row_kinds.insert(kind as u8, match kind {
                FactKind::Counter => "id: CounterKind token; value: count / 10",
                FactKind::AddedSubtype => "id: added subtype token; value: 1",
                FactKind::SetName => "id: set-name token; value: 1",
                FactKind::SetCreatureType => "id: creature-type token; value: 1",
                FactKind::UsedAbility => "id: zero-based activated ability index; value: 1",
                FactKind::Effect => "id: Modification token; detail: keyword/subtype/color token or 0; value: power or amount / 10; value_2: toughness / 10; duration, timestamp and source_definition identify record provenance",
                FactKind::Copy => "id: CopyEffect token; source_definition: copied definition; timestamp: copy rank; duration: 2",
                FactKind::CopyException => "Copy exception: same columns as a Modification effect, retained even when active effects move to shared arrays",
                FactKind::Transforms => "id: 0 for permanent, frame index + 1 for captured source; value: unscaled transform count",
                FactKind::CombatRestriction => "id: frame index + 1; detail: attacker-predicate token",
                FactKind::EffectProgram => "Copy-exception program: id: Modification token; detail: signed i32 learning token value; value: TokenKind; value_2: zero-based ordinal; duration, timestamp, source_definition retain provenance",
                FactKind::EffectiveColor => "id: Color token (Colorless for empty colors); value: 1",
                FactKind::KeywordCounterTimestamp => "id: CounterKind token; timestamp: counter timestamp rank",
                FactKind::ContinuationProgram => "object: controller; id: frame index + 1; detail: signed i32 learning token value; value: TokenKind; value_2: zero-based ordinal. Source program then effect execution are separate occurrences, each resetting ordinal to 0",
            });
        }
        facts.columns[fact::KIND].value.range = if version < 8 {
            "1..10; 0 padding"
        } else {
            "1..14 excluding 6; 0 padding"
        };
        layout.insert("facts".into(), facts);
    }
    if version >= 7 {
        use super::shared_effects::{detail_rows, EFFECT_ROWS, MEMBER_ROWS};
        layout.insert(
            "effects".into(),
            ArrayLayout::features(&["effect"], &[Some(EFFECT_ROWS)], effect::COLUMNS),
        );
        let mut details =
            ArrayLayout::features(&["detail"], &[Some(detail_rows(version))], detail::COLUMNS);
        details.row_kinds = BTreeMap::from([
            (1, "Keyword token in value_high/value_low"),
            (2, "Subtype token in value_high/value_low"),
            (3, "Color token in value_high/value_low"),
        ]);
        if version >= 8 {
            for kind in 11..=19 {
                details.row_kinds.insert(kind, "Learning token: TokenKind = kind - 10; exact signed i32 value in value_high/value_low; row order is program order");
            }
        } else {
            details.columns[detail::KIND].value.range = "0 padding; 1..3";
        }
        layout.insert("effect_details".into(), details);
        layout.insert(
            "effect_members".into(),
            ArrayLayout::features(&["membership"], &[Some(MEMBER_ROWS)], member::COLUMNS),
        );
    }
    if version >= 9 {
        layout.insert(
            "name_candidates".into(),
            ArrayLayout::features(&["action"], &[None], name::COLUMNS),
        );
    }
    Ok(layout)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::observation_encoder::{ObservationEncodeError, ObservationEncoderConfig};

    #[test]
    fn named_layout_preserves_every_frozen_width_and_column() {
        for version in 1..=9 {
            for history in 0..=1 {
                let config = ObservationEncoderConfig {
                    policy_input_version: version,
                    policy_history: history == 1,
                    ..Default::default()
                };
                let layout = observation_layout(version, history).unwrap();
                for array in layout.values() {
                    assert_eq!(array.axes.len(), array.shape.len());
                    let names: std::collections::BTreeSet<_> =
                        array.columns.iter().map(|c| c.name).collect();
                    assert_eq!(names.len(), array.columns.len());
                    for (index, column) in array.columns.iter().enumerate() {
                        assert_eq!(column.index, index, "{}", column.name);
                        assert!(
                            !column.value.kind.is_empty()
                                && !column.value.encoding.is_empty()
                                && !column.value.range.is_empty()
                        );
                    }
                }
                for (key, width, frozen) in [
                    (
                        "agent_player",
                        config.player_dim(),
                        if version < 5 { 40 } else { 73 },
                    ),
                    (
                        "agent_cards",
                        config.card_dim(),
                        if version < 5 { 41 } else { 56 },
                    ),
                    (
                        "agent_permanents",
                        config.permanent_dim(),
                        if version < 5 { 29 } else { 38 },
                    ),
                    ("actions", super::action::WIDTH, 22),
                    (
                        "events",
                        config.event_dim(),
                        if history == 0 { 7 } else { 12 },
                    ),
                ] {
                    assert_eq!(layout[key].columns.len(), width);
                    assert_eq!(width, frozen);
                }
                assert_eq!(layout.contains_key("relations"), version >= 2);
                assert_eq!(layout.contains_key("facts"), version >= 5);
                assert_eq!(layout.contains_key("decklists"), version >= 5);
                assert_eq!(layout.contains_key("effects"), version >= 7);
                assert_eq!(layout.contains_key("name_candidates"), version >= 9);
                if version >= 2 {
                    assert_eq!(layout["relations"].columns.len(), config.relation_dim());
                    assert_eq!(layout["relations"].columns[31].name, "valid");
                }
                assert_eq!(layout["agent_permanents"].columns[24].name, "definition");
                assert_eq!(layout["agent_cards"].columns.last().unwrap().name, "valid");
                assert_eq!(
                    layout["agent_permanents"].columns.last().unwrap().name,
                    "valid"
                );
            }
        }
    }

    #[test]
    fn recommended_input_keeps_defaults_and_names_recovery() {
        assert_eq!(RECOMMENDED_POLICY_INPUT_VERSION, 9);
        assert_eq!(ObservationEncoderConfig::default().policy_input_version, 6);
        for error in [
            ObservationEncodeError::UnsupportedLayerInput(6),
            ObservationEncodeError::UnsupportedEffectExecutionInput(7),
        ] {
            let message = error.to_string();
            assert!(message.contains("select v8 or later"));
            assert!(message.contains("recommended for new consumers: v9"));
        }
        assert!(observation_layout(0, 0).is_err());
        assert!(observation_layout(10, 0).is_err());
        assert!(observation_layout(9, 2).is_err());
    }
}
