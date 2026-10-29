//! Every observed fact reaches the policy input (v5). The guard names, for
//! each `Observation` field, keyword and counter kind, the column or rows that
//! encode it, or why it is not a decision input. A new field fails to compile
//! until it is listed; every listed column is checked against the v5 widths,
//! and every v5 row column must be claimed. Directed tests put a game into
//! each state and compare the named column with an otherwise equal baseline
//! that v4 cannot tell apart. Viewer-safety tests check that hidden cards never
//! reach the encoding.
use crate::{
    helpers::Scenario,
    support::{cast, mode, setup, setup_unfunded},
};
use managym::{
    agent::{
        action::ActionSpaceKind,
        continuation::{
            ClauseTarget, ContinuationFrame, ManaOutput, VisibleReference, RELATION_DIM_V5,
            RELATION_KEYWORDS_INDEX, RELATION_X_INDEX,
        },
        observation::{
            ActionOption, ActionSpaceData, CardData, CardTypeData, EventData, KeywordData,
            Observation, PermanentData, PlayerData, SideboardCardData, StackObjectData,
            StackTargetData, TurnData, KEYWORD_COUNT,
        },
        observation_encoder::{
            encode, EncodedObservation, ObservationEncoderConfig, CARD_DIM_V5,
            CARD_FACE_DOWN_INDEX, PERMANENT_ABILITIES_LOST_INDEX, PERMANENT_ATTACHED_INDEX,
            PERMANENT_ATTACKING_INDEX, PERMANENT_BLOCKING_INDEX,
            PERMANENT_CHARACTERISTICS_SET_INDEX, PERMANENT_DIM_V5,
            PERMANENT_DOUBLE_STRIKE_COUNTERS_INDEX, PERMANENT_FIREBENDING_INDEX,
            PERMANENT_FIRE_INDEX, PERMANENT_GRANTED_LAND_INDEX,
            PERMANENT_OTHER_TEMP_SUBTYPES_INDEX, PERMANENT_OTHER_USED_ABILITIES_INDEX,
            PERMANENT_STUN_INDEX, PERMANENT_TEMP_SUBTYPES_INDEX, PERMANENT_USED_ABILITIES_INDEX,
            PLAYER_DIM_V5, PLAYER_KEEPS_RED_MANA_INDEX, PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
            PLAYER_SEAT_INDEX, PLAYER_TURN_INDEX, TEMP_SUBTYPES, USED_ABILITY_FLAGS,
        },
    },
    flow::turn::StepKind,
    state::{
        game_object::{PermanentId, PlayerId},
        permanent::CounterKind,
        zone::ZoneType,
    },
};
use std::collections::BTreeSet;

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
}

impl Buffer {
    /// V5 row width.
    fn width(self) -> usize {
        match self {
            Buffer::Player => PLAYER_DIM_V5,
            Buffer::Card => CARD_DIM_V5,
            Buffer::Permanent => PERMANENT_DIM_V5,
            Buffer::Relation => RELATION_DIM_V5,
            Buffer::Action => managym::agent::observation_encoder::ACTION_DIM,
        }
    }

    /// The row's validity column; player validity is a buffer of its own.
    fn validity(self) -> Option<usize> {
        match self {
            Buffer::Player => None,
            Buffer::Relation => Some(31),
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
    /// Not a decision input, for the stated reason.
    Excluded(&'static str),
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
        known_hand => [Rows("known_hand")],
        sideboard_counts => [Excluded(HIDDEN_SIDEBOARD)],
        remaining_sideboard_counts => [Excluded(HIDDEN_SIDEBOARD)],
        keeps_red_mana => [column(Player, PLAYER_KEEPS_RED_MANA_INDEX)],
        no_maximum_hand_size => [column(Player, PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX)],
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
        face_down => [column(Card, CARD_FACE_DOWN_INDEX)],
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
        plus1_counters => [column(Permanent, 4)],
        cant_be_blocked_this_turn => [column(Permanent, 5)],
        power => [column(Permanent, 6)],
        toughness => [column(Permanent, 7)],
        is_animated => [column(Permanent, 8)],
        has_exile_link => [column(Permanent, 9)],
        exile_if_dies_this_turn => [column(Permanent, 23)],
        temp_subtypes => [
            column(Permanent, 26),
            Columns(Permanent, PERMANENT_TEMP_SUBTYPES_INDEX, TEMP_SUBTYPES.len()),
            column(Permanent, PERMANENT_OTHER_TEMP_SUBTYPES_INDEX),
        ],
        keywords => [Rows("keywords below")],
        stun_counters => [column(Permanent, PERMANENT_STUN_INDEX)],
        fire_counters => [column(Permanent, PERMANENT_FIRE_INDEX)],
        double_strike_counters => [column(Permanent, PERMANENT_DOUBLE_STRIKE_COUNTERS_INDEX)],
        attached_to => [Join(Permanent, PERMANENT_ATTACHED_INDEX)],
        name => [column(Permanent, PERMANENT_CHARACTERISTICS_SET_INDEX)],
        creature_types => [column(Permanent, PERMANENT_CHARACTERISTICS_SET_INDEX)],
        is_granted_land => [column(Permanent, PERMANENT_GRANTED_LAND_INDEX)],
        firebending => [column(Permanent, PERMANENT_FIREBENDING_INDEX)],
        attacking => [column(Permanent, PERMANENT_ATTACKING_INDEX)],
        blocking => [Join(Permanent, PERMANENT_BLOCKING_INDEX)],
        abilities_lost => [column(Permanent, PERMANENT_ABILITIES_LOST_INDEX)],
        used_abilities => [
            Columns(Permanent, PERMANENT_USED_ABILITIES_INDEX, USED_ABILITY_FLAGS),
            column(Permanent, PERMANENT_OTHER_USED_ABILITIES_INDEX),
        ],
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

/// `(keyword, card column, permanent column)`; relation rows use
/// `RELATION_KEYWORDS_INDEX + position`. Exhaustive like [`guard`].
fn keyword_columns() -> Vec<(&'static str, usize, usize)> {
    #[allow(dead_code)]
    fn exhaustive(value: &KeywordData) {
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
        } = value;
    }
    vec![
        ("flying", 17, 10),
        ("reach", 18, 11),
        ("haste", 19, 12),
        ("flash", 28, 13),
        ("vigilance", 20, 14),
        ("trample", 21, 15),
        ("first_strike", 22, 16),
        ("double_strike", 23, 17),
        ("deathtouch", 24, 18),
        ("lifelink", 25, 19),
        ("defender", 26, 20),
        ("menace", 27, 21),
        ("hexproof", 36, 22),
        ("indestructible", 38, 25),
    ]
}

/// The permanent column that counts each counter kind.
fn counter_column(kind: CounterKind) -> usize {
    match kind {
        CounterKind::PlusOnePlusOne => 4,
        CounterKind::Stun => PERMANENT_STUN_INDEX,
        CounterKind::Fire => PERMANENT_FIRE_INDEX,
        CounterKind::DoubleStrike => PERMANENT_DOUBLE_STRIKE_COUNTERS_INDEX,
    }
}

/// Structural relation columns: row kind, frame index and reference ordinal.
const RELATION_STRUCTURE: [usize; 3] = [0, 1, 29];

#[test]
fn every_observed_field_names_its_v5_encoding_and_every_v5_column_is_claimed() {
    let mut claimed: BTreeSet<(Buffer, usize)> = BTreeSet::new();
    for (field, encodings) in guard() {
        for encoded in encodings {
            let (buffer, start, len) = match encoded {
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
    for (keyword, card, permanent) in keyword_columns() {
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
    for kind in CounterKind::ALL {
        claimed.insert((Buffer::Permanent, counter_column(kind)));
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
fn permanent_row(s: &Scenario, viewer: usize, id: PermanentId) -> (usize, usize) {
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
    let (side, row) = permanent_row(s, viewer, id);
    let encoded = encoded(s, viewer, 5);
    let rows = if side == 0 {
        &encoded.agent_permanents
    } else {
        &encoded.opponent_permanents
    };
    rows[row * PERMANENT_DIM_V5 + column]
}

/// Object index of a permanent row, as action focus and joins use it.
fn object_index(s: &Scenario, viewer: usize, id: PermanentId) -> usize {
    let defaults = ObservationEncoderConfig::default();
    let (side, row) = permanent_row(s, viewer, id);
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

/// Gray Ogres for both players, tapped, in player 0's main phase.
fn ogre_state() -> (Scenario, PermanentId) {
    let mut s = setup_unfunded(&["Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[ogre].as_mut().unwrap().tapped = true;
    (s, ogre)
}

#[test]
fn each_counter_kind_reaches_its_permanent_column() {
    for kind in CounterKind::ALL {
        let (mut baseline, ogre) = ogre_state();
        let (mut changed, _) = ogre_state();
        if kind == CounterKind::DoubleStrike {
            // CR 122.1b: the counter grants double strike, which v4 sees; an
            // until-end-of-turn grant is the state v4 cannot tell apart.
            baseline.game_mut().state.permanents[ogre]
                .as_mut()
                .unwrap()
                .temp_keywords
                .double_strike = true;
        }
        changed.game_mut().state.permanents[ogre]
            .as_mut()
            .unwrap()
            .change_counters(kind, 1);
        let column = counter_column(kind);
        assert_eq!(permanent_value(&changed, 0, ogre, column), 0.1, "{kind:?}");
        if kind == CounterKind::PlusOnePlusOne {
            continue; // v4 sees +1/+1 counters and the power they add.
        }
        assert_v5_alone_sees(&baseline, &changed, 0, ogre, column);
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
    assert_v5_alone_sees(&baseline, &stunned, 0, ogre, PERMANENT_STUN_INDEX);
}

fn equipped(on_first: bool) -> (Scenario, PermanentId, [PermanentId; 2]) {
    let mut s = setup_unfunded(&["Trusty Boomerang", "Gray Ogre"]);
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
    let mut s = setup(&["United Front", "Gray Ogre"]);
    s.set_player_mana_pool(0, &format!("{x}WW"));
    cast(&mut s, "United Front");
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    mode(&mut s, x);
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
    let mut s = setup_unfunded(&["Toph, the First Metalbender", "White Lotus Tile"]);
    let toph = s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    s.game_mut().state.permanents[toph]
        .as_mut()
        .unwrap()
        .abilities_lost = !toph_has_abilities;
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
fn granted_firebending_reaches_its_permanent_column() {
    let (baseline, ogre) = ogre_state();
    let (mut changed, _) = ogre_state();
    changed.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .firebending_this_turn = 4;
    assert_eq!(
        permanent_value(&changed, 0, ogre, PERMANENT_FIREBENDING_INDEX),
        0.4
    );
    assert_v5_alone_sees(&baseline, &changed, 0, ogre, PERMANENT_FIREBENDING_INDEX);
}

#[test]
fn each_temporary_subtype_reaches_its_permanent_column() {
    for (offset, subtype) in TEMP_SUBTYPES
        .iter()
        .copied()
        .enumerate()
        .map(|(o, s)| (PERMANENT_TEMP_SUBTYPES_INDEX + o, s))
        .chain([(PERMANENT_OTHER_TEMP_SUBTYPES_INDEX, "Citizen")])
    {
        let (baseline, ogre) = ogre_state();
        let (mut changed, _) = ogre_state();
        changed.game_mut().state.permanents[ogre]
            .as_mut()
            .unwrap()
            .temp_subtypes
            .push(subtype.into());
        assert_eq!(permanent_value(&changed, 0, ogre, offset), 1.0, "{subtype}");
        assert_v5_alone_sees(&baseline, &changed, 0, ogre, offset);
    }
}

#[test]
fn honest_work_marks_set_characteristics_and_joins_the_aura_to_its_host() {
    let mut s = setup(&["Honest Work", "Wind Drake"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    assert_eq!(
        permanent_value(&s, 0, drake, PERMANENT_CHARACTERISTICS_SET_INDEX),
        0.0
    );
    s.set_player_mana_pool(0, "U");
    cast(&mut s, "Honest Work");
    crate::support::pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(managym::state::game_object::Target::Permanent(drake)));
    crate::support::settle(&mut s);
    let work = s.battlefield_permanents_named(0, "Honest Work")[0];
    assert_eq!(
        permanent_value(&s, 0, drake, PERMANENT_CHARACTERISTICS_SET_INDEX),
        1.0
    );
    assert_eq!(
        permanent_value(&s, 0, work, PERMANENT_ATTACHED_INDEX),
        (object_index(&s, 0, drake) + 1) as f32
    );
}

#[test]
fn every_keyword_reaches_its_card_permanent_and_relation_columns() {
    let mut s = setup_unfunded(&["Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let observation = Observation::for_player(s.game(), PlayerId(0));
    for (position, (keyword, card, permanent)) in keyword_columns().into_iter().enumerate() {
        let mut flags = [false; KEYWORD_COUNT];
        flags[position] = true;
        let keywords = keyword_data(flags);
        let mut changed = observation.clone();
        changed.agent_cards[0].keywords = keywords.clone();
        changed.agent_permanents[0].keywords = keywords;
        let mut frame = ContinuationFrame {
            kind: 1,
            granted_keywords: flags,
            ..ContinuationFrame::default()
        };
        frame.controller = 0;
        changed.continuation = vec![frame];
        let encoded = encode(&changed, &config(5)).unwrap();
        let card_row = &encoded.agent_cards[..CARD_DIM_V5];
        let permanent_row = &encoded.agent_permanents[..PERMANENT_DIM_V5];
        for (_, other_card, other_permanent) in keyword_columns() {
            assert_eq!(card_row[other_card] == 1.0, other_card == card, "{keyword}");
            assert_eq!(
                permanent_row[other_permanent] == 1.0,
                other_permanent == permanent,
                "{keyword}"
            );
        }
        let relation = &encoded.relations[RELATION_KEYWORDS_INDEX..][..KEYWORD_COUNT];
        assert_eq!(relation.iter().position(|v| *v == 1.0), Some(position));
        assert_eq!(relation.iter().sum::<f32>(), 1.0, "{keyword}");
    }
}

fn keyword_data(flags: [bool; KEYWORD_COUNT]) -> KeywordData {
    let [flying, reach, haste, flash, vigilance, trample, first_strike, double_strike, deathtouch, lifelink, defender, menace, hexproof, indestructible] =
        flags;
    let keywords = KeywordData {
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
    };
    assert_eq!(keywords.flags(), flags, "flags() follows field order");
    keywords
}

#[test]
fn lo_and_li_grant_lifelink_to_a_lesson_on_the_stack_in_v5() {
    let lifelink = RELATION_KEYWORDS_INDEX + 9;
    let lesson = |with_lo_and_li: bool| {
        let mut s = setup_unfunded(&["Lo and Li, Twin Tutors", "Firebending Lesson", "Gray Ogre"]);
        if with_lo_and_li {
            s.force_permanent_on_battlefield(0, "Lo and Li, Twin Tutors");
        }
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.set_player_mana_pool(0, "R");
        cast(&mut s, "Firebending Lesson");
        crate::support::pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(managym::state::game_object::Target::Permanent(ogre)));
        encoded(&s, 1, 5).relations
    };
    let (granted, plain) = (lesson(true), lesson(false));
    assert_eq!(granted[3], 1.0, "the top frame is the spell");
    assert_eq!((granted[lifelink], plain[lifelink]), (1.0, 0.0));
}

#[test]
fn turn_number_and_seat_reach_the_player_rows() {
    let mut s = setup_unfunded(&[]);
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
        rows.as_chunks::<PERMANENT_DIM_V5>()
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
    let mut s = setup(&["Sozin's Comet", "United Front"]);
    let comet = crate::support::library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    let front = crate::support::to_hand(&mut s, "United Front");
    s.set_player_mana_pool(0, "3");
    s.game_mut().scenario_refresh_priority().unwrap();
    crate::support::choose(
        &mut s,
        |a| matches!(a, managym::agent::action::Action::Foretell { card, .. } if *card == comet),
    );
    assert_eq!(s.game().state.zones.zone_of(comet), Some(ZoneType::Exile));
    let hidden = [comet, front].map(|card| s.game().state.cards[card].definition_id.0);

    let owner = encoded(&s, 0, 5);
    let face_down = owner
        .agent_cards
        .as_chunks::<CARD_DIM_V5>()
        .0
        .iter()
        .filter(|row| row[CARD_FACE_DOWN_INDEX] == 1.0)
        .count();
    assert_eq!(face_down, 1, "its owner sees the foretold card, face down");
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
        .as_chunks::<CARD_DIM_V5>()
        .0
        .iter()
        .all(|row| row[CARD_FACE_DOWN_INDEX] == 0.0));
}

// ---------------------------------------------------------------------------
// Combat, lost abilities, used activations and player state
// ---------------------------------------------------------------------------

/// Two ogres for player 1 and one for player 0, in player 0's main phase.
fn combat_state() -> (Scenario, PermanentId, [PermanentId; 2]) {
    let mut s = setup_unfunded(&["Gray Ogre"]);
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

#[test]
fn attacking_creatures_differ_from_idle_ones_only_in_v5() {
    let (idle, _, [first, _]) = combat_state();
    let (mut attacking, ..) = combat_state();
    declare(&mut attacking, &[first], None);
    assert_v5_alone_sees(&idle, &attacking, 0, first, PERMANENT_ATTACKING_INDEX);
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
    changed.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .abilities_lost = true;
    assert_v5_alone_sees(&baseline, &changed, 0, ogre, PERMANENT_ABILITIES_LOST_INDEX);
}

fn mai(exhausted: bool) -> (Scenario, PermanentId) {
    let mut s = setup_unfunded(&["Mai, Jaded Edge"]);
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
    assert_eq!(
        permanent_value(&spent, 0, mai_id, PERMANENT_USED_ABILITIES_INDEX),
        1.0,
        "Mai's only activated ability is exhaust"
    );
    assert_eq!(
        permanent_value(&spent, 0, mai_id, PERMANENT_OTHER_USED_ABILITIES_INDEX),
        0.0
    );
    assert_v5_alone_sees(&unused, &spent, 0, mai_id, PERMANENT_USED_ABILITIES_INDEX);
}

#[test]
fn kept_red_mana_and_no_hand_size_reach_the_player_rows_only_in_v5() {
    for column in [
        PLAYER_KEEPS_RED_MANA_INDEX,
        PLAYER_NO_MAXIMUM_HAND_SIZE_INDEX,
    ] {
        let (baseline, _) = ogre_state();
        let (mut changed, _) = ogre_state();
        let player = &mut changed.game_mut().state.players[1];
        if column == PLAYER_KEEPS_RED_MANA_INDEX {
            player.keeps_red_mana = true;
        } else {
            player.no_maximum_hand_size = true;
        }
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
