//! Every viewer fact reaches the policy tensor.
//!
//! The guard functions destructure each observation row type without `..`,
//! so a new field fails to compile until it names the v5 tensor columns that
//! encode it, or says why it needs none. Directed tests then put real games
//! into each T2 state and check the named columns, and that hidden facts
//! (an opponent's foretold card) reach neither the tensor nor the JSON.
use std::ops::Range;

use crate::{
    helpers::Scenario,
    support::{cast, choose, library_card, mode, settle, setup},
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
            CARD_X_INDEX, PERMANENT_DIM, PLAYER_DIM, PLAYER_HAND_ONLY_INDEX,
            PLAYER_RESTRICTED_INDEX,
        },
    },
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PlayerId, Target},
        zone::ZoneType,
    },
};

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

fn keyword_columns(keywords: &KeywordData, card: bool) -> Vec<(&'static str, Encoding)> {
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
    columns.extend(keyword_columns(keywords, true));
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
            Columns(PLAYER_HAND_ONLY_INDEX..PLAYER_DIM),
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
        // Avatar is the only temporary subtype the current pool grants.
        ("temp_subtypes", Columns(26..27)),
    ];
    columns.extend(keyword_columns(keywords, false));
    columns
}

/// Observations of a game holding every row type, from both seats.
fn rich_observations() -> Vec<Observation> {
    let mut s = setup(&["Gray Ogre"]);
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

fn card_of(s: &Scenario, permanent: managym::state::game_object::PermanentId) -> CardId {
    s.game().state.permanents[permanent]
        .as_ref()
        .expect("on the battlefield")
        .card
}

fn object_id(s: &Scenario, card: CardId) -> i32 {
    s.game().state.cards[card].id.0 as i32
}

/// The encoded row of `card` on either side of `observation`, if listed.
fn card_row(observation: &Observation, card: i32) -> Option<Vec<f32>> {
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
    let mut s = setup(&["Airbending Lesson", "Gray Ogre"]);
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

    let row = card_row(&airbender, id).unwrap();
    assert_eq!(
        permission_columns(&row),
        &[1.0, 0.0, 1.0, 0.2, 0., 0., 0., 0., 0., 0., 0., 0.]
    );
    let owner = card_row(&Observation::for_player(s.game(), PlayerId(1)), id).unwrap();
    assert_eq!(owner[PERMISSION + 1], 1.0, "the owner may cast it");
}

/// Avatar's Wrath's lock is on the restricted player's row for both viewers,
/// and the permissions it overrides stay visible.
#[test]
fn hand_only_lock_reaches_both_viewers_tensors() {
    let mut s = setup(&["Avatar's Wrath", "Gray Ogre", "Craw Wurm"]);
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
    assert_eq!(card_row(&locked, wurm_id).unwrap()[PERMISSION], 1.0);
}

/// Restricted mana is split by what it may pay for: Jasmine Dragon Tea
/// Shop's green counts under Ally only.
#[test]
fn restricted_mana_is_encoded_by_spending_subtype() {
    let mut s = setup(&["Jasmine Dragon Tea Shop"]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    s.set_player_mana_pool(0, "");
    crate::support::activate(&mut s, shop, 0);
    mode(&mut s, 4);

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
    let mut s = setup(&["United Front"]);
    s.set_player_mana_pool(0, "3WW");
    let front = cast(&mut s, "United Front");
    mode(&mut s, 3);
    let id = object_id(&s, front);
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let card = observation.agent_cards.iter().find(|c| c.id == id).unwrap();
    assert_eq!(card.x, Some(3));
    let row = card_row(&observation, id).unwrap();
    assert_eq!(&row[CARD_X_INDEX..CARD_X_INDEX + 2], &[1.0, 0.3]);
}

// --- Viewer safety -------------------------------------------------------

/// A foretold card and its permission reach only its owner: the owner's
/// tensor marks it foretold and not yet castable; the opponent's tensor and
/// JSON hold no trace of it beyond the exile count.
#[test]
fn foretold_card_and_permission_reach_only_the_owner() {
    let mut s = setup(&["Sozin's Comet"]);
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
    let row = card_row(&owner, id).expect("the owner sees it");
    assert_eq!(row[PERMISSION], 1.0);
    assert_eq!(row[PERMISSION + 6], 1.0, "castable only on a later turn");
    assert_eq!(row[PERMISSION + 11], 1.0, "foretold");

    let opponent = Observation::for_player(s.game(), PlayerId(1));
    assert!(card_row(&opponent, id).is_none());
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
    let row = card_row(&Observation::for_player(s.game(), PlayerId(0)), id).unwrap();
    assert_eq!(row[PERMISSION + 6], 0.0);
}
