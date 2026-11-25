//! CR 712.12 — a modal double-faced card is played or cast as the face its
//! player chooses. One `(object, profile, route)` names each play: the zone
//! and permissions open routes, and the selected face decides which route
//! permits it. The rules fixtures `Tidal Errand // Tidal Haven` (sorcery //
//! land) and `Kelp Hollow // Kelp Warden` (land // creature) exercise both
//! directions.

use managym::{
    agent::action::Action,
    flow::cast_permission::PlayRoute,
    state::{
        card::SpellProfile,
        game_object::{ObjectId, PlayerId},
        zone::ZoneType,
    },
};

use crate::support::{
    game::{five_color_unfunded_setup, move_card, move_from_library, Scenario},
    inspect::{current_object, now, offer_labels, permanent_of},
    play::{choose, fork_choosing, settle},
};

/// The `(verb, profile, route)` of every offer that plays the card.
fn plays(s: &Scenario, object: ObjectId) -> Vec<(&'static str, SpellProfile, Option<PlayRoute>)> {
    let object = now(s, object);
    s.action_space()
        .actions
        .iter()
        .filter_map(|action| match *action {
            Action::PlayLand {
                object: played,
                profile,
                route,
                ..
            } if played == object => Some(("play", profile, route)),
            Action::CastSpell {
                object: cast,
                profile,
                route,
                ..
            } if cast == object => Some(("cast", profile, route)),
            _ => None,
        })
        .collect()
}

fn refresh(s: &mut Scenario) {
    s.game_mut().scenario_refresh_priority().unwrap();
}

#[test]
fn spell_land_back_face_is_played_back_face_up_once() {
    let mut s = five_color_unfunded_setup(&["Tidal Errand", "Omniscience"]);
    let errand = move_from_library(&mut s, 0, "Tidal Errand", ZoneType::Hand);
    let island = move_from_library(&mut s, 0, "Island", ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    refresh(&mut s);
    assert_eq!(
        plays(&s, errand),
        [
            ("cast", SpellProfile::Front, None),
            ("play", SpellProfile::BackFace, None)
        ]
    );
    // A plain land and a lone hand route keep the original wire shape.
    let wire = |object: ObjectId| {
        s.action_space()
            .actions
            .iter()
            .find(|action| matches!(action, Action::PlayLand { object: o, .. } if *o == object))
            .map(|action| serde_json::to_value(action).unwrap())
            .unwrap()
    };
    let plain = wire(island);
    assert!(
        plain["PlayLand"].get("adventure").is_none() && plain["PlayLand"].get("route").is_none()
    );
    assert_eq!(wire(errand)["PlayLand"]["adventure"], "BackFace");
    assert!(offer_labels(&s)
        .iter()
        .any(|label| label == "Play Tidal Haven"));

    let mut played = fork_choosing(
        &s,
        |action| matches!(action, Action::PlayLand { object, profile: SpellProfile::BackFace, .. } if *object == errand),
    );
    settle(&mut played);
    let land = current_object(&played, errand);
    // Playing the back face does not count as turning the permanent over.
    let permanent = permanent_of(&played, errand).unwrap();
    assert_eq!(
        played.game().state.permanents[permanent]
            .as_ref()
            .unwrap()
            .transforms,
        0
    );
    assert_eq!(land.name, "Tidal Haven");
    assert!(land.types.is_land());
    assert!(permanent_of(&played, errand).is_some());
    assert_eq!(played.game().state.turn.lands_played, 1);

    // Omniscience adds a free route for the sorcery only. Routes are named
    // per face: the land face keeps its lone, unnamed hand route, like Island.
    s.force_permanent_on_battlefield(0, "Omniscience");
    refresh(&mut s);
    assert_eq!(
        plays(&s, errand),
        [
            ("cast", SpellProfile::Front, Some(PlayRoute::Hand)),
            ("cast", SpellProfile::Front, Some(PlayRoute::FreeFromHand)),
            ("play", SpellProfile::BackFace, None),
        ]
    );
    assert_eq!(plays(&s, island), [("play", SpellProfile::Front, None)]);
}

/// Muldrotha's land permission reaches the land back of a sorcery; an
/// opponent's "cast only from hand" refuses spells, never land plays.
#[test]
fn spell_land_back_face_uses_muldrotha_under_hand_only_casting() {
    let mut s = five_color_unfunded_setup(&[
        "Tidal Errand",
        "Muldrotha, the Gravetide",
        "Drannith Magistrate",
    ]);
    s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    s.force_permanent_on_battlefield(1, "Drannith Magistrate");
    let errand = move_from_library(&mut s, 0, "Tidal Errand", ZoneType::Graveyard);
    refresh(&mut s);
    // The lone route stays unnamed on the wire; its label names Muldrotha.
    assert_eq!(plays(&s, errand), [("play", SpellProfile::BackFace, None)]);
    let labels = offer_labels(&s);
    assert!(
        labels
            .iter()
            .any(|label| label.starts_with("Play Tidal Haven using Muldrotha")),
        "{labels:?}"
    );
    choose(
        &mut s,
        |action| matches!(action, Action::PlayLand { object, .. } if *object == errand),
    );
    settle(&mut s);
    assert_eq!(current_object(&s, errand).name, "Tidal Haven");
    assert_eq!(s.game().state[now(&s, errand)].owner, PlayerId(0));
}

/// A land front with a creature back is a land in every zone, so only the
/// selected creature face can use spell permissions.
#[test]
fn land_creature_back_face_reaches_spell_permissions() {
    let mut s = five_color_unfunded_setup(&[
        "Kelp Hollow",
        "Omniscience",
        "Muldrotha, the Gravetide",
        "Vizier of the Menagerie",
    ]);
    s.force_permanent_on_battlefield(0, "Omniscience");
    let hollow = move_from_library(&mut s, 0, "Kelp Hollow", ZoneType::Hand);
    refresh(&mut s);
    assert_eq!(
        plays(&s, hollow),
        [
            ("play", SpellProfile::Front, None),
            (
                "cast",
                SpellProfile::BackFace,
                Some(PlayRoute::FreeFromHand)
            ),
        ],
        "the free route casts only the creature; {{2}}{{U}} is unaffordable"
    );

    let mut cast = fork_choosing(
        &s,
        |action| matches!(action, Action::CastSpell { object, profile: SpellProfile::BackFace, .. } if *object == hollow),
    );
    settle(&mut cast);
    assert_eq!(current_object(&cast, hollow).name, "Kelp Warden");
    assert!(permanent_of(&cast, hollow).is_some());
    assert_eq!(cast.game().state.turn.lands_played, 0);

    s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    let hollow = move_card(&mut s, hollow, ZoneType::Graveyard);
    s.set_player_mana_pool(0, "2U");
    refresh(&mut s);
    // Each face has one Muldrotha permission (Land for the land face,
    // Creature for the creature face), so neither Command names it.
    assert_eq!(
        plays(&s, hollow),
        [
            ("play", SpellProfile::Front, None),
            ("cast", SpellProfile::BackFace, None),
        ]
    );
    let labels = offer_labels(&s);
    for label in ["Play Kelp Hollow using Muldrotha", "Cast Kelp Warden"] {
        assert!(labels.iter().any(|l| l.starts_with(label)), "{labels:?}");
    }

    s.force_permanent_on_battlefield(0, "Vizier of the Menagerie");
    let hollow = move_card(&mut s, hollow, ZoneType::Library);
    assert_eq!(
        s.game().state.zones.top(ZoneType::Library, PlayerId(0)),
        Some(hollow)
    );
    refresh(&mut s);
    assert_eq!(
        plays(&s, hollow),
        [("cast", SpellProfile::BackFace, None)],
        "the top library card is cast as the creature, never played as a land"
    );
}

/// CR 305.9: a land face is played only through a permission to play lands.
/// A cast-only grant (Airbend, Zul, Hama, Azula: "you may cast") opens the
/// sorcery front alone; a play grant (Redcap, Flameshape: "you may play")
/// also plays the land back.
#[test]
fn cast_only_grant_never_plays_a_land_face() {
    use managym::flow::cast_permission::PlayGrant;
    for play_lands in [false, true] {
        let mut s = five_color_unfunded_setup(&["Tidal Errand"]);
        let errand = move_from_library(&mut s, 0, "Tidal Errand", ZoneType::Exile);
        s.game_mut().state.cast_permissions.grant(PlayGrant {
            play_lands,
            ..PlayGrant::new(errand, PlayerId(0))
        });
        s.set_player_mana_pool(0, "1U");
        refresh(&mut s);
        let mut expected = vec![("cast", SpellProfile::Front, None)];
        if play_lands {
            expected.push(("play", SpellProfile::BackFace, None));
        }
        assert_eq!(plays(&s, errand), expected, "play_lands {play_lands}");
    }
}

/// CR 305.9: a land creature can be played only as a land, so Muldrotha's
/// creature permission never casts it, and it uses one land permission.
#[test]
fn a_land_creature_uses_only_muldrothas_land_permission() {
    let mut s = five_color_unfunded_setup(&["Thornwood Arbor", "Muldrotha, the Gravetide"]);
    s.force_permanent_on_battlefield(0, "Muldrotha, the Gravetide");
    let arbor = move_from_library(&mut s, 0, "Thornwood Arbor", ZoneType::Graveyard);
    refresh(&mut s);
    assert_eq!(plays(&s, arbor), [("play", SpellProfile::Front, None)]);
    let labels = offer_labels(&s);
    assert!(
        labels.iter().any(
            |label| label.starts_with("Play Thornwood Arbor using Muldrotha")
                && label.contains("Land")
        ),
        "{labels:?}"
    );
}
