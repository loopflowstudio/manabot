//! Every entered source creates a separately castable copy of its own inset.
use crate::support::{
    game::{enter_battlefield, five_color_setup, plant_in_graveyard},
    play::{cast_card, castable, settle},
};
use managym::state::{game_object::PlayerId, zone::ZoneType};

pub(super) fn cast_prepared_insets(cards: &[(&str, &str)]) {
    for &(source, spell) in cards {
        let mut s = five_color_setup(&[source, "Gray Ogre", "Jungle Hollow"]);
        s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.force_permanent_on_battlefield(0, "Gray Ogre");
        plant_in_graveyard(&mut s, 0, "Gray Ogre");
        s.force_permanent_on_battlefield(0, "Jungle Hollow");
        enter_battlefield(&mut s, source);
        settle(&mut s);
        let copy = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Exile, PlayerId(0))
            .iter()
            .find(|id| s.game().state[**id].name == spell)
            .copied()
            .expect(source);
        assert!(
            castable(&mut s, copy),
            "{source}: {spell} should be castable"
        );
        let life = s.life(0);
        let opponent_life = s.life(1);
        let hand = s.zone_size(0, ZoneType::Hand);
        let library = s.zone_size(0, ZoneType::Library);
        cast_card(&mut s, copy);
        settle(&mut s);
        match spell {
            "Seed Suture" | "Have a Bite" => s.assert_life(0, life + 1),
            "Vicious Verse" => s.assert_life(1, opponent_life - 1),
            "Vibrant Idea" => s.assert_zone_size(0, ZoneType::Hand, hand + 2),
            "Rejoinder" | "Raise Dead" => s.assert_zone_size(0, ZoneType::Hand, hand + 1),
            "Omit Variables" => s.assert_zone_size(0, ZoneType::Library, library - 3),
            "Bind to Life" => s.assert_zone_size(0, ZoneType::Library, library - 7),
            "Careful Study" => {
                s.assert_zone_size(0, ZoneType::Library, library - 2);
                s.assert_zone_size(0, ZoneType::Hand, hand);
            }
            "Rocket Volley" => assert!(s
                .battlefield_permanents_named(0, "Jungle Hollow")
                .is_empty()),
            "Aqueous Aria" | "Craft with Pride" | "Soul Tether" | "Peer Review"
            | "Awaken the Ages" => {
                let (token, count) = match spell {
                    "Aqueous Aria" => ("Elemental", 1),
                    "Craft with Pride" => ("Treasure", 1),
                    "Soul Tether" => ("Heartwood", 1),
                    "Peer Review" => ("Cadet", 1),
                    _ => ("Spirit", 2),
                };
                assert_eq!(
                    s.battlefield_permanents_named(0, token).len(),
                    count,
                    "{spell}"
                );
            }
            _ => {}
        }
        assert!(
            !s.game()
                .state
                .zones
                .zone_objects(ZoneType::Exile, PlayerId(0))
                .contains(copy),
            "{source} retained its spent copy"
        );
    }
}

#[test]
fn prepared_insets_cast_from_exile() {
    cast_prepared_insets(&[
        ("Vastlands Scavenger", "Bind to Life"),
        ("Pigment Wrangler", "Striking Palette"),
        ("Jadzi, Steward of Fate", "Oracle's Gift"),
        ("Elite Interceptor", "Rejoinder"),
        ("Adventurous Eater", "Have a Bite"),
        ("Blazing Firesinger", "Seething Song"),
        ("Campus Composer", "Aqueous Aria"),
        ("Cheerful Osteomancer", "Raise Dead"),
        ("Goblin Glasswright", "Craft with Pride"),
        ("Honorbound Page", "Forum's Favor"),
        ("Infirmary Healer", "Stream of Life"),
        ("Landscape Painter", "Vibrant Idea"),
        ("Maelstrom Artisan", "Rocket Volley"),
        ("Quill-Blade Laureate", "Twofold Intent"),
        ("Skycoach Conductor", "All Aboard"),
        ("Spellbook Seeker", "Careful Study"),
        ("Strife Scholar", "Awaken the Ages"),
        ("Studious First-Year", "Rampant Growth"),
    ]);
}

#[test]
fn harmonized_trio_pays_three_distinct_taps_and_orders_brainstorm() {
    use crate::support::{
        game::clear_summoning_sickness,
        inspect::{card_of, hand, library, now, offered_cards, tapped},
        play::{activate, resolve_top, select_card},
    };
    use managym::agent::action::Action;
    let mut s = five_color_setup(&["Harmonized Trio", "Gray Ogre"]);
    let trio = s.force_permanent_on_battlefield(0, "Harmonized Trio");
    clear_summoning_sickness(&mut s, trio);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s.action_space().actions.iter().any(
        |action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == trio)
    ));
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    activate(&mut s, trio, 0);
    for payer in [first, second] {
        assert!(!offered_cards(&s).contains(&card_of(&s, trio)));
        let object = card_of(&s, payer);
        select_card(&mut s, object);
    }
    assert!([trio, first, second].into_iter().all(|id| tapped(&s, id)));
    settle(&mut s);
    let copy = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Exile, PlayerId(0))[0];
    assert_eq!(s.game().state[copy].name, "Brainstorm");
    cast_card(&mut s, copy);
    resolve_top(&mut s);
    let cards = hand(&s, 0);
    assert_eq!(cards.len(), 3);
    for (first, second) in [(cards[0], cards[1]), (cards[1], cards[0])] {
        let mut branch = s.clone();
        select_card(&mut branch, first);
        select_card(&mut branch, second);
        settle(&mut branch);
        let deck = library(&branch, 0);
        assert_eq!(
            &deck[deck.len() - 2..],
            &[now(&branch, first), now(&branch, second)]
        );
        assert_eq!(hand(&branch, 0), [cards[2]]);
    }
}

#[test]
fn joined_researchers_rechecks_hand_sizes_and_both_players_draw_three() {
    use crate::support::{
        game::{move_card, move_from_library},
        inspect::hand,
        play::choose,
    };
    use managym::{agent::action::Action, flow::turn::StepKind, state::target::Target};
    let mut s = five_color_setup(&["Joined Researchers"]);
    s.force_permanent_on_battlefield(0, "Joined Researchers");
    let mut equal = s.clone();
    equal.advance_to_active_step(0, StepKind::End);
    assert!(equal.game().state.stack_objects.is_empty());
    let opposing = move_from_library(&mut s, 1, "Island", ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::End);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    let mut no_longer_ahead = s.clone();
    move_card(&mut no_longer_ahead, opposing, ZoneType::Graveyard);
    settle(&mut no_longer_ahead);
    no_longer_ahead.assert_zone_size(0, ZoneType::Exile, 0);
    settle(&mut s);
    let copy = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Exile, PlayerId(0))[0];
    assert_eq!(s.game().state[copy].name, "Secret Rendezvous");
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "1WW");
    let sizes = [0, 1].map(|p| hand(&s, p).len());
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(&mut s, copy);
    choose(&mut s, |action| {
        matches!(
            action,
            Action::ChooseTarget {
                target: Target::Player(PlayerId(1)),
                ..
            }
        )
    });
    settle(&mut s);
    for (p, before) in sizes.into_iter().enumerate() {
        assert_eq!(hand(&s, p).len(), before + 3);
    }
}
