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

#[test]
fn copying_a_prepared_creature_and_removing_its_abilities_preserve_one_cast() {
    use crate::support::{game::move_from_library, play::cast};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;

    // The Researchers prepare at end step. A copy entering later has the
    // printed trigger, but hasn't resolved it or prepared a spell of its own.
    let mut s = five_color_setup(&["Joined Researchers", "Clone Legion", "Honest Work"]);
    let original = s.force_permanent_on_battlefield(0, "Joined Researchers");
    move_from_library(&mut s, 1, "Island", ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::End);
    settle(&mut s);
    let prepared = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Exile, PlayerId(0))[0];
    let mut copied = s.clone();
    copied.advance_to_active_step(0, StepKind::Main);
    copied.set_player_mana_pool(0, "7UU");
    cast(&mut copied, "Clone Legion");
    assert!(copied.choose_target(Target::Player(PlayerId(0))));
    settle(&mut copied);
    assert_eq!(
        copied
            .battlefield_permanents_named(0, "Joined Researchers")
            .len(),
        2
    );
    copied.assert_zone_size(0, ZoneType::Exile, 1);
    copied.set_player_mana_pool(0, "1WW");
    assert!(castable(&mut copied, prepared));

    s.advance_to_active_step(1, StepKind::Main);
    let aura = move_from_library(&mut s, 1, "Honest Work", ZoneType::Hand);
    s.set_player_mana_pool(1, "U");
    cast_card(&mut s, aura);
    assert!(s.choose_target(Target::Permanent(original)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(original).name, "Humble Merchant");
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "1WW");
    let before = [0, 1].map(|p| s.zone_size(p, ZoneType::Hand));
    cast_card(&mut s, prepared);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    for (p, hand_size) in before.into_iter().enumerate() {
        s.assert_zone_size(p, ZoneType::Hand, hand_size + 3);
    }
    s.assert_zone_size(0, ZoneType::Exile, 0);
}

/// Sanar's Treasure activation waits for an instant or sorcery cast this
/// turn, however that spell has since left the stack; a creature spell does
/// not count. Its prepared Wild Idea finds an instant or sorcery, reveals it
/// into its caster's hand and shuffles.
#[test]
fn sanar_makes_treasure_after_an_instant_and_wild_idea_finds_one() {
    use crate::support::{
        game::{clear_summoning_sickness, move_from_library},
        inspect::{find_owned_card_off_battlefield, hand, offered_cards},
        play::{activate, offers_ability, resolve_top, select_card, target_opponent},
    };
    let mut s = five_color_setup(&["Sanar, Unfinished Genius", "Lightning Bolt", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let sanar = enter_battlefield(&mut s, "Sanar, Unfinished Genius");
    settle(&mut s);
    clear_summoning_sickness(&mut s, sanar);
    assert!(!offers_ability(&mut s, sanar, 0), "no spell cast yet");

    let ogre = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    cast_card(&mut s, ogre);
    settle(&mut s);
    assert!(
        !offers_ability(&mut s, sanar, 0),
        "a creature spell is not enough"
    );

    let bolt = move_from_library(&mut s, 0, "Lightning Bolt", ZoneType::Hand);
    cast_card(&mut s, bolt);
    target_opponent(&mut s);
    settle(&mut s);
    assert!(
        offers_ability(&mut s, sanar, 0),
        "the resolved instant still counts"
    );
    activate(&mut s, sanar, 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);

    let wild_idea = find_owned_card_off_battlefield(&s, 0, "Wild Idea");
    cast_card(&mut s, wild_idea);
    resolve_top(&mut s);
    let choices = offered_cards(&s);
    assert!(!choices.is_empty());
    assert!(
        choices
            .iter()
            .all(|&id| s.game().state[id].types.is_spell()),
        "only instant and sorcery cards"
    );
    select_card(&mut s, choices[0]);
    settle(&mut s);
    let [found] = hand(&s, 0)[..] else {
        panic!("one card found")
    };
    assert_eq!(s.game().state[found].name, "Lightning Bolt");
    let definition = s.game().state[found].definition_id;
    assert_eq!(
        s.game().state.players[0].known_hand.get(&definition),
        Some(&1),
        "the found card is revealed"
    );
}
