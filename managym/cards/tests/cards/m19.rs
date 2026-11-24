#[test]
fn token_producer_heroic_reinforcements() {
    use crate::support::{game::five_color_setup, play::play_and_settle};
    let mut s = five_color_setup(&["Heroic Reinforcements", "Gray Ogre"]);
    let earlier = s.force_permanent_on_battlefield(0, "Gray Ogre");
    play_and_settle(&mut s, "Heroic Reinforcements");
    assert_eq!(s.game().characteristics(earlier).pt(), (3, 3));
    assert!(s
        .game()
        .characteristics(earlier)
        .keywords
        .has(managym::state::card::Keyword::Haste));
    crate::assert_creature_tokens(
        &s,
        "Soldier",
        2,
        (2, 2),
        &[managym::state::mana::Color::White],
        &[managym::state::card::Keyword::Haste],
    );
    let late = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert_eq!(s.game().characteristics(late).pt(), (2, 2));
    assert!(!s
        .game()
        .characteristics(late)
        .keywords
        .has(managym::state::card::Keyword::Haste));
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    for id in s.battlefield_permanents_named(0, "Soldier") {
        let face = s.game().characteristics(id);
        assert_eq!(face.pt(), (1, 1));
        assert!(!face.keywords.has(managym::state::card::Keyword::Haste));
    }
}

#[test]
fn token_producer_lathliss_dragon_queen() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Lathliss, Dragon Queen", "Shivan Dragon"]);
    enter_and_settle(&mut s, "Lathliss, Dragon Queen");
    enter_and_settle(&mut s, "Shivan Dragon");
    crate::assert_creature_tokens(
        &s,
        "Dragon",
        1,
        (5, 5),
        &[managym::state::mana::Color::Red],
        &[managym::state::card::Keyword::Flying],
    );
    let queen = crate::support::inspect::permanent_named(&s, "Lathliss, Dragon Queen");
    crate::support::play::activate(&mut s, queen, 0);
    crate::support::play::settle(&mut s);
    assert_eq!(s.game().characteristics(queen).pt(), (7, 6));
    for name in ["Shivan Dragon", "Dragon"] {
        let dragon = crate::support::inspect::permanent_named(&s, name);
        assert_eq!(s.game().characteristics(dragon).pt(), (6, 5));
    }
}

#[test]
fn printed_creature_gigantosaurus() {
    crate::printed_creature_combat("Gigantosaurus", 10, 10, &[]);
}

#[test]
fn foundations_permanent_meteor_golem() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{cast, pass_until, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Meteor Golem", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(1, "Island");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    cast(&mut s, "Meteor Golem");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!s.choose_target(Target::Permanent(land)));
    assert!(!s.choose_target(Target::Permanent(own)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
}

#[test]
fn foundations_grant_aggressive_mammoth() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, keywords};
    use managym::state::card::Keyword;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Aggressive Mammoth", "Gray Ogre", "Hill Giant"]);
    let ally = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(0, "Hill Giant");
    let base = s.game().characteristics(ally).pt();
    let source = s.force_permanent_on_battlefield(0, "Aggressive Mammoth");
    assert!(keywords(&s, ally).has(Keyword::Trample));
    assert!(!keywords(&s, enemy).has(Keyword::Trample));
    assert!(keywords(&s, other).has(Keyword::Trample));
    assert_eq!(s.game().characteristics(enemy).pt(), base);
    let card = card_of(&s, source);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(ally).pt(), base);
    assert!(!keywords(&s, ally).has(Keyword::Trample));
}

#[test]
fn foundations_5f_exclusion_mage() {
    crate::bounces_an_opposing_creature("Exclusion Mage");
}

#[test]
fn foundations_5f_herald_of_faith() {
    let (s, ..) = crate::attack_and_resolve("Herald of Faith");
    assert_eq!((s.life(0), s.life(1)), (22, 20));
}

/// "It deals 1 damage to any target": both players and every creature are
/// offered, its own controller's included.
#[test]
fn foundations_5f_skeleton_archer() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::offered_targets;
    use crate::support::play::{enter_until_target, settle};
    use managym::state::game_object::{PlayerId, Target};
    let mut s = five_color_setup(&["Skeleton Archer", "Gray Ogre"]);
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    enter_until_target(&mut s, "Skeleton Archer");
    let offers = offered_targets(&s);
    for target in [
        Target::Player(PlayerId(0)),
        Target::Player(PlayerId(1)),
        Target::Permanent(ours),
        Target::Permanent(theirs),
    ] {
        assert!(offers.contains(&target), "{target:?}");
    }
    let mut player = s.clone();
    assert!(player.choose_target(Target::Player(PlayerId(1))));
    settle(&mut player);
    player.assert_life(1, 19);
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert_eq!(
        s.game().state.permanents[theirs].as_ref().unwrap().damage,
        1
    );
    s.assert_life(1, 20);
}

/// "It deals 2 damage to target player or planeswalker": the shared
/// player-or-planeswalker target never offers a creature.
#[test]
fn foundations_5f_viashino_pyromancer() {
    use crate::support::game::five_color_setup;
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::permanent::CounterKind;
    let mut s = five_color_setup(&["Viashino Pyromancer", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let walker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    crate::make_planeswalker(&mut s, walker);
    crate::support::game::set_counters(&mut s, walker, CounterKind::Loyalty, 3);
    let offers = [
        Target::Player(PlayerId(0)),
        Target::Player(PlayerId(1)),
        Target::Permanent(walker),
    ];
    let mut at_walker = s.clone();
    crate::enter_choosing(
        &mut at_walker,
        "Viashino Pyromancer",
        &offers,
        Target::Permanent(walker),
    );
    assert_eq!(
        crate::support::inspect::counters_of(&at_walker, walker, CounterKind::Loyalty),
        1
    );
    at_walker.assert_life(1, 20);
    crate::enter_choosing(
        &mut s,
        "Viashino Pyromancer",
        &offers,
        Target::Player(PlayerId(1)),
    );
    s.assert_life(1, 18);
}

/// "{3}{U}{U}: Draw two cards."
#[test]
fn foundations_5f_mystic_archaeologist() {
    use crate::support::{
        game::five_color_setup,
        inspect::hand,
        play::{activate, settle, spend_colorless},
    };
    let mut s = five_color_setup(&["Mystic Archaeologist"]);
    let archaeologist = s.force_permanent_on_battlefield(0, "Mystic Archaeologist");
    s.set_player_mana_pool(0, "3UU");
    activate(&mut s, archaeologist, 0);
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 2);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

/// "{2}, {T}: Each opponent loses 1 life and you gain 1 life": a tap cost,
/// so a summoning-sick Neonate is not offered the ability.
#[test]
fn foundations_5f_vampire_neonate() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        inspect::can_activate,
        play::{activate, settle, spend_colorless},
    };
    let mut s = five_color_setup(&["Vampire Neonate"]);
    let neonate = s.force_permanent_on_battlefield(0, "Vampire Neonate");
    s.set_player_mana_pool(0, "2");
    assert!(!can_activate(&s, neonate));
    clear_summoning_sickness(&mut s, neonate);
    activate(&mut s, neonate, 0);
    spend_colorless(&mut s);
    settle(&mut s);
    assert_eq!((s.life(0), s.life(1)), (21, 19));
}

/// "As long as you control a Dragon, this creature has flying": an
/// opponent's Dragon grants nothing; flying ends when the Dragon leaves.
#[test]
fn foundations_5f_kargan_dragonrider() {
    use crate::support::{
        game::{five_color_setup, move_card},
        inspect::{card_of, keywords},
    };
    use managym::state::{card::Keyword, zone::ZoneType};
    let mut s = five_color_setup(&["Kargan Dragonrider", "Lathliss, Dragon Queen"]);
    let rider = s.force_permanent_on_battlefield(0, "Kargan Dragonrider");
    assert!(!keywords(&s, rider).has(Keyword::Flying));
    s.force_permanent_on_battlefield(1, "Lathliss, Dragon Queen");
    assert!(!keywords(&s, rider).has(Keyword::Flying));
    let dragon = s.force_permanent_on_battlefield(0, "Lathliss, Dragon Queen");
    assert!(keywords(&s, rider).has(Keyword::Flying));
    let card = card_of(&s, dragon);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert!(!keywords(&s, rider).has(Keyword::Flying));
}
