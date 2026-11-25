#[test]
fn printed_creature_bishops_soldier() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat("Bishop's Soldier", 2, 2, &[Keyword::Lifelink]);
}

#[test]
fn foundations_spell_rivers_rebuke() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{cast, settle};
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["River's Rebuke", "Sol Ring", "Island", "Gray Ogre"]);
    let artifact = s.force_permanent_on_battlefield(1, "Sol Ring");
    let land = s.force_permanent_on_battlefield(1, "Island");
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let object = card_of(&s, artifact);
    cast(&mut s, "River's Rebuke");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Hand));
    assert_eq!(s.game().characteristics(mine).pt(), (2, 2));
    assert!(s.game().state.permanents[land].is_some());
}

#[test]
fn foundations_spell_dive_down() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Dive Down", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    cast(&mut s, "Dive Down");
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 6));
    assert!(s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Hexproof));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 3));
    assert!(!s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Hexproof));
}

/// "When this Equipment enters, attach it to target Pirate you control":
/// only the controller's Pirate is offered, and it gets +2/+1; with no
/// Pirate the Cutlass enters unattached.
#[test]
fn foundations_5f_pirate_s_cutlass() {
    use crate::support::{
        game::five_color_setup,
        inspect::{offered_targets, permanent_named},
        play::{enter_and_settle, enter_until_target, settle},
    };
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Pirate's Cutlass", "Kitesail Corsair", "Gray Ogre"]);
    let pirate = s.force_permanent_on_battlefield(0, "Kitesail Corsair");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Kitesail Corsair");
    let mut alone = s.clone();
    enter_until_target(&mut s, "Pirate's Cutlass");
    assert_eq!(offered_targets(&s), vec![Target::Permanent(pirate)]);
    assert!(s.choose_target(Target::Permanent(pirate)));
    settle(&mut s);
    let cutlass = permanent_named(&s, "Pirate's Cutlass");
    assert_eq!(
        s.game().state.permanents[cutlass]
            .as_ref()
            .unwrap()
            .attached_to,
        Some(crate::support::inspect::card_of(&s, pirate))
    );
    assert_eq!(s.game().characteristics(pirate).pt(), (4, 2));

    for id in [pirate] {
        let card = crate::support::inspect::card_of(&alone, id);
        crate::support::game::move_card(
            &mut alone,
            card,
            managym::state::zone::ZoneType::Graveyard,
        );
    }
    enter_and_settle(&mut alone, "Pirate's Cutlass");
    let cutlass = permanent_named(&alone, "Pirate's Cutlass");
    assert_eq!(
        alone.game().state.permanents[cutlass]
            .as_ref()
            .unwrap()
            .attached_to,
        None
    );
}

#[test]
fn new_horizons_grants_the_land_two_mana_of_one_color() {
    use crate::support::{
        game::five_color_setup,
        inspect::{counters, permanent_named, tapped},
        play::{activate, cast, choose_mode, pass_until, settle},
    };
    use managym::{agent::action::ActionSpaceKind, state::game_object::Target};
    let mut s = five_color_setup(&["New Horizons", "Gray Ogre", "Leonin Skyhunter"]);
    s.allow_priority_floats();
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "2G");
    cast(&mut s, "New Horizons");
    assert!(s.choose_target(Target::Permanent(land)));
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(counters(&s, ogre), 1);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, land, 0);
    choose_mode(&mut s, 0);
    assert!(tapped(&s, land));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 2);
    cast(&mut s, "Leonin Skyhunter");
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert_eq!(
        s.game()
            .characteristics(permanent_named(&s, "Leonin Skyhunter"))
            .pt(),
        (2, 2)
    );
}
