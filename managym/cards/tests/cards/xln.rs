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
fn entry_choice_sorcerous_spyglass() {
    use crate::support::{
        game::{five_color_setup, move_to_hand},
        inspect::{has_action, permanent_named},
        play::{cast, choose_entry_value, offers_ability, resolve_top, spend_colorless},
    };
    use managym::agent::action::Action;
    let mut s = five_color_setup(&["Sorcerous Spyglass", "Giant Koi", "Llanowar Elves"]);
    let koi = s.force_permanent_on_battlefield(0, "Giant Koi");
    let hand = move_to_hand(&mut s, "Giant Koi");
    let elf = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    crate::support::game::clear_summoning_sickness(&mut s, elf);
    cast(&mut s, "Sorcerous Spyglass");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    let mut fork = s.clone();
    choose_entry_value(&mut fork, "Lightning Bolt");
    let glass = permanent_named(&fork, "Sorcerous Spyglass");
    assert_eq!(
        fork.game().state.permanents[glass]
            .as_ref()
            .unwrap()
            .chosen
            .as_deref(),
        Some(&managym::state::permanent::EntryChoice::CardName(
            "Lightning Bolt".into()
        ))
    );
    assert!(offers_ability(&mut fork, koi, 1));
    choose_entry_value(&mut s, "Giant Koi");
    assert!(!offers_ability(&mut s, koi, 1));
    assert!(!has_action(
        &s,
        |a| matches!(a,Action::ActivateCardAbility{object,..} if *object==hand)
    ));
    cast(&mut s, "Sorcerous Spyglass");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    choose_entry_value(&mut s, "Llanowar Elves");
    s.allow_priority_floats();
    assert!(has_action(
        &s,
        |a| matches!(a,Action::TapForMana {permanent,..} if *permanent==elf)
    ));
}

#[test]
fn entry_choice_simultaneous_spyglass_and_banner() {
    use crate::support::{
        game::{five_color_setup, move_card},
        inspect::{card_of, permanent_named},
        play::{cast, choose_entry_value, resolve_top, settle},
    };
    use managym::state::{game_object::Target, zone::ZoneType};
    let mut s = five_color_setup(&["Brought Back", "Sorcerous Spyglass", "Heraldic Banner"]);
    let mut buried = vec![];
    for name in ["Sorcerous Spyglass", "Heraldic Banner"] {
        let id = s.force_permanent_on_battlefield(0, name);
        let card = card_of(&s, id);
        buried.push(move_card(&mut s, card, ZoneType::Graveyard));
    }
    s.force_card_in_hand(1, "Forest");
    cast(&mut s, "Brought Back");
    for &object in &buried {
        assert!(s.choose_target(Target::GraveyardCard(object)));
    }
    resolve_top(&mut s);
    crate::support::inspect::assert_choice_contract(&s);
    for first in [0, 1] {
        let mut fork = s.clone();
        assert_eq!(fork.action_space().actions.len(), 2);
        assert!(fork.game().state.players[1].known_hand.is_empty());
        choose_entry_value(
            &mut fork,
            ["for Sorcerous Spyglass", "for Heraldic Banner"][first],
        );
        assert_eq!(
            fork.game().state.players[1].known_hand.is_empty(),
            first == 1
        );
        for i in [first, 1 - first] {
            assert!(fork
                .battlefield_permanents_named(0, "Heraldic Banner")
                .is_empty());
            assert!(fork
                .battlefield_permanents_named(0, "Sorcerous Spyglass")
                .is_empty());
            choose_entry_value(&mut fork, ["Mountain", "Red"][i]);
        }
        settle(&mut fork);
        for name in ["Heraldic Banner", "Sorcerous Spyglass"] {
            assert!(fork.game().state.permanents[permanent_named(&fork, name)]
                .as_ref()
                .unwrap()
                .chosen
                .is_some());
        }
    }
}
