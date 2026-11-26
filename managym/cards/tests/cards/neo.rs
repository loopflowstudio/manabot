//! Entry choices of the Kamigawa: Neon Dynasty lands.

use crate::support::{
    game::{five_color_setup, move_to_hand, set_counters, Scenario},
    inspect::{card_of, permanent_named},
    play::{activate, castable, choose, choose_entry_value, choose_mode},
};
use managym::{
    agent::action::Action,
    state::{
        card::CardType,
        game_object::ObjectId,
        mana::{Mana, SpendingContext, SpendingObject},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

fn play(s: &mut Scenario, name: &str) -> ObjectId {
    let land = move_to_hand(s, name);
    choose(
        s,
        |a| matches!(a,Action::PlayLand {object,..} if *object == land),
    );
    land
}

#[test]
fn entry_choice_uncharted_haven() {
    let mut s = five_color_setup(&["Uncharted Haven"]);
    play(&mut s, "Uncharted Haven");
    assert_eq!(s.action_space().actions.len(), 5);
    choose_entry_value(&mut s, "Blue");
    let id = permanent_named(&s, "Uncharted Haven");
    assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
    assert_eq!(
        s.game().characteristics(id).activated_abilities[0].mana_choices(),
        Some(vec![Mana::parse("U")])
    );
}

#[test]
fn entry_choice_zhao_suppresses_both_land_choices() {
    for name in ["Uncharted Haven", "Secluded Courtyard"] {
        let mut s = five_color_setup(&[name, "Zhao, the Moon Slayer"]);
        let zhao = s.force_permanent_on_battlefield(1, "Zhao, the Moon Slayer");
        set_counters(&mut s, zhao, CounterKind::Conqueror, 1);
        play(&mut s, name);
        let land = permanent_named(&s, name);
        assert!(s.game().state.permanents[land]
            .as_ref()
            .unwrap()
            .chosen
            .is_none());
        let object = card_of(&s, zhao);
        s.game_mut().move_object(object, ZoneType::Graveyard);
        let choices = s.game().characteristics(land).activated_abilities[0]
            .mana_choices()
            .unwrap();
        if name == "Uncharted Haven" {
            assert_eq!(choices, vec![Mana::default()]);
        } else {
            assert!(choices.iter().all(|m| !m.classes()[0]
                .0
                .restriction
                .as_ref()
                .unwrap()
                .allows(&SpendingContext::cast(["Elf"], [CardType::Creature]))));
        }
    }
}

#[test]
fn entry_choice_secluded_courtyard() {
    let mut s = five_color_setup(&["Secluded Courtyard", "Llanowar Elves"]);
    play(&mut s, "Secluded Courtyard");
    choose_entry_value(&mut s, "Elf");
    let land = permanent_named(&s, "Secluded Courtyard");
    s.set_player_mana_pool(0, "");
    s.allow_priority_floats();
    activate(&mut s, land, 0);
    choose_mode(&mut s, 4);
    let elf = move_to_hand(&mut s, "Llanowar Elves");
    assert!(castable(&mut s, elf));
    let pool = s.game().state.players[0].mana_pool.mana();
    let restriction = pool.classes()[0].0.restriction.as_ref().unwrap();
    assert_eq!(pool.classes()[0].1, [0, 0, 0, 0, 1, 0]);
    assert!(restriction.allows(&SpendingContext::cast(["Elf"], [CardType::Creature])));
    assert!(!restriction.allows(&SpendingContext::cast(
        ["Elf"],
        [CardType::Kindred, CardType::Instant]
    )));
    for zone in [ZoneType::Battlefield, ZoneType::Hand] {
        for creature in [false, true] {
            assert_eq!(
                restriction.allows(&SpendingContext::Activate {
                    zone,
                    source: SpendingObject::new(["Elf"], false, creature)
                }),
                creature
            );
        }
    }
    crate::support::play::cast_card(&mut s, elf);
    assert!(s.game().state.players[0]
        .mana_pool
        .mana()
        .classes()
        .is_empty());
}

#[test]
fn entry_choice_courtyard_prices_the_creature_ability_in_its_zone() {
    use crate::support::{
        inspect::has_action,
        play::{activate_card, offers_ability},
    };
    let mut s = five_color_setup(&["Secluded Courtyard", "Giant Koi"]);
    play(&mut s, "Secluded Courtyard");
    choose_entry_value(&mut s, "Fish");
    let land = permanent_named(&s, "Secluded Courtyard");
    let koi = s.force_permanent_on_battlefield(0, "Giant Koi");
    let hand = move_to_hand(&mut s, "Giant Koi");
    s.game_mut().state.permanents[koi].as_mut().unwrap().tapped = true;
    s.set_player_mana_pool(0, "1");
    s.allow_priority_floats();
    activate(&mut s, land, 0);
    choose_mode(&mut s, 0);
    let mut kindred = s.clone();
    kindred.game_mut().state[hand].types =
        managym::state::card::CardTypes::new([CardType::Kindred]);
    kindred.game_mut().scenario_refresh_priority().unwrap();
    assert!(!has_action(
        &kindred,
        |a| matches!(a, Action::ActivateCardAbility { object, .. } if *object == hand)
    ));
    let mut cycling = s.clone();
    activate_card(&mut cycling, hand, 0);
    assert!(cycling.game().state.players[0]
        .mana_pool
        .mana()
        .classes()
        .is_empty());
    assert!(!offers_ability(&mut s, koi, 1));
    s.game_mut().state.permanents[koi].as_mut().unwrap().tapped = false;
    assert!(offers_ability(&mut s, koi, 1));
}
