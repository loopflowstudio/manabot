//! Tests for cards filed in `src/ktk.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle, spend_colorless, spend_named},
};
use managym::agent::action::ActionSpaceKind;

#[test]
fn prowess_resolves_above_the_noncreature_spell() {
    let mut s = five_color_setup(&["Monastery Swiftspear", "Abandon Attachments"]);
    let swift = s.force_permanent_on_battlefield(0, "Monastery Swiftspear");
    cast(&mut s, "Abandon Attachments");
    spend_named(&mut s, |label| label == "Spend {U} (expires at step end)");
    spend_colorless(&mut s);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().characteristics(swift).power, 2);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
}

/// The ten gain-lands enter tapped, gain 1 life as they enter and pay either
/// of their two colors, and no other.
#[test]
fn gain_lands_enter_tapped_gain_one_life_and_supply_either_color() {
    use crate::support::{
        game::move_to_hand,
        inspect::tapped,
        play::{cast_card, castable, play_land},
    };
    let lands = [
        ("Bloodfell Caves", "BR"),
        ("Blossoming Sands", "GW"),
        ("Dismal Backwater", "UB"),
        ("Jungle Hollow", "BG"),
        ("Rugged Highlands", "RG"),
        ("Scoured Barrens", "WB"),
        ("Swiftwater Cliffs", "UR"),
        ("Thornwood Falls", "GU"),
        ("Tranquil Cove", "WU"),
        ("Wind-Scarred Crag", "RW"),
    ];
    let one_drops = [
        ('W', "Healer's Hawk"),
        ('U', "Wind Drake"),
        ('B', "Typhoid Rats"),
        ('R', "Raging Goblin"),
        ('G', "Llanowar Elves"),
    ];
    for (name, colors) in lands {
        for (color, spell) in one_drops {
            let mut s = five_color_setup(&[name, spell]);
            let land = play_land(&mut s, name);
            assert!(tapped(&s, land), "{name} enters tapped");
            settle(&mut s);
            s.assert_life(0, 21);
            // Wind Drake is {2}{U}: the pool pays its generic part.
            s.set_player_mana_pool(0, if color == 'U' { "2" } else { "0" });
            let id = move_to_hand(&mut s, spell);
            assert!(!castable(&mut s, id), "{name} is tapped");
            s.game_mut().state.permanents[land].as_mut().unwrap().tapped = false;
            assert_eq!(
                castable(&mut s, id),
                colors.contains(color),
                "{name} for {color}"
            );
            if colors.contains(color) {
                cast_card(&mut s, id);
                settle(&mut s);
                assert!(tapped(&s, land), "{name} pays {color}");
                assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            }
        }
    }
}
