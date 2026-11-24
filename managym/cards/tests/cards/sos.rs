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
