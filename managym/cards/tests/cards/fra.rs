use super::sos::cast_prepared_insets;

#[test]
fn prepared_insets_cast_from_exile() {
    cast_prepared_insets(&[
        ("Prudent Fateseer", "Peer Review"),
        ("Diviner of Victory", "Unwind History"),
        ("Blossom-Blessed Angel", "Seed Suture"),
        ("Carnivorous Cultivator", "Enroot"),
        ("Emergency Phytomedic", "Seed Suture"),
        ("Fatehold Chronologist", "Peer Review"),
        ("Hallway Heckler", "Vicious Verse"),
        ("Konstrari Improviser", "Soul Tether"),
        ("Pompous Battlemage", "Improvised Act"),
        ("Semester Foreseer", "Peer Review"),
        ("Theorix Metamage", "Omit Variables"),
        ("Vigorbloom Vanguard", "Seed Suture"),
        ("Void Extrapolator", "Omit Variables"),
    ]);
}

/// Heartwood Crafter's colorless mana pays Soul Tether cast from its
/// prepared exile copy but no spell from hand. Codie copies the prepared
/// spell; every viewer sees which stack spell was prepared, and the mana's
/// restriction reaches the policy as a pooled-mana fact row.
#[test]
fn heartwood_mana_pays_its_prepared_spell_but_no_hand_spell() {
    use crate::support::{
        game::{clear_summoning_sickness, enter_battlefield, five_color_unfunded_setup},
        inspect::find_owned_card_off_battlefield,
        play::{cast_card, castable, resolve_top, settle, tap_for_mana},
    };
    use managym::agent::{
        observation::Observation,
        observation_encoder::{
            encode, unrepresentable_state, ObservationEncoderConfig, FACT_DIM, POOLED_MANA,
            STACK_DIM,
        },
        observation_layout::{fact, stack},
    };
    use managym::state::{game_object::PlayerId, zone::ZoneType};
    let mut s =
        five_color_unfunded_setup(&["Codie, Ravenous Codex", "Heartwood Crafter", "Gray Ogre"]);
    s.game_mut()
        .scenario_force_battlefield(PlayerId(0), "Codie, Ravenous Codex", true)
        .unwrap();
    let crafter = enter_battlefield(&mut s, "Heartwood Crafter");
    settle(&mut s);
    clear_summoning_sickness(&mut s, crafter);
    s.set_player_mana_pool(0, "RR");
    s.allow_priority_floats();
    tap_for_mana(&mut s, crafter);

    let ogre = crate::support::game::move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    assert!(
        !castable(&mut s, ogre),
        "the {{C}} cannot pay a spell from hand"
    );
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert!(obs.agent.restricted_mana.is_empty());
    let [pooled] = &obs.agent.pooled_mana_classes[..] else {
        panic!("one pooled class: {:?}", obs.agent.pooled_mana_classes)
    };
    assert_eq!(
        pooled.restriction.as_deref(),
        Some("anything but spells from hand")
    );
    assert_eq!(pooled.mana, [0, 0, 0, 0, 0, 1]);
    assert_eq!(obs.agent.lesson_mana, [0, 0, 0, 0, 0, 1]);
    assert_eq!(unrepresentable_state(&obs), None);
    let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
    let pooled_rows = input
        .facts
        .chunks_exact(FACT_DIM)
        .filter(|row| {
            let id = ((row[fact::ID_HIGH] as u32) << 16) | row[fact::ID_LOW] as u32;
            id == managym::cards::tokens::name_value(POOLED_MANA) as u32
        })
        .count();
    assert!(pooled_rows > 0, "the restricted class is a fact program");

    let tether = find_owned_card_off_battlefield(&s, 0, "Soul Tether");
    // The pool pays exactly, so the cast completes without a manual choice.
    cast_card(&mut s, tether);
    assert_eq!(
        s.game().state.players[0].mana_pool.total(),
        0,
        "Soul Tether spent the {{C}} and both red"
    );
    resolve_top(&mut s);
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        let prepared: Vec<_> = obs
            .stack_objects
            .iter()
            .map(|object| (object.from_prepared, object.flashback))
            .collect();
        // Top first: Codie's copy above the prepared original.
        assert_eq!(prepared, [(false, false), (true, false)]);
        let input = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        let rows: Vec<_> = input
            .stack
            .chunks_exact(STACK_DIM)
            .take(2)
            .map(|row| row[stack::FROM_PREPARED])
            .collect();
        assert_eq!(rows, [0.0, 1.0]);
        let restored: Observation = serde_json::from_str(&obs.to_json()).unwrap();
        assert_eq!(restored.to_json(), obs.to_json());
    }
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Heartwood").len(), 2);
}

/// Codie copies a spell cast from a prepared copy. Its copy is not a prepared
/// spell, an ordinary cast does not trigger it, and its activation prepares
/// each unprepared creature with a prepare spell without duplicating a copy.
#[test]
fn codie_copies_prepared_spells_and_prepares_each_creature() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup},
        play::{activate, cast_card, pay_rest, resolve_top, settle},
    };
    use managym::state::{
        game_object::{ObjectId, PlayerId},
        stack_object::StackObject,
        zone::ZoneType,
    };
    let prepared_spells = |s: &crate::support::game::Scenario| -> Vec<bool> {
        s.game()
            .state
            .stack_objects
            .iter()
            .filter_map(|object| match object {
                StackObject::Spell(spell) => Some(spell.from_prepared),
                _ => None,
            })
            .collect()
    };
    let tethers = |s: &crate::support::game::Scenario| -> Vec<ObjectId> {
        s.game()
            .state
            .zones
            .zone_objects(ZoneType::Exile, PlayerId(0))
            .iter()
            .copied()
            .filter(|id| s.game().state[*id].name == "Soul Tether")
            .collect()
    };
    let mut s = five_color_setup(&["Codie, Ravenous Codex", "Konstrari Improviser", "Gray Ogre"]);
    let codie = s
        .game_mut()
        .scenario_force_battlefield(PlayerId(0), "Codie, Ravenous Codex", true)
        .unwrap();
    enter_battlefield(&mut s, "Konstrari Improviser");
    settle(&mut s);
    let [copy] = tethers(&s)[..] else {
        panic!("one prepared Soul Tether")
    };
    cast_card(&mut s, copy);
    pay_rest(&mut s);
    assert_eq!(prepared_spells(&s), [true]);
    resolve_top(&mut s);
    assert_eq!(prepared_spells(&s), [true, false]);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Heartwood").len(), 2);
    assert!(tethers(&s).is_empty());

    let ogre = crate::support::game::move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    cast_card(&mut s, ogre);
    pay_rest(&mut s);
    assert_eq!(prepared_spells(&s), [false]);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);

    enter_battlefield(&mut s, "Konstrari Improviser");
    settle(&mut s);
    let kept = tethers(&s);
    assert_eq!(kept.len(), 1);
    s.set_player_mana_pool(0, "WUBRG");
    activate(&mut s, codie, 0);
    settle(&mut s);
    let prepared = tethers(&s);
    assert_eq!(prepared.len(), 2, "one copy for each Improviser");
    assert!(prepared.contains(&kept[0]));
}

#[test]
fn commons_and_annex_lands_check_their_controllers_planeswalkers_and_pay_either_color() {
    use crate::support::{
        game::{five_color_unfunded_setup, set_counters},
        inspect::tapped,
        play::{play_land, tapped_dual_land_pays_either_color},
    };
    use managym::state::permanent::CounterKind;

    for (name, colors) in [
        ("Dedicated Commons", "RW"),
        ("Fatehold Annex", "WU"),
        ("Formidable Commons", "BG"),
        ("Innovative Commons", "UR"),
        ("Konstrari Annex", "RG"),
        ("Meticulous Commons", "WB"),
        ("Stingerquill Annex", "BR"),
        ("Theorix Annex", "UB"),
        ("Transformative Commons", "GU"),
        ("Vigorbloom Annex", "GW"),
    ] {
        tapped_dual_land_pays_either_color(name, colors, |_, _| {});
        for controller in [0, 1] {
            let mut s = five_color_unfunded_setup(&[name, "Gray Ogre"]);
            let walker = s.force_permanent_on_battlefield(controller, "Gray Ogre");
            crate::make_planeswalker(&mut s, walker);
            set_counters(&mut s, walker, CounterKind::Loyalty, 3);
            let land = play_land(&mut s, name);
            assert_eq!(
                tapped(&s, land),
                controller != 0,
                "{name}, planeswalker seat {controller}"
            );
        }
    }
}
