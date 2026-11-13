use managym::{
    agent::action::ActionType,
    flow::turn::StepKind,
    state::{mana::Color, zone::ZoneType},
};

use crate::support::game::*;

/// CR 106.3 — Mana abilities can be activated to produce mana used for casting.
#[test]
fn cr_106_3_casting_uses_mana_abilities() {
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Mountain", 40)]),
        3,
    );

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Forest");
    s.force_card_in_hand(0, "Llanowar Elves");

    s.assert_action_available(ActionType::PriorityPlayLand);
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    s.assert_action_available(ActionType::PriorityCastSpell);
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));

    assert_eq!(s.zone_size(0, ZoneType::Stack), 1);

    s.pass_priority();
    s.pass_priority();

    assert!(
        !s.battlefield_permanents_named(0, "Llanowar Elves")
            .is_empty(),
        "cast creature should resolve to the battlefield"
    );

    let forest = s
        .battlefield_permanents_named(0, "Forest")
        .into_iter()
        .next()
        .expect("forest should be on battlefield");
    let tapped = s
        .game()
        .state
        .permanents
        .get(forest.0)
        .and_then(|perm| perm.as_ref())
        .map(|perm| perm.tapped)
        .unwrap_or(false);
    assert!(tapped, "land used for mana should be tapped");
}

/// CR 106.4 — Unspent mana empties from players' mana pools as steps end.
#[test]
fn cr_106_4_mana_pool_empties_between_steps() {
    let mut s = Scenario::new(deck(&[("Mountain", 40)]), deck(&[("Mountain", 40)]), 9);

    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "3R");
    assert_eq!(
        s.game().state.players[0].mana_pool.mana[Color::Red as usize],
        1
    );

    s.pass_priority();
    s.pass_priority();

    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

/// Negative: a spell is not castable without enough producible mana.
#[test]
fn cr_106_negative_spell_not_available_without_mana() {
    let mut s = Scenario::new(deck(&[("Gray Ogre", 40)]), deck(&[("Mountain", 40)]), 15);

    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Gray Ogre");

    s.assert_action_not_available(ActionType::PriorityCastSpell);
}

/// CR 107.4b/c and 118.3: a generic cost accepts any mana; C requires
/// colorless. Exhaust all 0/1 six-type pools against one fixed pip and 0..2
/// generic. The oracle is integer resource counting, independent of payment.
#[test]
fn audit_generic_and_specific_mana_boundary_matrix() {
    use managym::state::mana::{Mana, ManaCost};

    for bits in 0_u8..64 {
        let pool = Mana {
            mana: std::array::from_fn(|slot| (bits >> slot) & 1),
            ..Default::default()
        };
        for (slot, symbol) in ["W", "U", "B", "R", "G", "C"].iter().enumerate() {
            for generic in 0..=2 {
                let cost = ManaCost::parse(&format!("{generic}{symbol}"));
                let expected = pool.mana[slot] == 1 && bits.count_ones() > generic;
                assert_eq!(
                    pool.can_pay(&cost),
                    expected,
                    "bits={bits}, cost={generic}{symbol}"
                );
                if expected {
                    let mut remaining = pool.clone();
                    let mut owed = cost.clone();
                    while owed.mana_value > 0 {
                        // This fixture chooses one valid path; the engine offers
                        // every distinct unit instead of preferring a color.
                        let choice = remaining.spending_choices(&owed).remove(0);
                        remaining.spend(&choice.unit, &owed.purpose);
                        owed = choice.remaining;
                    }
                    assert_eq!(remaining.total(), pool.total() - generic as u16 - 1);
                    assert_eq!(remaining.mana[slot], 0);
                    assert!(remaining
                        .mana
                        .iter()
                        .zip(pool.mana)
                        .all(|(left, before)| *left <= before));
                }
            }
        }
    }
}

/// CR 106.6: spending restrictions survive storage and cannot pay an
/// unrelated cost. Clearing a pool loses both restricted and ordinary mana.
#[test]
fn audit_restricted_mana_never_pays_unrelated_costs() {
    use managym::state::mana::{Mana, ManaCost, ManaRestriction, SpendingContext};

    let lesson = ManaRestriction::spell(["Lesson"]);
    for slot in 0..6 {
        for eligible in [false, true] {
            let mut amounts = [0; 6];
            amounts[slot] = 1;
            let mut pool = Mana::restricted_to(lesson.clone(), amounts);
            let mut cost = ManaCost::parse("1");
            if eligible {
                cost.purpose = SpendingContext::cast(["Lesson"], []);
            }
            assert_eq!(pool.can_pay(&cost), eligible);
            let choices = pool.spending_choices(&cost);
            assert_eq!(choices.len(), usize::from(eligible));
            if eligible {
                pool.spend(&choices[0].unit, &cost.purpose);
                assert_eq!(pool.total(), 0);
            }
            pool.clear();
            assert_eq!(pool, Mana::default());
        }
    }
}

/// CR 106.6: a restriction may name any subtype. Mana for Ogre spells casts
/// Gray Ogre and nothing else; mana for abilities of Scarecrow sources pays
/// Scarecrow Guide's activation but not a Scarecrow spell. The observation
/// keeps each restriction whole, and the policy input, whose columns name
/// only Lesson, Shrine and Ally purposes, refuses it rather than drop it.
#[test]
fn restrictions_on_any_subtype_pay_exactly_their_purposes() {
    use managym::{
        agent::{
            action::Action,
            observation::{Observation, RestrictedManaData},
            observation_encoder::{encode, ObservationEncodeError, ObservationEncoderConfig},
        },
        state::{
            game_object::PlayerId,
            mana::{Mana, ManaRestriction},
        },
    };

    use crate::support::{inspect::can_cast, play::command};

    let ogre_spells = ManaRestriction::spell(["Ogre"]);
    let mut s = five_color_unfunded_setup(&["Gray Ogre", "Lightning Strike"]);
    let ogre = move_to_hand(&mut s, "Gray Ogre");
    let strike = move_to_hand(&mut s, "Lightning Strike");
    s.game_mut().state.players[0].mana_pool =
        Mana::restricted_to(ogre_spells.clone(), [0, 0, 0, 3, 0, 0]);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, ogre));
    assert!(!can_cast(&s, strike));
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let restricted = &observation.agent.restricted_mana;
    assert_eq!(restricted.len(), 1);
    assert_eq!(restricted[0].spells, ["Ogre"]);
    assert_eq!(restricted[0].mana, [0, 0, 0, 3, 0, 0]);
    assert!(matches!(
        encode(&observation, &ObservationEncoderConfig::default()),
        Err(ObservationEncodeError::Unrepresentable(state)) if state == "mana restricted to Ogre spells"
    ));
    // The columns sum Lesson, Shrine and Ally purposes, so only the classes
    // they solve for encode: these two pools would share one encoding.
    let class = |spells: &[&str], abilities: &[&str]| RestrictedManaData {
        spells: spells.iter().map(|name| name.to_string()).collect(),
        abilities: abilities.iter().map(|name| name.to_string()).collect(),
        except_nonartifact_spells: false,
        mana: [0, 0, 0, 1, 0, 0],
    };
    for (pool, refused) in [
        (
            [
                class(&["Lesson"], &[]),
                class(&["Shrine", "Ally"], &["Ally"]),
            ],
            "mana restricted to Ally or Shrine spells or abilities of Ally sources",
        ),
        (
            [
                class(&["Lesson", "Shrine"], &[]),
                class(&["Ally"], &["Ally"]),
            ],
            "",
        ),
        (
            [
                class(&["Lesson", "Shrine"], &[]),
                class(&["Ally", "Lesson"], &["Ally"]),
            ],
            "mana restricted to Ally or Lesson spells or abilities of Ally sources",
        ),
    ] {
        let mut observation = observation.clone();
        observation.agent.restricted_mana = pool.to_vec();
        match encode(&observation, &ObservationEncoderConfig::default()) {
            Err(ObservationEncodeError::Unrepresentable(state)) => assert_eq!(state, refused),
            Ok(_) => assert!(refused.is_empty()),
            Err(error) => panic!("{error}"),
        }
    }
    command(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == ogre),
    );
    // The restricted pool pays the whole cost, so payment is automatic.
    assert_eq!(s.game().state.players[0].mana_pool, Mana::default());
    assert_eq!(s.game().state.zones.zone_of(ogre), Some(ZoneType::Stack));

    for (restriction, activates) in [
        (ManaRestriction::only([], ["Scarecrow"]), true),
        (ManaRestriction::spell(["Scarecrow"]), false),
    ] {
        let mut s = five_color_unfunded_setup(&["Scarecrow Guide"]);
        let guide = s.force_permanent_on_battlefield(0, "Scarecrow Guide");
        s.game_mut().state.players[0].mana_pool =
            Mana::restricted_to(restriction.clone(), [0, 0, 0, 0, 0, 1]);
        s.game_mut().scenario_refresh_priority().unwrap();
        let offered =
            s.action_space().actions.iter().any(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == guide),
            );
        assert_eq!(offered, activates, "{}", restriction.describe());
        if activates {
            command(
                &mut s,
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == guide),
            );
            assert_eq!(s.game().state.players[0].mana_pool, Mana::default());
            command(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("W"));
        }
    }
}
