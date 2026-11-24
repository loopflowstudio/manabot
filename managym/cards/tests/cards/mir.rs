//! Mirage card behavior.

#[test]
fn foundations_5d_pacifism() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, permanent_named};
    use crate::support::play::{cast, settle};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::flow::turn::StepKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Pacifism", "Gray Ogre", "Hill Giant"]);
    let target = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let free = s.force_permanent_on_battlefield(1, "Hill Giant");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    for id in [ours, target, free] {
        crate::support::game::clear_summoning_sickness(&mut s, id);
    }
    cast(&mut s, "Pacifism");
    assert!(s.choose_target(Target::Permanent(target)));
    settle(&mut s);
    let aura = permanent_named(&s, "Pacifism");
    let source = card_of(&s, aura);
    let mut removed = s.clone();
    move_card(&mut removed, source, ZoneType::Graveyard);
    for (branch, allowed) in [(&mut s, false), (&mut removed, true)] {
        branch.advance_to_active_step(0, StepKind::DeclareAttackers);
        branch.declare_attack();
        branch.advance_until(
            |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
            "blockers".into(),
        );
        let mut blockers = Vec::new();
        while branch.action_space().kind == ActionSpaceKind::DeclareBlocker {
            for action in &branch.action_space().actions {
                if let Action::DeclareBlocker {
                    blocker,
                    attacker: Some(_),
                    ..
                } = action
                {
                    blockers.push(*blocker);
                }
            }
            branch.decline_block();
        }
        assert!(blockers.contains(&free));
        assert_eq!(blockers.contains(&target), allowed);
        branch.advance_until(
            |s| s.active_player() == 1 && s.action_space().kind == ActionSpaceKind::DeclareAttacker,
            "attacker declaration".into(),
        );
        let mut attackers = Vec::new();
        while branch.action_space().kind == ActionSpaceKind::DeclareAttacker {
            for action in &branch.action_space().actions {
                if let Action::DeclareAttacker {
                    permanent,
                    attack: true,
                    ..
                } = action
                {
                    attackers.push(*permanent);
                }
            }
            branch.decline_attack();
        }
        assert!(attackers.contains(&free));
        assert_eq!(attackers.contains(&target), allowed);
    }
}
