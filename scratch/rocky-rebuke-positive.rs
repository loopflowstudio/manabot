#[test]
fn rocky_rebuke_uses_creature_power_and_preserves_damage_source() {
    let mut s = setup(&["Rocky Rebuke", "Healer's Hawk", "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let before = s.life(0);
    cast(&mut s, "Rocky Rebuke");
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a,Action::ChooseTarget{target:Target::Permanent(p),..} if *p==victim)));
    assert!(s.choose_target(Target::Permanent(source)));
    assert!(s.choose_target(Target::Permanent(victim)));
    settle(&mut s);
    assert_eq!(s.life(0), before + 1);
    assert_eq!(
        s.game().state.permanents[victim].as_ref().unwrap().damage,
        1
    );
}

