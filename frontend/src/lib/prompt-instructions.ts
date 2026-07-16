// Client-authored instructions for the engine's ActionSpaceEnum families.
// The release prompt matrix checks that every reachable family is present.
export const DECISION_PROMPTS: Readonly<Record<string, string>> = {
  PRIORITY: 'Choose a legal action or pass priority.',
  DECLARE_ATTACKER: 'Declare attackers — choose an attack or finish declaring attackers.',
  DECLARE_BLOCKER: 'Declare blockers — choose a block or finish declaring blockers.',
  SCRY: 'Scry — keep each card on top or put it on the bottom.',
  LOOK_AND_SELECT: 'Look at the revealed cards — choose what to take.',
  PAY_OR_NOT: 'Optional cost — pay it or decline.',
  MODAL: 'Choose a mode.',
  DISCARD_THEN_DRAW: 'Learn — discard a card to draw a card, or keep your hand.',
  WATERBEND: 'Waterbend — tap permanents to help pay the cost.',
  CHOOSE_TARGET: 'Choose a target.',
};
