//! Legal prefixes of a simultaneous block declaration (CR 509.1b–c).

use super::game::Game;
use crate::state::{
    card::{Keyword, Restriction},
    effect::Modification,
    game_object::PermanentId,
};

#[derive(Default)]
struct BlockRequirements {
    first: usize,
    each: usize,
    menace: bool,
    limited: bool,
}

impl Game {
    fn block_requirements(&self, attacker: PermanentId) -> BlockRequirements {
        let mut rules = BlockRequirements {
            menace: self.layer_view(attacker).keywords.has(Keyword::Menace),
            limited: self.has_restriction(attacker, Restriction::CantBeBlockedByMoreThanOne),
            ..Default::default()
        };
        for effect in self.effects_on(attacker) {
            match effect.modification {
                Modification::MustBeBlocked => rules.first += 1,
                Modification::MustBeBlockedByAllAble => rules.each += 1,
                _ => {}
            }
        }
        rules
    }

    pub(crate) fn blocks_are_coupled(&self, attacker: PermanentId) -> bool {
        let r = self.block_requirements(attacker);
        r.first + r.each > 0 || r.menace || r.limited
    }

    pub(crate) fn block_actions(
        &self,
        player: crate::state::game_object::PlayerId,
        blocker: PermanentId,
    ) -> Vec<crate::agent::action::Action> {
        let remaining = &self
            .state
            .combat
            .as_ref()
            .expect("declaring blockers")
            .blockers_to_declare;
        self.block_options(blocker, remaining)
            .into_iter()
            .map(|attacker| crate::agent::action::Action::DeclareBlocker {
                player,
                blocker,
                attacker,
            })
            .collect()
    }

    /// All choices for this blocker that extend to a legal declaration.
    /// Preview may ask about a later blocker first, with the others still free.
    pub(crate) fn block_options(
        &self,
        blocker: PermanentId,
        remaining: &[PermanentId],
    ) -> Vec<Option<PermanentId>> {
        let _scan = self.board_scan();
        let combat = self.state.combat.as_ref().expect("declaring blockers");
        let attackers = &combat.attackers;
        let options = attackers
            .iter()
            .enumerate()
            .filter(|(_, &a)| self.blocker_can_block_attacker(blocker, a))
            .map(|(i, _)| Some(i))
            .chain([None]);
        let rules: Vec<_> = attackers
            .iter()
            .map(|&id| self.block_requirements(id))
            .collect();
        if rules.iter().all(|r| !r.menace && r.first + r.each == 0) {
            return options.map(|choice| choice.map(|i| attackers[i])).collect();
        }
        let mut counts: Vec<_> = attackers
            .iter()
            .map(|a| combat.attacker_to_blockers.get(a).map_or(0, Vec::len))
            .collect();
        let edges: Vec<Vec<usize>> = attackers
            .iter()
            .map(|&a| {
                remaining
                    .iter()
                    .enumerate()
                    .filter(|(_, &b)| self.blocker_can_block_attacker(b, a))
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        let mut best = None;
        let mut legal = Vec::new();
        for choice in options {
            if let Some(i) = choice {
                counts[i] += 1;
            }
            let bound = block_requirement_bound(
                &rules,
                &edges,
                remaining.len(),
                &counts,
                &mut vec![None; rules.len()],
            );
            if let Some(i) = choice {
                counts[i] -= 1;
            }
            if bound.is_some() && bound >= best {
                if bound > best {
                    legal.clear();
                    best = bound;
                }
                legal.push(choice.map(|i| attackers[i]));
            }
        }
        legal
    }
}

/// Match remaining blockers to slots on attackers. The first slot earns all
/// "must be blocked" requirements and every slot earns all lure requirements.
/// Slots matchable to distinct blockers form a transversal matroid: taking
/// the highest-value augmentable slot first maximizes total reward. Mandatory
/// slots (a menace attacker already committed to being blocked) go first.
///
/// A relaxed optimum with exactly one blocker on an undecided menace attacker
/// bounds both legal alternatives: leave it open, or insist on at least two.
/// Only those ambiguous attackers branch; ordinary declarations use one match.
fn block_requirement_bound(
    rules: &[BlockRequirements],
    edges: &[Vec<usize>],
    blockers: usize,
    counts: &[usize],
    blocked: &mut [Option<bool>],
) -> Option<usize> {
    let mut slots = Vec::new();
    let mut score = 0;
    for (i, r) in rules.iter().enumerate() {
        let n = counts[i];
        score += 2 * (usize::from(n > 0) * r.first + n * r.each);
        if blocked[i] == Some(false) {
            continue;
        }
        let required = if r.menace && (n > 0 || blocked[i] == Some(true)) {
            2usize.saturating_sub(n)
        } else {
            0
        };
        let capacity = if r.limited {
            (1usize.saturating_sub(n)).min(edges[i].len())
        } else {
            edges[i].len()
        };
        if required > capacity {
            return None;
        }
        if r.menace && n + capacity < 2 {
            continue;
        }
        // Score in halves: a menace pair earns the first-block bonus together.
        // The relaxation can use a half, but a legal completion scores whole
        // requirements. Equal-weight slots on one attacker sort together.
        for slot in 0..capacity {
            let first = if r.menace {
                usize::from(n == 0 && slot < 2) * r.first
            } else {
                2 * usize::from(n + slot == 0) * r.first
            };
            let reward = 2 * r.each + first;
            if reward == 0 && slot >= required {
                break;
            }
            slots.push((slot < required, reward, i));
        }
    }
    slots.sort_unstable_by(|a, b| b.cmp(a));
    let mut assigned = vec![None; blockers];
    let mut total = counts.to_vec();
    let mut seen = vec![false; blockers];
    for (required, reward, attacker) in slots {
        seen.fill(false);
        if augment(attacker, edges, &mut assigned, &mut seen) {
            total[attacker] += 1;
            score += reward;
        } else if required {
            return None;
        }
    }
    if let Some(i) = rules
        .iter()
        .enumerate()
        .position(|(i, r)| r.menace && total[i] == 1)
    {
        debug_assert_eq!(counts[i], 0);
        blocked[i] = Some(true);
        let mut best = block_requirement_bound(rules, edges, blockers, counts, blocked);
        if best != Some(score / 2) {
            blocked[i] = Some(false);
            best = best.max(block_requirement_bound(
                rules, edges, blockers, counts, blocked,
            ));
        }
        blocked[i] = None;
        best
    } else {
        Some(score / 2)
    }
}

fn augment(
    attacker: usize,
    edges: &[Vec<usize>],
    assigned: &mut [Option<usize>],
    seen: &mut [bool],
) -> bool {
    for &blocker in &edges[attacker] {
        if std::mem::replace(&mut seen[blocker], true) {
            continue;
        }
        if assigned[blocker].is_none_or(|other| augment(other, edges, assigned, seen)) {
            assigned[blocker] = Some(attacker);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaust all two-attacker / three-blocker edge sets and declarations,
    /// including fixed prefixes. Count each overlapping requirement separately.
    #[test]
    fn matching_agrees_with_all_small_block_declarations() {
        fn enumerate(
            rules: &[BlockRequirements],
            edges: &[Vec<usize>],
            counts: &mut [usize],
            left: usize,
        ) -> Option<usize> {
            if left == 0 {
                return rules.iter().zip(counts.iter()).try_fold(0, |sum, (r, &n)| {
                    (!(r.menace && n == 1 || r.limited && n > 1))
                        .then_some(sum + usize::from(n > 0) * r.first + n * r.each)
                });
            }
            let mut best = enumerate(rules, edges, counts, left - 1);
            for i in 0..rules.len() {
                if edges[i].contains(&(left - 1)) {
                    counts[i] += 1;
                    best = best.max(enumerate(rules, edges, counts, left - 1));
                    counts[i] -= 1;
                }
            }
            best
        }
        for mask in 0..64 {
            let edges: Vec<Vec<_>> = (0..2)
                .map(|i| (0..3).filter(|b| mask & (1 << (3 * i + b)) != 0).collect())
                .collect();
            for flags in 0..16 {
                let rules: Vec<_> = (0..2)
                    .map(|i| BlockRequirements {
                        first: 2 - i,
                        each: i,
                        menace: flags & (1 << i) != 0,
                        limited: flags & (4 << i) != 0,
                    })
                    .collect();
                for mut counts in [[0, 0], [1, 0], [0, 1], [1, 1]] {
                    let expected = enumerate(&rules, &edges, &mut counts, 3);
                    assert_eq!(
                        block_requirement_bound(&rules, &edges, 3, &counts, &mut [None; 2]),
                        expected,
                        "edges={mask}, rules={flags}, counts={counts:?}"
                    );
                }
            }
        }
    }
}
