//! Complete block legality shared by sequential supports and draft validation.
//!
//! Pair eligibility is immutable during declaration. Counts couple menace and
//! maximum-one restrictions; lure requirements maximize the number satisfied
//! among legal full plans (CR 509.1b–c), including any committed prefix.

use crate::{
    agent::action::{Action, AgentError, CapacityError},
    flow::game::Game,
    state::{
        card::{Keyword, Restriction},
        game_object::PermanentId,
    },
};

pub(crate) const COMBAT_SOLVER_LIMIT: usize = 65_536;

/// `None` is editable; `Some(None)` is a fixed decline.
type Assignment = Option<Option<usize>>;

#[derive(Clone, Copy, Default)]
struct BlockRestrictions {
    menace: bool,
    maximum_one: bool,
    lure: bool,
}

impl BlockRestrictions {
    fn for_attacker(game: &Game, attacker: PermanentId) -> Self {
        Self {
            menace: game.characteristics(attacker).keywords.has(Keyword::Menace),
            maximum_one: game.has_restriction(attacker, Restriction::CantBeBlockedByMoreThanOne),
            lure: game.must_be_blocked_by_all_able(attacker),
        }
    }

    fn coupled(self) -> bool {
        self.menace || self.maximum_one || self.lure
    }
}

pub(crate) struct BlockProblem {
    pub(crate) blockers: Vec<PermanentId>,
    pub(crate) attackers: Vec<PermanentId>,
    supports: Vec<Vec<usize>>,
    restrictions: Vec<BlockRestrictions>,
}

struct Work {
    limit: usize,
    visited: usize,
}

impl Work {
    fn visit(&mut self) -> Result<(), AgentError> {
        self.visited += 1;
        if self.visited > self.limit {
            return Err(AgentError::Capacity(CapacityError {
                resource: "combat_solver",
                limit: self.limit,
                visited: self.visited,
            }));
        }
        Ok(())
    }
}

struct Search<'a> {
    problem: &'a BlockProblem,
    assignments: &'a [Assignment],
    choices: Vec<Vec<Option<usize>>>,
    next: Vec<usize>,
    chosen: Vec<Option<usize>>,
    counts: Vec<usize>,
    row: usize,
    score: usize,
    target: Option<usize>,
    ceiling: usize,
    best: Option<usize>,
    work: &'a mut Work,
}

impl Search<'_> {
    fn upper(&self) -> usize {
        self.score
            + (self.row..self.problem.blockers.len())
                .filter(|index| match self.assignments[*index] {
                    Some(Some(attacker)) => self.problem.restrictions[attacker].lure,
                    Some(None) => false,
                    None => self.problem.supports[*index]
                        .iter()
                        .any(|attacker| self.problem.restrictions[*attacker].lure),
                })
                .count()
    }

    fn can_complete(&self) -> bool {
        let upper = self.upper();
        if self.target.is_some_and(|target| upper < target)
            || self.best.is_some_and(|best| upper <= best)
        {
            return false;
        }
        // A singleton must still have a possible second blocker. Maximum-one
        // plus menace permits only zero, so reject it without enumerating.
        self.counts.iter().enumerate().all(|(attacker, count)| {
            !self.problem.restrictions[attacker].menace
                || *count != 1
                || !self.problem.restrictions[attacker].maximum_one
                    && (self.row..self.problem.blockers.len()).any(|index| {
                        self.problem.supports[index].contains(&attacker)
                            && self.assignments[index].is_none_or(|fixed| fixed == Some(attacker))
                    })
        })
    }

    fn backtrack(&mut self) -> bool {
        if self.row == 0 {
            return false;
        }
        self.row -= 1;
        if let Some(attacker) = self.chosen[self.row] {
            self.counts[attacker] -= 1;
            self.score -= usize::from(self.problem.restrictions[attacker].lure);
        }
        true
    }

    /// An explicit DFS stack keeps large declarations off the native call
    /// stack. Only newly entered nodes consume the deterministic work budget.
    fn run(&mut self) -> Result<(), AgentError> {
        let mut entering = true;
        loop {
            if entering {
                self.work.visit()?;
                if !self.can_complete() {
                    if !self.backtrack() {
                        return Ok(());
                    }
                    entering = false;
                    continue;
                }
                if self.row == self.problem.blockers.len() {
                    if self.target.is_none_or(|target| self.score >= target) {
                        self.best = Some(self.score);
                        if self.target.is_some() || self.score == self.ceiling {
                            return Ok(());
                        }
                    }
                    if !self.backtrack() {
                        return Ok(());
                    }
                    entering = false;
                    continue;
                }
            }
            if self.next[self.row] == self.choices[self.row].len() {
                if !self.backtrack() {
                    return Ok(());
                }
                entering = false;
                continue;
            }
            let attacker = self.choices[self.row][self.next[self.row]];
            self.next[self.row] += 1;
            if let Some(attacker) = attacker {
                if !self.problem.supports[self.row].contains(&attacker)
                    || self.problem.restrictions[attacker].maximum_one && self.counts[attacker] != 0
                {
                    entering = false;
                    continue;
                }
                self.counts[attacker] += 1;
                self.score += usize::from(self.problem.restrictions[attacker].lure);
            }
            self.chosen[self.row] = attacker;
            self.row += 1;
            if self.row < self.problem.blockers.len() {
                self.next[self.row] = 0;
            }
            entering = true;
        }
    }
}

impl BlockProblem {
    pub(crate) fn from_game(game: &Game) -> Result<Self, AgentError> {
        let _scan = game.board_scan();
        let combat = game
            .state
            .combat
            .as_ref()
            .ok_or_else(|| AgentError::Rules("no combat declaration".into()))?;
        let blockers = game.eligible_blockers(game.non_active_player());
        let attackers: Vec<_> = combat
            .attackers
            .iter()
            .copied()
            .filter(|attacker| game.permanent_object(*attacker).is_some())
            .collect();
        let supports = blockers
            .iter()
            .map(|blocker| {
                attackers
                    .iter()
                    .enumerate()
                    .filter_map(|(index, attacker)| {
                        game.blocker_pair_eligible(*blocker, *attacker)
                            .then_some(index)
                    })
                    .collect()
            })
            .collect();
        let restrictions = attackers
            .iter()
            .map(|attacker| BlockRestrictions::for_attacker(game, *attacker))
            .collect();
        Ok(Self {
            blockers,
            attackers,
            supports,
            restrictions,
        })
    }

    /// Recover fixed blocks and declines from the actual queue, never a browser
    /// plan. `current` has already been popped by progression (or is published).
    fn prefix(&self, game: &Game, current: Option<PermanentId>) -> Vec<Assignment> {
        let combat = game.state.combat.as_ref().expect("problem has combat");
        self.blockers
            .iter()
            .map(|blocker| {
                if Some(*blocker) == current || combat.blockers_to_declare.contains(blocker) {
                    None
                } else {
                    Some(
                        combat
                            .attacker_to_blockers
                            .iter()
                            .find_map(|(attacker, blockers)| {
                                blockers
                                    .contains(blocker)
                                    .then(|| self.attackers.iter().position(|id| id == attacker))
                                    .flatten()
                            }),
                    )
                }
            })
            .collect()
    }

    /// Exact optimum/feasibility search; no optimum assignment escapes this
    /// authority. A bound prunes scores that cannot improve the best result.
    fn solve(
        &self,
        assignments: &[Assignment],
        target: Option<usize>,
        work: &mut Work,
    ) -> Result<Option<usize>, AgentError> {
        let choices = assignments
            .iter()
            .enumerate()
            .map(|(row, fixed)| {
                if let Some(fixed) = fixed {
                    return vec![*fixed];
                }
                // Required choices first reach the all-requirements bound early;
                // no chosen plan ever escapes the solver.
                [true, false]
                    .into_iter()
                    .flat_map(|required| {
                        self.supports[row]
                            .iter()
                            .copied()
                            .filter(move |attacker| self.restrictions[*attacker].lure == required)
                            .map(Some)
                    })
                    .chain(std::iter::once(None))
                    .collect()
            })
            .collect();
        let mut search = Search {
            problem: self,
            assignments,
            choices,
            next: vec![0; self.blockers.len()],
            chosen: vec![None; self.blockers.len()],
            counts: vec![0; self.attackers.len()],
            row: 0,
            score: 0,
            target,
            best: None,
            work,
            ceiling: 0,
        };
        search.ceiling = search.upper();
        search.run()?;
        Ok(search.best)
    }

    pub(crate) fn validate(
        &self,
        game: &Game,
        current: Option<PermanentId>,
        submitted: &[(PermanentId, Option<PermanentId>)],
    ) -> Result<(), AgentError> {
        let mut constraints = self.prefix(game, current);
        for (blocker, attacker) in submitted {
            let row = self
                .blockers
                .iter()
                .position(|id| id == blocker)
                .ok_or_else(|| AgentError::Rules("submitted blocker is not eligible".into()))?;
            if constraints[row].is_some() {
                return Err(AgentError::Rules(
                    "duplicate or already committed blocker".into(),
                ));
            }
            constraints[row] = Some(match attacker {
                None => None,
                Some(attacker) => Some(
                    self.attackers
                        .iter()
                        .position(|id| id == attacker)
                        .ok_or_else(|| {
                            AgentError::Rules("submitted attacker is not in combat".into())
                        })?,
                ),
            });
        }
        if constraints.iter().any(Option::is_none) {
            return Err(AgentError::Rules(
                "block declaration omits a pending blocker".into(),
            ));
        }
        if !self
            .restrictions
            .iter()
            .any(|restriction| restriction.coupled())
        {
            if constraints
                .iter()
                .enumerate()
                .all(|(row, assignment)| match assignment {
                    Some(Some(attacker)) => self.supports[row].contains(attacker),
                    Some(None) => true,
                    None => false,
                })
            {
                return Ok(());
            }
            return Err(AgentError::Rules(
                "block declaration contains an ineligible pair".into(),
            ));
        }
        let mut work = Work {
            limit: game.combat_solver_limit,
            visited: 0,
        };
        let optimum = self
            .solve(&vec![None; self.blockers.len()], None, &mut work)?
            .ok_or_else(|| AgentError::Rules("no legal complete block declaration".into()))?;
        if self
            .solve(&constraints, Some(optimum), &mut work)?
            .is_none()
        {
            return Err(AgentError::Rules(
                "block declaration violates pair eligibility, menace, maximum-one or required blocks".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn actions(
        &self,
        game: &Game,
        current: PermanentId,
    ) -> Result<Vec<Action>, AgentError> {
        let row = self
            .blockers
            .iter()
            .position(|id| *id == current)
            .ok_or_else(|| AgentError::Rules("current blocker is not eligible".into()))?;
        let mut work = Work {
            limit: game.combat_solver_limit,
            visited: 0,
        };
        let optimum = self
            .solve(&vec![None; self.blockers.len()], None, &mut work)?
            .ok_or_else(|| AgentError::Rules("no legal full block declaration".into()))?;
        let mut prefix = self.prefix(game, Some(current));
        let mut actions = Vec::new();
        for attacker in self.supports[row]
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::once(None))
        {
            prefix[row] = Some(attacker);
            if self.solve(&prefix, Some(optimum), &mut work)?.is_some() {
                actions.push(Action::DeclareBlocker {
                    player: game.non_active_player(),
                    blocker: current,
                    attacker: attacker.map(|index| self.attackers[index]),
                });
            }
        }
        if actions.is_empty() {
            return Err(AgentError::Rules(
                "committed block prefix has no legal completion".into(),
            ));
        }
        Ok(actions)
    }
}

impl Game {
    pub(crate) fn block_declaration_actions(
        &self,
        blocker: PermanentId,
    ) -> Result<Vec<Action>, AgentError> {
        let _scan = self.board_scan();
        let combat = self
            .state
            .combat
            .as_ref()
            .ok_or_else(|| AgentError::Rules("no combat declaration".into()))?;
        let coupled = combat
            .attackers
            .iter()
            .copied()
            .filter(|attacker| self.permanent_object(*attacker).is_some())
            .any(|attacker| BlockRestrictions::for_attacker(self, attacker).coupled());
        if coupled {
            return BlockProblem::from_game(self)?.actions(self, blocker);
        }
        // Independent blocks need no whole-board assignment model per subject.
        // Complete validation still checks the final declaration at its boundary.
        Ok(combat
            .attackers
            .iter()
            .copied()
            .filter(|attacker| self.blocker_pair_eligible(blocker, *attacker))
            .map(Some)
            .chain(std::iter::once(None))
            .map(|attacker| Action::DeclareBlocker {
                player: self.non_active_player(),
                blocker,
                attacker,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{game_object::PlayerId, player::PlayerConfig};
    use std::collections::BTreeMap;

    fn problem() -> BlockProblem {
        let deck = BTreeMap::from([("Mountain".into(), 24), ("Gray Ogre".into(), 16)]);
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck.clone()),
                PlayerConfig::new("b", deck),
            ],
            22,
            false,
        )
        .unwrap();
        let attackers = (0..2)
            .map(|_| {
                game.scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
                    .unwrap()
            })
            .collect();
        let blockers = (0..3)
            .map(|_| {
                game.scenario_force_battlefield(PlayerId(1), "Gray Ogre", true)
                    .unwrap()
            })
            .collect();
        BlockProblem {
            attackers,
            blockers,
            supports: vec![vec![0, 1]; 3],
            restrictions: vec![BlockRestrictions::default(); 2],
        }
    }

    #[test]
    fn combat_solver_matches_small_exhaustive_oracle() {
        let mut problem = problem();
        for supports in 0..64 {
            for menace in 0..4 {
                for maximum_one in 0..4 {
                    for lure in 0..4 {
                        problem.supports = (0..3)
                            .map(|row| {
                                (0..2)
                                    .filter(|column| supports & (1 << (row * 2 + column)) != 0)
                                    .collect()
                            })
                            .collect();
                        problem.restrictions = (0..2)
                            .map(|column| BlockRestrictions {
                                menace: menace & (1 << column) != 0,
                                maximum_one: maximum_one & (1 << column) != 0,
                                lure: lure & (1 << column) != 0,
                            })
                            .collect();
                        let mut legal_plans = Vec::new();
                        let mut expected = 0;
                        for plan in 0..27 {
                            let mut remaining = plan;
                            let mut counts = [0; 2];
                            let mut score = 0;
                            let mut eligible = true;
                            let mut assignments = [None; 3];
                            for (row, assignment) in assignments.iter_mut().enumerate() {
                                let choice = remaining % 3;
                                remaining /= 3;
                                if choice != 2 {
                                    *assignment = Some(choice);
                                    eligible &= problem.supports[row].contains(&choice);
                                    counts[choice] += 1;
                                    score += usize::from(problem.restrictions[choice].lure);
                                }
                            }
                            let legal = eligible
                                && (0..2).all(|column| {
                                    (!problem.restrictions[column].menace || counts[column] != 1)
                                        && (!problem.restrictions[column].maximum_one
                                            || counts[column] <= 1)
                                });
                            if legal {
                                expected = expected.max(score);
                                legal_plans.push((assignments, score));
                            }
                        }
                        let actual = problem
                            .solve(
                                &[None; 3],
                                None,
                                &mut Work {
                                    limit: COMBAT_SOLVER_LIMIT,
                                    visited: 0,
                                },
                            )
                            .unwrap();
                        assert_eq!(actual, Some(expected), "supports {supports}, menace {menace}, max-one {maximum_one}, lure {lure}");
                        // Native declarations fix a suffix of the board-order
                        // blockers because the queue is popped from the back.
                        // Check every one- and two-choice prefix, including
                        // declines and prefixes with no optimal completion.
                        for length in 1..=2 {
                            for prefix in 0..3_usize.pow(length) {
                                let mut constraints = [None; 3];
                                let mut remaining = prefix;
                                for row in (3 - length as usize..3).rev() {
                                    let choice = remaining % 3;
                                    remaining /= 3;
                                    constraints[row] = Some((choice != 2).then_some(choice));
                                }
                                let feasible = legal_plans.iter().any(|(plan, score)| {
                                    *score == expected
                                        && constraints.iter().zip(plan).all(|(fixed, choice)| {
                                            fixed.is_none_or(|fixed| fixed == *choice)
                                        })
                                });
                                let actual = problem
                                    .solve(
                                        &constraints,
                                        Some(expected),
                                        &mut Work {
                                            limit: COMBAT_SOLVER_LIMIT,
                                            visited: 0,
                                        },
                                    )
                                    .unwrap();
                                assert_eq!(actual.is_some(), feasible, "supports {supports}, menace {menace}, max-one {maximum_one}, lure {lure}, prefix {constraints:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn combat_solver_couples_two_lures_with_menace_and_fixed_prefixes() {
        let mut problem = problem();
        problem.blockers.pop();
        problem.supports.pop();
        problem.restrictions.fill(BlockRestrictions {
            menace: true,
            lure: true,
            maximum_one: false,
        });
        let solve = |prefix: &[Assignment]| {
            problem
                .solve(
                    prefix,
                    Some(2),
                    &mut Work {
                        limit: COMBAT_SOLVER_LIMIT,
                        visited: 0,
                    },
                )
                .unwrap()
        };
        assert_eq!(solve(&[Some(Some(0)), Some(Some(1))]), None);
        assert_eq!(solve(&[Some(Some(0)), None]), Some(2));
        assert_eq!(solve(&[Some(None), None]), None);
        problem.blockers.pop();
        problem.supports.pop();
        assert_eq!(
            problem
                .solve(
                    &[None],
                    None,
                    &mut Work {
                        limit: COMBAT_SOLVER_LIMIT,
                        visited: 0
                    }
                )
                .unwrap(),
            Some(0)
        );
    }

    #[test]
    fn combat_solver_budget_is_a_typed_failure() {
        let problem = problem();
        let error = problem
            .solve(
                &[None; 3],
                None,
                &mut Work {
                    limit: 1,
                    visited: 0,
                },
            )
            .unwrap_err();
        assert_eq!(
            error,
            AgentError::Capacity(CapacityError {
                resource: "combat_solver",
                limit: 1,
                visited: 2
            })
        );
    }
}

/// A constructed declaration whose last child reaches a coupled blocker scan.
#[cfg(test)]
pub(crate) fn capacity_attack_root() -> Game {
    use crate::{
        flow::turn::PhaseKind,
        state::{game_object::PlayerId, player::PlayerConfig},
    };
    use std::collections::BTreeMap;
    let attack_deck = BTreeMap::from([("Mountain".into(), 24), ("Boggart Brute".into(), 16)]);
    let block_deck = BTreeMap::from([("Mountain".into(), 24), ("Gray Ogre".into(), 16)]);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("a", attack_deck),
            PlayerConfig::new("b", block_deck),
        ],
        364,
        false,
    )
    .unwrap();
    game.scenario_force_battlefield(PlayerId(0), "Boggart Brute", true)
        .unwrap();
    for _ in 0..2 {
        game.scenario_force_battlefield(PlayerId(1), "Gray Ogre", true)
            .unwrap();
    }
    game.state.turn.current_phase = PhaseKind::Combat as usize;
    game.state.turn.current_step = 1;
    game.state.turn.step_initialized = false;
    game.tick().unwrap();
    game.skip_trivial = true;
    game.combat_solver_limit = 1;
    game
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    use crate::agent::structured_offer::{
        ChoiceAnswer, ChoiceStep, OfferSubmission, StructuredOfferError,
    };

    #[test]
    fn combat_capacity_quarantines_scalar_and_search_without_terminal_success() {
        for mode in ["scalar", "search", "semantic"] {
            let mut game = capacity_attack_root();
            let result = match mode {
                "search" => {
                    let offers = game.structured_search_offers().unwrap();
                    game.apply_offer_submission(
                        &offers,
                        &OfferSubmission {
                            offer_id: offers.projection().offers[0].id,
                            answers: vec![],
                        },
                    )
                    .map(|_| ())
                    .map_err(AgentError::from)
                }
                "semantic" => {
                    let frame = game.semantic_decision_frame().unwrap();
                    game.execute_semantic_command(&crate::decision::Command {
                        command_id: "capacity-proof".into(),
                        expected_revision: frame.revision,
                        offer_id: frame.offers[0].id.0,
                        answers: vec![],
                        object_preconditions: vec![],
                    })
                    .map(|_| ())
                    .map_err(AgentError::from)
                }
                _ => game.step(0).map(|_| ()),
            };
            let error = result.unwrap_err();
            assert_eq!(error.capacity().unwrap().resource, "combat_solver");
            assert!(!game.is_game_over());
            assert!(game.action_space().is_none());
            assert_eq!(game.step(0).unwrap_err(), error);
            assert_eq!(game.play().unwrap_err(), error);
            assert_eq!(
                crate::agent::rollout_pool::RolloutPool::from_game(&game, 1, 1, 7, 1).unwrap_err(),
                error
            );
            assert_eq!(game.random_playout(1, None).unwrap_err(), error);
            let mut env = crate::agent::env::Env::new(364, true, false, false);
            env.fixture_set_game(game.clone());
            assert_eq!(env.semantic_observation(0).unwrap_err(), error);
            assert_eq!(env.semantic_viewer_state_hash(0).unwrap_err(), error);
            assert_eq!(env.capture_decision(0, 0).unwrap_err(), error);

            assert!(matches!(
                game.structured_search_offers(),
                Err(StructuredOfferError::Execution(AgentError::Capacity(_)))
            ));
        }
    }

    #[test]
    fn combat_capacity_rolls_back_a_declaration_final_child_continuation() {
        let game = capacity_attack_root();
        let offers = game.structured_attacker_offers().unwrap();
        let offer = &offers.projection().offers[0];
        let ChoiceStep::Select {
            role, candidates, ..
        } = &offer.choices[0];
        let submission = OfferSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role: *role,
                candidates: candidates
                    .initial
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|candidate| candidate.id)
                    .collect(),
            }],
        };
        let before = crate::search_state::witness(&game);
        let error = match game.prepare_declaration(&offers.decode(&submission).unwrap()) {
            Ok(_) => panic!("expected capacity failure"),
            Err(error) => AgentError::from(error),
        };
        assert_eq!(error.capacity().unwrap().limit, 1);
        assert_eq!(crate::search_state::witness(&game), before);
        assert!(game.ensure_playable().is_ok());
    }

    #[test]
    fn combat_capacity_readonly_block_validation_preserves_the_root() {
        let mut game = capacity_attack_root();
        game.combat_solver_limit = COMBAT_SOLVER_LIMIT;
        game.step(0).unwrap();
        assert_eq!(
            game.action_space().unwrap().kind,
            crate::agent::action::ActionSpaceKind::DeclareBlocker
        );
        game.combat_solver_limit = 1;
        let offers = game.compound_offers().unwrap();
        let offer = &offers.projection().offers[0];
        let submission = OfferSubmission {
            offer_id: offer.id,
            answers: offer
                .choices
                .iter()
                .map(|choice| {
                    let ChoiceStep::Select {
                        role, candidates, ..
                    } = choice;
                    ChoiceAnswer::Candidates {
                        role: *role,
                        candidates: vec![candidates.initial.as_ref().unwrap()[0].id],
                    }
                })
                .collect(),
        };
        let before = crate::search_state::witness(&game);
        let error = AgentError::from(
            game.prepare_declaration(&offers.decode(&submission).unwrap())
                .err()
                .unwrap(),
        );
        assert_eq!(error.capacity().unwrap().resource, "combat_solver");
        assert_eq!(crate::search_state::witness(&game), before);
        assert!(game.ensure_playable().is_ok());
    }

    #[test]
    fn combat_capacity_rollout_pool_propagates_a_branch_failure_without_scoring() {
        let game = capacity_attack_root();
        let before = crate::search_state::witness(&game);
        let error =
            crate::agent::rollout_pool::RolloutPool::from_game(&game, 1, 1, 7, 1).unwrap_err();
        assert_eq!(error.capacity().unwrap().resource, "combat_solver");
        assert_eq!(crate::search_state::witness(&game), before);
    }
}
