//! Deterministic, viewer-safe rule-of-thumb opponent shared by scalar and batch
//! play. Heuristic owns only decision history; Board borrows an Observation,
//! never a Game. Knowledge is compiled from public card definitions.
//!
//! Selection rules:
//! - Mana: play a land, activate fetch lands, then cast the highest-mana-value
//!   spell with a worthwhile use. Let our spell resolve before casting another;
//!   legal offers determine affordability. Pay kicker when untapped lands cover
//!   spell plus kicker.
//! - Instant timing: delay instants in our upkeep and draw step. Untargeted
//!   instants (draw, selection) and draw/life abilities wait for the opponent's
//!   end step, leaving mana for permanents first. Fetch abilities are considered
//!   before spell timing.
//! - Targets: derive polarity from card definitions, with damage, destroy,
//!   bounce, counter and exile harmful; pumps, untaps and keyword grants
//!   beneficial. Aim harmful effects at the opponent's side and beneficial ones
//!   at ours. Hold damage removal until it kills a creature or the opponent,
//!   accounting for estimated kicker mana and graveyard Lessons; hold counters
//!   for an opposing spell on the stack.
//! - Card choices: Learn takes a Lesson, preferring removal while opposing
//!   creatures remain. Discard spare lands once five are in play, otherwise the
//!   costliest card; decline optional discards without a spare land. A
//!   look-and-select favors a land while hand and battlefield together have
//!   fewer than four.
//! - Combat: hold back the toughest attackers while the estimated swing back
//!   is lethal. Avoid an attack if an untapped blocker can kill the attacker
//!   and survive; attack with everything when unblockable damage is lethal.
//!   Plan blocks once per combat, one blocker per attacker: prefer kills that
//!   survive, trades up and free absorption, then chump the largest remaining
//!   attackers while incoming damage is lethal.
//! - Own tricks: an affordable pump in hand can rescue one dominated attacker.
//!   After blocks, cast pumps when they improve a one-on-one fight or make
//!   unblocked damage lethal; activate self-pumps on unblocked attackers.
//!   Hypothetical opposing combat tricks are ignored.
//! - Optional costs: pay offered ward and "unless pays" costs.
//! - Fallback for unhandled prompts: at priority prefer land, spell, then pass;
//!   attack or block when offered; otherwise take the first non-decline offer,
//!   or the last offer when all decline.
//!
//! Ties are deterministic for the same visible state and decision history.
//! Equal-value casts, equal-ranked targets and Lessons use the first legal
//! offer; equal-ranked discard and look-and-select choices use the last.
//! Block pairs rank by outcome, attacker power, lower blocker mana value, then
//! descending blocker and attacker IDs. Pump targets rank by fight improvement,
//! power, then descending ID. Other equal combat rankings keep observation order.
//!
//! These are estimates, not exact combat search. Mana counts untapped lands,
//! ignoring colors and other sources. Creature identity uses visible card
//! definitions (or an animated permanent); without a definition, positive
//! toughness counts as a creature. Modal bodies are opaque to spell ratings.
//! Combat assignments use the observation's attacker/blocker events and current
//! permanent IDs, remembered for the turn. No hidden hand or library order is
//! consulted.
mod knowledge;
use crate::agent::{
    action::{ActionSpaceKind as Space, ActionType as Action, AgentError},
    observation::{ActionOption, CardData, EventType, KeywordData, Observation, PermanentData},
};
use crate::{flow::turn::StepKind, state::zone::ZoneType};
use knowledge::{knowledge, Knowledge, Polarity, Spell};
use std::collections::{BTreeMap, BTreeSet, HashMap};

struct Board<'a> {
    raw: &'a Observation,
    actions: &'a [ActionOption],
    cards: HashMap<i32, &'a CardData>,
    by_key: HashMap<i32, &'a CardData>,
    permanents: HashMap<i32, &'a PermanentData>,
}
impl<'a> Board<'a> {
    fn new(raw: &'a Observation, max_actions: Option<usize>) -> Self {
        let cards = raw.agent_cards.iter().chain(&raw.opponent_cards);
        Self {
            raw,
            actions: &raw.action_space.actions[..max_actions
                .unwrap_or(usize::MAX)
                .min(raw.action_space.actions.len())],
            cards: cards.clone().map(|c| (c.id, c)).collect(),
            by_key: cards.map(|c| (c.registry_key, c)).collect(),
            permanents: raw
                .agent_permanents
                .iter()
                .chain(&raw.opponent_permanents)
                .map(|p| (p.id, p))
                .collect(),
        }
    }
    /// Keep observation order; callers collect only when ranking or reusing a side.
    fn creatures(&self, own: bool) -> impl Iterator<Item = &'a PermanentData> + '_ {
        let side = if own {
            &self.raw.agent_permanents
        } else {
            &self.raw.opponent_permanents
        };
        side.iter().filter(|p| {
            p.is_animated
                || self
                    .by_key
                    .get(&p.registry_key)
                    .map_or(p.toughness > 0, |c| c.card_types.is_creature)
        })
    }
    fn own(&self, p: &PermanentData) -> bool {
        p.controller_id == self.raw.agent.id
    }
    fn own_turn(&self) -> bool {
        self.raw.turn.active_player_id == self.raw.agent.id
    }
    fn mana_value(&self, p: &PermanentData) -> i32 {
        self.by_key
            .get(&p.registry_key)
            .map_or(0, |c| c.mana_cost.mana_value as i32)
    }
    fn lands(&self, untapped: bool) -> i32 {
        self.raw
            .agent_permanents
            .iter()
            .filter(|p| {
                (!untapped || !p.tapped)
                    && self
                        .by_key
                        .get(&p.registry_key)
                        .is_some_and(|c| c.card_types.is_land)
            })
            .count() as i32
    }
    fn can_kick(&self, c: &CardData) -> bool {
        c.kicker_cost > 0 && self.lands(true) >= c.mana_cost.mana_value as i32 + c.kicker_cost
    }
    fn hand(&self) -> impl Iterator<Item = &'a CardData> {
        self.raw
            .agent_cards
            .iter()
            .filter(|c| c.zone == ZoneType::Hand)
    }
    fn spell_on_stack(&self, own: bool) -> bool {
        self.raw.stack_objects.iter().any(|o| {
            o.kind == crate::agent::observation::StackObjectKindData::Spell
                && (o.controller_id == self.raw.agent.id) == own
        })
    }
    fn first(&self, kind: Action) -> Option<usize> {
        self.actions.iter().position(|a| a.action_type == kind)
    }
    fn declared(&self, yes: bool) -> Option<usize> {
        self.actions.iter().position(|a| a.declared == Some(yes))
    }
    /// Card choices prefer the last equal-ranked offer. Callers decide whether
    /// lands are valuable (100) or best avoided (-1) for this decision.
    fn best_card(&self, land_rank: i32) -> Option<(i32, usize)> {
        self.actions
            .iter()
            .enumerate()
            .filter(|(_, a)| a.action_type != Action::DeclineChoice)
            .filter_map(|(i, a)| {
                let card = self.cards.get(a.focus.first()?)?;
                let rank = if card.card_types.is_land {
                    land_rank
                } else {
                    card.mana_cost.mana_value as i32
                };
                Some((rank, i))
            })
            .max()
    }
    fn damage(&self, spell: &Spell, kicked: bool) -> i32 {
        (if kicked {
            spell.kicked_damage
        } else {
            spell.damage
        }) + if spell.lesson_damage {
            self.raw.agent.graveyard_lessons
        } else {
            0
        }
    }
}
#[derive(Clone, Copy)]
struct Body<'a> {
    power: i32,
    toughness: i32,
    damage: i32,
    keywords: &'a KeywordData,
}
impl<'a> Body<'a> {
    fn of(p: &'a PermanentData, pump: (i32, i32)) -> Self {
        Self {
            power: p.power + pump.0,
            toughness: p.toughness + pump.1,
            damage: p.damage,
            keywords: &p.keywords,
        }
    }
    fn dies(self, damage: i32, deathtouch: bool) -> bool {
        damage > 0 && (deathtouch || damage >= self.toughness - self.damage)
    }
    fn first_strike(self) -> bool {
        self.keywords.first_strike || self.keywords.double_strike
    }
}
fn fight(a: Body<'_>, b: Body<'_>) -> (bool, bool) {
    let mut a_dies = a.dies(b.power, b.keywords.deathtouch);
    let mut b_dies = b.dies(a.power, a.keywords.deathtouch);
    if a.first_strike() && !b.first_strike() && b_dies {
        a_dies = false;
    }
    if b.first_strike() && !a.first_strike() && a_dies {
        b_dies = false;
    }
    (a_dies, b_dies)
}
/// Higher is better for us: their creature dying, ours surviving. This
/// approximate fight is symmetric; attacker/blocker order does not affect it.
fn fight_value(ours: &PermanentData, theirs: &PermanentData, pump: (i32, i32)) -> i32 {
    let (ours_dies, theirs_dies) = fight(Body::of(ours, pump), Body::of(theirs, (0, 0)));
    2 * i32::from(theirs_dies) + i32::from(!ours_dies)
}
fn can_block(b: &PermanentData, a: &PermanentData) -> bool {
    !a.cant_be_blocked_this_turn
        && !b.tapped
        && (!a.keywords.flying || b.keywords.flying || b.keywords.reach)
}

/// One player's memory. Env clears it on batch reset; scalar games construct
/// fresh players. Direct callers reset between games, with turn rollback as a
/// backstop.
#[derive(Clone, Debug, Default)]
pub struct Heuristic {
    turn: Option<u32>,
    kicked: HashMap<i32, bool>,
    attack_pool: Option<BTreeSet<i32>>,
    held: BTreeSet<i32>,
    blocks: BTreeMap<i32, BTreeSet<i32>>,
    block_plan: Option<BTreeMap<i32, i32>>,
    trick_reserved: bool,
}
impl Heuristic {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    /// Return an original legal-offer index. A prefix is opt-in for registered
    /// first-legal policies; it never changes the authoritative observation.
    pub fn select_action(
        &mut self,
        obs: &Observation,
        max_actions: Option<usize>,
    ) -> Result<usize, AgentError> {
        if max_actions == Some(0) {
            return Err(AgentError("heuristic max_actions must be positive".into()));
        }
        let b = Board::new(obs, max_actions);
        if b.actions.is_empty() {
            return Err(AgentError("heuristic received no legal offers".into()));
        }
        self.observe(obs);
        let choice = match obs.action_space.action_space_type {
            Space::Priority => self.priority(&b),
            Space::ChooseTarget => self.target(&b),
            Space::DeclareAttacker => self.attack(&b),
            Space::DeclareBlocker => self.block(&b),
            Space::PayOrNot => self.pay(&b),
            Space::Learn => self.learn(&b),
            Space::Discard => self.discard(&b),
            Space::LookAndSelect => self.select(&b),
            _ => None,
        };
        Ok(choice.unwrap_or_else(|| fallback(&b)))
    }
    fn observe(&mut self, obs: &Observation) {
        if self.turn.is_some_and(|turn| obs.turn.turn_number < turn) {
            self.kicked.clear();
        }
        if self.turn != Some(obs.turn.turn_number) {
            self.turn = Some(obs.turn.turn_number);
            self.attack_pool = None;
            self.held.clear();
            self.blocks.clear();
            self.block_plan = None;
            self.trick_reserved = false;
        }
        for e in &obs.recent_events {
            if e.event_type == EventType::BlockersDeclared as i32 {
                self.blocks
                    .entry(e.source_id)
                    .or_default()
                    .insert(e.target_id);
            } else if e.event_type == EventType::CombatAttackersDeclared as i32 {
                self.blocks.entry(e.source_id).or_default();
            }
        }
    }
    fn priority(&self, b: &Board<'_>) -> Option<usize> {
        if let Some(land) = b.first(Action::PriorityPlayLand) {
            return Some(land);
        }
        let passing = b
            .actions
            .iter()
            .rposition(|a| a.action_type == Action::PriorityPassPriority);
        let mut casts = vec![];
        let mut activations = vec![];
        for (i, a) in b.actions.iter().enumerate() {
            if a.action_type == Action::PriorityCastSpell && a.focus.len() == 1 {
                if let Some(c) = b.cards.get(&a.focus[0]) {
                    casts.push((-(c.mana_cost.mana_value as i32), i, *c));
                }
            } else if a.action_type == Action::PriorityActivateAbility {
                if let Some(id) = a.focus.first() {
                    activations.push((i, *id));
                }
            }
        }
        if casts.is_empty() && activations.is_empty() || b.spell_on_stack(true) {
            return passing;
        }
        if let Some(i) = self.ability(b, &activations, |k| k.fetch) {
            return Some(i);
        }
        if b.own_turn() && b.raw.turn.step == StepKind::DeclareBlockers {
            let unblocked: Vec<_> = activations
                .iter()
                .copied()
                .filter(|(_, id)| {
                    self.blocks.get(id).is_some_and(BTreeSet::is_empty)
                        && b.permanents.get(id).is_some_and(|p| p.tapped)
                })
                .collect();
            if let Some(i) = self.ability(b, &unblocked, |k| k.pump) {
                return Some(i);
            }
        }
        casts.sort_by_key(|(mv, i, _)| (*mv, *i));
        for (_, i, c) in casts {
            if self.timely(b, c) && self.worth_casting(b, c) {
                return Some(i);
            }
        }
        if !b.own_turn() && b.raw.turn.step == StepKind::End {
            if let Some(i) = self.ability(b, &activations, |k| k.spare) {
                return Some(i);
            }
        }
        passing
    }
    fn ability(
        &self,
        b: &Board<'_>,
        actions: &[(usize, i32)],
        matches: impl Fn(&Knowledge) -> bool,
    ) -> Option<usize> {
        actions.iter().find_map(|(i, id)| {
            let p = b.permanents.get(id)?;
            let c = b.by_key.get(&p.registry_key)?;
            knowledge(&c.name).filter(|k| matches(k)).map(|_| *i)
        })
    }
    fn timely(&self, b: &Board<'_>, c: &CardData) -> bool {
        let Some(k) = knowledge(&c.name) else {
            return true;
        };
        if !k.instant {
            return true;
        }
        if b.own_turn() && matches!(b.raw.turn.step, StepKind::Upkeep | StepKind::Draw) {
            return false;
        }
        if k.spell.as_ref().is_some_and(|s| s.polarity.is_some()) {
            return true;
        }
        !b.own_turn() && b.raw.turn.step == StepKind::End
    }
    fn worth_casting(&self, b: &Board<'_>, c: &CardData) -> bool {
        let Some(s) = knowledge(&c.name)
            .and_then(|k| k.spell.as_ref())
            .filter(|s| s.polarity.is_some())
        else {
            return true;
        };
        if s.polarity == Some(Polarity::Beneficial) {
            return !self.pump_targets(b, s).is_empty();
        }
        if matches!(s.selector.as_str(), "Spell" | "SpellOrPermanent") {
            return b.spell_on_stack(false);
        }
        let damage = b.damage(s, s.kicked_damage > 0 && b.can_kick(c));
        damage <= 0
            || (s.selector == "AnyTarget" && damage >= b.raw.opponent.life)
            || b.creatures(false)
                .any(|p| Body::of(p, (0, 0)).dies(damage, false))
    }
    fn pay(&mut self, b: &Board<'_>) -> Option<usize> {
        if b.raw.action_space.focus.len() == 1 {
            if let Some(c) = b
                .cards
                .get(&b.raw.action_space.focus[0])
                .filter(|c| c.kicker_cost > 0 && c.owner_id == b.raw.agent.id)
            {
                let kick = b.can_kick(c);
                self.kicked.insert(c.id, kick);
                return b.first(if kick {
                    Action::PayCost
                } else {
                    Action::DeclineChoice
                });
            }
        }
        b.first(Action::PayCost)
    }
    fn learn(&self, b: &Board<'_>) -> Option<usize> {
        let removal_wanted = b.creatures(false).next().is_some();
        let mut best = None;
        for (i, a) in b
            .actions
            .iter()
            .enumerate()
            .filter(|(_, a)| a.action_type == Action::LearnTakeLesson)
        {
            let removal = a
                .focus
                .first()
                .and_then(|id| b.raw.agent_sideboard.iter().find(|c| c.candidate_id == *id))
                .and_then(|c| knowledge(&c.name))
                .and_then(|k| k.spell.as_ref())
                .is_some_and(|s| s.polarity == Some(Polarity::Harmful));
            let rank = if removal == removal_wanted { 2 } else { 1 };
            if best.is_none_or(|(r, _)| rank > r) {
                best = Some((rank, i));
            }
        }
        best.map(|(_, i)| i).or_else(|| self.discard(b))
    }
    fn discard(&self, b: &Board<'_>) -> Option<usize> {
        let decline = b
            .actions
            .iter()
            .rposition(|a| a.action_type == Action::DeclineChoice);
        let best = b.best_card(if b.lands(false) >= 5 { 100 } else { -1 });
        match best {
            Some((r, i)) if decline.is_none() || r >= 100 => Some(i),
            _ => decline,
        }
    }
    fn select(&self, b: &Board<'_>) -> Option<usize> {
        let short = b.lands(false) + (b.hand().filter(|c| c.card_types.is_land).count() as i32) < 4;
        b.best_card(if short { 100 } else { -1 }).map(|(_, i)| i)
    }
    fn pump_targets(&self, b: &Board<'_>, s: &Spell) -> Vec<i32> {
        if b.raw.turn.step != StepKind::DeclareBlockers {
            return vec![];
        }
        let power: i32 = self
            .blocks
            .iter()
            .filter(|(_, bs)| bs.is_empty())
            .filter_map(|(id, _)| b.permanents.get(id))
            .filter(|p| b.own(p))
            .map(|p| p.power)
            .sum();
        let mut gains = vec![];
        for (id, bs) in &self.blocks {
            let Some(a) = b.permanents.get(id) else {
                continue;
            };
            if bs.is_empty() {
                if b.own(a) && power + s.pump.0 >= b.raw.opponent.life {
                    gains.push((3, a.power, *id));
                }
            } else if bs.len() == 1 {
                let Some(blocker) = b.permanents.get(bs.first().unwrap()) else {
                    continue;
                };
                let (ours, theirs) = if b.own(a) {
                    (*a, *blocker)
                } else {
                    (*blocker, *a)
                };
                if !b.own(ours) {
                    continue;
                }
                let before = fight_value(ours, theirs, (0, 0));
                let after = fight_value(ours, theirs, s.pump);
                if after > before {
                    gains.push((after - before, ours.power, ours.id));
                }
            }
        }
        gains.sort_by(|a, b| b.cmp(a));
        gains.into_iter().map(|(_, _, id)| id).collect()
    }
    fn trick_in_hand(&self, b: &Board<'_>) -> Option<&'static Spell> {
        let lands = b.lands(true);
        b.hand().find_map(|c| {
            knowledge(&c.name)?.spell.as_ref().filter(|s| {
                s.polarity == Some(Polarity::Beneficial)
                    && s.pump != (0, 0)
                    && c.mana_cost.mana_value as i32 <= lands
            })
        })
    }
    fn target(&self, b: &Board<'_>) -> Option<usize> {
        let source = b
            .raw
            .action_space
            .focus
            .iter()
            .find_map(|id| b.cards.get(id))?;
        let spell = knowledge(&source.name)?.spell.as_ref()?;
        let polarity = spell.polarity?;
        let damage = b.damage(spell, self.kicked.get(&source.id).copied().unwrap_or(false));
        let pumps = if polarity == Polarity::Beneficial {
            self.pump_targets(b, spell)
        } else {
            vec![]
        };
        let mut best: Option<(Vec<i32>, usize)> = None;
        let mut decline = None;
        for (i, a) in b.actions.iter().enumerate() {
            if a.action_type == Action::DeclineChoice {
                decline = Some(i);
                continue;
            }
            if a.action_type != Action::ChooseTarget || a.focus.len() != 1 {
                continue;
            }
            let id = a.focus[0];
            let own = if id == b.raw.agent.id {
                Some(true)
            } else if id == b.raw.opponent.id {
                Some(false)
            } else {
                b.permanents
                    .get(&id)
                    .map(|p| p.controller_id)
                    .or_else(|| b.cards.get(&id).map(|c| c.owner_id))
                    .map(|c| c == b.raw.agent.id)
            };
            if own != Some(polarity == Polarity::Beneficial) {
                continue;
            }
            let key = if let Some(p) = b.permanents.get(&id) {
                if polarity == Polarity::Beneficial {
                    if let Some(pos) = pumps.iter().position(|p| *p == id) {
                        vec![2, (pumps.len() - pos) as i32]
                    } else {
                        vec![i32::from(self.blocks.contains_key(&id)), p.power]
                    }
                } else {
                    vec![
                        if damage == 0 || Body::of(p, (0, 0)).dies(damage, false) {
                            2
                        } else {
                            0
                        },
                        p.power,
                        b.mana_value(p),
                    ]
                }
            } else if b.cards.contains_key(&id) {
                vec![3]
            } else {
                vec![if damage >= b.raw.opponent.life { 4 } else { 1 }]
            };
            if best.as_ref().is_none_or(|(k, _)| key > *k) {
                best = Some((key, i));
            }
        }
        best.filter(|(k, _)| k[0] > 0 || decline.is_none())
            .map(|(_, i)| i)
            .or(decline)
    }
    fn attack(&mut self, b: &Board<'_>) -> Option<usize> {
        let a = b.permanents.get(b.actions.first()?.focus.first()?)?;
        if self.attack_pool.is_none() {
            self.attack_pool = Some(
                b.creatures(true)
                    .filter(|p| !p.tapped && (!p.is_summoning_sick || p.keywords.haste))
                    .map(|p| p.id)
                    .collect(),
            );
            self.held = self.defenders_to_hold(b);
        }
        let blockers: Vec<_> = b.creatures(false).filter(|p| !p.tapped).collect();
        let pool = self.attack_pool.as_ref().unwrap();
        let unblockable: i32 = b
            .creatures(true)
            .filter(|p| pool.contains(&p.id) && !blockers.iter().any(|bl| can_block(bl, p)))
            .map(|p| p.power)
            .sum();
        if unblockable >= b.raw.opponent.life {
            return b.declared(true);
        }
        if self.held.contains(&a.id) {
            return b.declared(false);
        }
        let mut dominators: Vec<_> = blockers
            .iter()
            .filter(|bl| {
                can_block(bl, a)
                    && fight(Body::of(a, (0, 0)), Body::of(bl, (0, 0))) == (true, false)
            })
            .collect();
        if a.keywords.menace && blockers.len() < 2 {
            dominators.clear();
        }
        if !dominators.is_empty() && !self.trick_reserved {
            if let Some(trick) = self.trick_in_hand(b) {
                if dominators
                    .iter()
                    .all(|bl| fight(Body::of(a, trick.pump), Body::of(bl, (0, 0))) != (true, false))
                {
                    self.trick_reserved = true;
                    dominators.clear();
                }
            }
        }
        b.declared(dominators.is_empty())
    }
    fn defenders_to_hold(&self, b: &Board<'_>) -> BTreeSet<i32> {
        let mut incoming: Vec<_> = b.creatures(false).map(|p| p.power).collect();
        incoming.sort_by(|a, b| b.cmp(a));
        let pool = self.attack_pool.as_ref().unwrap();
        let defenders = b.creatures(true).filter(|p| !pool.contains(&p.id)).count();
        let mut candidates: Vec<_> = b.creatures(true).filter(|p| pool.contains(&p.id)).collect();
        candidates.sort_by_key(|p| (-p.toughness, p.power));
        let mut held = BTreeSet::new();
        for (defenders, p) in (defenders..).zip(candidates) {
            if incoming.iter().skip(defenders).sum::<i32>() < b.raw.agent.life {
                break;
            }
            held.insert(p.id);
        }
        held
    }
    fn block(&mut self, b: &Board<'_>) -> Option<usize> {
        if self.block_plan.is_none() {
            self.block_plan = Some(self.plan_blocks(b));
        }
        let blocker = b.actions.iter().find_map(|a| a.focus.first());
        let target = blocker.and_then(|id| self.block_plan.as_ref().unwrap().get(id));
        b.actions
            .iter()
            .position(|a| a.focus.len() == 2 && a.focus.get(1) == target)
            .or_else(|| b.actions.iter().position(|a| a.focus.len() == 1))
    }
    fn plan_blocks(&self, b: &Board<'_>) -> BTreeMap<i32, i32> {
        let offered: BTreeSet<_> = b
            .actions
            .iter()
            .filter(|a| a.focus.len() == 2)
            .map(|a| a.focus[1])
            .collect();
        let mut attackers: Vec<_> = b
            .creatures(false)
            .filter(|p| offered.contains(&p.id) || (self.blocks.contains_key(&p.id) && p.tapped))
            .collect();
        attackers.sort_by_key(|p| -p.power);
        let blockers: Vec<_> = b.creatures(true).filter(|p| !p.tapped).collect();
        let mut pairs = vec![];
        for a in &attackers {
            for bl in &blockers {
                if !can_block(bl, a) {
                    continue;
                }
                let (ad, bd) = fight(Body::of(a, (0, 0)), Body::of(bl, (0, 0)));
                let rank = if ad && !bd {
                    3
                } else if ad && b.mana_value(a) >= b.mana_value(bl) {
                    2
                } else if !bd {
                    1
                } else {
                    continue;
                };
                pairs.push((rank, a.power, -b.mana_value(bl), bl.id, a.id));
            }
        }
        pairs.sort_by(|a, b| b.cmp(a));
        let mut plan = BTreeMap::new();
        let mut blocked = BTreeSet::new();
        for (_, _, _, bl, a) in pairs {
            if !plan.contains_key(&bl) && !blocked.contains(&a) {
                plan.insert(bl, a);
                blocked.insert(a);
            }
        }
        let unblocked: Vec<_> = attackers
            .into_iter()
            .filter(|a| !blocked.contains(&a.id))
            .collect();
        let mut spare: Vec<_> = blockers
            .into_iter()
            .filter(|bl| !plan.contains_key(&bl.id))
            .collect();
        spare.sort_by_key(|p| (b.mana_value(p), p.power));
        let mut incoming: i32 = unblocked.iter().map(|a| a.power).sum();
        for a in unblocked {
            if incoming < b.raw.agent.life {
                break;
            }
            if let Some(i) = spare.iter().position(|bl| can_block(bl, a)) {
                let bl = spare.remove(i);
                plan.insert(bl.id, a.id);
                incoming -= a.power;
            }
        }
        plan
    }
}
fn fallback(b: &Board<'_>) -> usize {
    let desired: &[Action] = match b.raw.action_space.action_space_type {
        Space::Priority => &[
            Action::PriorityPlayLand,
            Action::PriorityCastSpell,
            Action::PriorityPassPriority,
        ],
        Space::DeclareAttacker => &[Action::DeclareAttacker],
        Space::DeclareBlocker => &[Action::DeclareBlocker],
        _ => &[],
    };
    desired
        .iter()
        .find_map(|a| b.first(*a))
        .or_else(|| {
            b.actions
                .iter()
                .position(|a| a.action_type != Action::DeclineChoice)
        })
        .unwrap_or(b.actions.len() - 1)
}

#[cfg(test)]
mod tests;
