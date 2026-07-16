use std::{
    mem::size_of,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    agent::{action::ActionSpace, behavior_tracker::BehaviorTracker},
    flow::{
        combat::CombatState,
        decision::SuspendedResolution,
        event_log::EventLog,
        game::{Game, PendingChoice},
        priority::PriorityState,
        trigger::{DelayedTrigger, ExileLink, PendingTrigger},
        turn::TurnState,
    },
    state::{
        game_object::{CardId, Incarnation, ObjectLki, ObjectRef, PermanentId, PlayerId},
        mana::Mana,
        player::Player,
        stack_object::StackObject,
        zone::{ZoneManager, ZoneType},
    },
};

#[derive(Debug, Default)]
pub struct JournalStats {
    next_branch_id: AtomicU64,
    marks: AtomicU64,
    commits: AtomicU64,
    rollbacks: AtomicU64,
    entries: AtomicU64,
    peak_entries: AtomicU64,
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JournalStatsSnapshot {
    pub marks: u64,
    pub commits: u64,
    pub rollbacks: u64,
    pub entries: u64,
    pub peak_entries: u64,
    pub peak_bytes: u64,
}

impl JournalStats {
    pub fn next_branch_id(&self) -> u64 {
        self.next_branch_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    pub fn snapshot(&self) -> JournalStatsSnapshot {
        JournalStatsSnapshot {
            marks: self.marks.load(Ordering::Relaxed),
            commits: self.commits.load(Ordering::Relaxed),
            rollbacks: self.rollbacks.load(Ordering::Relaxed),
            entries: self.entries.load(Ordering::Relaxed),
            peak_entries: self.peak_entries.load(Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(Ordering::Relaxed),
        }
    }

    fn add_bytes(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let current = self
            .current_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed)
            + bytes as u64;
        self.peak_bytes.fetch_max(current, Ordering::Relaxed);
    }

    fn subtract_bytes(&self, bytes: usize) {
        if bytes != 0 {
            self.current_bytes
                .fetch_sub(bytes as u64, Ordering::Relaxed);
        }
    }

    fn add_entry(&self) {
        let current = self.entries.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_entries.fetch_max(current, Ordering::Relaxed);
    }

    fn remove_entries(&self, count: usize) {
        if count != 0 {
            self.entries.fetch_sub(count as u64, Ordering::Relaxed);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EventQueue {
    Events,
    Pending,
    Observation,
}

/// The mutable half of a [`Player`]. `deck` and `name` are fixed at setup and
/// never change during a match, so journaling them would copy a 60-entry vector
/// and a string on every life, mana, and damage change. Recording only the
/// mutable scalars keeps the entry allocation-free.
#[derive(Clone, Debug)]
struct PlayerVitals {
    life: i32,
    drew_when_empty: bool,
    alive: bool,
    mana_pool: Mana,
    combat_mana_pool: Mana,
}

impl PlayerVitals {
    fn capture(player: &Player) -> Self {
        Self {
            life: player.life,
            drew_when_empty: player.drew_when_empty,
            alive: player.alive,
            mana_pool: player.mana_pool.clone(),
            combat_mana_pool: player.combat_mana_pool.clone(),
        }
    }

    fn restore(self, player: &mut Player) {
        player.life = self.life;
        player.drew_when_empty = self.drew_when_empty;
        player.alive = self.alive;
        player.mana_pool = self.mana_pool;
        player.combat_mana_pool = self.combat_mana_pool;
    }
}

#[derive(Debug)]
enum UndoEntry {
    Player(PlayerId, PlayerVitals),
    /// Inverse of one card's zone change, including the exact index it held.
    /// The hot zone path: a whole-`ZoneManager` clone here would copy fifteen
    /// vectors for every card that moves.
    ZoneMove {
        card: CardId,
        owner: PlayerId,
        from: Option<(ZoneType, usize)>,
        card_zones_len: usize,
    },
    Permanent(PermanentId, Option<crate::state::permanent::Permanent>),
    CardToPermanent(CardId, Option<PermanentId>),
    Incarnation(CardId, Incarnation),
    ObjectLki(ObjectRef, Option<ObjectLki>),
    /// Bulk zone rewrites (reshuffle, reorder, determinize) are rare, so they
    /// are boxed: inline they would be 360 bytes and, as the largest variant,
    /// would set that size for every entry in the journal.
    Zones(Box<ZoneManager>),
    Turn(Box<TurnState>),
    Priority(PriorityState),
    Stack(Vec<StackObject>),
    Combat(Box<Option<CombatState>>),
    ManaCache(usize, Option<Mana>),
    EventQueue(EventQueue, EventLog),
    EventLength(EventQueue, usize),
    PendingTriggers(Vec<PendingTrigger>),
    PendingTriggerChoice(Box<Option<PendingTrigger>>),
    DelayedTriggers(Vec<DelayedTrigger>),
    ExileLinks(Vec<ExileLink>),
    SuspendedDecision(Box<Option<SuspendedResolution>>),
    TriggerEnqueueCounter(u64),
    ActionSpace(Box<Option<ActionSpace>>),
    DecisionEpoch(u64),
    PendingChoice(Box<Option<PendingChoice>>),
    SkipTrivialCount(usize),
    Trackers(Box<[BehaviorTracker; 2]>),
}

fn vec_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(size_of::<T>())
}

impl UndoEntry {
    fn owned_bytes(&self) -> usize {
        match self {
            Self::Zones(zones) => zones.allocated_bytes(),
            Self::Turn(turn) => turn
                .ability_resolutions_this_turn
                .len()
                .saturating_mul(size_of::<((usize, usize), u32)>()),
            Self::Stack(values) => vec_bytes(values),
            Self::Combat(combat) => (**combat).as_ref().map_or(0, |combat| {
                size_of::<CombatState>()
                    + vec_bytes(&combat.attackers)
                    + vec_bytes(&combat.attackers_to_declare)
                    + vec_bytes(&combat.blockers_to_declare)
                    + combat
                        .attacker_to_blockers
                        .values()
                        .map(vec_bytes)
                        .sum::<usize>()
            }),
            Self::EventQueue(_, values) => values.owned_bytes(),
            Self::PendingTriggers(values) => vec_bytes(values),
            Self::DelayedTriggers(values) => vec_bytes(values),
            Self::ExileLinks(values) => vec_bytes(values),
            Self::ActionSpace(space) => (**space).as_ref().map_or(0, |space| {
                size_of::<ActionSpace>() + vec_bytes(&space.actions) + vec_bytes(&space.focus)
            }),
            Self::PendingChoice(choice) => match &**choice {
                Some(PendingChoice::ChooseTargets {
                    chosen,
                    chosen_req_indices,
                    legal_targets,
                    ..
                }) => {
                    size_of::<PendingChoice>()
                        + vec_bytes(chosen)
                        + vec_bytes(chosen_req_indices)
                        + vec_bytes(legal_targets)
                }
                Some(_) => size_of::<PendingChoice>(),
                None => 0,
            },
            Self::PendingTriggerChoice(trigger) => (**trigger)
                .as_ref()
                .map_or(0, |_| size_of::<PendingTrigger>()),
            Self::SuspendedDecision(decision) => (**decision)
                .as_ref()
                .map_or(0, |_| size_of::<SuspendedResolution>()),
            Self::Trackers(_) => size_of::<[BehaviorTracker; 2]>(),
            Self::Player(_, _)
            | Self::ZoneMove { .. }
            | Self::Permanent(_, _)
            | Self::CardToPermanent(_, _)
            | Self::Incarnation(_, _)
            | Self::ObjectLki(_, _)
            | Self::Priority(_)
            | Self::ManaCache(_, _)
            | Self::EventLength(_, _)
            | Self::TriggerEnqueueCounter(_)
            | Self::DecisionEpoch(_)
            | Self::SkipTrivialCount(_) => 0,
        }
    }

    fn restore(self, game: &mut Game) {
        match self {
            Self::Player(player, old) => old.restore(&mut game.state.players[player.0]),
            Self::ZoneMove {
                card,
                owner,
                from,
                card_zones_len,
            } => game
                .state
                .zones
                .undo_move(card, owner, from, card_zones_len),
            Self::Permanent(permanent, old) => game.state.permanents[permanent] = old,
            Self::CardToPermanent(card, old) => game.state.card_to_permanent[card] = old,
            Self::Incarnation(card, old) => game.state.object_incarnations[card] = old,
            Self::ObjectLki(object, Some(old)) => {
                game.state.object_lki.insert(object, old);
            }
            Self::ObjectLki(object, None) => {
                game.state.object_lki.remove(&object);
            }
            Self::Zones(old) => game.state.zones = *old,
            Self::Turn(old) => game.state.turn = *old,
            Self::Priority(old) => game.state.priority = old,
            Self::Stack(old) => game.state.stack_objects = old,
            Self::Combat(old) => game.state.combat = *old,
            Self::ManaCache(player, old) => game.state.mana_cache[player] = old,
            Self::EventQueue(EventQueue::Events, old) => game.state.events = old,
            Self::EventQueue(EventQueue::Pending, old) => game.state.pending_events = old,
            Self::EventQueue(EventQueue::Observation, old) => {
                game.state.observation_events = old;
            }
            Self::EventLength(EventQueue::Events, old) => game.state.events.truncate(old),
            Self::EventLength(EventQueue::Pending, old) => {
                game.state.pending_events.truncate(old);
            }
            Self::EventLength(EventQueue::Observation, old) => {
                game.state.observation_events.truncate(old);
            }
            Self::PendingTriggers(old) => game.state.pending_triggers = old,
            Self::PendingTriggerChoice(old) => game.state.pending_trigger_choice = *old,
            Self::DelayedTriggers(old) => game.state.delayed_triggers = old,
            Self::ExileLinks(old) => game.state.exile_links = old,
            Self::SuspendedDecision(old) => game.state.suspended_decision = *old,
            Self::TriggerEnqueueCounter(old) => game.state.trigger_enqueue_counter = old,
            Self::ActionSpace(old) => game.current_action_space = *old,
            Self::DecisionEpoch(old) => game.decision_epoch = old,
            Self::PendingChoice(old) => game.pending_choice = *old,
            Self::SkipTrivialCount(old) => game.skip_trivial_count = old,
            Self::Trackers(old) => game.trackers = *old,
        }
    }
}

#[derive(Debug)]
pub(crate) struct UndoJournal {
    branch_id: u64,
    next_mark_id: u64,
    active_marks: Vec<u64>,
    entries: Vec<UndoEntry>,
    payload_bytes: usize,
    accounted_base_bytes: usize,
    stats: Arc<JournalStats>,
}

impl UndoJournal {
    fn new(branch_id: u64, stats: Arc<JournalStats>) -> Self {
        Self {
            branch_id,
            next_mark_id: 0,
            active_marks: Vec::new(),
            entries: Vec::new(),
            payload_bytes: 0,
            accounted_base_bytes: 0,
            stats,
        }
    }

    fn active(&self) -> bool {
        !self.active_marks.is_empty()
    }

    fn sync_base_bytes(&mut self) {
        let base = self
            .entries
            .capacity()
            .saturating_mul(size_of::<UndoEntry>())
            + self
                .active_marks
                .capacity()
                .saturating_mul(size_of::<u64>());
        if base > self.accounted_base_bytes {
            self.stats.add_bytes(base - self.accounted_base_bytes);
        }
        self.accounted_base_bytes = base;
    }

    fn record(&mut self, entry: UndoEntry) {
        if !self.active() {
            return;
        }
        let owned = entry.owned_bytes();
        self.entries.push(entry);
        self.sync_base_bytes();
        self.payload_bytes += owned;
        self.stats.add_bytes(owned);
        self.stats.add_entry();
    }

    fn pop_entry(&mut self) -> Option<UndoEntry> {
        let entry = self.entries.pop()?;
        let owned = entry.owned_bytes();
        self.payload_bytes -= owned;
        self.stats.subtract_bytes(owned);
        self.stats.remove_entries(1);
        Some(entry)
    }

    fn clear_entries(&mut self) {
        let count = self.entries.len();
        self.entries.clear();
        self.stats.subtract_bytes(self.payload_bytes);
        self.stats.remove_entries(count);
        self.payload_bytes = 0;
    }
}

impl Drop for UndoJournal {
    fn drop(&mut self) {
        self.stats
            .subtract_bytes(self.accounted_base_bytes + self.payload_bytes);
        self.stats.remove_entries(self.entries.len());
    }
}

#[derive(Clone, Debug)]
pub struct ClonePlusUndoMark {
    branch_id: u64,
    mark_id: u64,
    depth: usize,
    cursor: usize,
    rng_seed: [u8; 32],
    rng_stream: u64,
    rng_word_pos: u128,
    id_watermark: u32,
    cards_len: usize,
    permanents_len: usize,
    card_to_permanent_len: usize,
    incarnations_len: usize,
}

impl Game {
    pub(crate) fn enable_undo(&mut self, stats: Arc<JournalStats>, branch_id: u64) {
        self.undo = Some(UndoJournal::new(branch_id, stats));
    }

    pub(crate) fn undo_mark(&mut self) -> ClonePlusUndoMark {
        let journal = self.undo.as_mut().expect("undo driver state is enabled");
        journal.next_mark_id = journal
            .next_mark_id
            .checked_add(1)
            .expect("mark ID exhausted");
        let mark_id = journal.next_mark_id;
        let depth = journal.active_marks.len();
        journal.active_marks.push(mark_id);
        journal.sync_base_bytes();
        journal.stats.marks.fetch_add(1, Ordering::Relaxed);
        ClonePlusUndoMark {
            branch_id: journal.branch_id,
            mark_id,
            depth,
            cursor: journal.entries.len(),
            rng_seed: self.state.rng.get_seed(),
            rng_stream: self.state.rng.get_stream(),
            rng_word_pos: self.state.rng.get_word_pos(),
            id_watermark: self.state.id_gen.watermark(),
            cards_len: self.state.cards.len(),
            permanents_len: self.state.permanents.len(),
            card_to_permanent_len: self.state.card_to_permanent.len(),
            incarnations_len: self.state.object_incarnations.len(),
        }
    }

    pub(crate) fn undo_commit(&mut self, mark: ClonePlusUndoMark) {
        let journal = self.undo.as_mut().expect("undo driver state is enabled");
        Self::assert_top_mark(journal, &mark);
        journal.active_marks.pop();
        journal.stats.commits.fetch_add(1, Ordering::Relaxed);
        if journal.active_marks.is_empty() {
            journal.clear_entries();
        }
    }

    pub(crate) fn undo_rollback(&mut self, mark: ClonePlusUndoMark) {
        let mut journal = self.undo.take().expect("undo driver state is enabled");
        Self::assert_top_mark(&journal, &mark);
        while journal.entries.len() > mark.cursor {
            journal
                .pop_entry()
                .expect("journal cursor checked")
                .restore(self);
        }
        self.state.cards.truncate(mark.cards_len);
        self.state.permanents.truncate(mark.permanents_len);
        self.state
            .card_to_permanent
            .truncate(mark.card_to_permanent_len);
        self.state
            .object_incarnations
            .truncate(mark.incarnations_len);
        self.state.id_gen.restore(mark.id_watermark);
        let mut rng = ChaCha8Rng::from_seed(mark.rng_seed);
        rng.set_stream(mark.rng_stream);
        rng.set_word_pos(mark.rng_word_pos);
        self.state.rng = rng;
        journal.active_marks.pop();
        journal.stats.rollbacks.fetch_add(1, Ordering::Relaxed);
        self.undo = Some(journal);
    }

    fn assert_top_mark(journal: &UndoJournal, mark: &ClonePlusUndoMark) {
        assert_eq!(
            journal.branch_id, mark.branch_id,
            "mark belongs to another branch"
        );
        assert_eq!(
            journal.active_marks.len(),
            mark.depth + 1,
            "mark depth is stale"
        );
        assert_eq!(
            journal.active_marks.last(),
            Some(&mark.mark_id),
            "marks must be LIFO"
        );
        assert!(mark.cursor <= journal.entries.len(), "mark cursor is stale");
    }

    fn record_undo(&mut self, entry: UndoEntry) {
        if let Some(journal) = self.undo.as_mut() {
            journal.record(entry);
        }
    }

    fn journal_active(&self) -> bool {
        self.undo.as_ref().is_some_and(UndoJournal::active)
    }

    pub(crate) fn journal_player(&mut self, player: PlayerId) {
        self.record_undo(UndoEntry::Player(
            player,
            PlayerVitals::capture(&self.state.players[player.0]),
        ));
    }

    /// Journal one card's zone change. Must be called before the move, while
    /// the card still sits at the index this records.
    pub(crate) fn journal_zone_move(&mut self, card: CardId, owner: PlayerId) {
        if !self.journal_active() {
            return;
        }
        let from = self.state.zones.locate(card, owner);
        let card_zones_len = self.state.zones.card_zones_len();
        self.record_undo(UndoEntry::ZoneMove {
            card,
            owner,
            from,
            card_zones_len,
        });
    }

    pub(crate) fn journal_permanent(&mut self, permanent: PermanentId) {
        self.record_undo(UndoEntry::Permanent(
            permanent,
            self.state.permanents[permanent].clone(),
        ));
    }

    pub(crate) fn journal_card_to_permanent(&mut self, card: CardId) {
        if let Some(old) = self.state.card_to_permanent.get(card.0).copied() {
            self.record_undo(UndoEntry::CardToPermanent(card, old));
        }
    }

    pub(crate) fn journal_incarnation(&mut self, card: CardId) {
        self.record_undo(UndoEntry::Incarnation(
            card,
            self.state.object_incarnations[card],
        ));
    }

    pub(crate) fn journal_object_lki(&mut self, object: ObjectRef) {
        self.record_undo(UndoEntry::ObjectLki(
            object,
            self.state.object_lki.get(&object).copied(),
        ));
    }

    pub(crate) fn journal_zones(&mut self) {
        self.record_undo(UndoEntry::Zones(Box::new(self.state.zones.clone())));
    }

    pub(crate) fn journal_turn(&mut self) {
        self.record_undo(UndoEntry::Turn(Box::new(self.state.turn.clone())));
    }

    pub(crate) fn journal_priority(&mut self) {
        self.record_undo(UndoEntry::Priority(self.state.priority.clone()));
    }

    pub(crate) fn journal_stack(&mut self) {
        self.record_undo(UndoEntry::Stack(self.state.stack_objects.clone()));
    }

    pub(crate) fn journal_combat(&mut self) {
        self.record_undo(UndoEntry::Combat(Box::new(self.state.combat.clone())));
    }

    pub(crate) fn journal_mana_cache(&mut self, player: PlayerId) {
        self.record_undo(UndoEntry::ManaCache(
            player.0,
            self.state.mana_cache[player.0].clone(),
        ));
    }

    pub(crate) fn journal_events(&mut self) {
        self.record_undo(UndoEntry::EventQueue(
            EventQueue::Events,
            self.state.events.journal_snapshot(),
        ));
    }

    pub(crate) fn journal_event_append(&mut self) {
        self.record_undo(UndoEntry::EventLength(
            EventQueue::Events,
            self.state.events.len(),
        ));
    }

    /// Journal the choice, decision, and trigger queues that a transition may
    /// mutate through many interior paths (`play`, `triggers`, `decision`).
    ///
    /// These fields are recorded whole at the transition boundary rather than
    /// at each interior write. Entries restore LIFO, so the value recorded here
    /// is the last one applied and therefore wins over any interior record —
    /// one record at the outermost boundary is complete for each field. The
    /// queues are drained by the end of a transition, so at this point they are
    /// empty or near-empty and the clone is cheap.
    pub(crate) fn journal_transition_queues(&mut self) {
        self.journal_pending_choice();
        self.journal_suspended_decision();
        self.journal_pending_events();
        self.journal_pending_triggers();
        self.journal_pending_trigger_choice();
        self.journal_delayed_triggers();
        self.journal_exile_links();
        self.journal_trigger_enqueue_counter();
    }

    pub(crate) fn journal_pending_events(&mut self) {
        self.record_undo(UndoEntry::EventQueue(
            EventQueue::Pending,
            self.state.pending_events.journal_snapshot(),
        ));
    }

    pub(crate) fn journal_pending_event_append(&mut self) {
        self.record_undo(UndoEntry::EventLength(
            EventQueue::Pending,
            self.state.pending_events.len(),
        ));
    }

    pub(crate) fn journal_observation_events(&mut self) {
        self.record_undo(UndoEntry::EventQueue(
            EventQueue::Observation,
            self.state.observation_events.journal_snapshot(),
        ));
    }

    pub(crate) fn journal_observation_event_append(&mut self) {
        self.record_undo(UndoEntry::EventLength(
            EventQueue::Observation,
            self.state.observation_events.len(),
        ));
    }

    pub(crate) fn journal_pending_triggers(&mut self) {
        self.record_undo(UndoEntry::PendingTriggers(
            self.state.pending_triggers.clone(),
        ));
    }

    pub(crate) fn journal_pending_trigger_choice(&mut self) {
        self.record_undo(UndoEntry::PendingTriggerChoice(Box::new(
            self.state.pending_trigger_choice.clone(),
        )));
    }

    pub(crate) fn journal_delayed_triggers(&mut self) {
        self.record_undo(UndoEntry::DelayedTriggers(
            self.state.delayed_triggers.clone(),
        ));
    }

    pub(crate) fn journal_exile_links(&mut self) {
        self.record_undo(UndoEntry::ExileLinks(self.state.exile_links.clone()));
    }

    pub(crate) fn journal_suspended_decision(&mut self) {
        self.record_undo(UndoEntry::SuspendedDecision(Box::new(
            self.state.suspended_decision.clone(),
        )));
    }

    pub(crate) fn journal_trigger_enqueue_counter(&mut self) {
        self.record_undo(UndoEntry::TriggerEnqueueCounter(
            self.state.trigger_enqueue_counter,
        ));
    }

    pub(crate) fn journal_action_space(&mut self) {
        self.record_undo(UndoEntry::ActionSpace(Box::new(
            self.current_action_space.clone(),
        )));
    }

    pub(crate) fn journal_decision_epoch(&mut self) {
        self.record_undo(UndoEntry::DecisionEpoch(self.decision_epoch));
    }

    pub(crate) fn journal_pending_choice(&mut self) {
        self.record_undo(UndoEntry::PendingChoice(Box::new(
            self.pending_choice.clone(),
        )));
    }

    pub(crate) fn journal_skip_trivial_count(&mut self) {
        self.record_undo(UndoEntry::SkipTrivialCount(self.skip_trivial_count));
    }

    pub(crate) fn journal_trackers(&mut self) {
        self.record_undo(UndoEntry::Trackers(Box::new(self.trackers.clone())));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The journal's whole point is staying small per mutation, and an enum is
    /// as large as its largest variant. Before boxing the bulk variants an
    /// entry cost 360 bytes -- the inline size of `ZoneManager` -- so a 20-byte
    /// player record paid for a whole zone snapshot. Keep the bulk payloads
    /// behind a box.
    ///
    /// 72 bytes is the floor worth reaching: what remains are `Permanent` and
    /// `ObjectLki`, the two hottest entries. Boxing those would shrink the
    /// entry but add a heap allocation to every permanent mutation, trading
    /// the size win for a larger time cost.
    #[test]
    fn an_undo_entry_stays_compact() {
        assert!(
            size_of::<UndoEntry>() <= 72,
            "UndoEntry grew to {} bytes; box the variant that widened it",
            size_of::<UndoEntry>()
        );
        assert!(size_of::<PlayerVitals>() <= 24);
    }
}
