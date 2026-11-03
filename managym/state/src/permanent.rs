use smallvec::SmallVec;

use super::{
    card::{ActivatedAbilityDefinition, Card, Keyword, Keywords},
    game_object::{CardId, ObjectId, ObjectRef, PlayerId},
};

/// Counter kinds the engine knows (CR 122). A new kind is one variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterKind {
    PlusOnePlusOne,
    /// CR 122.1d: replaces the next untap.
    Stun,
    /// No rules meaning of its own; cards count it.
    Fire,
    /// CR 122.1b keyword counter.
    DoubleStrike,
    /// No rules meaning of its own; cards count it (the Ascensions).
    Quest,
    /// CR 122.1h: a permanent with one that would die is exiled instead.
    Finality,
    /// No rules meaning of its own; cards count it (Zhao, the Moon Slayer).
    Conqueror,
    /// CR 714.2: a Saga's chapter abilities trigger as these are put on it.
    Lore,
}

impl CounterKind {
    /// The serialized name (`stun`, `double_strike`, ...).
    pub fn name(self) -> &'static str {
        match self {
            CounterKind::PlusOnePlusOne => "plus_one_plus_one",
            CounterKind::Stun => "stun",
            CounterKind::Fire => "fire",
            CounterKind::DoubleStrike => "double_strike",
            CounterKind::Quest => "quest",
            CounterKind::Finality => "finality",
            CounterKind::Conqueror => "conqueror",
            CounterKind::Lore => "lore",
        }
    }

    pub const ALL: [CounterKind; 8] = [
        CounterKind::PlusOnePlusOne,
        CounterKind::Stun,
        CounterKind::Fire,
        CounterKind::DoubleStrike,
        CounterKind::Quest,
        CounterKind::Finality,
        CounterKind::Conqueror,
        CounterKind::Lore,
    ];
}

/// Counters on one object by kind (CR 122.1), +1/+1 included. Stored
/// sparsely: only kinds with a positive count, in `CounterKind` order, so a
/// permanent carries nothing for the kinds it lacks however many the engine
/// knows. Serialized as a map of those entries, so hashes do not see absent
/// kinds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counters(SmallVec<[(CounterKind, i32); 2]>);

impl Counters {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, kind: CounterKind) -> i32 {
        self.0
            .iter()
            .find(|(present, _)| *present == kind)
            .map_or(0, |(_, count)| *count)
    }

    /// Add `delta` counters of `kind`, clamped at zero. Returns whether the
    /// count changed.
    pub fn change(&mut self, kind: CounterKind, delta: i32) -> bool {
        match self.0.binary_search_by_key(&kind, |(present, _)| *present) {
            Ok(index) => {
                let old = self.0[index].1;
                let new = old.saturating_add(delta).max(0);
                if new == 0 {
                    self.0.remove(index);
                } else {
                    self.0[index].1 = new;
                }
                new != old
            }
            Err(index) if delta > 0 => {
                self.0.insert(index, (kind, delta));
                true
            }
            Err(_) => false,
        }
    }

    /// The kinds present, with their counts, in `CounterKind` order.
    pub fn iter(&self) -> impl Iterator<Item = (CounterKind, i32)> + '_ {
        self.0.iter().copied()
    }
}

impl serde::Serialize for Counters {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(self.iter())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Permanent {
    /// Number of transformations of this incarnation (CR 701.27f).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub transforms: u32,
    pub id: ObjectId,
    pub card: CardId,
    pub controller: PlayerId,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: i32,
    pub deathtouch_damage: bool,
    /// Dealt damage this turn ("if it was dealt damage this turn"), whether
    /// or not that damage is still marked; cleared during cleanup.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub dealt_damage_this_turn: bool,
    /// Counters of every kind (CR 122.1). Any permanent can carry them,
    /// including lands; +1/+1 counters change P/T only through
    /// `Game::characteristics`.
    #[serde(skip_serializing_if = "Counters::is_empty")]
    pub counters: Counters,
    /// Its timestamp (CR 613.7d/e): when it entered, renewed when an Aura or
    /// Equipment becomes attached. Static abilities apply in this order among
    /// continuous-effect records.
    pub timestamp: ObjectId,
    pub attacking: bool,
    /// True only for this incarnation entering as a resolving cast spell.
    pub was_cast: bool,
    /// The X announced for that spell (CR 107.3m), read by its own
    /// entering-the-battlefield triggers; 0 otherwise.
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub cast_x: u8,
    /// The exact object this Equipment is attached to (CR 301.5). State-based
    /// actions clear it once that object is gone or no longer a creature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<ObjectRef>,
    /// Exhaust abilities (indices into the card's activated abilities) this
    /// object has activated; a new object starts fresh (CR 400.7).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub exhausted: Vec<usize>,
    /// "Once each turn" abilities this object has activated this turn,
    /// cleared at cleanup.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub activated_this_turn: Vec<usize>,
}

impl Permanent {
    /// Whether this object has used up the ability: exhaust once ever
    /// (CR 702.177a), "once each turn" already this turn (CR 602.5b).
    pub fn activation_used(&self, index: usize, ability: &ActivatedAbilityDefinition) -> bool {
        (ability.exhaust && self.exhausted.contains(&index))
            || (ability.once_each_turn && self.activated_this_turn.contains(&index))
    }

    /// Record an activation against the ability's use limits.
    pub fn record_activation(&mut self, index: usize, ability: &ActivatedAbilityDefinition) {
        if ability.exhaust {
            self.exhausted.push(index);
        }
        if ability.once_each_turn {
            self.activated_this_turn.push(index);
        }
    }

    pub fn new(id: ObjectId, card_id: CardId, card: &Card) -> Self {
        Self {
            transforms: 0,
            id,
            card: card_id,
            controller: card.owner,
            tapped: false,
            // CR 302.6 applies to any permanent that is a creature; only
            // creatures and Vehicles can attack or pay {T} as creatures here.
            summoning_sick: (card.types.is_creature() || card.has_subtype("Vehicle"))
                && !card.keywords.has(Keyword::Haste),
            damage: 0,
            deathtouch_damage: false,
            dealt_damage_this_turn: false,
            timestamp: id,
            counters: Counters::default(),
            attacking: false,
            was_cast: false,
            cast_x: 0,
            attached_to: None,
            exhausted: Vec::new(),
            activated_this_turn: Vec::new(),
        }
    }

    /// Printed keywords plus keyword counters. An input to
    /// `Game::characteristics`, which applies continuous effects and static
    /// grants in layer 6.
    pub fn intrinsic_keywords(&self, card: &Card) -> Keywords {
        let mut keywords = card.keywords;
        // CR 122.1b: a keyword counter grants its keyword while it remains.
        if self.counters.get(CounterKind::DoubleStrike) > 0 {
            keywords.insert(Keyword::DoubleStrike);
        }
        keywords
    }

    /// CR 122.1d: a tapped permanent with a stun counter that would become
    /// untapped stays tapped and loses one stun counter instead. Every untap
    /// path (untap step, effects, costs) goes through here.
    pub fn untap(&mut self) {
        if self.tapped && self.counters.get(CounterKind::Stun) > 0 {
            self.counters.change(CounterKind::Stun, -1);
        } else {
            self.tapped = false;
        }
    }

    pub fn counter_count(&self, kind: CounterKind) -> i32 {
        self.counters.get(kind)
    }

    /// Add `delta` counters of `kind`, clamped at zero. Returns whether the
    /// count changed.
    pub fn change_counters(&mut self, kind: CounterKind, delta: i32) -> bool {
        self.counters.change(kind, delta)
    }

    /// "With a counter on it" / "with no counters on it" read every kind.
    pub fn has_counters(&self) -> bool {
        !self.counters.is_empty()
    }

    pub fn tap(&mut self) {
        self.tapped = true;
    }

    pub fn clear_damage(&mut self) {
        self.damage = 0;
        self.deathtouch_damage = false;
    }

    pub fn take_damage(&mut self, amount: i32) {
        self.damage += amount;
        if amount > 0 {
            self.dealt_damage_this_turn = true;
        }
    }

    /// "Once each turn" limits and "dealt damage this turn" reset in the
    /// cleanup step.
    pub fn clear_turn_history(&mut self) {
        self.dealt_damage_this_turn = false;
        self.activated_this_turn.clear();
    }
}
