use super::{
    card::{Card, Keywords},
    game_object::{CardId, ObjectId, PlayerId},
    mana::Mana,
};

/// Counter kinds the engine knows (CR 122). Absence in authored IR means
/// +1/+1, the kind every earlier card uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterKind {
    PlusOnePlusOne,
    /// CR 122.1d: replaces the next untap.
    Stun,
    /// No rules meaning of its own; cards count it.
    Fire,
    /// CR 122.1b keyword counter.
    DoubleStrike,
}

impl CounterKind {
    pub const ALL: [CounterKind; 4] = [
        CounterKind::PlusOnePlusOne,
        CounterKind::Stun,
        CounterKind::Fire,
        CounterKind::DoubleStrike,
    ];

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "plus_one_plus_one" => Some(Self::PlusOnePlusOne),
            "stun" => Some(Self::Stun),
            "fire" => Some(Self::Fire),
            "double_strike" => Some(Self::DoubleStrike),
            _ => None,
        }
    }
}

/// Non-+1/+1 counters on one permanent. Serialized only when present so
/// state hashes of games without them are unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Counters {
    #[serde(skip_serializing_if = "is_zero")]
    pub stun: i32,
    #[serde(skip_serializing_if = "is_zero")]
    pub fire: i32,
    #[serde(skip_serializing_if = "is_zero")]
    pub double_strike: i32,
}

impl Counters {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

fn is_zero(value: &i32) -> bool {
    *value == 0
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Permanent {
    pub id: ObjectId,
    pub card: CardId,
    pub controller: PlayerId,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: i32,
    pub deathtouch_damage: bool,
    pub temp_power: i32,
    pub temp_toughness: i32,
    /// Until-EOT effects that set base power or toughness (CR 613.4b), cleared
    /// during cleanup. The latest effect wins, so a setter overwrites.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_base_power: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_base_toughness: Option<i32>,
    /// +1/+1 counters (CR 122). Any permanent can carry them, including
    /// lands — they contribute to P/T only via effective_power/toughness.
    /// Other kinds live in `counters`; `counter_count` reads either.
    pub plus1_counters: i32,
    /// Counters of every kind other than +1/+1 (CR 122.1).
    #[serde(skip_serializing_if = "Counters::is_empty")]
    pub counters: Counters,
    /// "Can't be blocked this turn" — cleared during cleanup.
    pub cant_be_blocked_this_turn: bool,
    /// Earthbend animation: this permanent is a 0/0 creature with haste in
    /// addition to its printed types (still a land). No duration — it lasts
    /// as long as the permanent does.
    pub animated: bool,
    /// Keywords granted until end of turn — cleared during cleanup.
    pub temp_keywords: Keywords,
    /// Additive subtype grants tied to this incarnation, removed at cleanup.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub temp_subtypes: Vec<String>,
    /// Object-local replacement, removed at cleanup or when this incarnation leaves.
    pub exile_if_dies_this_turn: bool,
    pub attacking: bool,
    /// Exhaust abilities this incarnation has activated (CR 702.177a).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub exhausted_abilities: Vec<usize>,
    /// True only for this incarnation entering as a resolving cast spell.
    pub was_cast: bool,
}

impl Permanent {
    pub fn new(id: ObjectId, card_id: CardId, card: &Card) -> Self {
        Self {
            id,
            card: card_id,
            controller: card.owner,
            tapped: false,
            summoning_sick: card.types.is_creature() && !card.keywords.haste,
            damage: 0,
            deathtouch_damage: false,
            temp_power: 0,
            temp_toughness: 0,
            temp_base_power: None,
            temp_base_toughness: None,
            plus1_counters: 0,
            counters: Counters::default(),
            cant_be_blocked_this_turn: false,
            animated: false,
            temp_keywords: Keywords::default(),
            temp_subtypes: Vec::new(),
            exile_if_dies_this_turn: false,
            attacking: false,
            exhausted_abilities: Vec::new(),
            was_cast: false,
        }
    }

    /// Whether this permanent is a creature — printed type or earthbend
    /// animation (a land that's also a 0/0 creature).
    pub fn is_creature(&self, card: &Card) -> bool {
        card.types.is_creature() || self.animated
    }

    /// Printed keywords plus until-EOT grants plus animation (earthbent
    /// lands have haste).
    pub fn effective_keywords(&self, card: &Card) -> Keywords {
        let mut keywords = card.keywords.union(&self.temp_keywords);
        if self.animated {
            keywords.haste = true;
        }
        // CR 122.1b: a keyword counter grants its keyword while it remains.
        if self.counters.double_strike > 0 {
            keywords.double_strike = true;
        }
        keywords
    }

    pub fn can_tap(&self, card: &Card) -> bool {
        // Haste bypasses sickness; a temporary grant must not erase the
        // restriction that applies again when the grant expires (CR 702.10).
        !(self.tapped
            || (self.summoning_sick
                && self.is_creature(card)
                && !self.effective_keywords(card).haste))
    }

    pub fn can_attack(&self, card: &Card) -> bool {
        self.is_creature(card) && !self.effective_keywords(card).defender && self.can_tap(card)
    }

    pub fn can_block(&self, card: &Card) -> bool {
        self.is_creature(card) && !self.tapped
    }

    pub fn producible_mana(&self, card: &Card) -> Mana {
        let mut total = Mana::default();
        if !self.can_tap(card) {
            return total;
        }
        for ability in &card.mana_abilities {
            total.add(&ability.mana);
        }
        total
    }

    /// CR 122.1d: a tapped permanent with a stun counter that would become
    /// untapped stays tapped and loses one stun counter instead. Every untap
    /// path (untap step, effects, costs) goes through here.
    pub fn untap(&mut self) {
        if self.tapped && self.counters.stun > 0 {
            self.counters.stun -= 1;
        } else {
            self.tapped = false;
        }
    }

    pub fn counter_count(&self, kind: CounterKind) -> i32 {
        match kind {
            CounterKind::PlusOnePlusOne => self.plus1_counters,
            CounterKind::Stun => self.counters.stun,
            CounterKind::Fire => self.counters.fire,
            CounterKind::DoubleStrike => self.counters.double_strike,
        }
    }

    /// Add `delta` counters of `kind`, clamped at zero. Returns whether the
    /// count changed.
    pub fn change_counters(&mut self, kind: CounterKind, delta: i32) -> bool {
        let count = match kind {
            CounterKind::PlusOnePlusOne => &mut self.plus1_counters,
            CounterKind::Stun => &mut self.counters.stun,
            CounterKind::Fire => &mut self.counters.fire,
            CounterKind::DoubleStrike => &mut self.counters.double_strike,
        };
        let old = *count;
        *count = old.saturating_add(delta).max(0);
        *count != old
    }

    /// Whether an exhaust ability at `ability_index` was already activated.
    pub fn exhausted(&self, ability_index: usize) -> bool {
        self.exhausted_abilities.contains(&ability_index)
    }

    /// "With a counter on it" / "with no counters on it" read every kind.
    pub fn has_counters(&self) -> bool {
        self.plus1_counters > 0 || !self.counters.is_empty()
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
    }

    pub fn clear_temporary_modifiers(&mut self) {
        self.exile_if_dies_this_turn = false;
        self.temp_power = 0;
        self.temp_toughness = 0;
        self.temp_base_power = None;
        self.temp_base_toughness = None;
        self.cant_be_blocked_this_turn = false;
        self.temp_keywords = Keywords::default();
        self.temp_subtypes.clear();
    }
}
