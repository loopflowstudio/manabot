use super::{
    card::{Card, Keywords},
    game_object::{CardId, ObjectId, PlayerId},
    mana::Mana,
};

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
    /// +1/+1 counters (CR 122). Any permanent can carry them, including
    /// lands — they contribute to P/T only via effective_power/toughness.
    pub plus1_counters: i32,
    /// "Can't be blocked this turn" — cleared during cleanup.
    pub cant_be_blocked_this_turn: bool,
    /// Earthbend animation: this permanent is a 0/0 creature with haste in
    /// addition to its printed types (still a land). No duration — it lasts
    /// as long as the permanent does.
    pub animated: bool,
    /// Keywords granted until end of turn — cleared during cleanup.
    pub temp_keywords: Keywords,
    pub attacking: bool,
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
            plus1_counters: 0,
            cant_be_blocked_this_turn: false,
            animated: false,
            temp_keywords: Keywords::default(),
            attacking: false,
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
        keywords
    }

    pub fn can_tap(&self, card: &Card) -> bool {
        !(self.tapped || (self.summoning_sick && self.is_creature(card)))
    }

    pub fn can_attack(&self, card: &Card) -> bool {
        self.is_creature(card)
            && !self.effective_keywords(card).defender
            && !self.tapped
            && !self.summoning_sick
    }

    pub fn can_block(&self, card: &Card) -> bool {
        self.is_creature(card) && !self.tapped
    }

    /// Base + counters + until-EOT deltas. Static continuous effects and
    /// characteristic-defining P/T need whole-game state — use
    /// `Game::effective_power` / `Game::effective_toughness` (this is their
    /// local component).
    pub fn local_power(&self, card: &Card) -> i32 {
        card.power.unwrap_or(0) + self.plus1_counters + self.temp_power
    }

    pub fn local_toughness(&self, card: &Card) -> i32 {
        card.toughness.unwrap_or(0) + self.plus1_counters + self.temp_toughness
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

    pub fn untap(&mut self) {
        self.tapped = false;
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
        self.temp_power = 0;
        self.temp_toughness = 0;
        self.cant_be_blocked_this_turn = false;
        self.temp_keywords = Keywords::default();
    }
}
