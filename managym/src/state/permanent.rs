use super::{
    card::Card,
    game_object::{CardId, ObjectId, PlayerId},
    mana::Mana,
};

#[derive(Clone, Debug, PartialEq, Eq)]
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
            attacking: false,
        }
    }

    pub fn can_tap(&self, card: &Card) -> bool {
        !(self.tapped || (self.summoning_sick && card.types.is_creature()))
    }

    pub fn can_attack(&self, card: &Card) -> bool {
        card.types.is_creature() && !card.keywords.defender && !self.tapped && !self.summoning_sick
    }

    pub fn can_block(&self, card: &Card) -> bool {
        card.types.is_creature() && !self.tapped
    }

    pub fn has_lethal_damage(&self, card: &Card) -> bool {
        if !card.types.is_creature() {
            return false;
        }
        if self.deathtouch_damage && self.damage > 0 {
            return true;
        }
        self.damage >= self.effective_toughness(card)
    }

    pub fn effective_power(&self, card: &Card) -> i32 {
        card.power.unwrap_or(0) + self.plus1_counters + self.temp_power
    }

    pub fn effective_toughness(&self, card: &Card) -> i32 {
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
    }
}
