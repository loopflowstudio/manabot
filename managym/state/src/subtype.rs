//! Subtype categories (CR 205.3): each subtype belongs to exactly one card
//! type's list, so "Cave" is a land type, "Shrine" an enchantment type and
//! "Lesson" a spell type wherever it appears. Definitions, modifications and
//! predicates keep subtypes as their printed names; this table is the one
//! authority that says which list a name is on. Rules that read creature
//! types (CR 205.3m: a creature's creature types, a creature-type replacement
//! that keeps other subtypes) ask `is_creature_type`, never infer it by
//! excluding a partial list of other categories.
//!
//! The table holds every subtype the catalog authors (the cards crate's
//! definition checks fail on an unlisted name) and the CR lists
//! of creature, land, artifact, enchantment and spell types. Creature types
//! follow CR 205.3m (2026-11-06), with Oracle's ASCII apostrophes.

/// The card type whose subtype list contains a name (CR 205.3g–m).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubtypeCategory {
    Artifact,
    Creature,
    Enchantment,
    Land,
    Planeswalker,
    Spell,
}

use SubtypeCategory::{Artifact, Creature, Enchantment, Land, Planeswalker, Spell};

/// Sorted by name for binary search; `subtype_table_is_sorted` checks it.
const SUBTYPES: &[(&str, SubtypeCategory)] = &[
    ("Adventure", Spell),
    ("Advisor", Creature),
    ("Aenar", Creature),
    ("Aetherborn", Creature),
    ("Ajani", Planeswalker),
    ("Alien", Creature),
    ("Ally", Creature),
    ("Andorian", Creature),
    ("Android", Creature),
    ("Angel", Creature),
    ("Antelope", Creature),
    ("Ape", Creature),
    ("Arcane", Spell),
    ("Archer", Creature),
    ("Archon", Creature),
    ("Armadillo", Creature),
    ("Army", Creature),
    ("Artificer", Creature),
    ("Assassin", Creature),
    ("Assembly-Worker", Creature),
    ("Astartes", Creature),
    ("Atog", Creature),
    ("Attraction", Artifact),
    ("Aura", Enchantment),
    ("Aurochs", Creature),
    ("Avatar", Creature),
    ("Azra", Creature),
    ("Background", Enchantment),
    ("Badger", Creature),
    ("Bajoran", Creature),
    ("Balloon", Creature),
    ("Barbarian", Creature),
    ("Bard", Creature),
    ("Basilisk", Creature),
    ("Bat", Creature),
    ("Bear", Creature),
    ("Beast", Creature),
    ("Beaver", Creature),
    ("Beeble", Creature),
    ("Beholder", Creature),
    ("Berserker", Creature),
    ("Betazoid", Creature),
    ("Bird", Creature),
    ("Bison", Creature),
    ("Blinkmoth", Creature),
    ("Blood", Artifact),
    ("Boar", Creature),
    ("Bobblehead", Artifact),
    ("Bolian", Creature),
    ("Book", Artifact),
    ("Borg", Creature),
    ("Brikar", Creature),
    ("Bringer", Creature),
    ("Brushwagg", Creature),
    ("C'tan", Creature),
    ("Caitian", Creature),
    ("Camarid", Creature),
    ("Camel", Creature),
    ("Capybara", Creature),
    ("Cardassian", Creature),
    ("Caribou", Creature),
    ("Carrier", Creature),
    ("Cartouche", Enchantment),
    ("Case", Enchantment),
    ("Cat", Creature),
    ("Cave", Land),
    ("Centaur", Creature),
    ("Chandra", Planeswalker),
    ("Child", Creature),
    ("Chimera", Creature),
    ("Chorus", Spell),
    ("Citizen", Creature),
    ("Class", Enchantment),
    ("Cleric", Creature),
    ("Cloud", Land),
    ("Clown", Creature),
    ("Clue", Artifact),
    ("Cockatrice", Creature),
    ("Construct", Creature),
    ("Contraption", Artifact),
    ("Coward", Creature),
    ("Coyote", Creature),
    ("Crab", Creature),
    ("Crocodile", Creature),
    ("Curse", Enchantment),
    ("Custodes", Creature),
    ("Cyberman", Creature),
    ("Cyclops", Creature),
    ("Dalek", Creature),
    ("Dauthi", Creature),
    ("Demigod", Creature),
    ("Demon", Creature),
    ("Denobulan", Creature),
    ("Desert", Land),
    ("Deserter", Creature),
    ("Detective", Creature),
    ("Devil", Creature),
    ("Dinosaur", Creature),
    ("Djinn", Creature),
    ("Doctor", Creature),
    ("Dog", Creature),
    ("Dragon", Creature),
    ("Drake", Creature),
    ("Dreadnought", Creature),
    ("Drix", Creature),
    ("Drone", Creature),
    ("Druid", Creature),
    ("Dryad", Creature),
    ("Dwarf", Creature),
    ("Echidna", Creature),
    ("Edosian", Creature),
    ("Efreet", Creature),
    ("Egg", Creature),
    ("El-Aurian", Creature),
    ("Elder", Creature),
    ("Eldrazi", Creature),
    ("Elemental", Creature),
    ("Elephant", Creature),
    ("Elf", Creature),
    ("Elk", Creature),
    ("Employee", Creature),
    ("Equipment", Artifact),
    ("Eternal", Creature),
    ("Excalbian", Creature),
    ("Eye", Creature),
    ("Faerie", Creature),
    ("Ferengi", Creature),
    ("Ferret", Creature),
    ("Fish", Creature),
    ("Flagbearer", Creature),
    ("Food", Artifact),
    ("Forest", Land),
    ("Fortification", Artifact),
    ("Fox", Creature),
    ("Fractal", Creature),
    ("Frog", Creature),
    ("Fungus", Creature),
    ("Gamer", Creature),
    ("Gamma", Creature),
    ("Gargoyle", Creature),
    ("Garruk", Planeswalker),
    ("Gate", Land),
    ("Germ", Creature),
    ("Giant", Creature),
    ("Giraffe", Creature),
    ("Gith", Creature),
    ("Glimmer", Creature),
    ("Gnoll", Creature),
    ("Gnome", Creature),
    ("Goat", Creature),
    ("Goblin", Creature),
    ("God", Creature),
    ("Gold", Artifact),
    ("Golem", Creature),
    ("Gorgon", Creature),
    ("Gorn", Creature),
    ("Graveborn", Creature),
    ("Gremlin", Creature),
    ("Griffin", Creature),
    ("Guest", Creature),
    ("Hag", Creature),
    ("Halfling", Creature),
    ("Hamster", Creature),
    ("Harpy", Creature),
    ("Heartwood", Artifact),
    ("Hedgehog", Creature),
    ("Hellion", Creature),
    ("Hero", Creature),
    ("Hippo", Creature),
    ("Hippogriff", Creature),
    ("Hirogen", Creature),
    ("Homarid", Creature),
    ("Homunculus", Creature),
    ("Horror", Creature),
    ("Horse", Creature),
    ("Human", Creature),
    ("Hydra", Creature),
    ("Hyena", Creature),
    ("Illusion", Creature),
    ("Illyrian", Creature),
    ("Imp", Creature),
    ("Incarnation", Creature),
    ("Incubator", Artifact),
    ("Infinity", Artifact),
    ("Inhuman", Creature),
    ("Inkling", Creature),
    ("Inquisitor", Creature),
    ("Insect", Creature),
    ("Island", Land),
    ("Jace", Planeswalker),
    ("Jackal", Creature),
    ("Jellyfish", Creature),
    ("Jem'Hadar", Creature),
    ("Juggernaut", Creature),
    ("Junk", Artifact),
    ("Kaito", Planeswalker),
    ("Kangaroo", Creature),
    ("Kavu", Creature),
    ("Kelpien", Creature),
    ("Kirin", Creature),
    ("Kithkin", Creature),
    ("Klingon", Creature),
    ("Knight", Creature),
    ("Kobold", Creature),
    ("Kor", Creature),
    ("Kraken", Creature),
    ("Kree", Creature),
    ("Kwejian", Creature),
    ("Lair", Land),
    ("Lamia", Creature),
    ("Lammasu", Creature),
    ("Lander", Artifact),
    ("Lanthanite", Creature),
    ("Leech", Creature),
    ("Lemur", Creature),
    ("Lesson", Spell),
    ("Leviathan", Creature),
    ("Lhurgoyf", Creature),
    ("Licid", Creature),
    ("Liliana", Planeswalker),
    ("Lizard", Creature),
    ("Llama", Creature),
    ("Lobster", Creature),
    ("Locus", Land),
    ("Manticore", Creature),
    ("Map", Artifact),
    ("Masticore", Creature),
    ("Medusan", Creature),
    ("Melkotian", Creature),
    ("Mercenary", Creature),
    ("Merfolk", Creature),
    ("Metathran", Creature),
    ("Mine", Land),
    ("Minion", Creature),
    ("Minotaur", Creature),
    ("Mite", Creature),
    ("Mole", Creature),
    ("Monger", Creature),
    ("Mongoose", Creature),
    ("Monk", Creature),
    ("Monkey", Creature),
    ("Moogle", Creature),
    ("Moonfolk", Creature),
    ("Mount", Creature),
    ("Mountain", Land),
    ("Mouse", Creature),
    ("Mutagen", Artifact),
    ("Mutant", Creature),
    ("Myr", Creature),
    ("Mystic", Creature),
    ("Na'kuhl", Creature),
    ("Nautilus", Creature),
    ("Necron", Creature),
    ("Nephilim", Creature),
    ("Nightmare", Creature),
    ("Nightstalker", Creature),
    ("Ninja", Creature),
    ("Noble", Creature),
    ("Noggle", Creature),
    ("Nomad", Creature),
    ("Nymph", Creature),
    ("Ocampa", Creature),
    ("Octopus", Creature),
    ("Officer", Creature),
    ("Ogre", Creature),
    ("Omen", Spell),
    ("Ooze", Creature),
    ("Orb", Creature),
    ("Orc", Creature),
    ("Orgg", Creature),
    ("Orion", Creature),
    ("Otter", Creature),
    ("Ouphe", Creature),
    ("Ox", Creature),
    ("Oyster", Creature),
    ("Pangolin", Creature),
    ("Peasant", Creature),
    ("Pegasus", Creature),
    ("Pentavite", Creature),
    ("Performer", Creature),
    ("Pest", Creature),
    ("Phelddagrif", Creature),
    ("Phoenix", Creature),
    ("Phyrexian", Creature),
    ("Pilot", Creature),
    ("Pincher", Creature),
    ("Pirate", Creature),
    ("Plains", Land),
    ("Plan", Enchantment),
    ("Plant", Creature),
    ("Platypus", Creature),
    ("Porcupine", Creature),
    ("Possum", Creature),
    ("Power-Plant", Land),
    ("Powerstone", Artifact),
    ("Praetor", Creature),
    ("Primarch", Creature),
    ("Prism", Creature),
    ("Processor", Creature),
    ("Q", Creature),
    ("Qu", Creature),
    ("Rabbit", Creature),
    ("Raccoon", Creature),
    ("Ranger", Creature),
    ("Rat", Creature),
    ("Rebel", Creature),
    ("Reflection", Creature),
    ("Rhino", Creature),
    ("Rigger", Creature),
    ("Robot", Creature),
    ("Rogue", Creature),
    ("Role", Enchantment),
    ("Romulan", Creature),
    ("Room", Enchantment),
    ("Rune", Enchantment),
    ("Sable", Creature),
    ("Saga", Enchantment),
    ("Salamander", Creature),
    ("Samurai", Creature),
    ("Sand", Creature),
    ("Saproling", Creature),
    ("Satyr", Creature),
    ("Scarecrow", Creature),
    ("Scientist", Creature),
    ("Scion", Creature),
    ("Scorpion", Creature),
    ("Scout", Creature),
    ("Sculpture", Creature),
    ("Seal", Creature),
    ("Serf", Creature),
    ("Serpent", Creature),
    ("Servo", Creature),
    ("Shade", Creature),
    ("Shaman", Creature),
    ("Shapeshifter", Creature),
    ("Shard", Enchantment),
    ("Shark", Creature),
    ("Sheep", Creature),
    ("Shi'ar", Creature),
    ("Shrine", Enchantment),
    ("Siren", Creature),
    ("Skeleton", Creature),
    ("Skrull", Creature),
    ("Skunk", Creature),
    ("Slith", Creature),
    ("Sliver", Creature),
    ("Sloth", Creature),
    ("Slug", Creature),
    ("Snail", Creature),
    ("Snake", Creature),
    ("Soldier", Creature),
    ("Soltari", Creature),
    ("Sorcerer", Creature),
    ("Spacecraft", Artifact),
    ("Spawn", Creature),
    ("Specter", Creature),
    ("Spellshaper", Creature),
    ("Sphere", Land),
    ("Sphinx", Creature),
    ("Spider", Creature),
    ("Spike", Creature),
    ("Spirit", Creature),
    ("Splinter", Creature),
    ("Sponge", Creature),
    ("Spy", Creature),
    ("Squid", Creature),
    ("Squirrel", Creature),
    ("Starfish", Creature),
    ("Stone", Artifact),
    ("Surrakar", Creature),
    ("Survivor", Creature),
    ("Swamp", Land),
    ("Symbiote", Creature),
    ("Synth", Creature),
    ("Talaxian", Creature),
    ("Talosian", Creature),
    ("Tamarian", Creature),
    ("Tellarite", Creature),
    ("Tentacle", Creature),
    ("Terminus", Artifact),
    ("Tetravite", Creature),
    ("Thalakos", Creature),
    ("Tholian", Creature),
    ("Thopter", Creature),
    ("Thrull", Creature),
    ("Tiefling", Creature),
    ("Time Lord", Creature),
    ("Tosk", Creature),
    ("Tower", Land),
    ("Town", Land),
    ("Toy", Creature),
    ("Trap", Spell),
    ("Treasure", Artifact),
    ("Treefolk", Creature),
    ("Tribble", Creature),
    ("Trill", Creature),
    ("Trilobite", Creature),
    ("Triskelavite", Creature),
    ("Troll", Creature),
    ("Turtle", Creature),
    ("Tyranid", Creature),
    ("Unicorn", Creature),
    ("Urza's", Land),
    ("Utrom", Creature),
    ("Vampire", Creature),
    ("Varmint", Creature),
    ("Vau-N'Akat", Creature),
    ("Vedalken", Creature),
    ("Vehicle", Artifact),
    ("Villain", Creature),
    ("Vivien", Planeswalker),
    ("Volver", Creature),
    ("Vorta", Creature),
    ("Vulcan", Creature),
    ("Wall", Creature),
    ("Walrus", Creature),
    ("Warlock", Creature),
    ("Warrior", Creature),
    ("Weasel", Creature),
    ("Weird", Creature),
    ("Werewolf", Creature),
    ("Whale", Creature),
    ("Wizard", Creature),
    ("Wolf", Creature),
    ("Wolverine", Creature),
    ("Wombat", Creature),
    ("Worm", Creature),
    ("Wraith", Creature),
    ("Wurm", Creature),
    ("Xindi", Creature),
    ("Yeti", Creature),
    ("Zombie", Creature),
    ("Zubera", Creature),
];

/// Every subtype in the table, in name order.
pub fn subtypes() -> impl Iterator<Item = &'static str> {
    SUBTYPES.iter().map(|(name, _)| *name)
}

/// Sorted CR 205.3m names for "choose a creature type" (CR 205.3e).
pub fn creature_type_names() -> &'static [&'static str] {
    static NAMES: std::sync::LazyLock<Vec<&str>> = std::sync::LazyLock::new(|| {
        SUBTYPES
            .iter()
            .filter_map(|&(name, kind)| (kind == Creature).then_some(name))
            .collect()
    });
    &NAMES
}

/// The list `subtype` is on, or `None` for a name outside the table.
pub fn subtype_category(subtype: &str) -> Option<SubtypeCategory> {
    SUBTYPES
        .binary_search_by(|(name, _)| (*name).cmp(subtype))
        .ok()
        .map(|index| SUBTYPES[index].1)
}

/// Whether `subtype` is on the list of a card type the object has
/// (CR 205.3d): kindred uses creature types (CR 205.3m), and instants and
/// sorceries share spell types (CR 205.3k). Planeswalker and battle types are
/// not in the table, so nothing corresponds to them.
pub fn subtype_corresponds_to(subtype: &str, card_type: crate::card::CardType) -> bool {
    use crate::card::CardType as T;
    let wanted = match card_type {
        T::Creature | T::Kindred => Creature,
        T::Land => Land,
        T::Artifact => Artifact,
        T::Enchantment => Enchantment,
        T::Instant | T::Sorcery => Spell,
        T::Planeswalker | T::Battle => return false,
    };
    subtype_category(subtype) == Some(wanted)
}

/// CR 205.3m. An unlisted name is not a creature type; debug builds reject it,
/// so a new subtype must be classified before any rule reads it.
pub fn is_creature_type(subtype: &str) -> bool {
    let category = subtype_category(subtype);
    debug_assert!(category.is_some(), "unclassified subtype {subtype:?}");
    category == Some(Creature)
}

/// An object's creature types (CR 205.3m): the ones it lists, or every
/// creature type at once ("all creature types", Soulstone Sanctuary). "All"
/// is one value rather than an enumeration of the creature-type list, so it
/// also covers types no card in the catalog names.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CreatureTypes {
    /// Sorted and distinct.
    Listed(Vec<String>),
    All,
}

impl Default for CreatureTypes {
    fn default() -> Self {
        Self::Listed(Vec::new())
    }
}

impl CreatureTypes {
    /// The distinct creature types among `subtypes`.
    pub fn among<'a>(subtypes: impl IntoIterator<Item = &'a String>) -> Self {
        let mut listed: Vec<String> = subtypes
            .into_iter()
            .filter(|subtype| is_creature_type(subtype))
            .cloned()
            .collect();
        listed.sort();
        listed.dedup();
        Self::Listed(listed)
    }

    /// The listed creature types; none for all creature types.
    #[inline]
    pub fn listed(&self) -> &[String] {
        match self {
            Self::Listed(listed) => listed,
            Self::All => &[],
        }
    }

    #[inline]
    pub fn is_all(&self) -> bool {
        matches!(self, Self::All)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Listed(listed) if listed.is_empty())
    }

    pub fn contains(&self, creature_type: &str) -> bool {
        match self {
            Self::Listed(listed) => listed.iter().any(|t| t == creature_type),
            Self::All => is_creature_type(creature_type),
        }
    }

    /// Whether the two have a creature type in common ("that share a
    /// creature type"); all creature types shares with any nonempty list.
    pub fn shares(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All, other) | (other, Self::All) => !other.is_empty(),
            (Self::Listed(a), Self::Listed(b)) => a.iter().any(|t| b.contains(t)),
        }
    }

    /// The creature types both have.
    pub fn intersection(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::All, other) | (other, Self::All) => other.clone(),
            (Self::Listed(a), Self::Listed(b)) => {
                Self::Listed(a.iter().filter(|t| b.contains(t)).cloned().collect())
            }
        }
    }
}

/// A listed set serializes as its names, so retained records keep their
/// bytes; all creature types is the string `"all"`.
impl serde::Serialize for CreatureTypes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Listed(listed) => listed.serialize(serializer),
            Self::All => serializer.serialize_str("all"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtype_table_is_sorted() {
        assert!(SUBTYPES.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn non_creature_subtypes_are_not_creature_types() {
        for name in ["Cave", "Shrine", "Lesson", "Saga", "Equipment", "Forest"] {
            assert!(!is_creature_type(name), "{name}");
        }
        assert_eq!(subtype_category("Shrine"), Some(Enchantment));
        assert_eq!(subtype_category("Cave"), Some(Land));
        assert!(is_creature_type("Citizen"));
        assert_eq!(subtype_category("Not A Subtype"), None);
    }
}
