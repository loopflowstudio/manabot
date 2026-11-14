# Pydantic direction

Jack Heart, 2026-11-13: "i think keep the pydantic config required for now";
"i am fine with using pydantic as basic a dependency / native protocol to etude".
Pydantic is Etude's required native config contract. Keep train(config) and
training(train, config=...); no kwargs transport or optional config adapter.
Full remarks: 382c653ec8:scratch/config-api-direction.md.
