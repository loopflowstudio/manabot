# Config interface review

2026-11-13, Jack Heart: "i think keep the pydantic config required for now".
He clarified: "i guess if there is a best-of-both worlds approach im fine with it, but i want to work BEST with systems that define their configs as pydantic objects".

Further direction: "i am fine with using pydantic as basic a dependency / native protocol to etude". Treat Pydantic as Etude's native configuration contract, not an optional adapter to abstract away. Learner independence does not require configuration-library independence.

Jack Heart: "Right. Etude is a system designed for people using the best first-in-class tools. it is allowed to be opinionated where it needs to be." Favor a deliberate stack and strong conventions over speculative compatibility layers. The execution boundary leaves training logic with its author; it does not require neutrality about configuration or infrastructure tools.

Jack Heart: "it wants to be flexible across usecases, learning approaches, compute providers, and lots of other things. but infrastructure design is something its meant to \"do for you so you dont have to think about\" and its allowed to require using an API that enables that to work most reliably". Flexibility across workloads and providers is compatible with a required execution contract. Prefer APIs that let Etude own infrastructure reliably, rather than exposing infrastructure choices merely for configurability. Pydantic configuration, run identity, artifact persistence, stopping and retry contracts can be prescribed while learning logic remains ordinary user code.

Keep `train(config: FrontierConfig)` and `training(train, config=...)` as the
primary interface. Pydantic owns defaults, constraints and serialization;
Etude validates and transports that config. The review sketch's proposed
`kwargs` transport is withdrawn. Ordinary function bodies and explicit loops
work with this interface; extracting execution responsibilities does not
require replacing typed configs or embedding an execution plan in them.
