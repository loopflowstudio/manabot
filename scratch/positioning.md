# Etude positioning (explore, 2026-11-13)

Page: `scratch/explore.html`. Nothing here is decided.

Jack Heart, 2026-11-13, on the goal of the cutover: "move as much execution
management outside of Manabot and into Etude as possible so that Loop code can
be really focused on algorithms and parameters". On Manabot: "kind of a library
that can help you design your own training experience experiments and you can
pick and choose different components of it". On the loop: "how does that
compare to a clean RL or a PufferRL-style file? These systems are
anti-frameworks and they're single-file". On sequencing: "I really want to
finalize and get the client experience first before we release the API".

Jack Heart, 2026-11-13, opening this exploration: compare against ClearML and
W&B ("clear tablestakes we should learn from, what did customers find
frustarting"), Prime Intellect and Tinker ("how do we skate to where the puck
is going without just being a follower-on"), "focusing specifically on
multiagent RL (but still wanting to keep open exanding beyond magic and its
close cousins, into stuff more in the space of what softmax.com says they
do)", and "having Etude evolve into having a cloud service where that has a
research ontology and experiment store which serves as higher level interface
for allocating agents to run experiments for you automatically".

Three directions shown, by central noun:

- A, the run: a small contract that makes any loop survive rented machines.
- B, the player and the match: a league; results are relations between players.
- C, the question: a research record (Question, Experiment, Run/Match,
  Evidence, Finding) that agents are assigned to, with budgets and review.

They stack (A floor, B domain model, C service). Recommendation offered, not
accepted: B leads in open source, A stays small, C's records are files now.

Jack Heart, 2026-11-13, after the first page: "right sf-compute's autoresearch
is another comp". Added to the page: SF Compute Autoresearch (launched
2026-09-11; agents get an H100 container over MCP/CLI, a harness that records
results, forkable sandboxes, managed agents "given source code, an objective
and a budget" in private preview). It splits C into C1 (Etude runs the agents,
competing with SF Compute) and C2 (Etude is the record and guard rails any
agent writes to, with SF Compute as one more place to rent). Question 7 asks
which.

Jack Heart, 2026-11-13, on question 7 (dictated; "A2" read as Etude and "gave
me a note" as givemeanode): "we do have Loopflow as kind of a core asset, which
is like an agent orchestrator. I do want to pitch A2 as actually sort of
directly steering the agents. Obviously you can have a chat UI with one of the
agents on top of it but it should be going through Loopflow basically. It's
possible we could start with C2."

"What I don't want to do is compete on compute. I think where I want to
position Etude is sort of like it's a compute-independent substrate I guess.
One analogy I would bring is that Loopflow has the idea: it's not a harness,
it's a meta-harness. It sits on top of all the harnesses and believes that
there's going to be lots of models and lots of competing harnesses."

"companies that have their own long-term compute are just going to want to
bring their own cloud. Other people are going to run rent-only. I would focus
more on the 'hey, if you don't have a lot of allocated compute and you're
trying to scrape it together in various ways, Loopflow kind of helps you do
that efficiently and without a lot of headache.'" (He says Loopflow here; the
launcher that scrapes compute together is Etude's. Question 8 asks.)

"to some degree, we're targeting a smaller audience than what SF Compute is."

Page updated: C now reads as Etude steering agents through Loopflow, never
running its own agent harness and never selling compute; EXP-58's machine
model is named as the compute-independence work. New questions 8–10.

Jack Heart, 2026-11-13, answering 8–10:

8: "8A/C -- for right now ive decided to keep the compute launching interfaces
different, and its not clear that we will need agentcompute launching
(possible that e.g. the agents run on your own laptop or its more you decide
where the agents run by using loopflow,)"

9: "question is the key to the Etude Cloud. Again maybe we need to change the
naming here but there's a difference between the Etude SDK, which is what
we're building right now, and the Etude Cloud. The Etude Cloud would involve
some combination of a hosted version of the SDK as well as this additional UI
and ontology around questions"

10: "I'm not really sure I quite understand the question. I am currently using
Loopflow to launch and run agents and then you launch things in Etude. I think
that overall experience or flow is correct. Over time some of that may move to
the cloud but I also think it's possible that the experience is primarily that
you continue to use agents on your own machine. Those agents are given some
combination of Loopflow and Etude SDKs to drive both the Etude cloud ontology
and questions and research and sort of input and output and drive questions
from there as well as to launch experiments. I think once the experiment
launches this may be a little bit different. It might be a different agent
than the agent you're using to decide which experiments to launch"

Corrections found in etude-studio after the first page: the question ontology
already exists as Inquiry (`docs/design/inquiry-model.md`, decisions 51–53,
STU-10); a hosted prototype runs on Fly; `placement.md` proposes where parts
go; STU-3 splits out a machines layer and overlaps EXP-58. The page's C
section predates reading these.

Filed: EXP-57 (live view), EXP-58 (machine model and metrics), STU-14 (hosted
single deployment with Loopflow Studio sign-in).

Jack Heart, 2026-11-13, on 11 (what the tending agent may do without asking:
a watch only, b also cancel/retry/continue within budget, c also change the
experiment within budget) and 12 (naming: a "Etude SDK"/"Etude Cloud", b the
hosted product gets its own name, c decide later): "11 b or c, 12 b or c".
So 11a and 12a are out; between b and c each stays open.

On STU-14: "It's supposed to be 'own bucket credentials' I think. I think I
want to be able to continue to run the [Etude] SDK from my laptop without
having to go to the server but then the server can still read them". Brief
updated: the server is a second reader of the bucket, never in a run's write
path.

Jack Heart, 2026-11-13, on 11: "actually 11c almost certainly. but probably
there are firm boundaries like the cant exceed the experiment budget. i dont
see it as a ig question."

Jack Heart, 2026-11-13, on the W&B complaints table, naming the slow
interface and unreliable sync, the single step axis, and history living in
someone else's service: "These are 3 of the things i experience with wandb.
Another major one i had was that we had a very hard time having 'company
dashboards' and getting people to have consistent views into ongoing
experiments that they understood. charts were always moving around, the step
function would hcange, etc". Added to the page as a fourth row; posted to
EXP-57 and STU-14 as a comment.

Jack Heart, 2026-11-13, on what defines Etude (bears on question 1):

"I think the Etude sort of takes for granted the idea that training runs are a
fundamental unit of doing RL, if not just ML in general, much in the same way
that a checkpoint is. Every product that does ML ops is built around
checkpoints in a certain way but it's not like checkpoints are what defines
the product. I think training runs are kind of like that for Etude."

"I think one of the things that really defines Etude is an operational tool. I
am currently trying to imagine the idea that Etude is a framework that is of
value, interesting, useful, worth using, and sort of worth adopting. I guess I
would say in a previous job I used Wandb and Sky Pilot. I would like to get to
the point where I can tell people, 'Hey if you use those tools and you're a
little bit unsatisfied with them, you should try using ours.'"

"That gets into the question of how much of their things should we directly
implement and compete on and say we're a drop-in replacement? Do you want to
do some of that? What are the things where we're totally differentiating? I
think that's the question we're trying to answer."

"I think one of the biggest places we're differentiating is the focus on
evals. I think my experience with Wandb was that it was a little bit weak on
the idea of how you evaluate what's going on. In particular this idea of
multi-agent evals is about being able to generate evals where you're comparing
different checkpoints with each other in certain environments."

Page: round four proposes three tiers (match, decline, differentiate) against
W&B and SkyPilot. Proposal only. SkyPilot points are from memory, unchecked.

Jack Heart, 2026-11-13, on the four "direction of travel" points ("A2" read as
Etude):

On "the loop goes back to the researcher": "Agreed. I will note that I think
Tinker is kind of an even more extreme example than what we're doing in terms
of them really saying, 'Just write the inner loop and we'll manage the rest.'
I think that really gets at the core of what we're trying to move Etude
towards in this current change of 'Can we make Etude do more of it for you?'
The difference is that we're not behind a closed API. We're just like, 'Yeah
we're a library so you don't have to worry about it. You can just call our
code.'"

On "the reusable thing is the environment and its scoring": "I do think we're
talking about evals, which is more similar to Prime. But we're saying, 'Oh
it's not just the environment,' because it is sort of the environment, but
only if, by the environment, you also include: what are the other policies
that are in the situation? With Magic there is a lot about: 'Okay are you
doing drafting? Are you doing deck building? What format are you doing?' I
think there is a lot of emphasis on, when you're doing an experiment, how you
set up the environment but it's also about who the other players are. This is
kind of a world where there are more different policies that are being trained
and it's more evolutionary."

Correcting "nobody is selling a Trainer class any more": "I think Tinker kind
of is saying, 'We'll sell you a trainer class.' and I don't want to do that,
right? At least not in A2. We're building a trainer class and Manabot, we're
building a trainer specifically for Magic: The Gathering. We're not trying to
sell that because, for obvious reasons. Maybe if that gets really good, we
might talk about how we evolve that into other domains but that's not what A2
does".

On "written for coding agents as much as for people": "100% this".

On the fourth point (experiments getting cheap; autoresearch), in his next
message: "Experiments are relatively cheap to write but they can be expensive
to run. More than this I think I'm moving to a world where designing the
experiment is kind of like the lingua franca of doing work. In a way it used
to be that writing an email or writing a task was sort of like the work of a
product manager or something and then you had engineers who turned that into
code. Well now we have the agents who turn that into code even more reliably.
Even more so you can think about experiment design as almost sort of like a
coding language because it can be 'compiled' and executed by the 'agent'
runtime". Page: round five; question 15.

Jack Heart, 2026-11-13, after round five (dictated; "Attune" read as Etude):
"I think we're getting a little ahead of ourselves. I think 'experiment' as a
formalized concept really starts to come up more when we get to Attune Cloud.
I think it's important to know that's kind of where we're going but right now
I want to focus on Attune SDK".

Jack Heart, 2026-11-13, on "a budget per experiment, not only per run": "Again
this is, I think, a formal definition of an experiment that sits with Etude
Cloud. I think an Etude SDK experiment is just kind of a nice name for a
label". On machine-readable output: "This is probably a nice isolated task"
(filed as EXP-59). On a README that opens with an evaluation: "Also could be
an isolated task but I think rewriting the README is probably in scope for
today for this work here". Draft: `scratch/etude-readme-draft.md`, not applied
to Etude.

Scope from here: the SDK. Parked as hosted-product questions, not to be
pursued now: 3, 4, 7, 9, 10, 12, 15, and the formal experiment/inquiry record.
Still open and about the SDK: 2 (how wide multi-agent is), 6 (what happens to
EXP-56), 13 (what evaluation covers in the README), 14 (how far to match
SkyPilot), and whether game setup stays an opaque conditions value. Question 1
was spoken to (runs are the assumed unit; evals differentiate) without a
letter.

Jack Heart, 2026-11-13, on existing Etude Tasks. STU-3: "Merge this into 58.
58 is probably more up to date". EXP-12: "Okay I agree. This is probably
something that should start as soon as we're done with 56". EXP-37: "Fold this
one into 56". EXP-14: "Close for now I guess". EXP-15: "Don't really know what
this one is. Let's leave it". On the bug where a refused or interrupted attempt
marks a finished run failed: "This should be part of 56". On an experiment
budget across launches: "Yeah okay, this probably is its own task". On the
three evaluation gaps (pool seat, training reading the population, rating over
every checkpoint): "Okay let's just file all these as one task, which is just
'Level Up Evaluation API.'"

Done: EXP-56 steered with the bug and EXP-37's remaining points; EXP-37, EXP-14
and STU-3 completed with reasons (STU-3's unmerged branch and worktree
`/Users/jack/src/etude.machines-and-runtimes` kept; summary on EXP-58); EXP-12
commented; EXP-60 (evaluation API) and EXP-61 (spending limit across launches
under one experiment label) filed. EXP-15 untouched.

Jack Heart, 2026-11-13 (dictated; "A2SDK" read as Etude SDK): "I want you to
rewrite the explore page with two different things to explore and then you can
talk about the matrix of them and how they relate. 1. What is the single-line
or above-the-fold landing page for A2SDK? What does the catchline look like?
What do we like? What is the video or image that we show to immediately
demonstrate the value and who should be using this product? 2. What are the
ways that we could design the more execution ops API in 56?"

Page rewritten for that: `scratch/explore.html`. The earlier page (rounds one
to five) is kept at `scratch/explore-rounds-1-5.html`.

Jack Heart, 2026-11-13, on the three earlier directions (dictated; "A2SDK"
read as Etude SDK): "In terms of your previous options of Run Contract, Run
Contract League, and Lab That Remembers, I think Lab That Remembers is A2SDK
and we're trying to build lower-level primitives for that. League is, I think,
closer to the face of A2SDK but what we're trying to build now is mostly
focused on the train API, which is the building block for the League API".
Then, correcting the first sentence: "Lab That Remembers is etude cloude".

The page's fronts (L1 league, L2 unsatisfied with W&B and SkyPilot, L3 the run
that survives, L4 the agent's bench), loop shapes (E1 handle as built, E2
`run.steps`, E3 Etude owns the loop, E4 W&B-shaped module) and state positions
(S1 directory as built, S2 `run.save`/`run.load`, S3 checkpoint is the resume
point) are proposals. Only E1 and S1 exist. Questions 1–6 on the page are open.

Jack Heart, 2026-11-13, first impressions of the rewritten page (question 1):
"in terms of the landing page, L1 definitely beats the other ones  I think
maybe we even need to go higher up the stack and go closer to Etude Lab before
we have something really compelling but I think L1 is probably the place to
aim for right now". ("Etude Lab" is his phrase here; earlier the same layer
was "Etude Cloud" and "Lab That Remembers". Whether it is a name is not
settled; naming question 12 stays "b or c".) Questions 2–6 not yet spoken to.

Jack Heart, 2026-11-13, on the loop shapes: "In terms of E1 through E4 I
definitely like E2 the best. I am somewhat intrigued by E3 but I am unsure
about how flexible that would be and whether that would fit as we start to do
things like, 'Oh we're training.' If we look at Ryan's Jace, for example
(I'll link that below), that is a sophisticated thing that has multiple models
and potentially nested loops and such. I don't want to go so far as to rule
out innovative learning patterns like that" (https://github.com/RyanSaxe/jace).

On who owns state (dictated; "pedantic" read as Pydantic): "I think the Etude
philosophy is: we write code, you run it on your machine, it's saved to your
machine, it's in normal-ass formats that you can read, and you own it. We do
not own it. We may separately build a product where we manage it for you but
at a fundamental level you can use our framework to save your state. Using our
framework to write your trainer does not lock you into us". "I think there's a
separate question about how that state is formatted. I think having a state
pedantic variable that makes it really clear what you get, what things we're
saving, and how they relate to each other, that's all good. I'm into that. I
think we can maybe hide more code behind Etude that you don't really care
about if you do that. But I do think, fundamentally, we're going to be
publishing directories, checkpoints, and JSON files. Probably mostly, a lot of
the exports of this, of all of the state, look the same. I agree that can be
`state.export` or whatever rather than sort of a direct call".

Jace as read 2026-11-13 (pushed 2026-11-12): `loops/limited.py` runs draft,
build, play, learn per iteration; five models, six optimizers (shared
embedding); opponents and other draft seats may be past versions; own
`tracking.py` (14 KB) writing identity.json, state/metrics/evaluations .jsonl,
`checkpoints/<model>-<update>.pt` and an `inputs` lineage list. Page round two
tests E2 and E3 against it and sketches a Pydantic State saved as
`state.json` plus the author's files. Not checked: whether `run.checkpoint`
in PR #103 records several named models at one point. Questions 7–11 open.

Jack Heart, 2026-11-13, asked "remind me what URL do we own for etude?", then
"i meant for etude studio", then of `etude.gg`: "i didnt intend to use
etude.gg for that". No domain for Etude (the SDK or Studio) is recorded in
either repository; only `etude-center.fly.dev`. Where the L1 landing page
would live is open.

Jack Heart, 2026-11-13: "can i see a new page which is a bigger sketch of what
manabot's loops, as well as what like the core user docs would say to define
the core etude train APIs would look like post exp-56". Page:
`scratch/sketch.html` (Etude "Training" docs page; Manabot `Saved`/`export`/
`restore`, `self_play`, the Elemental and belief recipes, the minimal loop;
six additions to PR #103; six questions). Proposals only. One design point
found while writing it: `run.steps` must raise `Stopped` on the pass after a
stop is asked for, not end quietly, or a phased recipe (belief) runs its next
phase on a half-trained policy; that in turn means `run.save` is called every
pass and Etude decides when the export runs.

Jack Heart, 2026-11-13, reading `scratch/sketch.html`. On the config: "you
bring your own config, we jsut require it to be pydantic?" then "docs should
make that clear. but sounds good." and "also make sure 'you write a file that
has a train func...'". On `with track(...) as run`: "I mostly like this, some
concerns about it being a little more than trakcing now, its really like
recording but i supose that is close to a snymonym. its just to me tracking
doesnt clearly say like 'save logs, and checkpoints, ...'". Page updated: the
docs open with the file and state the three requirements on config and
`train`; the name is question 7, open.

Jack Heart, 2026-11-13, quoting the sketch's State paragraph ("State is what
your next attempt needs to carry on. You describe it and write it; Etude makes
the write safe and carries it between machines."): "and define its output?"
Read two ways and answered both; which he meant is not confirmed.

Jack Heart, 2026-11-13, quoting the Jace-against-E2/E3 summary: "ok, lets
redesign the sketch to show how this would work". `scratch/sketch.html` is now
three parts: Etude's Training page; Jace as the hard case (a proposed
`run_limited.py` around its unchanged `limited_iteration`, the files it
leaves, a nested-loop variant with a second clock, and the same checkpoint
seated two ways); Manabot. Checked in PR #103 at `105abdd`: `run.checkpoint`
takes one artifact per call, and a different file at an already recorded
coordinate is kept as `CLOCK-VALUE.ordinal-N`, with no model name. Proposed
for that: a checkpoint may be a folder written by an export function with a
`checkpoint.json`, the same way state is written; `run.checkpoints()` lists
this run's. Not checked: how evaluation hands a folder to playing code.

Jack Heart, 2026-11-13, on the sketch: "i guess we should also add like:
writing this kind of code is relatively cheap. individually we shouldnt expect
that the way we manage checkpoints or even record metrics to be anything
someone cant do with an agent effectively for free. where it starts to add
value is just when its a shared and interplays with other parts of hte system"
and "the API (and the positioning) should be designed with that in mind".
Page: a section "The test for every call" lists, per thing the loop writes,
what else in Etude reads it. By that test the `state.json` record's fields,
`run.save(every=)` and a `trained_against` table are the weakest (docs
conventions, not API); the verdicts are mine, not his.

Jack Heart, 2026-11-13: "i think we should also consider whether any prior art
(Katago, AlphaStar, OpenAI 5, Ataraxos, etc) has more of a multi-stage
training setup where you first train A, then train B, that might want slightly
different set u than something like Jace (concurrent training of multpiple
policies). It seems like the end state may be allowing experiments to have
like a full DAG of training stages, each of may have multiple runs, and each
run having possibly multiple trained models, but the heavy heavy default
should be 1 stage 1 model, and the API should accept that but generalize where
needed". Page: Part 4 (prior art from memory, Ataraxos's staging not
established; a five-rung ladder; `runs(state)` with `start=` and
`run.input("start")` as the staged form, no stage noun). Built parts it leans
on, checked at `105abdd`: `Input`, `continue_from`, `ExperimentState.run`.

Jack Heart, 2026-11-13, on the config rules in the sketch: "is there no
Pydantic class that just defines both ?" and "i want to understand how we
differentiate at the CLI level between 'start a new run' and 'return to the
run i was previously running'". Checked at `105abdd`: `etude launch` always
makes a new run; a lost machine or `Stopped` is retried as a new attempt of
the same run with no command; `etude continue --run RUN --until` makes a NEW
child run from a copy of the parent's out (the sketch had wrongly said "the
same run, further"; corrected); locally the directory given to `local_launch`
is the identity (Manabot: `--out DIR`, rerun to resume). No command returns to
an ended, unfinished launched run as the same run. Sketch now proposes
`etude run` (new, local) and `etude resume RUN` (same run). Open.

Jack Heart, 2026-11-13, on the run/resume verbs: "seems fine, but run should
be able to take --resume or something probably? and also we should allow
people to define their own run id (not sure where we would enforce
uniqueness)". Sketch: `--resume RUN` on `etude run` and `etude launch`
(replacing a bare `etude resume`), `--name` for a chosen id, unique within an
experiment. Seen at `105abdd`: `Job.name` exists and `ExperimentState.run(name)`
looks runs up by it; locally the run id is the out directory's basename;
`track`'s docstring says identity and "whether it resumes" is settled by
`etude.run.declaration.declare`, which I did not read.

Jack Heart, 2026-11-13, on my "Saved appears only if you want resume state":
"why would you not be able to resume a run. that should be something we are
saying we handle automatically". Sketch docs: "State" replaced by "Carrying
on": `run.resume(model=..., optimizer=...)` registers objects with
`state_dict`/`load_state_dict`, loads the last save into them and returns the
clocks; `run.save()` takes no arguments. Costs Etude a per-framework writer
(PyTorch first). Parts 2 and 3 of the page not yet redrawn to it.

Jack Heart, 2026-11-13, on `format`, `world` and `counts` in Manabot's saved
state: "i would think anything like this is just on you to not change between
runs. like yeah if you change the code within attempts at a run, you get
undefined behavior. seems ifne". Sketch docs now say so. Consequence for
Manabot (mine): `format`, the `world` check and `StateMismatch` leave the
loop's state; `counts` is data, not a guard, and still rides along.

Jack Heart, 2026-11-13: "we also have names for run (is seeded) and job (is
not seeded). possible the api here should be around one or the other to make
this more clear?" Checked at `105abdd`: `Job` is "What to run, without a
seed"; `Run` is "A job and a seed: the thing that is launched";
`job.runs(*seeds)`; `Run(id="baseline-seed-7", job=job)` chooses a run id
(so chosen ids are built for launched runs; my earlier answer missed this);
the job receives `ETUDE_SEED`. Sketch: a "Jobs and runs" section; the seed
leaves the config and is read as `run.seed` (proposed). Not checked: whether a
job's identity digest includes the source revision, which would make
"resume with fixed code" a different job.

Jack Heart, 2026-11-13: "ok, great. i think this is enough to write a new
design. ; Be ambitious. Help us land in the new world where we're set up for
the most success in terms of adoption by Jace and evolution into the Etude
Cloud world. Don't worry about requiring a lot of churn on existing code of
Manabot or sticking with the design as currently written or proposed. None of
that matters." and "Really lean into all the things I emphasize today and
write a design for MTG-329 and EXP-56 that helps us get there. Then I will
tell 56 to then tell the 56 task worker and 329 workers to rebuild based on
this design." Design: `scratch/design-train-api.md`. Neither Task has been
steered with it. Checked while writing: a `Job`'s identity includes its
source, and `declare` today resumes a re-used run name.

Jack Heart, 2026-11-13: "lets walk through the otions of hte five open points",
then on point 1 (the name of `track`): "1: tracked_run?" A suggestion put as a
question; not recorded as chosen. Points 2–5 not yet spoken to.
Then: "or track_run?" Both `tracked_run` and `track_run` are his candidates; neither chosen.
On `Run` in a job file beside the object inside `train`: "yeah hmm and the Run
object returend by tracked_run is not a Run?" Fact at `105abdd`: two classes
are both named `Run`: `etude.launch.records.Run` ("A job and a seed: the
thing that is launched") and `etude.train.run.Run` (the handle `track`
returns). Also `RunRecord` (launcher) and `TrainingRun` (observed state).

Jack Heart, 2026-11-13: "i dont know i think we need a higher level ontology
review". Written: `scratch/ontology.md` (twelve-noun sentence; what has
several names; what one name means several things; nine proposals; six
questions). Found while writing it: `state.json` already exists as `RunState`
("how a tracking attempt ended"), so the design's `out/state/state.json`
reuses a taken name; and `Model` is already an Etude eval noun (checkpoint
plus how it plays), which the sketches' `model=` blurs.
While it was being written: "maybe Run and etude.run want ot change". Added to
`scratch/ontology.md` as option 1e. `etude.run` is the package
`src/etude/run/` (declaration, delivery, storage, records), distinct from
`etude.train.run`, where `track` and the live `Run` are defined.
Then: "ok, RunRequest works". Design and ontology updated: `Run` is the object
inside `train`; the declared job-and-seed is `RunRequest`. With `Run` free, the
`tracked_run`/`TrackedRun` idea is not needed; the name of `track` itself is
still as he left it ("I mostly like this", with the concern). Open: the new
name for the internal `etude.run` package.
On the declared side having no visible class: "still in favor of not needing
to know its called a RunRquest". Design: a job file returns `Job`; seeds are a
launch option (`training(..., seeds=(0, 1, 2))`); `RunRequest` stays
underneath. Not checked: what seed a run gets today when none is given
(`RunRequest.seed` may be None), or how run ids are derived now.
On the internal `etude.run` package's new name (I had suggested
`etude.tracking`): "or etude.track?" Not settled. Noted: `track` the function
is defined in `etude.train.run` today, with `Step`/`Time` in
`etude.train.records`; EXP-56's notes say its `AGENTS.md` asks for imports
from the defining module.
Then: "etude.tracking also fine iether way". Design takes `etude.tracking`,
so the package and the `track` function do not share a name; my choice
between two he allowed.
On Etude saying "player" and leaving "model" to the author: "works for me".
Written into the design (evaluation section) and ontology question 2. Not
checked: how many stored evaluation records carry the word `model`, and
whether main at v0.3.1 still names the class `Model`.

Jack Heart, 2026-11-13: "are 329 and 56 moving on the new design?" (they were
not: both at demo, nothing running, no commit since the morning) and "walk me
through the question". Asked, in this order: (1) two attempt classes: a one
record, b one noun over two records, c leave; (2) five checkpoint classes: a
reduce to `CheckpointRecord` plus one reference, b docs only, c leave; (3) the
stored run's names: a converge on `RunRecord` now, b later, c leave; (4) group
nouns: a experiment and job, b also cohort, c all four; (5) `state.json`
clash: a rename the existing one `outcome.json`, b rename the new one, c keep
both; (6) how much renaming in EXP-56: a only what the train API touches, b
all before release, c none. His answers: "1a -- i think RunAttempt is probably
better than Attempt though", "2a", "3a", "4a, or alternatively use cohort
instead of experiment as a more general word", "5a", "6b". Design: new Names
section. Neither Task steered yet.

Jack Heart, 2026-11-13: "once thats written please make sure to steer both
tasks, and copy the design over to the exp-56 worktree". And on the three
consequences (EXP-56 substantially bigger with single record classes across
the launcher database and tracked folder; EXP-12 more important; stored data
stays readable): "im fine with all of those". Done: design, ontology,
positioning, sketch and README draft copied to
`/Users/jack/src/etude.exp-56-give-a-training/scratch/`; EXP-56 and MTG-329
steered. A steer is read at a Task's next step; neither Graph was started.

Jack Heart, 2026-11-13, on where cohort is used: asked "where do we actually
use the cohort-as-launched-togetehr" (answer: a launcher envelope of
`parallel` and `max_usd` with a name, also a selector; not used by evaluation
or reports; label word still open). Then: "you can run both. i dont care that
manabot cant be test. just make sure to clarify it should build what ican
then stop". Done: MTG-329 steered with those words; both Graphs started from
`pursue` (MTG-329 process 2145874b, EXP-56 process 0e3cfdc0). Both stop at
the interactive demo step; nothing is delivered.

Jack Heart, 2026-11-13, on "Experiment: the label on runs, taken from the
file. No limits, nothing formal. Cohort: a named envelope of limits that runs
are admitted under, which a later launch can add to.": "fine for now". Design
updated; the joinable cohort is pointed at EXP-61. Not checked: whether a
launch under an existing cohort name is allowed today.
Then: "i think the future world where experiment=expriment makes sense". Read
as: the SDK's experiment label and Etude Cloud's formal experiment are the
same thing, so the word stays and the label grows into the record. My
reading; he did not spell it out.
