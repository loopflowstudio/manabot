---
requires: the experiment module and opening of its report
produces: routing JSON selecting `exploratory` or `finding`
action_style: procedural
---
Choose the experiment's tier from its design.

Return `{"path":"finding"}` only when `experiments/reports/<name>.md` opens
with a question paragraph committed before any scored run, to be cited as a
claim about a method or playing strength and reviewed by a person. Otherwise
return `{"path":"exploratory"}`. When unsure, return `{"path":"exploratory"}`.

An exploratory run whose numbers turn out to matter never becomes a finding by
relabeling; its numbers motivate a new experiment with its own committed
question.
