---
requires: the experiment's design
produces: selected exploratory or finding path
---
Choose the experiment's tier from its design.

Return the declared routing JSON with `{"path":"finding"}` only when the report
opens with a question paragraph committed before any scored run: the result
will be cited as a claim about a method or about playing strength, and a
person reviews what it established. Otherwise return
`{"path":"exploratory"}`: the report ends the experiment with no review. When
unsure, it is exploratory.

An exploratory experiment whose result turns out to matter does not become a
finding by relabeling. Its numbers motivate a new experiment with its own
committed question.
