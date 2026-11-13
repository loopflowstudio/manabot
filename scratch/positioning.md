Correcting my reading: "I'm thinking the other way around. I'm thinking that
in the future Cloud will have names for experiments and it will supply the
experiment name to the A2 SDK launcher" ("A2" read as Etude). Design: the
launcher is the authority for a run's experiment name; a name given at launch
wins over the file stem. Seen at `105abdd`: `Job.experiment` exists and
`track` refuses a name that differs from the launch's; `etude launch` has no
experiment option in the signature I read.
