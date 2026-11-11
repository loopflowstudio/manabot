---
requires: a change or behavior to assess
produces: a change that works for the people who will use it, with what you saw and what is still wrong
action_style: procedural
---
Use the change the way the people it is for will use it, and fix what breaks.

You are its first user. Nobody should open a demo or merge a PR and find it broken
on the first click; if they would, this step failed.

## Use the real thing

Run the real product, not its tests: the built app, the installed CLI, the live
page in a browser, the API over the network, the simulator or device. Drive it
with computer use, a browser, a terminal or whatever reaches it the way a person
would. Internal setters, test-only endpoints, mocks and fixtures do not count.

Getting there is your job. Build it, start its services, create accounts and
realistic data, sign in, set up whatever state a real user would already have.
When no route exists, make one. Hand back only what truly needs a person, such
as their own credentials, a physical device, a payment or a production approval,
and say exactly what that left unchecked.

Make the scenario as real as you can reach:

- Arrive the way a person arrives: a fresh install or first visit, then a
  returning user with existing data from before this change.
- Do the main job end to end, then the things people actually do: wrong input,
  empty and large data, going back, interrupting and resuming, a slow or failed
  network, two things at once.
- Check what it leaves behind as well as what it shows: files, rows, messages,
  processes, and what another part of the product now reads.
- Look at it. Read the output, check the screenshot, notice what is confusing,
  slow or ugly, not only what is wrong.

## Fix and repeat

Repair what breaks within the change's scope, then go through the experience
again from the start. Keep going until it works for the people it is for. A
change to what the product should be is not yours to make; describe it
instead. When asked for a read-only review, report instead of
fixing.

Run tests only where they tell you something the real path cannot reach, and
keep any you add about what a person would notice.

## Keep what you learned

How to launch and drive this product belongs in this skill. When you had to work
out a step it does not describe, or a step it describes no longer works, write
the working version here with `lf skill customize qa`, replacing what was wrong.

Show what you did in `scratch/qa.html`: each scenario, what you did, what you saw
with screenshots or transcripts, and what you fixed. Finish with what works now,
what you could not reach and why, and what is still wrong, worst first. Do not
publish, land or choose the next Graph step from here.

## Driving manabot training

- Smallest real run: `uv run manabot train --device cpu --steps 3 --seed 197 --out /tmp/<dir>/run1`
  (local recipe, about 15 s). Each update's metrics, checkpoints and author
  artifacts are under `OUT/self-play/etude/<run id>/`. Read them back with
  `manabot.training.update_metrics.read_updates(OUT, [TrainingRun])`, and with
  `manabot.reports.experiment.load_evidence` plus a report figure function
  (`--extra reports`). Copy the run elsewhere to check relocation.
- `--recipe` rejects `--resume-from`. To check resume, dump
  `manabot.recipes.local.local_smoke(steps=200, device="cpu")` to JSON with
  `schedule_clock="iteration_fraction"` and `recovery_max_microsteps` set. Run it
  with `--regime`, stop it after a few `self-play-update-*.pt` files appear, then
  start it again with `--resume-from <run id>` and a sibling `--out` (same
  parent). The resumed attempt writes into the parent's Etude run, so read from
  the shared parent directory.
- `manabot train` may ignore SIGINT for several updates; stop it with SIGTERM to
  its Python pid. When waiting with `pgrep -f`, the waiting shell's own command
  line matches the pattern. Wait on the pid (`kill -0`) instead.
- Older runs with historical metric shapes are often retained in other worktrees'
  `.runs/`. Copy one before reading it; never modify it.
