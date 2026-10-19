# Why the added model runs are still missing

2026-10-17, requested by Jack Heart with `$5whys`. This follows the earlier
study006 analysis; it does not replace that incident or its unresolved CUDA
question. Jack authorized the combined grid, 64x4 and 384x2 additions, necessary
MLOps delivery, and an inclusive $50 budget.

## Outcome and corrected diagnosis

Neither added shape has a training result. Twelve original grid cells and their
early-learning screen are verified. The replacement grouped campaign is prepared
but has not passed its full startup gate. Canary015 (`fe1b153c3183`) is terminal,
its provider pod is confirmed absent, and no worker receipt or training exists.
Conservative settled charges are $20.38, no rental caps remain open, and $5 is
reserved for ancillary costs. These are conservative ledger charges, not a claim
that the provider billed every estimated second.

I initially described015 as another runtime-dependency installation failure.
That was wrong. Its final visible phase was installation, but the provider stopped
the container about3.5seconds into it. The local Mac had slept during launch,
spending the startup allowance before the cloud machine existed.

All following times are UTC on October17; macOS power logs use UTC-7.

| Event | Time | Evidence |
|---|---|---|
| SDK records session claim; host enters **Idle Sleep** | 22:05:43 | SQLite `sessions.claimed_at`; `pmset -g log` |
| Session packet reaches S3 | 22:05:45 | Object LastModified |
| Host returns in DarkWake after772seconds | 22:18:35 | Power log |
| Prepared source copied; bootstrap published | 22:18:37–38 | Object LastModified |
| Provider creates pod | 22:18:39.813 | Provider creation receipt |
| Container reaches bootstrap entrypoint | 22:18:42.339 | Provider log timestamp |
| Client runtime installation begins | 22:18:44.465 | Provider log timestamp |
| Provider stops and removes container | 22:18:48 | Provider system log timestamp |
| Local controller finishes settlement after further sleep | 22:23:53.544 | Attempt receipt and power log |

Retained evidence is under `.runs/grid-007/worker-startup-probe/`:
`failure-review.json`, `host-sleep-events.json`, `object-times.json`,
`3544807c776f-provider-creation.json`, `boot-attempts.json`,
`rental-settlement.json`, and the provider stream. The stream's `ts` field matters:
arrival order alone placed a delayed install-start message after the stop event.
An independent S3 HEAD returns404 for this session's `out/receipt`.

## The causal chain

1. **Why are64x4 and384x2 still missing?** Their earlier attempts failed before
   learning, and the revised grouped campaign is gated on a complete setup/CUDA/
   two-update/checkpoint/cleanup canary. The last three canaries have not supplied
   that proof. This is missing execution, not evidence against either model.

2. **Why did the latest canary fail?** Creation happened776.56seconds after its
   recorded claim. The installed SDK's pre-start allowance is300seconds from that
   claim. It was already476.56seconds overdue when RunPod created the machine;
   the container was then stopped within seconds. There is no120-second installer
   timeout in this attempt and no CUDA execution.

3. **Why did so much time pass before creation?** The laptop entered idle sleep
   at the claim and stayed asleep772seconds. The controller resumed preparation
   during a brief wake and created the pod. Source/bootstrap publication and the
   provider's creation timestamp align with the wake. The pod itself reached the
   entrypoint2.53seconds after creation, contradicting a13-minute image-startup
   explanation.

4. **Why could a suspended launch still rent a GPU?** The SDK records a claim
   before `Executor.create` publishes the journal/source and calls the provider.
   `RunPodMachines.create`/`RunPodAPI.create` did not reject an already spent
   startup allowance. `Scheduler.wait(timeout=25)` checks its timeout only after
   `tick()` returns; it is not a deadline around the tick or its I/O. The independent
   guardian bounded the eventual rental, but it cannot keep a local controller
   awake or prevent an already stale creation request.

5. **Why did our operating process permit this?** Earlier launch instructions
   used `caffeinate`; this replacement controller was launched directly through
   Doppler/uv without it. Sleep protection was a property of a particular shell
   invocation, not a required launch invariant. Local tests covered forwarding,
   timeouts and retained artifacts but not a suspended controller between claim
   and provider creation. I kept following the last remote phase without first
   comparing host sleep, claim, creation and stop timestamps. The missing checks
   made that operational mistake possible and the initial diagnosis misleading.

## Other causes and limits

- Canary010 did reach the worker, then exhausted1200seconds of project setup.
  Its retained log shows large CUDA dependency downloads. The verified4.22GB
  mirror addresses that known network-dependent setup path;015 never exercised it.
- Canary014 installed the client in89.225seconds but produced no worker receipt.
  Its local host has no matching sleep event in the checked13:40–13:50Pacific
  interval. Do not attribute014 to015's cause. Import, source fetch, receipt upload
  and remote network behavior are still unresolved there.
- The earlier proven120-second SDK build timeout was addressed by a source-identical
  prebuilt client. Other attempts completed setup and twelve grid cells trained;
  this contradicts a universal incompatibility of the image or model code.
- The original CUDA/NVML failure and historical finite-weight anomaly remain open.
  Neither is explained by this host-sleep evidence.
- This does not prove that every minute of delay or every earlier failed request
  came from sleep. Nor does avoiding idle sleep make a laptop an always-on service.

## Prevention and verification

Implemented locally in `experiments/runners/etude_launch_guard.py`:

- A bounded, process-owned macOS idle-sleep assertion, verified in `pmset` before
  admission and released when the context exits. It does not override explicit
  sleep, lid closure or power loss.
- A check immediately before the provider create call against the SDK's actual
  recorded claim. More than120seconds spent before creation, clock reversal,
  expired launcher lifetime, a lost sleep assertion or an already settled session
  refuses creation. This leaves at least180seconds of the unchanged300-second
  pre-start window. No clock is reset and no failed attempt is replayed.
- Decision evidence excludes provider arguments, environment values and credentials.
  A refusal occurs before the provider call; ambiguous provider errors still
  propagate unchanged, without a retry. The SDK's generic `capacity_refused` label
  is accompanied by the specific local refusal reason.

Thirty-six focused checks passed, including the recorded776.56-second pause,
clock reversal, a settled session, lifetime expiry and unchanged provider error
propagation. A separate local probe uses the real macOS assertion and verifies
that a stale request makes zero provider transport calls, then checks assertion
cleanup. Its result is `.runs/grid-007/launch-guard-proof/verification.json`.
These are local prevention proofs, not a successful GPU canary.

The immediate next action is one separately recorded replacement of015 with this
guard, the same diagnostic code and immutable dependency/scientific inputs. Keep
all existing GPU/deadline/output gates and budget limits. No further dependency
packaging or increased timeout is justified by015. Longer unattended orchestration
belongs on an always-on controller; relocating it is a proposed architectural
improvement, not a completed migration or a reason to silently change accounts.

PR306 merged after all CI passed; it adds worker diagnostics but did not fix
this newly established host-sleep cause. The Task and experiment goal remain
incomplete. Task automation is disabled and the CI watcher is not running.
