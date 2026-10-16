"""One bounded diagnostic evaluator beside future baseline learners.

The existing CheckpointQueue owns admission, evaluator leases and failure/cost
receipts. This supervisor only discovers the learner's 500-update exports and
requests 1,000-update diagnostic cohorts. It never changes learner weights.
"""

from pathlib import Path
import threading
import time

from manabot.arena.matchup import AuthoredMatchup
from manabot.training.checkpoint_queue import CheckpointQueue, MonitoringBudget
from manabot.training.execution import atomic_json
from manabot.training.monitor_evaluation import MonitorProtocol


class DiagnosticMonitor:
    """An explicitly bounded sidecar; failures remain diagnostic, not strength."""

    def __init__(
        self,
        source: Path,
        output: Path,
        *,
        updates: tuple[int, ...] = tuple(range(1000, 10000, 1000)),
        seconds: float = 9900,
    ) -> None:
        matchup = AuthoredMatchup(
            pack_key="ur-lessons-vs-gw-allies", role_a="ur_lessons", role_b="gw_allies"
        )
        protocols = tuple(
            MonitorProtocol(
                authored_matchup=matchup,
                include_mirrors=True,
                purpose="supplemental-matchup-diagnostic",
                deal_seeds=tuple(range(1_226_900_000, 1_226_900_013)),
                opponent=anchor,
                bootstrap_replicates=2000,
                bootstrap_seed=12290,
            )
            for anchor in ("scripted_greedy", "random")
        )
        self.source, self.output = source, output
        self.config = MonitoringBudget(
            seconds=seconds,
            attempt_seconds=900,
            updates=updates,
            protocol=protocols[0],
            additional_protocols=(protocols[1],),
        )
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._watch, name="baseline-diagnostics", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        self.thread.join()

    def _watch(self) -> None:
        queue: CheckpointQueue | None = None
        try:
            queue = CheckpointQueue(self.output, self.config)
            while True:
                if self.source.exists():
                    queue.tick([self.source])
                if (
                    self.stop.is_set()
                    and queue.process is None
                    and (
                        queue.pending == 0
                        or self.config.seconds - queue.charged_seconds
                        < self.config.attempt_seconds
                    )
                ):
                    break
                # Polling only receipts; no optimizer, RNG or live policy access.
                time.sleep(30)
        except Exception as error:
            atomic_json(
                self.output / "supervisor-error.json",
                {"error": f"{type(error).__name__}: {error}"},
            )
        finally:
            if queue is not None:
                queue.close()
