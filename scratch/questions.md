# Fix-forward fixture choice — 2026-11-12

Assumption: this experiment/notebook acceptance test needs real training and
live complete-game monitoring, but does not require complete-game training.
Use the existing one-seat exact-quota collector for its eight-transition
workload; keep all four attempts and every existing assertion. This bounds
the fixture without changing production defaults or the shared WDL baseline.
The 120-second training allowance is a hang guard, not a timing claim; gate
still needs Ubuntu evidence under the unchanged 300-second test limit and
480-second ordinary-job target.
