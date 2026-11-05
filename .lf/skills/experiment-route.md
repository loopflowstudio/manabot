---
requires: reviewed experiment and current software admission evidence
produces: selected software or ready path
---
Choose whether the reviewed experiment needs software before its final freeze.

Return the declared routing JSON with `{"path":"software"}` when concrete implementation
or admission work remains. Use `{"path":"ready"}` only when the existing executable
protocol and software already meet the reviewed design. Existing checkpoints
are sufficient when the question only needs evaluation; do not invent fitting
or a placeholder PR. A missing scientific decision belongs to design review,
not to a software detour or an implicit execution allowance.
