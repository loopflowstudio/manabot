"""Plan actions with frozen policies and viewer-relative hidden-world samples.

puct owns tree search; conditional compares the same root under different
beliefs. Search never fits model parameters; training consumes its targets."""
