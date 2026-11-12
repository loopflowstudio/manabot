# Reversible assumption

Diagnose the incomplete engine upgrade when importing `manabot.managym`, not
while importing the root package. This preserves the repair message and the
existing offline-report import contract; both paths have passing tests.
