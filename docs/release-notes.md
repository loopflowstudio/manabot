## Python compatibility and downstream dependencies

- Supports ordinary CPython 3.12 and 3.13, with 3.13 as the development default.
- Source installs and wheels declare Torch >=2.10.0 and Etude SDK >=0.2.15,
  allowing downstream projects to resolve compatible newer versions.
- Removes the unused direct Pandas dependency. Notebook generation remains
  available in ordinary experiment reporting; notebook execution uses the
  `notebook` extra.
- Supplies bundled-engine wheels for both interpreters on Linux x86-64
  (glibc 2.28+) and Apple Silicon macOS 14+. No free-threaded ABI support.
- Requires both PyPI publication and a matching GitHub Release. Retrying a
  completed upload verifies artifact digests and never overwrites PyPI files.

Choose the CPU or CUDA backend explicitly; see the
[installation guide](https://github.com/loopflowstudio/manabot/blob/main/docs/development.md#install).
Retained experiments keep their original interpreter and source pins.
