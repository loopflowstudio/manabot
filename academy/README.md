# Academy frontend

The experience server ships in managym. Install and launch it with:

```bash
pip install 'managym[academy]'
managym academy
```

See [Running Academy](../docs/academy.md#running-academy) for installation,
play, replay inspection and troubleshooting, and the
[server guide](../managym/academy/README.md) for development details.
Frontend source stays in `academy/ui`; use `./scripts/play` for hot reload.
The top-level `academy` Python import is deprecated and removed in 0.4.0;
use `managym.academy`.
