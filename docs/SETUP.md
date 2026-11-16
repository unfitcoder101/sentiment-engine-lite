# Setup

Needs Python 3.11 or newer. Everything runs locally on SYNTHETIC data; the only network use is `pip install`.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_demo.py            # about 25 seconds; writes var/demo.sqlite and web/data.json
.venv/bin/python -m se.cli serve        # dashboard + API on http://127.0.0.1:8000/
```

Without the API: `.venv/bin/python serve.py` serves the dashboard from `web/data.json` (interactive tools are off).

Other commands:

```
.venv/bin/python -m pytest -q           # tests (one test needing real model weights is skipped)
sh scripts/check_all.sh                 # tests, compile check, script check, verify
.venv/bin/python -m se.cli gates        # gates table
.venv/bin/python -m se.cli verify       # schema, row counts, hashes, banner flags
.venv/bin/python -m se.cli prereg-status
```

Do not run `prereg-freeze` again unless you intend to start a new pre-registration: it re-hashes the config and
the pre-registration document. `runs/prereg.json` is kept in git; the rest of `runs/` and `var/` are not.

Optional and not installed: torch and transformers (only for a local FinBERT directory you hold yourself).
