"""Command line: python -m se.cli <command>. Run from the project folder."""
import argparse
import sys

from . import config, gates


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m se.cli", description="Sentiment Engine Lite (synthetic data only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prereg-freeze", help="hash configs/profile.yaml and the pre-registration into runs/prereg.json")
    sub.add_parser("prereg-status", help="compare current hashes with runs/prereg.json")
    sub.add_parser("run", help="run the synthetic demo (both worlds, robustness) and write web/data.json")
    sub.add_parser("gates", help="print the gates table")
    sub.add_parser("serve", help="serve the dashboard and API on 127.0.0.1:8000")
    sub.add_parser("verify", help="check schema version, row counts, hashes and banner flags")
    for name in ("import-prices", "import-news"):
        p = sub.add_parser(name, help="bring-your-own CSV import (does NOT clear any licence)")
        p.add_argument("csv")
        p.add_argument("--db", default="var/user.sqlite")
        p.add_argument("--source-name")
        p.add_argument("--licence-note")
        p.add_argument("--source-tz", default=None)
        p.add_argument("--market", default="US")
    args = ap.parse_args(argv)

    if args.cmd == "prereg-freeze":
        rec = config.prereg_freeze()
        print(f"frozen {rec['frozen_on']}: config {rec['config_sha256'][:12]}..., prereg doc {rec['prereg_doc_sha256'][:12]}...")
    elif args.cmd == "prereg-status":
        st = config.prereg_status()
        print(("MATCH: " if st["match"] else "MISMATCH: ") + st["reason"])
        return 0 if st["match"] else 1
    elif args.cmd == "run":
        sys.path.insert(0, str(config.ROOT))
        import run_demo
        run_demo.main()
    elif args.cmd == "gates":
        print(gates.table(gates.load()))
    elif args.cmd == "serve":
        from . import api
        api.serve()
    elif args.cmd == "verify":
        from . import verify
        return verify.main()
    elif args.cmd in ("import-prices", "import-news"):
        from . import importers
        return importers.cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
