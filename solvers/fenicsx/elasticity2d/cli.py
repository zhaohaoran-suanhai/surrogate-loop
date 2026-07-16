from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from solvers.fenicsx.elasticity2d.quality import (
    generate_datasets,
    run_calibration,
    software_versions,
)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(arguments)
    try:
        if namespace.action == "doctor":
            payload = {"status": "ok", **software_versions()}
        elif namespace.action == "calibrate":
            output = run_calibration(namespace.job, namespace.output_dir)
            payload = {
                "status": "ok",
                "manifest": str(output.resolve()),
                "summary": {"action": "calibrate"},
            }
        else:
            manifest = generate_datasets(namespace.job, namespace.output_dir)
            payload = {
                "status": "ok",
                "manifest": str(manifest.manifest_path.resolve()),
                "summary": {
                    "development_samples": manifest.development_samples,
                    "sealed_test_samples": manifest.sealed_test_samples,
                },
            }
    except (OSError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fenicsx-elasticity2d")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("doctor")
    for action in ("calibrate", "generate"):
        subparser = subparsers.add_parser(action)
        subparser.add_argument("--job", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
