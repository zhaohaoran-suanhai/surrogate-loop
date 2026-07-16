from __future__ import annotations

import argparse
from collections.abc import Sequence

from surrogate_loop import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surrogate-loop",
        description="标量代理模型最小闭环命令行入口",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
