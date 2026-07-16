from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from surrogate_loop import __version__
from surrogate_loop.config import load_spec
from surrogate_loop.inference import predict_endpoint
from surrogate_loop.pipeline import run_pipeline


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


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
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="校验结构化运行配置")
    validate.add_argument("--config", type=Path, required=True)

    run = subparsers.add_parser("run", help="执行数据生成、训练、选模和验收")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--request", default="通过结构化配置启动训练")

    report = subparsers.add_parser("report", help="读取已完成运行的报告")
    report.add_argument("--run-dir", type=Path, required=True)

    predict = subparsers.add_parser("predict", help="加载已验收模型并预测 u(1)")
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--gamma", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            spec = load_spec(arguments.config)
            _print_json({"status": "valid", "mode": spec.mode})
        elif arguments.command == "run":
            spec = load_spec(arguments.config)
            if arguments.smoke and spec.mode != "smoke":
                raise ValueError("--smoke 只能与 mode=smoke 的配置一起使用")
            result = run_pipeline(arguments.config, arguments.runs_dir, arguments.request)
            _print_json(
                {
                    "run_dir": str(result.run_dir.resolve()),
                    "status": result.status,
                    "selected_model": result.selected_model,
                    "test_metrics": result.test_metrics,
                }
            )
        elif arguments.command == "report":
            manifest = json.loads(
                (arguments.run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (arguments.run_dir / "test_metrics.json").read_text(encoding="utf-8")
            )
            _print_json({**manifest, "test_metrics": metrics})
        elif arguments.command == "predict":
            value = predict_endpoint(arguments.run_dir, arguments.gamma)
            _print_json({"gamma": arguments.gamma, "u_at_1": value})
        else:
            parser.print_help()
        return 0
    except Exception as error:  # noqa: BLE001 - CLI 边界统一转换为稳定退出码
        print(f"错误：{error}", file=sys.stderr)
        return 2
