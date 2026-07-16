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

    operator = subparsers.add_parser("operator", help="一维热传导神经算子闭环")
    operator_commands = operator.add_subparsers(dest="operator_command")

    operator_validate = operator_commands.add_parser("validate", help="校验神经算子配置")
    operator_validate.add_argument("--config", type=Path, required=True)

    operator_run = operator_commands.add_parser("run", help="训练并验收 DeepONet")
    operator_run.add_argument("--config", type=Path, required=True)
    operator_run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    operator_run.add_argument("--request", default="通过结构化配置启动神经算子训练")

    operator_report = operator_commands.add_parser("report", help="读取神经算子运行报告")
    operator_report.add_argument("--run-dir", type=Path, required=True)

    operator_predict = operator_commands.add_parser("predict", help="执行点预测或场预测")
    operator_predict.add_argument("--run-dir", type=Path, required=True)
    operator_predict.add_argument("--alpha", type=float, required=True)
    operator_predict.add_argument("--a", type=float, required=True)
    operator_predict.add_argument("--b", type=float, required=True)
    operator_predict.add_argument("--x", type=float)
    operator_predict.add_argument("--t", type=float)
    operator_predict.add_argument("--nx", type=int)
    operator_predict.add_argument("--nt", type=int)
    operator_predict.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "operator":
            _handle_operator(arguments)
        elif arguments.command == "validate":
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


def _handle_operator(arguments: argparse.Namespace) -> None:
    if arguments.operator_command == "validate":
        from surrogate_loop.operator.config import load_operator_spec

        spec = load_operator_spec(arguments.config)
        _print_json({"status": "valid", "mode": spec.mode, "template": spec.problem.template})
        return
    if arguments.operator_command == "run":
        try:
            from surrogate_loop.operator.pipeline import run_operator_pipeline
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise RuntimeError(
                    "神经算子依赖未安装，请运行 uv sync --extra operator --all-groups"
                ) from error
            raise
        result = run_operator_pipeline(
            arguments.config,
            arguments.runs_dir,
            arguments.request,
        )
        _print_json(
            {
                "run_dir": str(result.run_dir.resolve()),
                "status": result.status,
                "deeponet_metrics": result.deeponet_metrics,
                "pod_metrics": result.pod_metrics,
            }
        )
        return
    if arguments.operator_command == "report":
        manifest = json.loads(
            (arguments.run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        test_metrics = json.loads(
            (arguments.run_dir / "test_metrics.json").read_text(encoding="utf-8")
        )
        pod_metrics = json.loads(
            (arguments.run_dir / "pod_metrics.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            (arguments.run_dir / "training_history.json").read_text(encoding="utf-8")
        )
        _print_json(
            {
                **manifest,
                "deeponet_metrics": test_metrics,
                "pod_metrics": pod_metrics,
                "training": training,
            }
        )
        return
    if arguments.operator_command == "predict":
        try:
            import numpy as np

            from surrogate_loop.operator.inference import (
                load_operator_bundle,
                predict_field,
                predict_point,
            )
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise RuntimeError(
                    "神经算子依赖未安装，请运行 uv sync --extra operator --all-groups"
                ) from error
            raise
        bundle = load_operator_bundle(arguments.run_dir)
        point_requested = arguments.x is not None or arguments.t is not None
        if point_requested:
            if arguments.x is None or arguments.t is None:
                raise ValueError("点预测必须同时提供 --x 和 --t")
            value = predict_point(
                bundle,
                arguments.alpha,
                arguments.a,
                arguments.b,
                x=arguments.x,
                t=arguments.t,
            )
            _print_json({"x": arguments.x, "t": arguments.t, "u": value})
            return
        nx = arguments.nx or bundle.spec.grid.nx
        nt = arguments.nt or bundle.spec.grid.nt
        if nx < 2 or nt < 2:
            raise ValueError("场预测的 nx 和 nt 必须至少为 2")
        x = np.linspace(0.0, 1.0, nx)
        t = np.linspace(0.0, 1.0, nt)
        field = predict_field(
            bundle,
            arguments.alpha,
            arguments.a,
            arguments.b,
            x=x,
            t=t,
        )
        output = arguments.output or arguments.run_dir / "predicted_field.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, x=x, t=t, u=field)
        _print_json({"output": str(output.resolve()), "shape": list(field.shape)})
        return
    raise ValueError("请为 operator 指定 validate、run、report 或 predict 子命令")
