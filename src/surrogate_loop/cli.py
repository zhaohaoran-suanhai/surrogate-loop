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

    elasticity = subparsers.add_parser(
        "elasticity2d",
        help="二维线弹性神经算子闭环",
        description="二维线弹性神经算子闭环",
    )
    elasticity_commands = elasticity.add_subparsers(dest="elasticity_command")
    elasticity_commands.add_parser("doctor", help="诊断隔离 FEniCSx 环境")
    elasticity_validate = elasticity_commands.add_parser("validate", help="校验二维弹性配置")
    elasticity_validate.add_argument("--config", type=Path, required=True)
    elasticity_calibrate = elasticity_commands.add_parser("calibrate", help="运行求解器校准")
    elasticity_calibrate.add_argument("--config", type=Path, required=True)
    elasticity_calibrate.add_argument("--output-dir", type=Path, required=True)
    elasticity_run = elasticity_commands.add_parser("run", help="训练并验收二维弹性算子")
    elasticity_run.add_argument("--config", type=Path, required=True)
    elasticity_run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    elasticity_run.add_argument("--request", default="通过结构化配置启动二维弹性训练")
    elasticity_run.add_argument("--reuse-data-from", type=Path)
    elasticity_report = elasticity_commands.add_parser("report", help="读取二维弹性报告")
    elasticity_report.add_argument("--run-dir", type=Path, required=True)
    elasticity_predict = elasticity_commands.add_parser("predict", help="二维弹性点或场预测")
    elasticity_predict.add_argument("--run-dir", type=Path, required=True)
    for name in ("e", "nu", "p", "theta", "y0", "w"):
        elasticity_predict.add_argument(f"--{name}", type=float, required=True)
    elasticity_predict.add_argument("--x", type=float)
    elasticity_predict.add_argument("--y", type=float)
    elasticity_predict.add_argument("--nx", type=int)
    elasticity_predict.add_argument("--ny", type=int)
    elasticity_predict.add_argument("--output", type=Path)

    cavity = subparsers.add_parser(
        "cavity2d",
        help="二维顶盖驱动方腔 POD-RBF 闭环",
    )
    cavity_commands = cavity.add_subparsers(dest="cavity_command")
    cavity_validate = cavity_commands.add_parser("validate", help="校验方腔配置")
    cavity_validate.add_argument("--config", type=Path, required=True)
    cavity_plan = cavity_commands.add_parser("plan", help="生成确定性 Fluent 请求")
    cavity_plan.add_argument("--config", type=Path, required=True)
    cavity_plan.add_argument("--output-dir", type=Path, required=True)
    cavity_verify = cavity_commands.add_parser(
        "verify-solver",
        help="验证垂直切片或校准 Fluent 协议",
    )
    cavity_verify.add_argument("--config", type=Path, required=True)
    cavity_verify.add_argument("--fluent-pipeline", type=Path, required=True)
    cavity_verify.add_argument("--output-dir", type=Path, required=True)
    cavity_run = cavity_commands.add_parser("run", help="训练并验收方腔 POD-RBF")
    cavity_run.add_argument("--config", type=Path, required=True)
    cavity_run.add_argument("--fluent-pipeline", type=Path, required=True)
    cavity_run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    cavity_run.add_argument("--request", default="通过真实 Fluent 数据训练二维方腔代理模型")
    cavity_report = cavity_commands.add_parser("report", help="读取方腔运行报告")
    cavity_report.add_argument("--run-dir", type=Path, required=True)
    cavity_predict = cavity_commands.add_parser("predict", help="执行域内方腔场预测")
    cavity_predict.add_argument("--run-dir", type=Path, required=True)
    cavity_predict.add_argument("--re", type=float, required=True)
    cavity_predict.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "cavity2d":
            _handle_cavity(arguments)
        elif arguments.command == "elasticity2d":
            _handle_elasticity(arguments)
        elif arguments.command == "operator":
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
        from surrogate_loop.operator.inference import verify_operator_run

        verified = verify_operator_run(arguments.run_dir)
        _print_json(
            {
                **verified.manifest,
                "deeponet_metrics": verified.test_metrics.to_dict(),
                "pod_metrics": verified.pod_metrics,
                "training": verified.training,
            }
        )
        return
    if arguments.operator_command == "predict":
        try:
            import numpy as np

            from surrogate_loop.operator.artifacts import REQUIRED_HASHED_FILES
            from surrogate_loop.operator.inference import (
                load_operator_bundle,
                load_operator_spec_metadata,
                predict_field,
                predict_point,
                validate_field_grid,
                validate_prediction_request,
            )
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise RuntimeError(
                    "神经算子依赖未安装，请运行 uv sync --extra operator --all-groups"
                ) from error
            raise
        point_requested = arguments.x is not None or arguments.t is not None
        spec = load_operator_spec_metadata(arguments.run_dir)
        validate_prediction_request(
            spec,
            arguments.alpha,
            arguments.a,
            arguments.b,
            x=arguments.x if point_requested else None,
            t=arguments.t if point_requested else None,
            nx=arguments.nx,
            nt=arguments.nt,
        )
        bundle = load_operator_bundle(arguments.run_dir)
        if point_requested:
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
        nx = bundle.spec.grid.nx if arguments.nx is None else arguments.nx
        nt = bundle.spec.grid.nt if arguments.nt is None else arguments.nt
        validate_field_grid(nx, nt)
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
        protected = {
            (arguments.run_dir.resolve() / name).resolve()
            for name in (*REQUIRED_HASHED_FILES, "manifest.json")
        }
        if output.resolve() in protected:
            raise ValueError("输出路径不能覆盖受保护运行产物")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, x=x, t=t, u=field)
        _print_json({"output": str(output.resolve()), "shape": list(field.shape)})
        return
    raise ValueError("请为 operator 指定 validate、run、report 或 predict 子命令")


def _handle_cavity(arguments: argparse.Namespace) -> None:
    from surrogate_loop.operator.cavity2d.config import load_cavity_spec
    from surrogate_loop.operator.cavity2d.inference import (
        predict_accepted_cavity,
        read_cavity_report,
        verify_solver_pipeline,
    )
    from surrogate_loop.operator.cavity2d.pipeline import run_cavity_pipeline
    from surrogate_loop.operator.cavity2d.sampling import (
        build_cavity_sample_plan,
        write_solver_request,
    )

    command = arguments.cavity_command
    if command == "validate":
        spec = load_cavity_spec(arguments.config)
        _print_json({"status": "valid", "mode": spec.mode})
        return
    if command == "plan":
        spec = load_cavity_spec(arguments.config)
        request = write_solver_request(
            arguments.output_dir,
            spec,
            build_cavity_sample_plan(spec),
        )
        _print_json(
            {
                "status": "planned",
                "mode": spec.mode,
                "solver_request": str(request.resolve()),
            }
        )
        return
    if command == "verify-solver":
        _print_json(
            verify_solver_pipeline(
                arguments.config,
                arguments.fluent_pipeline,
                arguments.output_dir,
            )
        )
        return
    if command == "run":
        result = run_cavity_pipeline(
            arguments.config,
            arguments.fluent_pipeline,
            arguments.runs_dir,
            arguments.request,
        )
        _print_json(
            {
                "run_dir": str(result.run_dir.resolve()),
                "status": result.status,
                "selected_model": result.selected_model,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
            }
        )
        return
    if command == "report":
        _print_json(read_cavity_report(arguments.run_dir))
        return
    if command == "predict":
        _print_json(
            predict_accepted_cavity(
                arguments.run_dir,
                arguments.re,
                arguments.output,
            )
        )
        return
    raise ValueError(
        "请为 cavity2d 指定 validate、plan、verify-solver、run、report 或 predict 子命令"
    )


def _handle_elasticity(arguments: argparse.Namespace) -> None:
    import numpy as np

    from surrogate_loop.operator import external_solver
    from surrogate_loop.operator.elasticity2d.artifacts import (
        verify_freeze_manifest,
    )
    from surrogate_loop.operator.elasticity2d.config import load_elasticity_spec
    from surrogate_loop.operator.elasticity2d.dataset import write_solver_job
    from surrogate_loop.operator.elasticity2d.inference import (
        load_elasticity_bundle,
        load_elasticity_spec_metadata,
        predict_elasticity_points,
        read_elasticity_report,
        validate_elasticity_request,
    )
    from surrogate_loop.operator.elasticity2d.sampling import build_sample_plan

    command = arguments.elasticity_command
    if command == "doctor":
        _print_json(external_solver.doctor_solver_environment(Path.cwd()))
        return
    if command == "validate":
        spec = load_elasticity_spec(arguments.config)
        _print_json({"status": "valid", "mode": spec.mode, "template": spec.problem.template})
        return
    if command == "calibrate":
        spec = load_elasticity_spec(arguments.config)
        if spec.mode != "calibration":
            raise ValueError("calibrate 只接受 calibration 配置")
        job = write_solver_job(spec, build_sample_plan(spec), arguments.output_dir)
        completed = external_solver.run_solver_process(
            "calibrate",
            ("--job", str(job), "--output-dir", str(arguments.output_dir)),
            Path.cwd(),
            3600.0,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "FEniCSx 校准失败")
        _print_json(external_solver.parse_solver_json(completed.stdout, "calibrate"))
        return
    if command == "run":
        from surrogate_loop.operator.elasticity2d.pipeline import run_elasticity_pipeline

        result = run_elasticity_pipeline(
            arguments.config,
            arguments.runs_dir,
            arguments.request,
            reuse_data_from=arguments.reuse_data_from,
        )
        _print_json(
            {
                "run_dir": str(result.run_dir.resolve()),
                "status": result.status,
                "deeponet_metrics": result.deeponet_metrics,
                "pod_rbf_metrics": result.pod_rbf_metrics,
            }
        )
        return
    if command == "report":
        state, payload = read_elasticity_report(arguments.run_dir)
        _print_json({"state": state.value, **payload})
        return
    if command == "predict":
        point_requested = arguments.x is not None or arguments.y is not None
        if point_requested and (arguments.x is None or arguments.y is None):
            raise ValueError("点预测必须同时提供 --x 和 --y")
        parameters = np.array(
            [[arguments.e, arguments.nu, arguments.p, arguments.theta, arguments.y0, arguments.w]],
            dtype=np.float64,
        )
        coordinates = (
            np.array([[arguments.x, arguments.y]], dtype=np.float64)
            if point_requested
            else None
        )
        spec = load_elasticity_spec_metadata(arguments.run_dir)
        validate_elasticity_request(
            spec, parameters, coordinates, nx=arguments.nx, ny=arguments.ny
        )
        bundle = load_elasticity_bundle(arguments.run_dir)
        if point_requested:
            displacement = predict_elasticity_points(bundle, parameters, coordinates)[0]
            _print_json({"x": arguments.x, "y": arguments.y, "u": displacement.tolist()})
            return
        if arguments.output is None:
            raise ValueError("场预测必须提供 --output")
        nx = spec.observation.nx if arguments.nx is None else arguments.nx
        ny = spec.observation.ny if arguments.ny is None else arguments.ny
        x, y = np.meshgrid(np.linspace(0.0, 4.0, nx), np.linspace(0.0, 1.0, ny))
        points = np.column_stack((x.ravel(), y.ravel()))
        protected = {
            (arguments.run_dir.resolve() / name).resolve()
            for name in (
                *verify_freeze_manifest(arguments.run_dir).files,
                "freeze_manifest.json",
                "status.json",
                "acceptance.json",
                "acceptance_stage.json",
                "sealed_test_summary.json",
            )
        }
        if arguments.output.resolve() in protected:
            raise ValueError("输出路径不能覆盖受保护运行产物")
        displacement = predict_elasticity_points(bundle, parameters, points)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.output, coordinates=points, displacement=displacement)
        _print_json({"output": str(arguments.output.resolve()), "shape": [ny, nx, 2]})
        return
    raise ValueError("请为 elasticity2d 指定固定子命令")
