from __future__ import annotations

import hashlib
import json
from pathlib import Path

from surrogate_loop.operator.field_data import sha256_file

_LEGACY_STAGE_FIELDS = {
    "schema_version",
    "status",
    "result_sha256",
    "diagnostic_sha256",
}
_CURRENT_STAGE_FIELDS = _LEGACY_STAGE_FIELDS | {"dataset_provenance_sha256"}
_LEGACY_REPORT_FIELDS = {
    "schema_version",
    "status",
    "deeponet_metrics",
    "pod_rbf_metrics",
    "training",
    "timing",
}
_CURRENT_REPORT_FIELDS = _LEGACY_REPORT_FIELDS | {
    "model_architecture",
    "data_provenance",
    "directional_metrics",
}
_REUSE_EVIDENCE_FIELDS = {
    "schema_version",
    "source_run_dir",
    "source_request_sha256",
    "source_manifest_sha256",
    "development_sha256",
    "sealed_test_sha256",
    "target_job_sha256",
}
_DIAGNOSTIC_FILES = {
    "diagnostics/displacement_comparison.png",
    "diagnostics/fenicsx_stress_summary.png",
}


def read_verified_development_report(run_dir: Path) -> dict[str, object]:
    directory = run_dir.resolve()
    expected_version, request_identity = _verified_request_identity(directory)
    stage = _read_json(directory / "development_stage.json")
    report_path = directory / "development_evaluation.json"
    report = _read_json(report_path)

    expected_stage_fields = (
        _LEGACY_STAGE_FIELDS if expected_version == 5 else _CURRENT_STAGE_FIELDS
    )
    expected_report_fields = (
        _LEGACY_REPORT_FIELDS if expected_version == 5 else _CURRENT_REPORT_FIELDS
    )
    if stage.get("schema_version") != expected_version or set(stage) != expected_stage_fields:
        raise RuntimeError("二维弹性 Smoke 报告版本与请求身份不一致")
    if (
        report.get("schema_version") != expected_version
        or report.get("status") != "development_complete"
        or set(report) != expected_report_fields
    ):
        raise RuntimeError("二维弹性 Smoke 报告字段无效")
    if (
        stage.get("status") != "complete"
        or stage.get("result_sha256") != sha256_file(report_path)
        or not _diagnostics_match(directory, stage.get("diagnostic_sha256"))
    ):
        raise RuntimeError("二维弹性 Smoke 报告完整性校验失败")

    if expected_version == 5:
        if (directory / "dataset_reuse.json").exists():
            raise RuntimeError("旧版二维弹性报告不得包含数据复用证据")
        return report

    model = request_identity["spec"]["model"]
    if report.get("model_architecture") != model["architecture"]:
        raise RuntimeError("二维弹性 Smoke 模型架构与请求身份不一致")
    _verify_provenance(directory, request_identity, stage, report)
    return report


def _verified_request_identity(run_dir: Path) -> tuple[int, dict[str, object]]:
    payload = _read_json(run_dir / "request.json")
    identity = {name: value for name, value in payload.items() if name != "identity_sha256"}
    base_fields = {"request", "spec"}
    reused_fields = base_fields | {
        "reuse_data_from",
        "reuse_manifest_sha256",
        "reuse_source_request_sha256",
    }
    if set(identity) not in (base_fields, reused_fields):
        raise RuntimeError("二维弹性 Smoke 请求身份字段无效")
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if payload.get("identity_sha256") != digest:
        raise RuntimeError("二维弹性 Smoke 请求身份摘要无效")
    spec = identity.get("spec")
    if not isinstance(spec, dict) or spec.get("mode") != "smoke":
        raise RuntimeError("二维弹性开发报告请求必须为 Smoke")
    if run_dir.name != f"elasticity-smoke-{digest[:12]}":
        raise RuntimeError("二维弹性 Smoke 运行目录与请求身份不一致")
    model = spec.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("二维弹性 Smoke 请求模型身份无效")
    architecture = model.get("architecture")
    if architecture == "directional_linear_v2":
        return 6, identity
    if architecture is None and set(identity) == base_fields:
        return 5, identity
    raise RuntimeError("二维弹性 Smoke 请求模型版本无效")


def _verify_provenance(
    run_dir: Path,
    request: dict[str, object],
    stage: dict[str, object],
    report: dict[str, object],
) -> None:
    evidence_path = run_dir / "dataset_reuse.json"
    request_is_reused = "reuse_data_from" in request
    digest = stage.get("dataset_provenance_sha256")
    if not request_is_reused:
        if (
            digest is not None
            or evidence_path.exists()
            or report.get("data_provenance") != {"mode": "generated"}
        ):
            raise RuntimeError("二维弹性 Smoke 生成数据来源身份无效")
        return
    if not isinstance(digest, str) or len(digest) != 64 or not evidence_path.is_file():
        raise RuntimeError("二维弹性 Smoke 复用数据来源摘要无效")
    evidence = _read_json(evidence_path)
    if set(evidence) != _REUSE_EVIDENCE_FIELDS or evidence.get("schema_version") != 1:
        raise RuntimeError("二维弹性 Smoke 复用数据来源字段无效")
    expected_hashes = {
        "source_manifest_sha256": request.get("reuse_manifest_sha256"),
        "target_job_sha256": sha256_file(run_dir / "solver_job.json"),
        "development_sha256": sha256_file(
            run_dir / "solver_output" / "datasets" / "development.npz"
        ),
        "sealed_test_sha256": sha256_file(
            run_dir / "solver_output" / "datasets" / "sealed_test.npz"
        ),
    }
    local_manifest_hash = sha256_file(
        run_dir / "solver_output" / "datasets" / "dataset_manifest.json"
    )
    if (
        sha256_file(evidence_path) != digest
        or evidence.get("source_run_dir") != request.get("reuse_data_from")
        or evidence.get("source_request_sha256")
        != request.get("reuse_source_request_sha256")
        or local_manifest_hash != request.get("reuse_manifest_sha256")
        or any(evidence.get(name) != value for name, value in expected_hashes.items())
        or report.get("data_provenance")
        != {"mode": "reused", "evidence": evidence}
    ):
        raise RuntimeError("二维弹性 Smoke 复用数据来源完整性校验失败")


def _diagnostics_match(run_dir: Path, payload: object) -> bool:
    if not isinstance(payload, dict) or set(payload) != _DIAGNOSTIC_FILES:
        return False
    return all(
        _is_sha256(digest) and sha256_file(run_dir / relative) == digest
        for relative, digest in payload.items()
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取二维弹性开发报告文件：{path.name}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"二维弹性开发报告文件必须是对象：{path.name}")
    return payload
