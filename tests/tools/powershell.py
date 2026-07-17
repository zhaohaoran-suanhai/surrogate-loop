from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("powershell.exe")
POWERSHELL_7 = shutil.which("pwsh.exe")
POWERSHELL_EXECUTABLES = tuple(
    executable for executable in (POWERSHELL, POWERSHELL_7) if executable is not None
)


def ps_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def run_powershell(
    code: str,
    *,
    cwd: Path | None = None,
    executable: str | None = None,
) -> subprocess.CompletedProcess[str]:
    prelude = (
        "$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false);"
    )
    encoded = base64.b64encode((prelude + code).encode("utf-16le")).decode("ascii")
    return subprocess.run(
        [executable or POWERSHELL, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
