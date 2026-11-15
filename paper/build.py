#!/usr/bin/env python3
"""跨平台编译入口：自动检测操作系统并调用对应的构建脚本。

用法:
    python build.py        # 在 paper/ 目录下
    python paper/build.py  # 在仓库根目录

分发策略:
    Windows  -> build.ps1  (PowerShell)
    Linux    -> build.sh   (bash)
    macOS    -> build.sh   (bash)
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent


def main() -> int:
    system: str = platform.system()
    print(f"[build.py] 检测到操作系统: {system}")

    if system == "Windows":
        # 优先 PowerShell Core（pwsh），否则回退到 Windows 自带的 Windows PowerShell
        ps: str = "pwsh" if shutil.which("pwsh") else "powershell"
        script: Path = HERE / "build.ps1"
        print(f"[build.py] 使用 {ps} 运行 {script.name}")
        return subprocess.call(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        )

    if system in ("Linux", "Darwin"):
        script = HERE / "build.sh"
        print(f"[build.py] 使用 bash 运行 {script.name}")
        return subprocess.call(["bash", str(script)])

    print(f"[build.py] 错误：不支持的操作系统 {system}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
