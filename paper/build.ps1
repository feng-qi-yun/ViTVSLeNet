# 编译课程报告：XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX
# 适用平台：Windows（PowerShell）
# 前置依赖：系统 PATH 中有 xelatex 与 bibtex（MiKTeX 或 TeX Live）
# 用法：cd paper; .\build.ps1
$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    # 依赖检查：xelatex 与 bibtex 必须在 PATH 中
    foreach ($cmd in @('xelatex', 'bibtex')) {
        if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
            throw "找不到 $cmd。请先安装 MiKTeX 或 TeX Live，参见 README.md「论文」一节。"
        }
    }

    Write-Host '==> xelatex (第 1 遍)'
    xelatex -interaction=nonstopmode main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "xelatex 第 1 遍失败（退出码 $LASTEXITCODE），检查 main.log" }

    Write-Host '==> bibtex'
    bibtex main | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "bibtex 失败（退出码 $LASTEXITCODE），检查 main.blg" }

    Write-Host '==> xelatex (第 2 遍)'
    xelatex -interaction=nonstopmode main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "xelatex 第 2 遍失败（退出码 $LASTEXITCODE），检查 main.log" }

    Write-Host '==> xelatex (第 3 遍)'
    xelatex -interaction=nonstopmode main.tex | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "xelatex 第 3 遍失败（退出码 $LASTEXITCODE），检查 main.log" }

    if (Test-Path main.pdf) {
        $match = Select-String -Path main.log -Pattern 'Output written on main\.pdf \((\d+) pages'
        $pages = if ($match) { $match.Matches[0].Groups[1].Value } else { '?' }
        Write-Output "OK: main.pdf ($pages pages)"
    } else {
        throw 'main.pdf 未生成，检查 main.log'
    }
}
finally {
    Pop-Location
}
