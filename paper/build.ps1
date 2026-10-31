# 编译课程报告（XeLaTeX -> BibTeX -> XeLaTeX*2）
# 用法: cd paper; .\build.ps1
$ErrorActionPreference = 'Stop'
$texbin = Join-Path $PSScriptRoot '..\tools\texlive\2026\bin\windows'
$env:PATH = (Resolve-Path $texbin).Path + ';' + $env:PATH

Push-Location $PSScriptRoot
try {
    xelatex -interaction=nonstopmode main.tex | Out-Null
    bibtex main | Out-Null
    xelatex -interaction=nonstopmode main.tex | Out-Null
    xelatex -interaction=nonstopmode main.tex | Out-Null
    if (Test-Path main.pdf) {
        $pages = (Select-String -Path main.log -Pattern 'Output written on main\.pdf \((\d+) pages').Matches[0].Groups[1].Value
        Write-Output "OK: main.pdf ($pages pages)"
    } else {
        Write-Error 'main.pdf 未生成，检查 main.log'
    }
}
finally {
    Pop-Location
}
