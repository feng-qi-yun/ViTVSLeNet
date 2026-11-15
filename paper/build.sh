#!/usr/bin/env bash
# 编译课程报告：XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX
# 适用平台：Linux / macOS（两个平台共用）
# 前置依赖：系统 PATH 中有 xelatex 与 bibtex（TeX Live 或 MacTeX）
# 用法：cd paper && ./build.sh
set -euo pipefail

# 切到脚本所在目录（paper/），保证相对路径稳定
cd "$(dirname "$0")"

# 依赖检查：xelatex 与 bibtex 必须在 PATH 中
for cmd in xelatex bibtex; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "错误：找不到 $cmd。" >&2
        echo "请先安装 TeX Live（Linux）或 MacTeX（macOS），参见 README.md「论文」一节。" >&2
        exit 1
    fi
done

echo '==> xelatex (第 1 遍)'
xelatex -interaction=nonstopmode main.tex

echo '==> bibtex'
bibtex main

echo '==> xelatex (第 2 遍)'
xelatex -interaction=nonstopmode main.tex

echo '==> xelatex (第 3 遍)'
xelatex -interaction=nonstopmode main.tex

if [[ -f main.pdf ]]; then
    pages="$(grep -oE 'Output written on main\.pdf \([0-9]+ pages' main.log \
        | grep -oE '[0-9]+' | tail -1 || true)"
    echo "OK: main.pdf (${pages:-?} pages)"
else
    echo "错误：main.pdf 未生成，请检查 main.log。" >&2
    exit 1
fi
