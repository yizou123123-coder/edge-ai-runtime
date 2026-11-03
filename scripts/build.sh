#!/usr/bin/env bash
set -euo pipefail

# 参数1：构建预设名称
preset="${1:?Usage: $0 <preset>}"

# 定位项目根目录
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

# 配置
cmake --preset "$preset"

# 编译
cmake --build --preset "$preset" -j4


