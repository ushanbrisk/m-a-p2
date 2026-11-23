#!/usr/bin/env bash
# 一键创建 YuE2 推理环境.
#
#   bash setup_env.sh              # 建 env + 装依赖 + 装 wheel + 自检
#   bash setup_env.sh --recreate   # 先删掉同名 env 再建
#
# 幂等: 已经装好就直接跳过; 依赖版本对不上会用 requirements.txt 里的版本覆盖。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/home/luke/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-yue2}"
# 注意: 不能用 $CONDA_PREFIX —— 当前 shell 可能激活的是 base, 会导致脚本
# 用错 python。按名字从 conda env 列表里查, 查不到再退回默认路径。
ENV_PREFIX="$("$CONDA_BIN" env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
ENV_PREFIX="${ENV_PREFIX:-/home/luke/miniconda3/envs/$ENV_NAME}"
PY="$ENV_PREFIX/bin/python"
WHEEL="$PROJECT_DIR/yue2_infer-0.1.5-py3-none-any.whl"

if [[ "${1:-}" == "--recreate" ]]; then
    echo "==> 删除旧 env: $ENV_NAME"
    "$CONDA_BIN" env remove -n "$ENV_NAME" -y || true
fi

if [[ ! -x "$PY" ]]; then
    echo "==> 创建 conda env: $ENV_NAME (python 3.12)"
    "$CONDA_BIN" create -n "$ENV_NAME" python=3.12 -y
else
    echo "==> env 已存在: $ENV_NAME"
fi

# 已经装好且版本对得上就跳过 pip —— 每次 pip install 都要联网查一遍索引,
# 本机到 pypi 的链路很慢, 重跑一次要十几分钟。
if "$PY" "$PROJECT_DIR/scripts/check_env.py" >/dev/null 2>&1; then
    echo "==> 依赖版本已就位, 跳过 pip install (想强制重装就加 --force)"
    if [[ "${1:-}" == "--force" ]]; then
        echo "==> --force: 重新安装依赖"
        "$PY" -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"
        "$PY" -m pip install --no-deps "$WHEEL"
    fi
else
    echo "==> 安装 Python 依赖 (torch 2.10.0+cu128 等, 约 8GB 下载)"
    "$PY" -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"

    echo "==> 安装 yue2 wheel (--no-deps: 依赖上面已经按 METADATA 装好了)"
    "$PY" -m pip install --no-deps "$WHEEL"
fi

echo "==> 记录本环境的精确版本快照"
"$PY" -m pip freeze > "$PROJECT_DIR/requirements.lock.txt"

echo "==> 自检"
"$PY" "$PROJECT_DIR/scripts/check_env.py"

cat <<EOF

环境就绪。使用方法:

    conda activate $ENV_NAME
    cd $PROJECT_DIR
    python scripts/infer_yue2.py \\
        --style_txt examples/style.txt \\
        --lyrics_txt examples/lyrics.txt \\
        --output_dir output --id my_song

EOF
