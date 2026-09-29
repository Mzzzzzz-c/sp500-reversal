#!/bin/bash
# 双击运行：更新数据 → 回测 → 打开报告。完整输出保存在 logs/backtest_log.txt
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee logs/backtest_log.txt) 2>&1
echo "开始时间：$(date)"
PY="../venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" -c "import pandas, numpy, yaml, requests, lxml" 2>/dev/null || \
  "$PY" -m pip install -q -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com pandas numpy pyyaml requests lxml
set -e
"$PY" scripts/update_data.py
"$PY" scripts/run_backtest.py "$@"
open results/report.html
echo "✅ 完成"
read -n 1 -s -r -p "按任意键关闭..."
