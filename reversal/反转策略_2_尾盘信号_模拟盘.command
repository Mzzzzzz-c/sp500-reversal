#!/bin/bash
# 美东 15:30–15:45 双击运行（北京时间：夏令时 03:30–03:45，冬令时 04:30–04:45）
# 生成当天的收盘价市价单（MOC）清单，并按“全部成交”记入模拟盘；随后更新模拟盘净值。
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee "logs/live_$(date +%Y%m%d_%H%M).txt") 2>&1
PY="../venv/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" scripts/live_signal.py --assume-filled "$@"
echo
"$PY" scripts/paper_track.py || true
read -n 1 -s -r -p "按任意键关闭..."
