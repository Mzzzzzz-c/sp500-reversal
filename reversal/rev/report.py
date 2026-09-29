"""生成不依赖任何第三方库的 HTML 回测报告（图表为内嵌 SVG）。"""
from __future__ import annotations

import html
import math
from typing import Dict

import numpy as np
import pandas as pd

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563"]


def svg_lines(series: Dict[str, pd.Series], title: str, log: bool = True, width: int = 900,
              height: int = 320) -> str:
    ser = {k: v.dropna() for k, v in series.items() if v is not None and len(v.dropna()) > 1}
    if not ser:
        return ""
    x0 = min(s.index.min() for s in ser.values())
    x1 = max(s.index.max() for s in ser.values())
    vals = np.concatenate([s.values for s in ser.values()])
    vals = vals[vals > 0] if log else vals
    lo, hi = float(np.min(vals)), float(np.max(vals))
    f = (lambda v: math.log10(v)) if log else (lambda v: v)
    ylo, yhi = f(lo), f(hi)
    if yhi - ylo < 1e-9:
        yhi = ylo + 1
    L, R, T, B = 60, 20, 30, 30
    W, Hh = width - L - R, height - T - B
    span = (x1 - x0).days or 1

    def xy(d, v):
        return L + W * (d - x0).days / span, T + Hh * (1 - (f(v) - ylo) / (yhi - ylo))

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">',
           f'<text x="{L}" y="18" font-size="14" font-weight="600">{html.escape(title)}</text>']
    # y 轴刻度
    for k in range(5):
        yv = ylo + (yhi - ylo) * k / 4
        v = 10 ** yv if log else yv
        y = T + Hh * (1 - k / 4)
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{L - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="#6b7280">{v:.2f}</text>')
    for yr in range(x0.year + 1, x1.year + 1):
        d = pd.Timestamp(f"{yr}-01-01")
        x = L + W * (d - x0).days / span
        out.append(f'<text x="{x:.1f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="#6b7280">{yr}</text>')
    for i, (k, s) in enumerate(ser.items()):
        s = s[s > 0] if log else s
        step = max(1, len(s) // 600)
        pts = " ".join("%.1f,%.1f" % xy(d, v) for d, v in list(s.items())[::step] + [(s.index[-1], s.iloc[-1])])
        c = COLORS[i % len(COLORS)]
        out.append(f'<polyline fill="none" stroke="{c}" stroke-width="1.6" points="{pts}"/>')
        out.append(f'<rect x="{L + 10}" y="{T + 8 + i * 16}" width="10" height="10" fill="{c}"/>'
                   f'<text x="{L + 26}" y="{T + 17 + i * 16}" font-size="12">{html.escape(k)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def table(df: pd.DataFrame) -> str:
    return df.to_html(classes="t", border=0, na_rep="—", escape=True)


def page(title: str, sections: list) -> str:
    css = """
    body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#111827}
    h1{font-size:22px} h2{font-size:18px;margin-top:32px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}
    table.t{border-collapse:collapse;font-size:13px;margin:8px 0;overflow-x:auto;display:block}
    table.t th,table.t td{border:1px solid #e5e7eb;padding:4px 8px;text-align:right;white-space:nowrap}
    table.t th{background:#f9fafb} .note{color:#4b5563;font-size:14px;line-height:1.6}
    .warn{background:#fef2f2;border:1px solid #fecaca;padding:10px 14px;border-radius:6px}
    """
    body = [f"<h1>{html.escape(title)}</h1>"]
    for s in sections:
        body.append(s)
    return f'<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body>{"".join(body)}</body></html>'
