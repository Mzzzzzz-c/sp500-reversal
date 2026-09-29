"""标普 500 历史成分股（按日期）+ 行业分类。

来源：github.com/fja05680/sp500 —— 1996 年至今每次调整后的完整成分名单（代码为当时的代码）。
再用维基百科的当前成分表补上它最后一次更新之后的调整。

雅虎只保存【现存代码】的历史，所以：
- 改过代码的公司（FB→META、ANTM→ELV …）需要映射到现在的代码，否则会丢掉它们改名前的历史；
- 已退市/被收购的公司没有行情，只能缺失 —— 这是剩余的幸存者偏差，报告里会说明。
- 代码被别的公司重新使用的情况（如旧 IR 与新 IR、旧 SUN）需要单独处理，防止张冠李戴。
"""
from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import pandas as pd
import requests

HIST_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
            "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# 旧代码 → 雅虎上现在的代码（仅限同一家公司改代码；已用价格水平核对过）
RENAMES = {
    "FB": "META", "ANTM": "ELV", "BLL": "BALL", "PKI": "RVTY", "RE": "EG", "ABC": "COR", "FLT": "CPAY",
    "WLTW": "WTW", "FI": "FISV", "MMC": "MRSH", "BK": "BNY", "SATS": "ECHO", "BHGE": "BKR",
    "SYMC": "GEN", "NLOK": "GEN", "HCP": "DOC", "PEAK": "DOC", "JEC": "J", "UTX": "RTX", "ARNC": "HWM",
    "CTL": "LUMN", "DISCA": "WBD", "HRS": "LHX", "TMK": "GL", "DPS": "KDP",
}
# 代码被别家重用：在给定日期之前属于另一家公司 → 这段时间映射到别的代码或丢弃（None）
REUSED = {
    "IR": ("2020-03-03", "TT"),     # 2020-03-02 前的 IR 是 Ingersoll-Rand plc（现 TT）；之后是原 Gardner Denver
    "SUN": ("2099-01-01", None),     # 旧 Sunoco（2012 年退出），雅虎的 SUN 是 Sunoco LP
    "PARA": ("2099-01-01", None),    # 雅虎 PARA 数据异常
    "VIAC": ("2099-01-01", None),
    "CBS": ("2099-01-01", None),
    "CPWR": ("2099-01-01", None),    # 旧 Compuware；雅虎 CPWR 是别的证券
    "EP": ("2099-01-01", None),      # 旧 El Paso
    "GENZ": ("2099-01-01", None),    # 旧 Genzyme
    "BBT": ("2099-01-01", None),     # BB&T（已并入 TFC）；雅虎 BBT 是优先股
    "MI": ("2099-01-01", None),      # 旧 Marshall & Ilsley
}

GICS_ETF = {
    "Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE", "Industrials": "XLI",
    "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC",
}


def fetch_history(cache_path: str, refresh: bool = False) -> pd.DataFrame:
    if refresh or not os.path.exists(cache_path):
        r = requests.get(HIST_URL, timeout=60)
        r.raise_for_status()
        open(cache_path, "wb").write(r.content)
    return pd.read_csv(cache_path)


def fetch_current(cache_path: str, refresh: bool = False) -> pd.DataFrame:
    """维基百科当前成分表：Symbol, Security, GICS Sector, Date added。"""
    if refresh or not os.path.exists(cache_path):
        html = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        df = pd.read_html(io.StringIO(html))[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df[["Symbol", "Security", "GICS Sector", "Date added"]].to_csv(cache_path, index=False)
    return pd.read_csv(cache_path)


def membership(hist: pd.DataFrame, current: Optional[pd.DataFrame], trading_days: pd.DatetimeIndex,
               start: str = "2010-01-01") -> pd.DataFrame:
    """返回 交易日 × 代码（雅虎代码）的布尔矩阵：当天是否为标普 500 成分。"""
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date")
    rows = [(d, set(x.strip().replace(".", "-") for x in t.split(","))) for d, t in zip(h["date"], h["tickers"])]
    if current is not None and len(current):
        cur = set(current["Symbol"].str.replace(".", "-", regex=False))
        last_d, last_set = rows[-1]
        if cur != last_set:
            added = pd.to_datetime(current.loc[current["Symbol"].isin(cur - last_set), "Date added"], errors="coerce")
            d = added.max() if added.notna().any() else pd.Timestamp.today().normalize()
            if pd.notna(d) and d > last_d:
                rows.append((d, cur))
    td = pd.DatetimeIndex(trading_days)
    td = td[td >= pd.Timestamp(start)]
    dates = pd.DatetimeIndex([r[0] for r in rows])
    pos = dates.searchsorted(td, side="right") - 1        # 每个交易日对应的最近一次名单
    names = sorted(set().union(*[s for _, s in rows]))
    col = {n: i for i, n in enumerate(names)}
    mat = np.zeros((len(td), len(names)), dtype=bool)
    for k, p in enumerate(pos):
        if p >= 0:
            mat[k, [col[x] for x in rows[p][1]]] = True
    m = pd.DataFrame(mat, index=td, columns=names)
    # 代码映射
    out = {}
    for n in names:
        s = m[n]
        if n in REUSED:
            cut, alt = REUSED[n]
            early = s & (s.index < pd.Timestamp(cut))
            late = s & (s.index >= pd.Timestamp(cut))
            if alt:
                out[alt] = out.get(alt, False) | early
            out[n] = out.get(n, False) | late
            continue
        tgt = RENAMES.get(n, n)
        out[tgt] = out.get(tgt, False) | s
    return pd.DataFrame(out).astype(bool)


def drop_junk(member: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
              min_median_dollar_vol: float = 5e6) -> tuple:
    """成分期内成交额中位数过低的代码，多半是雅虎把代码给了别的小证券 —— 整段剔除。"""
    c = close.reindex(index=member.index, columns=member.columns)
    v = volume.reindex(index=member.index, columns=member.columns)
    med = (c * v).where(member).median()
    bad = sorted(med[med < min_median_dollar_vol].index)
    m = member.copy()
    m[bad] = False
    return m, bad


# SEC 行业代码（SIC）→ GICS 大类（粗略，仅用于已退出指数、拿不到 GICS 的股票）
def sic_to_gics(sic) -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return "Unknown"
    if s in (1311, 1381, 1382, 1389, 2911, 2990, 4610, 4612, 4613, 4922, 4923, 5171, 5172, 1221, 1222, 6792):
        return "Energy"
    if 1000 <= s < 1500 or 2400 <= s < 2500 or 2600 <= s < 2700 or 2800 <= s < 2830 or 2840 <= s < 2900 \
            or 3000 <= s < 3400:
        return "Materials"
    if 2830 <= s < 2840 or 3841 <= s <= 3851 or 8000 <= s < 8100 or s in (5122, 6324, 8731):
        return "Health Care"
    if 2000 <= s < 2200 or s in (5140, 5141, 5150, 5180, 5411, 5412, 5912) or 2840 <= s <= 2844:
        return "Consumer Staples"
    if 3570 <= s < 3580 or 3660 <= s < 3700 or 3820 <= s < 3830 or 7370 <= s < 7380 or s in (3825, 3826, 3827):
        return "Information Technology"
    if 4900 <= s < 4950 or 4950 <= s < 4970:
        return "Utilities"
    if 4800 <= s < 4900 or 2710 <= s < 2790 or 7810 <= s < 7830:
        return "Communication Services"
    if 6000 <= s < 6500 and s != 6798:
        return "Financials"
    if s == 6798 or 6500 <= s < 6600:
        return "Real Estate"
    if 5000 <= s < 6000 or 2300 <= s < 2400 or 3710 <= s < 3720 or 3940 <= s < 3950 or 7000 <= s < 7100 \
            or 5800 <= s < 5900 or 2500 <= s < 2600 or 1520 <= s < 1540:
        return "Consumer Discretionary"
    return "Industrials"


def sectors(symbols, current: Optional[pd.DataFrame] = None, extra: Optional[pd.DataFrame] = None,
            sec_info: Optional[dict] = None) -> pd.Series:
    """每只股票的 GICS 大类：当前成分表 > 额外表（symbol, sector）> SEC 行业代码推断。"""
    out = {}
    if sec_info:
        for s, v in sec_info.items():
            out[s] = sic_to_gics(v.get("sic"))
    if extra is not None:
        for s, sec in zip(extra["symbol"], extra["sector"]):
            if isinstance(sec, str):
                out[str(s).replace(".", "-")] = sec
    if current is not None:
        for s, sec in zip(current["Symbol"], current["GICS Sector"]):
            out[str(s).replace(".", "-")] = sec
    for old, new in RENAMES.items():
        if new in out and old not in out:
            out[old] = out[new]
    return pd.Series({s: out.get(s, "Unknown") for s in symbols})
