"""回测方案：你最初的设想 vs 完善后的计划规格，以及三档成本。"""
from __future__ import annotations

from .config import override

# 你最初的设想：每天跌幅最大的 30 只，等权，不做任何过滤和风控
ORIGINAL = {
    "signal.rank_by": "raw", "signal.z_max": None, "signal.resid_max": None, "signal.raw_max": None,
    "signal.max_names": 30, "signal.sector_max_names": 30,
    "filters.min_adv_usd": 0, "filters.earnings_recent": False, "filters.earnings_next": False,
    "filters.ex_dividend": False, "filters.gap_news": "keep", "filters.volume_spike": None,
    "portfolio.weighting": "equal", "portfolio.max_weight": 1 / 30, "portfolio.sector_max_weight": 1.0,
    "portfolio.adv_max_frac": None, "portfolio.rebalance_band": 0.0,
    "risk.vix_half": None, "risk.vix_stop": None, "risk.dd_half": None, "risk.dd_stop": None,
    "risk.daily_loss_stop": None,
}

# 回撤熔断（影子净值回撤 ≥20% 暂停，修复到 <10% 才恢复）会让一个本身不赚钱的策略早早停掉且再也不恢复，
# 为了看清策略本身的表现，主要方案不开熔断；单独列一个开熔断的方案作对照。
NO_DD = {"risk.dd_half": None, "risk.dd_stop": None, "risk.daily_loss_stop": None}

STRATEGIES = {
    "原始设想：跌幅前30只": ORIGINAL,
    "原始设想：跌超5%": {**ORIGINAL, "signal.raw_max": -0.05},
    "原始设想：次日开盘卖": {**ORIGINAL, "portfolio.exit": "open"},
    "计划规格": NO_DD,
    "计划规格+对冲SPY": {**NO_DD, "portfolio.hedge": True},
    "计划规格+回撤熔断": {},
}

# 成本档位
SMALL = "小账户1万美元"
COSTS = {
    "零成本（理想）": {"costs.bps_per_side": 0, "costs.fees_bps": 0, "costs.min_commission": None,
                 "costs.hedge_bps": 0},
    "机构成本5bp": {"costs.bps_per_side": 5, "costs.fees_bps": 0, "costs.min_commission": None},
    "小账户1万美元": {},     # config.yaml 里的 IBKR 阶梯佣金 + 账户规模（account.value，默认 1 万美元示例）
}


def make(cfg: dict, strategy: str, cost: str) -> dict:
    return override(override(cfg, STRATEGIES[strategy]), COSTS[cost])
