# reversal —— 标普 500 尾盘超跌反转策略

这是 qsys 里的一个独立模块，不改动 qsys 原有的任何代码和数据。
策略：每天美东 15:30 找出标普 500 里“异常大跌”的股票，收盘前用 MOC 单买入，第二天收盘前卖出。

> **回测结论（2026-09-30）：这个策略在标普 500 里扣除成本后不赚钱，没有通过上线标准。**
> 详细数字见 `results/report.html`。程序默认只做**模拟跟踪**。

| 入选股票次日相对 SPY 的超额（零成本，每笔平均） | 训练 2011–19 | 验证 2020–22 | 测试 2023–26 | 15:30 真实信号 2023-10 起 |
|---|---|---|---|---|
| 原始设想：跌幅前 30 只 | -0.5bp | +0.1bp | +4.4bp | +5.1bp |
| 原始设想：跌超 5% | -4.3bp | -22.0bp | +13.3bp | +13.5bp |
| 计划规格（残差 z≤-2、剔除财报等） | -5.1bp | -14.0bp | -0.5bp | -1.0bp |

来回交易一次的成本：机构约 10bp，1 万美元的 IBKR 小账户约 20–25bp（每笔最低佣金固定，账户越小越贵）。
15:30 真实信号（2023-10 至 2026-09）按 1 万美元小账户的成本：原始设想年化 -43.6%，计划规格 -8.9%，同期 SPY +25.1%。
96 组参数网格里，训练期表现最好的一组到了验证期就变成负的。

## 目录
```
config.yaml                    所有参数（调参只改这里）
rev/data/yahoo.py              雅虎行情：日线、60 分钟线、实时快照
rev/data/sec_earnings.py       SEC 8-K 财报发布时间（精确到秒，区分盘前/盘后）
rev/data/universe.py           标普 500 历史成分股（按日期）、代码改名映射、行业分类
rev/data/store.py              本地缓存（data/）
rev/dataset.py                 把缓存整理成回测用的面板
rev/strategy.py                信号、过滤、选股、定权（回测与实盘共用）
rev/backtest.py                逐日回测引擎（MOC 成交、IBKR 阶梯佣金、风控）
rev/live.py                    实盘/模拟盘：新浪/雅虎实时行情 + Nasdaq 财报/除息日历 → 订单清单
rev/metrics.py rev/report.py   绩效统计、HTML 报告
rev/scenarios.py               回测方案（你最初的设想 / 计划规格 / 三档成本）
scripts/update_data.py         更新数据
scripts/run_backtest.py        回测（--grid 另跑参数网格）
scripts/live_signal.py         尾盘信号与 MOC 订单清单
scripts/paper_track.py         模拟盘记账
positions.csv                  本策略自己的持仓（symbol,quantity），和你其他策略分开
反转策略_1_更新数据并回测.command     双击：更新数据 → 回测 → 打开报告
反转策略_2_尾盘信号_模拟盘.command     双击：尾盘生成订单清单并记入模拟盘
```

## 使用

> GitHub 上的版本没有附带雅虎行情（`data/daily`、`data/intraday60`、`data/events`），第一次使用先运行 `python scripts/update_data.py`。

在 `quant/reversal` 目录下（用 qsys 的 venv：`../venv/bin/python`）：
```
python scripts/update_data.py          # 更新数据（首次约 5–10 分钟）
python scripts/run_backtest.py         # 回测，输出 results/report.html
python scripts/live_signal.py          # 美东 15:30–15:45 运行：当天信号 + MOC 订单清单
python scripts/paper_track.py          # 模拟盘净值
```
或者直接双击两个 `.command` 文件。

### 每天的时间表（北京时间）
| 美东 | 北京（夏令时 / 11 月 1 日后冬令时） | 做什么 |
|---|---|---|
| 收盘后 | 上午 | `update_data.py` 更新日线、60 分钟线、财报时间 |
| 15:30–15:45 | 03:30–03:45 / 04:30–04:45 | `live_signal.py` 生成当天 MOC 清单 |
| 15:50 前 | 03:50 / 04:50 前 | （实盘才需要）在 IBKR 提交 MOC 单。NYSE 截止 15:50，Nasdaq 15:55 |

## 数据来源（全部免费）
| 数据 | 来源 | 说明 |
|---|---|---|
| 日线 OHLCV、分红、拆股 | 雅虎财经 | 2010 年起；只有现存代码 |
| 60 分钟 K 线 | 雅虎财经 | 只能取最近约 730 天；本地会一直累积，越用越长 |
| 历史成分股 | github.com/fja05680/sp500 + 维基百科 | 按日期的完整名单 |
| 财报发布时间 | SEC EDGAR 8-K Item 2.02 | 精确到秒，能区分盘前/盘后 |
| 实时行情 | 新浪美股（国内直连）/ 雅虎 | 实盘建议换成券商行情 |
| 未来财报、除息日历 | Nasdaq 官网接口 | 实盘过滤 F2/F3/F4 用 |
| 3 个月国债利率 | qsys 的 data/meta/tbill_3m.csv | 现金利息、夏普比率 |

国内访问雅虎需要代理时，在 `config.yaml` 的 `data.proxy` 填写，例如 `http://127.0.0.1:1087`（ShadowsocksX-NG 默认 HTTP 端口）。

## 已知局限
1. **幸存者偏差没有完全消除**：退市、被收购公司的行情雅虎没有。2011 年每天约有 340 只成分股有数据，2026 年约 500 只。缺的主要是后来被收购的公司。
2. 60 分钟线只有约 3 年，所以“15:30 真实信号”版本只能回测 2023-10 以后；更早的年份用“收盘价信号”近似，会有轻微前视偏差。报告里对比了两者在同一时间段的差别。
3. 行业分类用当前 GICS；已退出指数的公司用 SEC 行业代码推断。
4. 回测里按权重计算收益，没有模拟整数股取整。小账户用整数股时，高价股可能买不起（清单会自动放弃）。
5. IBKR 碎股订单支持的订单类型有限（能不能用 MOC 以 IBKR 的说明为准），所以默认用整数股。

## 只生成清单，不会自动下单
下单需要你自己在 IBKR 里确认。
