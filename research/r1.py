import sys, warnings; sys.path.insert(0,'.')
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from rev.config import load_config, override
from rev import dataset, strategy as ST, metrics as MT
cfg=load_config(); D=dataset.load(cfg); H=pd.read_pickle("/home/claude/H.pkl")
adj=D["P"]["adjclose"]; opn=D["P"]["open"]; close=D["P"]["close"]; div=D["P"]["div"].fillna(0)
spyadj=adj["SPY"]
fwd={}
for k in [1,2,3,5]:
    f=adj.shift(-k)/adj-1; fwd[k]=f.sub(f["SPY"],axis=0)   # excess vs SPY
on=(opn.shift(-1)+div.shift(-1))/close-1; fwd["on"]=on.sub(on["SPY"],axis=0)
periods={"train 2011-19":("2011-01-01","2019-12-31"),"valid 2020-22":("2020-01-01","2022-12-31"),"test 2023-26":("2023-01-01","2026-12-31")}
def stats(mask, S, label):
    row={"label":label}
    for pn,(a,b) in periods.items():
        m=mask.loc[a:b]
        n=int(m.values.sum())
        x=fwd[1].reindex(m.index).where(m).stack()
        row[pn+" n/day"]=round(n/ max(len(m),1),1)
        row[pn+" 1d bp"]=round(x.mean()*1e4,1)
        # t-stat on daily portfolio means (equal-weight per day), robust to cross-correlation
        dm=fwd[1].reindex(m.index).where(m).mean(axis=1).dropna()
        row[pn+" t"]=round(dm.mean()/dm.std()*np.sqrt(len(dm)),2) if len(dm)>5 else np.nan
    return row
def variants(mode):
    S=ST.signal_features(D,H,cfg,mode)
    mem=D["member"].reindex(index=S["idx"],columns=S["r"].columns,fill_value=False)&S["r"].notna()&S["z"].notna()
    ok,_=ST.eligibility(D,H,S,cfg)
    base_liq=mem&(S["px"]>=5)&(H["adv"].reindex(S["idx"])>=5e7)
    rk=S["r"].where(mem).rank(axis=1,method="first")
    zr=S["z"].where(ok).rank(axis=1,method="first")
    out=[]
    out.append(stats(rk<=30,S,"A 原始：跌幅最大30只（无过滤）"))
    out.append(stats(mem&(S["r"]<=-0.05),S,"B 原始：跌幅超5%（无过滤）"))
    out.append(stats(mem&(S["r"]<=-0.10),S,"B2 跌幅超10%（无过滤）"))
    out.append(stats(base_liq&(S["z"]<=-2),S,"C z≤-2（仅流动性过滤）"))
    cand=ok&(S["z"]<=-2)&(S["e"]<=-0.02)
    out.append(stats(cand,S,"D 计划规格：z≤-2 且残差≤-2% + 全部过滤"))
    out.append(stats(cand&(S["z"]<=-3),S,"D2 计划规格 + z≤-3"))
    # earnings-driven drops
    er=D["earn_react"].reindex(index=S["idx"],columns=S["r"].columns,fill_value=False)
    out.append(stats(base_liq&er&(S["r"]<=-0.05),S,"E 财报当天跌超5%（对照）"))
    # market down day
    mdown=(S["spy_r"]<=-0.01)
    out.append(stats(cand&mdown.values[:,None],S,"F 计划规格，仅大盘跌≥1%的日子"))
    out.append(stats(cand&(~mdown).values[:,None],S,"G 计划规格，大盘未大跌的日子"))
    return pd.DataFrame(out).set_index("label")
for mode in ["daily","intraday"]:
    print("\n######", mode, "— 次日（收盘→收盘）相对 SPY 超额收益，bp")
    print(variants(mode).to_string())
