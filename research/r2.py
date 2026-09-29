import sys, warnings; sys.path.insert(0,'.')
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from rev.config import load_config
from rev import dataset, strategy as ST
cfg=load_config(); D=dataset.load(cfg); H=pd.read_pickle("/home/claude/H.pkl")
adj=D["P"]["adjclose"]; opn=D["P"]["open"]; close=D["P"]["close"]; div=D["P"]["div"].fillna(0)
fwd={}
for k in [1,2,3,5]:
    f=adj.shift(-k)/adj-1; fwd[f"{k}d"]=f.sub(f["SPY"],axis=0)/k   # per-day excess
on=(opn.shift(-1)+div.shift(-1))/close-1; fwd["隔夜"]=on.sub(on["SPY"],axis=0)
periods={"训练11-19":("2011-01-01","2019-12-31"),"验证20-22":("2020-01-01","2022-12-31"),"测试23-26":("2023-01-01","2026-12-31")}
S=ST.signal_features(D,H,cfg,"daily")
mem=D["member"].reindex(index=S["idx"],columns=S["r"].columns,fill_value=False)&S["r"].notna()&S["z"].notna()
ok,_=ST.eligibility(D,H,S,cfg)
liq=mem&(S["px"]>=5)&(H["adv"].reindex(S["idx"])>=5e7)
rk=S["r"].where(mem).rank(axis=1,method="first")
V={"A 跌幅前30":rk<=30,"B 跌超5%":mem&(S["r"]<=-0.05),"B2 跌超10%":mem&(S["r"]<=-0.10),
   "C z≤-2":liq&(S["z"]<=-2),"D 计划规格":ok&(S["z"]<=-2)&(S["e"]<=-0.02)}
rows=[]
for vn,m in V.items():
    for h,f in fwd.items():
        row={"变体":vn,"持有":h}
        for pn,(a,b) in periods.items():
            mm=m.loc[a:b]; dm=f.reindex(mm.index).where(mm).mean(axis=1).dropna()
            row[pn+" bp/天"]=round(dm.mean()*1e4,1); row[pn+" t"]=round(dm.mean()/dm.std()*np.sqrt(len(dm)),2)
        rows.append(row)
print(pd.DataFrame(rows).set_index(["变体","持有"]).to_string())
