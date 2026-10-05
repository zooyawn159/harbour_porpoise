import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = glob.glob("**/*Porpoise_responses_to_construction*.csv", recursive=True)[0]
df = pd.read_csv(path)

# ── 1. 라벨 점검 ──
print("라벨 결측:", df[["resp24_50", "resp12_50"]].isna().sum().to_dict())
for w in ("24", "12"):
    na = df[df[f"resp{w}_50"].isna()]
    print(f"[{w}h] 라벨 NaN {len(na)}행 중 dph NaN: {na[f'dph{w}'].isna().sum()}, base==0: {(na[f'base{w}'] == 0).sum()}")
    print(f"base{w} 분포:\n{df[f'base{w}'].describe().round(1).to_string()}")
    print(f"base{w} <= 3 비율:", round((df[f"base{w}"] <= 3).mean(), 3))
    ok = df[f"resp{w}_50"].notna()
    recomputed = ((df[f"dph{w}"] - df[f"base{w}"]) / df[f"base{w}"] <= -0.5).astype(float)
    print(f"라벨 재계산 일치율({w}h):", (recomputed[ok] == df.loc[ok, f"resp{w}_50"]).mean())
print("24h vs 12h 라벨 교차표 (-1 = 결측)")
print(pd.crosstab(df["resp24_50"].fillna(-1), df["resp12_50"].fillna(-1)))

# ── 2. 변수 분포 + 반응률 ──
df["dist_bin"] = pd.qcut(df["distance"], 8, duplicates="drop")
df["order_grp"] = pd.cut(df["piling_order"], [0, 12, 47, 70, 90])
df["vessel_grp"] = pd.cut(df["vessels24_1km"], [-1, 0, 10, 100, np.inf])

def rate(col, y="resp24_50"):   # 12시간으로 보려면 y="resp12_50"
    return df.dropna(subset=[y]).groupby(col, observed=True)[y].agg(["mean", "count"])

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].hist(df["distance"].dropna(), bins=30); ax[0, 0].set_title("distance (km)")
ax[0, 1].hist(np.log1p(df["vessels24_1km"]), bins=30); ax[0, 1].set_title("log1p(vessels24_1km)")
ax[0, 2].hist(df["base24"].dropna(), bins=24); ax[0, 2].set_title("base24 (baseline detection-positive hours)")
for a, col in zip(ax[1], ["dist_bin", "order_grp", "vessel_grp"]):
    g = rate(col)
    a.bar(range(len(g)), g["mean"])
    a.set_xticks(range(len(g)))
    a.set_xticklabels([str(i) for i in g.index], rotation=45, ha="right")
    for i, n in enumerate(g["count"]):
        a.text(i, g["mean"].iloc[i], f"n={n}", ha="center", va="bottom", fontsize=8)
    a.set_title(f"response rate (24h) by {col}"); a.set_ylim(0, 1)
plt.tight_layout(); plt.savefig("eda_overview.png", dpi=150); plt.show()

# ── 3. 터빈 단위 그림 ──
t = df.groupby("turbine").agg(order=("piling_order", "first"), add=("ADD", "first"),
                              n=("dep_no", "size"), rate=("resp24_50", "mean"))
fig2, a2 = plt.subplots(figsize=(7, 5))
for k, c in (("Y", "tab:blue"), ("N", "tab:red")):
    s = t[t["add"] == k]
    a2.scatter(s["order"], s["rate"], c=c, s=s["n"] * 3, label=f"ADD={k}")
a2.set_xlabel("piling_order"); a2.set_ylabel("mean response rate (24h)"); a2.legend()
plt.savefig("eda_turbine.png", dpi=150); plt.show()

# ── 4. 터빈 안에서의 거리 효과 (행이 많은 6개 터빈) ──
big = t.sort_values("n", ascending=False).index[:6]
fig3, a3 = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for a, tb in zip(a3.ravel(), big):
    s = df[df["turbine"] == tb].dropna(subset=["resp24_50"]).sort_values("distance")
    a.scatter(s["distance"], s["resp24_50"] + np.random.uniform(-0.04, 0.04, len(s)), s=10)
    a.plot(s["distance"], s["resp24_50"].rolling(9, center=True, min_periods=3).mean(), c="r")
    a.set_title(f"{tb} (order {int(t.loc[tb, 'order'])}, n={len(s)})")
    a.set_xlabel("distance (km)")
plt.tight_layout(); plt.savefig("eda_within_turbine.png", dpi=150); plt.show()