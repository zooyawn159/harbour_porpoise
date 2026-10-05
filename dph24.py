import glob, os, re
import pandas as pd

resp = pd.read_csv(glob.glob("**/*Porpoise_responses_to_construction*.csv", recursive=True)[0])
pil = pd.read_csv(glob.glob("**/*Piling_summary_by_turbine*.csv", recursive=True)[0])
for c in ("start_time", "end_time"):
    pil[c] = pd.to_datetime(pil[c], dayfirst=True)
pil = pil.set_index("turbine")

# 클릭 파일: 파일 이름 맨 앞 숫자가 dep_no
files = {int(os.path.basename(f).split()[0]): f
         for f in glob.glob("**/*.csv", recursive=True)
         if re.match(r"\d+ \d+ \d{4}", os.path.basename(f))}

def hourly(dep):
    d = pd.read_csv(files[dep], usecols=["ChunkEnd", "Nfiltered", "MinsOn"])
    d["t"] = pd.to_datetime(d["ChunkEnd"], dayfirst=True)
    d = d[d["MinsOn"] == 1].set_index("t")
    return d["Nfiltered"].resample("h").sum() > 0   # 가설: 한 시간에 filtered click이 있으면 positive

H = pd.Timedelta(hours=1)
rows = []
sample = resp.dropna(subset=["dph24", "base24"]).sample(10, random_state=0)
for _, r in sample.iterrows():
    h = hourly(int(r["dep_no"]))
    end = pil.loc[r["turbine"], "end_time"].floor("h")
    start = pil.loc[r["turbine"], "start_time"].floor("h")
    rows.append({
        "dep_no": int(r["dep_no"]), "turbine": r["turbine"],
        "dph24_표": r["dph24"],
        "A_종료후24h": h[end:end + 23 * H].sum(),
        "B_종료까지24h": h[end - 23 * H:end].sum(),
        "base24_표": r["base24"],
        "base_시작직전24h": h[start - 24 * H:start - H].sum(),
    })
print(pd.DataFrame(rows).to_string())