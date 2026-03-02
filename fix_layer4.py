import re

path = "/workspaces/algo-ensemble/layer_4_elimination.py"

new_fetch = """def fetch_top_n_from_comet(top_n: int) -> pd.DataFrame:
    api = API(api_key=COMET_API_KEY)
    experiments = api.get_experiments(COMET_WORKSPACE, COMET_PROJECT)
    exp_list = list(experiments)
    print(f"  [INFO] Total experiments in {COMET_PROJECT}: {len(exp_list)}")
    records = []
    for exp in exp_list:
        try:
            v = get_all_logged_values(exp)
            composite = float(v.get("composite_score", 0))
            if composite <= 0:
                continue
            symbol = extract_symbol(exp.name)
            records.append({
                "experiment_name":  exp.name,
                "symbol":           symbol,
                "COMPOSITE_SCORE":  composite,
                "SENTIMENT_SCORE":  float(v.get("sentiment_score",  0)),
                "VOLATILITY_SCORE": float(v.get("volatility_score", 0)),
                "RETURN_20C_PCT":   float(v.get("return_20c_pct",   0)),
                "HIGH_52W":         float(v.get("52w_high",         0)),
            })
        except Exception as e:
            print(f"  [WARN] Could not read {exp.name}: {e}")
    if not records:
        print("  [ERROR] No experiments with composite_score > 0 found.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.sort_values("COMPOSITE_SCORE", ascending=False, inplace=True)
    df.drop_duplicates(subset="symbol", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  [INFO] {len(df)} unique symbols found.")
    return df.head(top_n)"""

with open(path, "r") as f:
    code = f.read()

pattern = r'def fetch_top_n_from_comet\(top_n.*?return df\.head\(top_n\)'
new_code = re.sub(pattern, new_fetch, code, flags=re.DOTALL)

with open(path, "w") as f:
    f.write(new_code)

print("DONE. Keys used:")
for line in new_code.split("\n"):
    if 'v.get("' in line:
        print(" ", line.strip())
