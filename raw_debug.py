import os
from comet_ml import API
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

api = API(api_key=os.getenv("COMET_API_KEY"))
experiments = list(api.get_experiments("zamaan", "stock-screener"))

print(f"Total: {len(experiments)}")

# Check first 3 and last 3
for exp in experiments[:3] + experiments[-3:]:
    print(f"\n=== {exp.name} ===")
    metrics = exp.get_metrics_summary()
    params  = exp.get_parameters_summary()
    others  = exp.get_others_summary()
    print(f"  metrics={len(metrics)}  params={len(params)}  others={len(others)}")
    
    # Print ALL values from whichever has content
    for item in metrics + params + others:
        if item["name"] not in ["Name","storagesizebytes"] and not item["name"].startswith("sys."):
            print(f"  [{item['name']}] = {item['valueCurrent']}")
