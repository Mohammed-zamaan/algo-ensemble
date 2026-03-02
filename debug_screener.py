import os
from comet_ml import API
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

api = API(api_key=os.getenv("COMET_API_KEY"))
workspace = os.getenv("COMET_WORKSPACE")
project   = os.getenv("COMET_PROJECT_NAME")

print(f"Connecting to: {workspace}/{project}")
experiments = api.get_experiments(workspace, project)
exp_list = list(experiments)
print(f"Total experiments: {len(exp_list)}")

for exp in exp_list[:3]:
    print(f"\n--- {exp.name} ---")
    summary = exp.get_metrics_summary()
    print(f"  Metrics count: {len(summary)}")
    for m in summary:
        print(f"    name={m['name']}  valueCurrent={m['valueCurrent']}")
