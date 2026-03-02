import os
from comet_ml import API
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

api = API(api_key=os.getenv("COMET_API_KEY"))
experiments = api.get_experiments(
    os.getenv("COMET_WORKSPACE"),
    os.getenv("COMET_PROJECT_NAME")
)

exp_list = list(experiments)
print(f"Total experiments fetched: {len(exp_list)}")

for exp in exp_list[:3]:
    print(f"\n--- {exp.name} ---")
    try:
        summary = exp.get_metrics_summary()
        print(f"  get_metrics_summary() -> {len(summary)} items")
        for m in summary[:5]:
            print(f"    {m}")
    except Exception as e:
        print(f"  get_metrics_summary() ERROR: {e}")

    try:
        logged = exp.get_metrics()
        print(f"  get_metrics() -> {len(logged)} items")
        for m in list(logged)[:5]:
            print(f"    {m}")
    except Exception as e:
        print(f"  get_metrics() ERROR: {e}")
