import os
from comet_ml import API
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")
api = API(api_key=os.getenv("COMET_API_KEY"))
workspace = os.getenv("COMET_WORKSPACE")

for project in ["stock-screener", "algo-ensemble", "general"]:
    experiments = api.get_experiments(workspace, project)
    exp_list = list(experiments)
    print(f"\nProject: {project}  ({len(exp_list)} experiments)")
    for exp in exp_list[:5]:
        metrics = {m["name"]: m["valueCurrent"] for m in exp.get_metrics_summary()}
        has_composite = "COMPOSITE_SCORE" in metrics
        print(f"  {exp.name[:50]}  |  has_COMPOSITE_SCORE={has_composite}  |  metrics={list(metrics.keys())[:4]}")
