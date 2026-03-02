import os
from comet_ml import API
from dotenv import load_dotenv

load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")
api = API(api_key=os.getenv("COMET_API_KEY"))
experiments = api.get_experiments(os.getenv("COMET_WORKSPACE"), os.getenv("COMET_PROJECT_NAME"))
exp_list = list(experiments)

# Check first experiment that has "v2" in the name (the screener run)
for exp in exp_list:
    if "_v2" in exp.name:
        print(f"\n--- {exp.name} ---")
        print("  METRICS:")
        for m in exp.get_metrics_summary():
            print(f"    {m['name']} = {m['valueCurrent']}")
        print("  PARAMETERS:")
        for p in exp.get_parameters_summary():
            print(f"    {p['name']} = {p['valueCurrent']}")
        print("  OTHER (logged data):")
        try:
            others = exp.get_others_summary()
            for o in others:
                print(f"    {o['name']} = {o['valueCurrent']}")
        except:
            pass
        break
