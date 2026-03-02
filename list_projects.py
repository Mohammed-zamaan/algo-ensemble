import os
from comet_ml import API
from dotenv import load_dotenv
load_dotenv(dotenv_path="/workspaces/algo-ensemble/.env")

api = API(api_key=os.getenv("COMET_API_KEY"))
workspace = os.getenv("COMET_WORKSPACE")
print(f"Workspace: {workspace}")
projects = api.get_projects(workspace)
for p in projects:
    print(f"  Project: {p}")
