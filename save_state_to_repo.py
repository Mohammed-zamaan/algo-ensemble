import os, subprocess
from datetime import date
from dotenv import load_dotenv
load_dotenv()

STATE_FILES = [
    "state/positions_state.json",
    "state/equity_curve.csv",
    "state/trade_log.csv",
    "trade_candidates.csv",
    "trade_signals.csv",
    "trade_orders.csv",
]

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def commit_state_to_repo():
    today = str(date.today())
    print("\n[SAVE STATE] Committing state back to repo...")
    run_cmd('git config user.email "actions@github.com"')
    run_cmd('git config user.name "algo-ensemble-bot"')
    run_cmd("git pull --rebase origin main")
    for f in STATE_FILES:
        if os.path.exists(f):
            run_cmd(f"git add {f}")
            print(f"  [GIT] Staged: {f}")
    status = run_cmd("git diff --cached --name-only")
    if not status:
        print("  [GIT] No state changes — skipping push")
        return
    run_cmd(f'git commit -m "state: auto-update {today}"')
    run_cmd("git push origin main")
    print("[SAVE STATE] Done")

if __name__ == "__main__":
    commit_state_to_repo()
