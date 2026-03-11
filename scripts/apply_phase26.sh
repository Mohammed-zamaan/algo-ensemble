#!/usr/bin/env bash
set -e

echo "Applying Phase 26: IST timezone hardening"

ROOT="src/trading_ensemble"

mkdir -p "$ROOT/core"

########################################
# 1) Create core/timeutils.py
########################################
cat > "$ROOT/core/timeutils.py" <<'PY'
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def now_ist() -> datetime:
    return datetime.now(IST)


def now_utc() -> datetime:
    return datetime.now(UTC)


def fmt_ist(dt: datetime | None = None) -> str:
    value = dt or now_ist()
    return value.strftime("%Y-%m-%d %H:%M:%S")


def fmt_utc(dt: datetime | None = None) -> str:
    value = dt or now_utc()
    return value.strftime("%Y-%m-%d %H:%M:%S")
PY

########################################
# 2) Patch monitor/scheduler.py
########################################
python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/monitor/scheduler.py")
text = path.read_text()

if "from trading_ensemble.core.timeutils import now_ist, fmt_utc" not in text:
    text = text.replace(
        "from trading_ensemble.dashboard.command_center import build_command_center\n",
        "from trading_ensemble.dashboard.command_center import build_command_center\n"
        "from trading_ensemble.core.timeutils import now_ist, fmt_utc\n",
    )

text = text.replace(
    "    now = datetime.now()\n",
    "    now = now_ist()\n",
)

text = text.replace(
    '    print(f"  cycle_time           = {cycle_time}")\n',
    '    print(f"  cycle_time_ist       = {cycle_time}")\n'
    '    print(f"  cycle_time_utc       = {fmt_utc()}")\n',
)

path.write_text(text)
PY

########################################
# 3) Patch dashboard/command_center.py
########################################
python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/dashboard/command_center.py")
text = path.read_text()

if "from trading_ensemble.core.timeutils import fmt_ist" not in text:
    text = text.replace(
        "from datetime import datetime\n",
        "",
    )
    text = text.replace(
        "import pandas as pd\n",
        "import pandas as pd\n"
        "from trading_ensemble.core.timeutils import fmt_ist\n",
    )

text = text.replace(
    '        ("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),',
    '        ("generated_at", fmt_ist()),',
)

path.write_text(text)
PY

########################################
# 4) Patch pipeline/summary.py
########################################
python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/pipeline/summary.py")
if path.exists():
    text = path.read_text()

    if "from trading_ensemble.core.timeutils import fmt_ist" not in text:
        text = text.replace(
            "from datetime import datetime\n",
            "",
        )
        text = text.replace(
            "import pandas as pd\n",
            "import pandas as pd\n"
            "from trading_ensemble.core.timeutils import fmt_ist\n",
        )

    text = text.replace(
        '{"metric": "run_time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},',
        '{"metric": "run_time", "value": fmt_ist()},',
    )

    path.write_text(text)
PY

########################################
# 5) Patch monitor/trigger_engine.py
########################################
python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/monitor/trigger_engine.py")
text = path.read_text()

if "from trading_ensemble.core.timeutils import fmt_ist" not in text:
    text = text.replace(
        "from datetime import datetime\n",
        "",
    )
    text = text.replace(
        "import pandas as pd\n",
        "import pandas as pd\n"
        "from trading_ensemble.core.timeutils import fmt_ist\n",
    )

text = text.replace(
    '"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),',
    '"timestamp": fmt_ist(),',
)

text = text.replace(
    '    promoted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n',
    '    promoted_at = fmt_ist()\n',
)

path.write_text(text)
PY

########################################
# 6) Patch pipeline/stages/signals.py
########################################
python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/pipeline/stages/signals.py")
if path.exists():
    text = path.read_text()

    if "from trading_ensemble.core.timeutils import fmt_ist" not in text:
        text = text.replace(
            "from datetime import datetime, time as dt_time\n",
            "from datetime import time as dt_time\n",
        )
        text = text.replace(
            "import numpy as np\n",
            "import numpy as np\n"
            "from trading_ensemble.core.timeutils import fmt_ist\n",
        )

    text = text.replace(
        '"SIGNAL_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),',
        '"SIGNAL_TIME": fmt_ist(),',
    )

    path.write_text(text)
PY

########################################
# 7) Patch pipeline/stages/execution.py
########################################
python - <<'PY'
from pathlib import Path

path = Path("src/trading_ensemble/pipeline/stages/execution.py")
if path.exists():
    text = path.read_text()

    if "from trading_ensemble.core.timeutils import fmt_ist, now_ist" not in text:
        text = text.replace(
            "from datetime import datetime, date\n",
            "from datetime import date\n",
        )
        text = text.replace(
            "import pandas as pd\n",
            "import pandas as pd\n"
            "from trading_ensemble.core.timeutils import fmt_ist, now_ist\n",
        )

    text = text.replace(
        '        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n',
        '        now = fmt_ist()\n',
    )

    text = text.replace(
        "            broker_order_id = f\"PAPER-{date.today()}-{order['symbol']}\"\n",
        "            broker_order_id = f\"PAPER-{now_ist().date()}-{order['symbol']}\"\n",
    )

    path.write_text(text)
PY

echo "Phase 26 patch applied successfully."
sed -n '1,220p' scripts/apply_phase26.sh