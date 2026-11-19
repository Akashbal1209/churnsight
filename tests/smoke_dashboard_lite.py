"""Headless smoke test of the dashboard in deploy (LITE) mode.

Run with only requirements.txt installed (no torch):
    CHURNSIGHT_LITE=1 CHURNSIGHT_PROCESSED_DIR=data/app python tests/smoke_dashboard_lite.py
"""

import os
import sys

os.environ.setdefault("CHURNSIGHT_LITE", "1")
os.environ.setdefault("CHURNSIGHT_PROCESSED_DIR", "data/app")
os.environ.setdefault("LLM_PROVIDER", "none")

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file("../app/dashboard.py", default_timeout=120).run()
assert not at.exception, at.exception
print("tabs:", len(at.tabs), "| metrics:", [m.value for m in at.metric][:5])

search = next(b for b in at.button if b.label == "Search")
at.text_input[0].set_value("faltando item recebi apenas um")
search.click().run()
assert not at.exception, at.exception
print("search ok | info:", [i.value[:60] for i in at.info])
print("dataframes rendered:", len(at.dataframe))
sys.exit(0)
