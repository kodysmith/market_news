from __future__ import annotations

import os

from .app_factory import create_app

app = create_app()


if __name__ == "__main__":
    # Ensure data/report.json exists for the API to serve
    if not os.path.exists("data/report.json"):
        print("Warning: report.json not found in the current directory. Please run generate_report.py first.")
    app.run(host="0.0.0.0", port=5000, debug=True)