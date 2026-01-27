from __future__ import annotations

import os
import subprocess
import time

from flask import Blueprint, jsonify, send_from_directory

from .shared import REPORT_PATH

bp = Blueprint("report", __name__)


@bp.route("/report.json")
def get_report_json():
    try:
        # Try to find report.json in current directory or parent
        report_path = REPORT_PATH
        if not os.path.exists(report_path):
            # Try parent directory (apis/../data/report.json)
            parent_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "report.json"
            )
            if os.path.exists(parent_path):
                report_path = parent_path
            else:
                # If report doesn't exist, skip regeneration and return error
                return (
                    jsonify(
                        {
                            "error": "report.json not found",
                            "message": "Report file not available. Some features may not work.",
                        }
                    ),
                    404,
                )

        # Only try to regenerate if explicitly stale (skip for now to avoid errors)
        # if is_report_stale():
        #     try:
        #         subprocess.run(['python3', 'generate_report.py'], check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        #     except (subprocess.CalledProcessError, FileNotFoundError):
        #         pass  # Continue with existing file if regeneration fails

        # Serve the file
        abs_path = os.path.abspath(report_path)
        directory = os.path.dirname(abs_path)
        filename = os.path.basename(abs_path)
        return send_from_directory(directory, filename)
    except FileNotFoundError:
        return jsonify({"error": "report.json not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to serve report: {str(e)}"}), 500

