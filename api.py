from flask import Flask, send_from_directory, jsonify
import os
from flask_cors import CORS
import subprocess
import time

app = Flask(__name__)
CORS(app)  # This will allow all domains. For production, restrict origins!

REPORT_PATH = 'report.json'
REFRESH_INTERVAL = 30 * 60  # 30 minutes in seconds

def is_report_stale():
    if not os.path.exists(REPORT_PATH):
        return True
    mtime = os.path.getmtime(REPORT_PATH)
    return (time.time() - mtime) > REFRESH_INTERVAL

@app.route('/report.json')
def get_report_json():
    try:
        if is_report_stale():
            # Regenerate the report
            subprocess.run(['python3', 'generate_report.py'], check=True)
        return send_from_directory(os.getcwd(), REPORT_PATH)
    except FileNotFoundError:
        return jsonify({"error": "report.json not found"}), 404
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Failed to regenerate report: {e}"}), 500

@app.route('/news.json')
def get_news_json():
    try:
        return send_from_directory(os.getcwd(), "news.json")
    except FileNotFoundError:
        return jsonify({"error": "news.json not found"}), 404

@app.route('/')
def index():
    return "Market News API is running. Access /report.json for data."

if __name__ == '__main__':
    # Ensure report.json exists for the API to serve
    if not os.path.exists('report.json'):
        print("Warning: report.json not found in the current directory. Please run generate_report.py first.")
    app.run(host='0.0.0.0', port=5000, debug=True)