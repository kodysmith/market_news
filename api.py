from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__)

@app.route('/report.json')
def get_report_json():
    try:
        return send_from_directory(os.getcwd(), 'report.json')
    except FileNotFoundError:
        return jsonify({"error": "report.json not found"}), 404

@app.route('/')
def index():
    return "Market News API is running. Access /report.json for data."

if __name__ == '__main__':
    # Ensure report.json exists for the API to serve
    if not os.path.exists('report.json'):
        print("Warning: report.json not found in the current directory. Please run generate_report.py first.")
    app.run(host='0.0.0.0', port=5000, debug=True)