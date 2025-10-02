#!/usr/bin/env python3
"""
Chat API for QuantEngine Research Interfaces

Provides REST API endpoints for sector research and group analysis chat interfaces.
Integrates with the QuantEngine LLM capabilities for intelligent market research.

Endpoints:
- POST /chat/sector - Sector research analysis
- POST /chat/group - Group/portfolio analysis
- GET /chat/history - Get chat history
- GET /chat/sectors - Get available sectors
- GET /chat/groups - Get available stock groups
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify, render_template_string
import traceback

# Add QuantEngine root to path
quant_engine_root = Path(__file__).parent.parent / 'QuantEngine'
if str(quant_engine_root) not in sys.path:
    sys.path.insert(0, str(quant_engine_root))

# Import chat interfaces
try:
    from sector_research_chat import SectorResearchChat
    from group_analysis_chat import GroupAnalysisChat
    CHAT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Chat interfaces not available: {e}")
    CHAT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global chat instances
sector_chat = None
group_chat = None

def initialize_chat_interfaces():
    """Initialize chat interfaces"""
    global sector_chat, group_chat
    
    if not CHAT_AVAILABLE:
        logger.error("❌ Chat interfaces not available")
        return False
    
    try:
        # Initialize sector research chat
        sector_chat = SectorResearchChat()
        logger.info("✅ Sector Research Chat initialized")
        
        # Initialize group analysis chat
        group_chat = GroupAnalysisChat()
        logger.info("✅ Group Analysis Chat initialized")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize chat interfaces: {e}")
        return False

# Initialize on startup
if not initialize_chat_interfaces():
    logger.warning("⚠️ Chat interfaces not available - API will return errors")

@app.route('/')
def index():
    """Main API documentation page"""
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>QuantEngine Chat API</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .method { color: #007bff; font-weight: bold; }
        .path { color: #28a745; font-family: monospace; }
        .description { margin-top: 10px; }
        .example { background: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 3px; font-family: monospace; }
    </style>
</head>
<body>
    <h1>🤖 QuantEngine Chat API</h1>
    <p>AI-powered market research and analysis through conversational interfaces.</p>
    
    <h2>Available Endpoints</h2>
    
    <div class="endpoint">
        <div class="method">POST</div>
        <div class="path">/chat/sector</div>
        <div class="description">Research specific sectors or groups of stocks</div>
        <div class="example">
{
  "question": "Research the technology sector outlook for the next 3 months",
  "sectors": ["technology"],
  "analysis_type": "outlook",
  "time_horizon": "3 months"
}
        </div>
    </div>
    
    <div class="endpoint">
        <div class="method">POST</div>
        <div class="path">/chat/group</div>
        <div class="description">Analyze groups of stocks or portfolios</div>
        <div class="example">
{
  "question": "Compare mega cap tech vs dividend aristocrats",
  "groups": ["mega_cap_tech", "dividend_aristocrats"],
  "analysis_type": "comparison"
}
        </div>
    </div>
    
    <div class="endpoint">
        <div class="method">GET</div>
        <div class="path">/chat/sectors</div>
        <div class="description">Get available sectors for analysis</div>
    </div>
    
    <div class="endpoint">
        <div class="method">GET</div>
        <div class="path">/chat/groups</div>
        <div class="description">Get available stock groups for analysis</div>
    </div>
    
    <div class="endpoint">
        <div class="method">GET</div>
        <div class="path">/chat/history</div>
        <div class="description">Get chat history</div>
    </div>
    
    <h2>Example Questions</h2>
    <ul>
        <li>"Research the technology sector outlook for the next 3 months"</li>
        <li>"Compare healthcare and financial sectors performance"</li>
        <li>"Analyze AAPL, MSFT, GOOGL for correlation analysis"</li>
        <li>"What's the risk profile of the energy sector?"</li>
        <li>"Compare mega cap tech vs dividend aristocrats"</li>
    </ul>
</body>
</html>
    """)

@app.route('/chat/sector', methods=['POST'])
def sector_research():
    """Sector research endpoint"""
    if not sector_chat:
        return jsonify({"error": "Sector research chat not available"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            analysis = loop.run_until_complete(sector_chat.ask_question(question))
        finally:
            loop.close()
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Sector research error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/group', methods=['POST'])
def group_analysis():
    """Group analysis endpoint"""
    if not group_chat:
        return jsonify({"error": "Group analysis chat not available"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get('question', '')
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            analysis = loop.run_until_complete(group_chat.ask_question(question))
        finally:
            loop.close()
        
        return jsonify({
            "success": True,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Group analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/sectors', methods=['GET'])
def get_sectors():
    """Get available sectors for analysis"""
    if not sector_chat:
        return jsonify({"error": "Sector research chat not available"}), 500
    
    try:
        sectors = {}
        for sector_name, sector_info in sector_chat.sector_definitions.items():
            sectors[sector_name] = {
                'name': sector_name,
                'description': sector_info['description'],
                'etf': sector_info['etf'],
                'key_stocks': sector_info['key_stocks'][:5],  # Top 5
                'subsectors': sector_info['subsectors']
            }
        
        return jsonify({
            "success": True,
            "sectors": sectors,
            "count": len(sectors),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Get sectors error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/groups', methods=['GET'])
def get_groups():
    """Get available stock groups for analysis"""
    if not group_chat:
        return jsonify({"error": "Group analysis chat not available"}), 500
    
    try:
        groups = {}
        for group_name, group_info in group_chat.predefined_groups.items():
            groups[group_name] = {
                'name': group_info['name'],
                'description': group_info['description'],
                'stocks': group_info['stocks'][:10],  # Top 10
                'category': group_info['category']
            }
        
        return jsonify({
            "success": True,
            "groups": groups,
            "count": len(groups),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Get groups error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history from both interfaces"""
    try:
        history = {
            "sector_research": sector_chat.research_history if sector_chat else [],
            "group_analysis": group_chat.research_history if group_chat else []
        }
        
        return jsonify({
            "success": True,
            "history": history,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Get history error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/status', methods=['GET'])
def get_chat_status():
    """Get chat system status"""
    try:
        status = {
            "sector_chat_available": sector_chat is not None,
            "group_chat_available": group_chat is not None,
            "llm_available": sector_chat.llm_client is not None if sector_chat else False,
            "data_broker_available": sector_chat.data_broker is not None if sector_chat else False,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify({
            "success": True,
            "status": status
        })
        
    except Exception as e:
        logger.error(f"❌ Get status error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/chat/example', methods=['GET'])
def get_example_questions():
    """Get example questions for testing"""
    try:
        examples = {
            "sector_research": [
                "Research the technology sector outlook for the next 3 months",
                "Compare healthcare and financial sectors performance",
                "What are the best performing stocks in the energy sector?",
                "Analyze the consumer discretionary sector for trading opportunities",
                "What's the risk profile of the real estate sector?",
                "How will the Fed decision impact the financial sector?",
                "Research semiconductor stocks in the technology sector"
            ],
            "group_analysis": [
                "Analyze the mega cap tech group performance",
                "Compare dividend aristocrats vs growth stocks",
                "Research AAPL, MSFT, GOOGL, AMZN, META for correlation analysis",
                "What's the risk profile of the fintech group?",
                "Analyze diversification in the biotech group",
                "Compare value stocks vs growth stocks performance",
                "Research TSLA, NVDA, AMD for momentum analysis"
            ]
        }
        
        return jsonify({
            "success": True,
            "examples": examples,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Get examples error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🤖 Starting QuantEngine Chat API...")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  / - API documentation")
    print("  POST /chat/sector - Sector research")
    print("  POST /chat/group - Group analysis")
    print("  GET  /chat/sectors - Available sectors")
    print("  GET  /chat/groups - Available groups")
    print("  GET  /chat/history - Chat history")
    print("  GET  /chat/status - System status")
    print("  GET  /chat/example - Example questions")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
