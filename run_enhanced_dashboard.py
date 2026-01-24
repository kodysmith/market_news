#!/usr/bin/env python3
"""
Enhanced Dashboard Launcher
Launches the enhanced Streamlit dashboard with AI chat integration
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the enhanced dashboard"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    dashboard_path = script_dir / "QuantEngine" / "enhanced_dashboard_with_chat.py"
    
    print("🚀 Starting Enhanced Market Scanner Dashboard with AI Chat...")
    print("=" * 60)
    print("Features:")
    print("  📊 Real-time market data for expanded symbol list")
    print("  🤖 AI-powered sector and group analysis")
    print("  💬 Conversational research assistant")
    print("  📈 Interactive charts and metrics")
    print("=" * 60)
    print(f"Dashboard: {dashboard_path}")
    print("=" * 60)
    
    # Check if dashboard file exists
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()



