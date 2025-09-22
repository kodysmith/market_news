#!/usr/bin/env python3
"""
Startup script for all MarketNews APIs
"""

import subprocess
import time
import os
import signal
import sys
from threading import Thread

def run_command(command, cwd=None):
    """Run a command in a subprocess"""
    return subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

def monitor_process(process, name):
    """Monitor a process and print its output"""
    try:
        for line in iter(process.stdout.readline, ''):
            print(f"[{name}] {line.rstrip()}")
    except:
        pass

def main():
    """Start all APIs"""
    print("🚀 Starting MarketNews APIs...")
    
    processes = []
    
    try:
        # Start main API server
        print("Starting main API server on port 5000...")
        main_api = run_command("python3 api.py", cwd="/Users/kody/base/MarketNews")
        processes.append(("Main API", main_api))
        
        # Start options scanner
        print("Starting options scanner on port 8083...")
        options_scanner = run_command("python3 -m http.server 8083", cwd="/Users/kody/base/MarketNews/options_scanner")
        processes.append(("Options Scanner", options_scanner))
        
        # Start Buffett screener
        print("Starting Buffett screener on port 8084...")
        buffett_screener = run_command("python3 -m http.server 8084", cwd="/Users/kody/base/MarketNews/buffett_screener")
        processes.append(("Buffett Screener", buffett_screener))
        
        # Start monitoring threads
        monitor_threads = []
        for name, process in processes:
            thread = Thread(target=monitor_process, args=(process, name))
            thread.daemon = True
            thread.start()
            monitor_threads.append(thread)
        
        print("\n✅ All APIs started successfully!")
        print("📊 Main API: http://localhost:5000")
        print("📈 Options Scanner: http://localhost:8083/trade_ideas.html")
        print("🏰 Buffett Screener: http://localhost:8084/web/buffett_screener.html")
        print("\nPress Ctrl+C to stop all services...")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"❌ {name} process died unexpectedly")
                    return
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        
        # Terminate all processes
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except:
                try:
                    process.kill()
                    print(f"🔪 {name} force killed")
                except:
                    pass
        
        print("👋 All services stopped")

if __name__ == "__main__":
    main()

