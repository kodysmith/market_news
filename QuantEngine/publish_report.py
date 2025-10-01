#!/usr/bin/env python3
"""
Publish QuantEngine reports to Firebase for the Market News App
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
import re

# Add QuantEngine to path
quant_engine_dir = Path(__file__).parent
sys.path.insert(0, str(quant_engine_dir))

def parse_markdown_report(report_path: str) -> dict:
    """Parse markdown report into structured data"""
    with open(report_path, 'r') as f:
        content = f.read()
    
    # Extract metadata
    lines = content.split('\n')
    title = ""
    generated_time = ""
    query = ""
    
    for line in lines:
        if line.startswith('# Research Report:'):
            title = line.replace('# Research Report:', '').strip()
        elif line.startswith('**Generated:**'):
            generated_time = line.replace('**Generated:**', '').strip()
        elif line.startswith('**Query:**'):
            query = line.replace('**Query:**', '').strip()
    
    # Extract current market data
    current_price = None
    daily_change = None
    volatility = None
    
    for i, line in enumerate(lines):
        if '**Current Price:**' in line:
            current_price = float(re.search(r'\$([\d.]+)', line).group(1))
        elif '**Daily Change:**' in line:
            daily_change = float(re.search(r'([+-]?[\d.]+)%', line).group(1))
        elif '**Volatility:**' in line:
            volatility = float(re.search(r'([\d.]+)%', line).group(1))
    
    # Extract support and resistance levels
    resistance_levels = []
    support_levels = []
    in_resistance_section = False
    in_support_section = False
    
    for line in lines:
        if '### Key Resistance Levels' in line:
            in_resistance_section = True
            in_support_section = False
            continue
        elif '### Key Support Levels' in line:
            in_resistance_section = False
            in_support_section = True
            continue
        elif line.startswith('###') and not in_resistance_section and not in_support_section:
            in_resistance_section = False
            in_support_section = False
            continue
        
        if in_resistance_section and line.startswith('- **$'):
            price_match = re.search(r'\$([\d.]+)', line)
            if price_match:
                resistance_levels.append(float(price_match.group(1)))
        elif in_support_section and line.startswith('- **$'):
            price_match = re.search(r'\$([\d.]+)', line)
            if price_match:
                support_levels.append(float(price_match.group(1)))
    
    # Extract scenarios
    scenarios = []
    in_scenarios = False
    current_scenario = {}
    
    for line in lines:
        if '## Price Scenarios' in line:
            in_scenarios = True
            continue
        elif line.startswith('##') and in_scenarios:
            in_scenarios = False
            continue
        
        if in_scenarios and line.startswith('###'):
            if current_scenario:
                scenarios.append(current_scenario)
            scenario_name = line.replace('###', '').strip()
            current_scenario = {
                'name': scenario_name,
                'probability': None,
                'target_price': None,
                'description': None
            }
        elif in_scenarios and current_scenario:
            if '**Probability:**' in line:
                prob_match = re.search(r'([\d.]+)%', line)
                if prob_match:
                    current_scenario['probability'] = float(prob_match.group(1))
            elif '**Target Price:**' in line:
                price_match = re.search(r'\$([\d.]+)', line)
                if price_match:
                    current_scenario['target_price'] = float(price_match.group(1))
            elif '**Description:**' in line:
                current_scenario['description'] = line.replace('**Description:**', '').strip()
    
    if current_scenario:
        scenarios.append(current_scenario)
    
    return {
        'title': title,
        'generated_time': generated_time,
        'query': query,
        'current_price': current_price,
        'daily_change': daily_change,
        'volatility': volatility,
        'resistance_levels': resistance_levels,
        'support_levels': support_levels,
        'scenarios': scenarios,
        'content': content
    }

def publish_to_firebase(report_data: dict, report_path: str):
    """Publish report to Firebase"""
    # Firebase Cloud Functions endpoint
    firebase_url = "https://api-hvi4gdtdka-uc.a.run.app"
    
    # Create the payload for the market news app
    payload = {
        "type": "quant_engine_report",
        "timestamp": datetime.now().isoformat(),
        "title": report_data['title'],
        "query": report_data['query'],
        "asset": report_data['title'].split(':')[0].strip() if ':' in report_data['title'] else "Unknown",
        "current_price": report_data['current_price'],
        "daily_change": report_data['daily_change'],
        "volatility": report_data['volatility'],
        "resistance_levels": report_data['resistance_levels'],
        "support_levels": report_data['support_levels'],
        "scenarios": report_data['scenarios'],
        "content": report_data['content'],
        "report_path": report_path,
        "generated_time": report_data['generated_time']
    }
    
    try:
        response = requests.post(
            f"{firebase_url}/publish-quant-report",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print(f"✅ Report published successfully: {report_data['title']}")
            return True
        else:
            print(f"❌ Failed to publish report: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error publishing report: {e}")
        return False

def main():
    """Main function to publish reports"""
    if len(sys.argv) < 2:
        print("Usage: python publish_report.py <report_path>")
        sys.exit(1)
    
    report_path = sys.argv[1]
    
    if not os.path.exists(report_path):
        print(f"❌ Report file not found: {report_path}")
        sys.exit(1)
    
    print(f"📄 Parsing report: {report_path}")
    
    # Parse the markdown report
    report_data = parse_markdown_report(report_path)
    
    print(f"📊 Report data extracted:")
    print(f"   Title: {report_data['title']}")
    print(f"   Asset: {report_data['title'].split(':')[0].strip() if ':' in report_data['title'] else 'Unknown'}")
    print(f"   Current Price: ${report_data['current_price']}")
    print(f"   Resistance Levels: {len(report_data['resistance_levels'])}")
    print(f"   Support Levels: {len(report_data['support_levels'])}")
    print(f"   Scenarios: {len(report_data['scenarios'])}")
    
    # Publish to Firebase
    print(f"🚀 Publishing to Firebase...")
    success = publish_to_firebase(report_data, report_path)
    
    if success:
        print("✅ Report published successfully!")
    else:
        print("❌ Failed to publish report")
        sys.exit(1)

if __name__ == "__main__":
    main()

