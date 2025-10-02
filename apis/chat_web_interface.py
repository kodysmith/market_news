#!/usr/bin/env python3
"""
Web Interface for QuantEngine Chat System

A modern web interface for interacting with the QuantEngine chat APIs.
Provides an intuitive chat interface for sector research and group analysis.

Features:
- Real-time chat interface
- Sector and group selection
- Response formatting and display
- Chat history
- Example questions
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, render_template_string, request, jsonify, session
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'quantengine_chat_secret_key_2024'

# Chat API configuration
CHAT_API_BASE = "http://localhost:5001"

# Initialize session storage
if 'chat_history' not in session:
    session['chat_history'] = []

@app.route('/')
def index():
    """Main chat interface"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantEngine Chat - AI Market Research</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .container {
            flex: 1;
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            gap: 20px;
        }
        
        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            height: fit-content;
        }
        
        .main-content {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 500px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
            max-height: 500px;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease-in;
        }
        
        .message.user {
            text-align: right;
        }
        
        .message.assistant {
            text-align: left;
        }
        
        .message-content {
            display: inline-block;
            max-width: 80%;
            padding: 15px 20px;
            border-radius: 20px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.assistant .message-content {
            background: white;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .message-time {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .chat-input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s ease;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s ease;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
        }
        
        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        .section {
            margin-bottom: 25px;
        }
        
        .section h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }
        
        .sector-list, .group-list {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .sector-item, .group-item {
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s ease;
            border: 1px solid #e0e0e0;
        }
        
        .sector-item:hover, .group-item:hover {
            background: #e9ecef;
        }
        
        .sector-item.selected, .group-item.selected {
            background: #667eea;
            color: white;
        }
        
        .example-questions {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .example-question {
            padding: 8px 12px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s ease;
            font-size: 0.9em;
            border-left: 3px solid #667eea;
        }
        
        .example-question:hover {
            background: #e9ecef;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online {
            background: #28a745;
        }
        
        .status-offline {
            background: #dc3545;
        }
        
        .analysis-section {
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .analysis-section h4 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .analysis-content {
            color: #666;
            line-height: 1.6;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #f5c6cb;
        }
        
        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #c3e6cb;
        }
        
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
                padding: 10px;
            }
            
            .sidebar {
                width: 100%;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 QuantEngine Chat</h1>
        <p>AI-Powered Market Research & Analysis</p>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="section">
                <h3>📊 Sectors</h3>
                <div class="sector-list" id="sectorList">
                    <div class="loading">Loading sectors...</div>
                </div>
            </div>
            
            <div class="section">
                <h3>📈 Stock Groups</h3>
                <div class="group-list" id="groupList">
                    <div class="loading">Loading groups...</div>
                </div>
            </div>
            
            <div class="section">
                <h3>💡 Example Questions</h3>
                <div class="example-questions" id="exampleQuestions">
                    <div class="loading">Loading examples...</div>
                </div>
            </div>
            
            <div class="section">
                <h3>🔧 System Status</h3>
                <div id="systemStatus">
                    <div class="loading">Checking status...</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-content">
                            <h4>Welcome to QuantEngine Chat! 🚀</h4>
                            <p>I'm your AI research assistant. I can help you with:</p>
                            <ul>
                                <li>📊 Sector research and analysis</li>
                                <li>📈 Group and portfolio analysis</li>
                                <li>🔍 Stock correlation studies</li>
                                <li>📉 Risk assessment and diversification</li>
                                <li>💡 Trading recommendations</li>
                            </ul>
                            <p>Try asking me about specific sectors or stock groups, or select from the examples on the left!</p>
                        </div>
                        <div class="message-time">{{ current_time }}</div>
                    </div>
                </div>
                
                <div class="loading" id="loadingIndicator">
                    <div class="spinner"></div>
                    <div>Analyzing your question...</div>
                </div>
                
                <div class="input-container">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Ask me about sectors, stock groups, or market analysis..." autocomplete="off">
                    <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedSectors = [];
        let selectedGroups = [];
        let chatHistory = [];
        
        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            loadSectors();
            loadGroups();
            loadExamples();
            checkSystemStatus();
            
            // Add event listeners
            document.getElementById('chatInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        });
        
        // Load available sectors
        async function loadSectors() {
            try {
                const response = await fetch('/api/sectors');
                const data = await response.json();
                
                if (data.success) {
                    const sectorList = document.getElementById('sectorList');
                    sectorList.innerHTML = '';
                    
                    Object.entries(data.sectors).forEach(([key, sector]) => {
                        const item = document.createElement('div');
                        item.className = 'sector-item';
                        item.innerHTML = `
                            <strong>${sector.name}</strong><br>
                            <small>${sector.description}</small>
                        `;
                        item.onclick = () => toggleSector(key, item);
                        sectorList.appendChild(item);
                    });
                } else {
                    document.getElementById('sectorList').innerHTML = '<div class="error-message">Failed to load sectors</div>';
                }
            } catch (error) {
                console.error('Error loading sectors:', error);
                document.getElementById('sectorList').innerHTML = '<div class="error-message">Error loading sectors</div>';
            }
        }
        
        // Load available groups
        async function loadGroups() {
            try {
                const response = await fetch('/api/groups');
                const data = await response.json();
                
                if (data.success) {
                    const groupList = document.getElementById('groupList');
                    groupList.innerHTML = '';
                    
                    Object.entries(data.groups).forEach(([key, group]) => {
                        const item = document.createElement('div');
                        item.className = 'group-item';
                        item.innerHTML = `
                            <strong>${group.name}</strong><br>
                            <small>${group.description}</small>
                        `;
                        item.onclick = () => toggleGroup(key, item);
                        groupList.appendChild(item);
                    });
                } else {
                    document.getElementById('groupList').innerHTML = '<div class="error-message">Failed to load groups</div>';
                }
            } catch (error) {
                console.error('Error loading groups:', error);
                document.getElementById('groupList').innerHTML = '<div class="error-message">Error loading groups</div>';
            }
        }
        
        // Load example questions
        async function loadExamples() {
            try {
                const response = await fetch('/api/examples');
                const data = await response.json();
                
                if (data.success) {
                    const exampleList = document.getElementById('exampleQuestions');
                    exampleList.innerHTML = '';
                    
                    // Combine sector and group examples
                    const allExamples = [...data.examples.sector_research, ...data.examples.group_analysis];
                    
                    allExamples.forEach(example => {
                        const item = document.createElement('div');
                        item.className = 'example-question';
                        item.textContent = example;
                        item.onclick = () => {
                            document.getElementById('chatInput').value = example;
                            sendMessage();
                        };
                        exampleList.appendChild(item);
                    });
                } else {
                    document.getElementById('exampleQuestions').innerHTML = '<div class="error-message">Failed to load examples</div>';
                }
            } catch (error) {
                console.error('Error loading examples:', error);
                document.getElementById('exampleQuestions').innerHTML = '<div class="error-message">Error loading examples</div>';
            }
        }
        
        // Check system status
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                if (data.success) {
                    const status = data.status;
                    const statusDiv = document.getElementById('systemStatus');
                    statusDiv.innerHTML = `
                        <div>
                            <span class="status-indicator ${status.sector_chat_available ? 'status-online' : 'status-offline'}"></span>
                            Sector Research: ${status.sector_chat_available ? 'Online' : 'Offline'}
                        </div>
                        <div>
                            <span class="status-indicator ${status.group_chat_available ? 'status-online' : 'status-offline'}"></span>
                            Group Analysis: ${status.group_chat_available ? 'Online' : 'Offline'}
                        </div>
                        <div>
                            <span class="status-indicator ${status.llm_available ? 'status-online' : 'status-offline'}"></span>
                            LLM: ${status.llm_available ? 'Online' : 'Offline'}
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Error checking status:', error);
                document.getElementById('systemStatus').innerHTML = '<div class="error-message">Status check failed</div>';
            }
        }
        
        // Toggle sector selection
        function toggleSector(sectorKey, element) {
            if (selectedSectors.includes(sectorKey)) {
                selectedSectors = selectedSectors.filter(s => s !== sectorKey);
                element.classList.remove('selected');
            } else {
                selectedSectors.push(sectorKey);
                element.classList.add('selected');
            }
        }
        
        // Toggle group selection
        function toggleGroup(groupKey, element) {
            if (selectedGroups.includes(groupKey)) {
                selectedGroups = selectedGroups.filter(g => g !== groupKey);
                element.classList.remove('selected');
            } else {
                selectedGroups.push(groupKey);
                element.classList.add('selected');
            }
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            
            // Clear input
            input.value = '';
            
            // Show loading
            showLoading(true);
            
            try {
                // Determine which API to use based on selected items or message content
                let apiEndpoint = '/api/sector';
                let requestData = { question: message };
                
                if (selectedGroups.length > 0 || message.toLowerCase().includes('group') || message.toLowerCase().includes('compare')) {
                    apiEndpoint = '/api/group';
                    requestData.groups = selectedGroups;
                }
                
                if (selectedSectors.length > 0) {
                    requestData.sectors = selectedSectors;
                }
                
                const response = await fetch(apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Format and display the analysis
                    displayAnalysis(data.analysis);
                } else {
                    addMessage(`Error: ${data.error}`, 'assistant', true);
                }
                
            } catch (error) {
                console.error('Error sending message:', error);
                addMessage(`Error: ${error.message}`, 'assistant', true);
            } finally {
                showLoading(false);
            }
        }
        
        // Add message to chat
        function addMessage(content, sender, isError = false) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            if (isError) {
                messageContent.innerHTML = `<div class="error-message">${content}</div>`;
            } else {
                messageContent.textContent = content;
            }
            
            const messageTime = document.createElement('div');
            messageTime.className = 'message-time';
            messageTime.textContent = new Date().toLocaleTimeString();
            
            messageDiv.appendChild(messageContent);
            messageDiv.appendChild(messageTime);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Store in history
            chatHistory.push({
                content,
                sender,
                timestamp: new Date().toISOString()
            });
        }
        
        // Display analysis results
        function displayAnalysis(analysis) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            // Format the analysis
            let formattedContent = '<h4>📊 Analysis Results</h4>';
            
            if (analysis.llm_analysis) {
                const llmAnalysis = analysis.llm_analysis;
                
                // Display each section
                const sections = [
                    { title: 'Overview', key: 'group_overview' },
                    { title: 'Technical Analysis', key: 'technical_analysis' },
                    { title: 'Risk Assessment', key: 'risk_assessment' },
                    { title: 'Recommendations', key: 'trading_recommendations' }
                ];
                
                sections.forEach(section => {
                    const content = llmAnalysis[section.key];
                    if (content && content.trim()) {
                        formattedContent += `
                            <div class="analysis-section">
                                <h4>${section.title}</h4>
                                <div class="analysis-content">${content}</div>
                            </div>
                        `;
                    }
                });
                
                // If no structured sections, show raw response
                if (llmAnalysis.raw_response) {
                    formattedContent += `
                        <div class="analysis-section">
                            <h4>Full Analysis</h4>
                            <div class="analysis-content">${llmAnalysis.raw_response}</div>
                        </div>
                    `;
                }
            } else {
                formattedContent += '<p>Analysis completed successfully.</p>';
            }
            
            messageContent.innerHTML = formattedContent;
            
            const messageTime = document.createElement('div');
            messageTime.className = 'message-time';
            messageTime.textContent = new Date().toLocaleTimeString();
            
            messageDiv.appendChild(messageContent);
            messageDiv.appendChild(messageTime);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Show/hide loading indicator
        function showLoading(show) {
            const loading = document.getElementById('loadingIndicator');
            const sendButton = document.getElementById('sendButton');
            
            if (show) {
                loading.classList.add('show');
                sendButton.disabled = true;
            } else {
                loading.classList.remove('show');
                sendButton.disabled = false;
            }
        }
    </script>
</body>
</html>
    """)

# API proxy endpoints
@app.route('/api/sectors')
def api_sectors():
    """Proxy to get sectors"""
    try:
        response = requests.get(f"{CHAT_API_BASE}/chat/sectors")
        return response.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/groups')
def api_groups():
    """Proxy to get groups"""
    try:
        response = requests.get(f"{CHAT_API_BASE}/chat/groups")
        return response.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/examples')
def api_examples():
    """Proxy to get examples"""
    try:
        response = requests.get(f"{CHAT_API_BASE}/chat/example")
        return response.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/status')
def api_status():
    """Proxy to get status"""
    try:
        response = requests.get(f"{CHAT_API_BASE}/chat/status")
        return response.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sector', methods=['POST'])
def api_sector():
    """Proxy to sector research"""
    try:
        data = request.get_json()
        response = requests.post(f"{CHAT_API_BASE}/chat/sector", json=data)
        return response.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/group', methods=['POST'])
def api_group():
    """Proxy to group analysis"""
    try:
        data = request.get_json()
        response = requests.post(f"{CHAT_API_BASE}/chat/group", json=data)
        return response.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    print("🌐 Starting QuantEngine Chat Web Interface...")
    print("=" * 60)
    print("Web Interface: http://localhost:5002")
    print("Chat API: http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5002, debug=True)
