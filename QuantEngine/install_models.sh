#!/bin/bash

# Install Latest 2025/2026 Models for Financial Analysis
# Run this after Ollama is installed

echo "🤖 Installing Latest 2025/2026 Models for Financial Analysis..."
echo "============================================================="

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "❌ Ollama is not running. Please start it first:"
    echo "   ollama serve &"
    exit 1
fi

echo "✅ Ollama is running"

# Install RTX 3090 24GB VRAM optimized models (SMALLER MODELS)
echo "📥 Installing Mistral 8x7B MoE (Primary - Real-time Analysis - 18-20GB)..."
ollama pull mistral:8x7b

echo "📥 Installing DeepSeek-R1 32B (Advanced Reasoning - 16-18GB)..."
ollama pull deepseek-r1:32b

echo "📥 Installing CodeLlama 34B (Code Generation - 18-20GB)..."
ollama pull codellama:34b

echo "📥 Installing Gemma 2 27B (Research - Google's Large Model - 14-16GB)..."
ollama pull gemma2:27b

echo "📥 Installing Llama 3.1 8B (Fast Analysis - 6-8GB)..."
ollama pull llama3.1:8b

# Verify installation
echo "🔍 Verifying model installation..."
ollama list

echo ""
echo "✅ Model Installation Complete!"
echo "==============================="
echo ""
echo "📊 Available Models (RTX 3090 24GB VRAM - SMALLER MODELS):"
echo "  🥇 mistral:8x7b       - Primary (Real-time Analysis - 18-20GB)"
echo "  🥈 deepseek-r1:32b    - Advanced Reasoning (16-18GB)"
echo "  🥉 codellama:34b      - Code Generation (18-20GB)"
echo "  🔬 gemma2:27b         - Research (Google's Large Model - 14-16GB)"
echo "  🚀 llama3.1:8b        - Fast Analysis (6-8GB)"
echo ""
echo "🎯 Recommended for RTX 3090 24GB VRAM Financial Analysis:"
echo "  - mistral:8x7b (Primary - Real-time decisions)"
echo "  - deepseek-r1:32b (Advanced reasoning tasks)"
echo "  - gemma2:27b (Research and analysis)"
echo ""
echo "🚀 Next Steps:"
echo "1. Test a model: ollama run mistral:8x7b 'Analyze AAPL stock'"
echo "2. Start the scanner: ./start_scanner_gpu.sh"
echo "3. Check the dashboard: ./start_dashboard.sh"
echo ""
echo "💡 RTX 3090 24GB Model Selection Tips:"
echo "  - Use mistral:8x7b for real-time market analysis (fast, good quality)"
echo "  - Use deepseek-r1:32b for complex financial modeling"
echo "  - Use gemma2:27b for research reports and detailed analysis"
echo "  - Use codellama:34b for strategy code generation"
echo "  - Use llama3.1:8b for quick analysis and testing"
