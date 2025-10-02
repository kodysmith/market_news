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

# Install RTX 3090 24GB VRAM optimized models
echo "📥 Installing Qwen 2.5 72B (Primary - Best Overall for RTX 3090)..."
ollama pull qwen2.5:72b

echo "📥 Installing Llama 3.1 70B (Deep Analysis - 405B parameters)..."
ollama pull llama3.1:70b

echo "📥 Installing Mistral 8x7B MoE (Real-time Analysis - Fast)..."
ollama pull mistral:8x7b

echo "📥 Installing CodeLlama 34B (Code Generation - Large)..."
ollama pull codellama:34b

echo "📥 Installing Gemma 2 27B (Research - Google's Large Model)..."
ollama pull gemma2:27b

echo "📥 Installing DeepSeek-R1 32B (Reasoning - Advanced)..."
ollama pull deepseek-r1:32b

# Verify installation
echo "🔍 Verifying model installation..."
ollama list

echo ""
echo "✅ Model Installation Complete!"
echo "==============================="
echo ""
echo "📊 Available Models (RTX 3090 24GB VRAM):"
echo "  🥇 qwen2.5:72b       - Primary (Best Overall - 72B parameters)"
echo "  🥈 llama3.1:70b      - Deep Analysis (405B parameters)"
echo "  🥉 mistral:8x7b      - Real-time (MoE Architecture)"
echo "  💻 codellama:34b     - Code Generation (Large)"
echo "  🔬 gemma2:27b        - Research (Google's Large Model)"
echo "  🧠 deepseek-r1:32b   - Reasoning (Advanced)"
echo ""
echo "🎯 Recommended for RTX 3090 Financial Analysis:"
echo "  - qwen2.5:72b (Primary - Comprehensive Analysis)"
echo "  - llama3.1:70b (Deep Research - 405B parameters)"
echo "  - mistral:8x7b (Real-time - Fast decisions)"
echo ""
echo "🚀 Next Steps:"
echo "1. Test a model: ollama run qwen2.5:72b 'Analyze AAPL stock'"
echo "2. Start the scanner: ./start_scanner_gpu.sh"
echo "3. Check the dashboard: ./start_dashboard.sh"
echo ""
echo "💡 RTX 3090 Model Selection Tips:"
echo "  - Use qwen2.5:72b for comprehensive market analysis"
echo "  - Use llama3.1:70b for deep research reports (405B parameters)"
echo "  - Use mistral:8x7b for real-time analysis (MoE architecture)"
echo "  - Use codellama:34b for strategy code generation"
echo "  - Use deepseek-r1:32b for advanced reasoning tasks"
