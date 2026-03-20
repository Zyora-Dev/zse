#!/bin/bash
#
# ZSE + OpenClaw Setup Script
# Run any local model with OpenClaw using ZSE inference engine
#
# Usage: curl -fsSL https://zllm.in/openclaw.sh | bash
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}║       ${GREEN}ZSE + OpenClaw Integration Setup${BLUE}                   ║${NC}"
echo -e "${BLUE}║       Run any local model with OpenClaw                   ║${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running on supported OS
OS="$(uname -s)"
case "$OS" in
    Linux*)     OS_TYPE=linux;;
    Darwin*)    OS_TYPE=macos;;
    *)          echo -e "${RED}Unsupported OS: $OS${NC}"; exit 1;;
esac

echo -e "${YELLOW}▸ Detected OS: $OS_TYPE${NC}"

# Check for NVIDIA GPU
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
            GPU_VRAM=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs | sed 's/ MiB//')
            GPU_VRAM_GB=$((GPU_VRAM / 1024))
            echo -e "${GREEN}▸ GPU detected: $GPU_NAME ($GPU_VRAM_GB GB VRAM)${NC}"
            return 0
        fi
    fi
    echo -e "${YELLOW}▸ No NVIDIA GPU detected. ZSE will run in CPU mode (slower).${NC}"
    GPU_VRAM_GB=0
    return 1
}

# Recommend model based on VRAM
recommend_model() {
    if [ "$GPU_VRAM_GB" -ge 40 ]; then
        RECOMMENDED_MODEL="Qwen/Qwen2.5-72B-Instruct"
        RECOMMENDED_DESC="72B model - Best quality"
    elif [ "$GPU_VRAM_GB" -ge 24 ]; then
        RECOMMENDED_MODEL="Qwen/Qwen2.5-32B-Instruct"
        RECOMMENDED_DESC="32B model - Great for complex tasks"
    elif [ "$GPU_VRAM_GB" -ge 12 ]; then
        RECOMMENDED_MODEL="Qwen/Qwen2.5-14B-Instruct"
        RECOMMENDED_DESC="14B model - Good balance"
    elif [ "$GPU_VRAM_GB" -ge 8 ]; then
        RECOMMENDED_MODEL="Qwen/Qwen2.5-7B-Instruct"
        RECOMMENDED_DESC="7B model - Fits 8GB GPUs"
    else
        RECOMMENDED_MODEL="Qwen/Qwen2.5-3B-Instruct"
        RECOMMENDED_DESC="3B model - Minimal resources"
    fi
    echo -e "${BLUE}▸ Recommended model: $RECOMMENDED_MODEL${NC}"
    echo -e "  ($RECOMMENDED_DESC)"
}

# Check if ZSE is installed
check_zse() {
    if command -v zse &> /dev/null; then
        ZSE_VERSION=$(zse --version 2>/dev/null | head -1 || echo "unknown")
        echo -e "${GREEN}▸ ZSE already installed: $ZSE_VERSION${NC}"
        return 0
    fi
    return 1
}

# Install ZSE
install_zse() {
    echo ""
    echo -e "${YELLOW}▸ Installing ZSE...${NC}"
    
    if [ "$GPU_VRAM_GB" -gt 0 ]; then
        pip install zllm-zse[cuda] --quiet
    else
        pip install zllm-zse --quiet
    fi
    
    if command -v zse &> /dev/null; then
        echo -e "${GREEN}▸ ZSE installed successfully!${NC}"
    else
        echo -e "${RED}▸ Failed to install ZSE. Please run: pip install zllm-zse[cuda]${NC}"
        exit 1
    fi
}

# Check Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo -e "${GREEN}▸ Python: $PYTHON_VERSION${NC}"
    else
        echo -e "${RED}▸ Python 3 not found. Please install Python 3.11+${NC}"
        exit 1
    fi
}

# Main setup
main() {
    echo ""
    echo -e "${YELLOW}Checking system requirements...${NC}"
    echo ""
    
    check_python
    check_gpu
    recommend_model
    
    echo ""
    
    if ! check_zse; then
        read -p "Install ZSE? (Y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
            install_zse
        else
            echo -e "${YELLOW}Skipping ZSE installation.${NC}"
        fi
    fi
    
    echo ""
    echo -e "${YELLOW}Choose your model:${NC}"
    echo ""
    echo "  1) $RECOMMENDED_MODEL (recommended for your GPU)"
    echo "  2) Qwen/Qwen2.5-7B-Instruct (7B - fast, fits 8GB)"
    echo "  3) Qwen/Qwen2.5-32B-Instruct (32B - powerful, needs 24GB)"
    echo "  4) meta-llama/Llama-3.1-8B-Instruct (Llama 3.1 8B)"
    echo "  5) Enter custom model name"
    echo ""
    read -p "Select [1-5]: " MODEL_CHOICE
    
    case $MODEL_CHOICE in
        1) MODEL="$RECOMMENDED_MODEL";;
        2) MODEL="Qwen/Qwen2.5-7B-Instruct";;
        3) MODEL="Qwen/Qwen2.5-32B-Instruct";;
        4) MODEL="meta-llama/Llama-3.1-8B-Instruct";;
        5) 
            read -p "Enter model name (HuggingFace format): " MODEL
            ;;
        *) MODEL="$RECOMMENDED_MODEL";;
    esac
    
    echo ""
    echo -e "${GREEN}▸ Selected model: $MODEL${NC}"
    echo ""
    
    # Start ZSE
    echo -e "${YELLOW}Starting ZSE server...${NC}"
    echo ""
    echo -e "Running: ${BLUE}zse serve $MODEL --port 8000${NC}"
    echo ""
    echo -e "${YELLOW}First run will download the model (this may take a while)${NC}"
    echo ""
    
    echo "──────────────────────────────────────────────────────────────"
    echo ""
    echo -e "${GREEN}OpenClaw Configuration:${NC}"
    echo ""
    echo "Add to your OpenClaw config.yaml:"
    echo ""
    echo -e "${BLUE}llm:"
    echo "  provider: openai-compatible"
    echo "  api_base: http://localhost:8000/v1"
    echo "  api_key: zse"
    echo -e "  model: default${NC}"
    echo ""
    echo "Or set environment variables:"
    echo ""
    echo -e "${BLUE}export OPENAI_API_BASE=http://localhost:8000/v1"
    echo -e "export OPENAI_API_KEY=zse${NC}"
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    
    # Run ZSE
    zse serve "$MODEL" --port 8000
}

main "$@"
