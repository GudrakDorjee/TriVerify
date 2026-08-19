#!/bin/bash
# Step 2.3: LoRA 微调脚本
set -e
export OMP_NUM_THREADS=1

cd /root/autodl-tmp/LlamaFactory-main
source venv/bin/activate

echo "============================================"
echo "Step 2.3: LoRA 微调"
echo "============================================"

CONFIG=${1:-"qwen_lora"}
CONFIG_PATH="baseline/lora_finetune/${CONFIG}.yaml"

echo "配置: ${CONFIG_PATH}"
echo ""

# 使用 llamafactory-cli 或直接调用
if command -v llamafactory-cli &> /dev/null; then
    llamafactory-cli train ${CONFIG_PATH}
else
    python3 -m llamafactory.train ${CONFIG_PATH}
fi

echo ""
echo "✅ 微调完成: ${CONFIG}"
