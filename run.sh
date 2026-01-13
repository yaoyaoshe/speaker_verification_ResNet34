#!/bin/bash

# =========================================================
# 说话人识别 DDP 训练启动脚本
# =========================================================

# 1. 显卡设置
# ---------------------------------------------------------
# 指定要使用的 GPU ID，例如 "0,1,2,3" 或 "0,1,2,3,4,5,6,7"
GPU_IDS="0,1,2,3,4,5,6,7"

# 自动计算 GPU 数量 (用于 torchrun --nproc_per_node)
NUM_GPUS=$(echo $GPU_IDS | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# 2. 路径配置
# ---------------------------------------------------------
# 检查点和日志保存目录
CHECKPOINT_DIR="/Netdata/2025/wjc/checkpoints_kuochong_duizhao"

# 3. 训练超参数
# ---------------------------------------------------------
# 单张显卡的 Batch Size (RTX 3090 建议 16~32，如果显存不够改回 8)
BATCH_SIZE=8

# 总轮次
TOTAL_EPOCHS=40

# 课程学习阶段配置
WARMUP_EPOCHS=5     # Margin=0 的轮数
FINETUNE_EPOCHS=15   # Margin=0.5 的轮数
# 中间的轮数 = TOTAL - WARMUP - FINETUNE (Margin 线性增加)

# 4. 功能开关
# ---------------------------------------------------------
# 是否开启语速音高扩增 (true/false)
ENABLE_SPEED_PERTURB=false

# 是否自动从最新断点恢复 (true/false)
AUTO_RESUME=false

# =========================================================
# 脚本逻辑 (通常不需要修改以下内容)
# =========================================================

# 构建 Python 参数字符串
PY_ARGS="--checkpoint_dir ${CHECKPOINT_DIR} \
         --batch_size ${BATCH_SIZE} \
         --epochs ${TOTAL_EPOCHS} \
         --warmup_epochs ${WARMUP_EPOCHS} \
         --fine_tune_epochs ${FINETUNE_EPOCHS}"

# 根据开关添加参数
if [ "$ENABLE_SPEED_PERTURB" = true ]; then
    PY_ARGS="${PY_ARGS} --speed_perturb"
fi

if [ "$AUTO_RESUME" = true ]; then
    PY_ARGS="${PY_ARGS} --auto_resume"
fi

# 打印运行信息
echo "======================================================="
echo "正在启动 DDP 训练..."
echo "使用显卡 ID: $GPU_IDS (共 $NUM_GPUS 张)"
echo "保存目录   : $CHECKPOINT_DIR"
echo "Batch Size : $BATCH_SIZE (单卡) / $((BATCH_SIZE * NUM_GPUS)) (全局)"
echo "扩增状态   : $ENABLE_SPEED_PERTURB"
echo "断点恢复   : $AUTO_RESUME"
echo "======================================================="

# 设置可见显卡环境变量
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 启动训练
# OMP_NUM_THREADS=4 防止 CPU 争抢导致死锁
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    /Netdata/2025/wjc/code/kuozeng/train_ddp.py $PY_ARGS