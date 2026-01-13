#!/bin/bash

# 指定使用的物理显卡 ID
GPU_IDS="0,1,2,3,4,5,6,7"
NUM_GPUS=$(echo $GPU_IDS | tr -cd ',' | wc -c)
NUM_GPUS=$((NUM_GPUS + 1))

# DDP 通信端口 (如果多任务同时运行，需修改此端口以防冲突)
MASTER_PORT=29500

# 限制 CPU 线程数 (防止数据加载时 CPU 争抢导致训练卡死)
export OMP_NUM_THREADS=4

# 训练结果、模型、日志保存目录
CHECKPOINT_DIR="/Netdata/2025/wjc/checkpoints_kuochong_ddp_final"


# 单张显卡的 Batch Size 
# 全局 Batch Size = BATCH_SIZE_PER_GPU * NUM_GPUS
BATCH_SIZE_PER_GPU=16
# 总训练轮次
TOTAL_EPOCHS=30

# LMFT
# 前 N 轮使用 Softmax (Margin=0)
WARMUP_EPOCHS=5
# 后 N 轮使用完整 Margin (Margin=0.5)
FINETUNE_EPOCHS=8

# 是否启用语速音高扩增 (数据量 x3)
# true: 开启 (推荐，提升鲁棒性)
# false: 关闭 (仅使用原始数据，速度快)
ENABLE_SPEED_PERTURB=true

# 是否启用环境噪声/混响增强
# true: 开启 (推荐)
# false: 关闭
ENABLE_ENV_AUGMENTATION=true


# 自动恢复: true/false
# 如果为 true，脚本会自动去 CHECKPOINT_DIR 找最新的 .pth 继续训练
AUTO_RESUME=false

# 指定恢复路径 (仅当 AUTO_RESUME=false 时生效)
# 如果留空 ""，且 AUTO_RESUME=false，则从头开始训练 (Epoch 1)
SPECIFIC_RESUME_PATH=""
# 示例: SPECIFIC_RESUME_PATH="/path/to/best_model_epoch_10.pth"



# 1. 设置可见显卡
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 2. 构建 Python 运行参数
PY_ARGS="--checkpoint_dir ${CHECKPOINT_DIR} \
         --batch_size ${BATCH_SIZE_PER_GPU} \
         --epochs ${TOTAL_EPOCHS} \
         --warmup_epochs ${WARMUP_EPOCHS} \
         --fine_tune_epochs ${FINETUNE_EPOCHS}"

# 处理开关逻辑
if [ "$ENABLE_SPEED_PERTURB" = true ]; then
    PY_ARGS="${PY_ARGS} --speed_perturb"
fi

if [ "$ENABLE_ENV_AUGMENTATION" = false ]; then
    # 注意：Python脚本中是 --disable_aug，所以这里反向逻辑
    PY_ARGS="${PY_ARGS} --disable_aug"
fi

if [ "$AUTO_RESUME" = true ]; then
    PY_ARGS="${PY_ARGS} --auto_resume"
elif [ -n "$SPECIFIC_RESUME_PATH" ]; then
    PY_ARGS="${PY_ARGS} --resume ${SPECIFIC_RESUME_PATH}"
fi

# 3. 打印运行信息
echo "======================================================="
echo "   🚀 启动 DDP 分布式训练"
echo "======================================================="
echo "   - 显卡列表    : $GPU_IDS (共 $NUM_GPUS 张)"
echo "   - 保存目录    : $CHECKPOINT_DIR"
echo "   - 全局 Batch  : $((BATCH_SIZE_PER_GPU * NUM_GPUS)) (单卡: $BATCH_SIZE_PER_GPU)"
echo "   - 总轮次      : $TOTAL_EPOCHS (Warmup: $WARMUP_EPOCHS, Finetune: $FINETUNE_EPOCHS)"
echo "   - 语速扩增    : $ENABLE_SPEED_PERTURB"
echo "   - 环境增强    : $ENABLE_ENV_AUGMENTATION"
if [ "$AUTO_RESUME" = true ]; then
    echo "   - 断点策略    : 自动恢复 (Auto Resume)"
elif [ -n "$SPECIFIC_RESUME_PATH" ]; then
    echo "   - 断点策略    : 指定文件 -> $SPECIFIC_RESUME_PATH"
else
    echo "   - 断点策略    : ⚠️ 从头开始训练 (Fresh Start)"
fi
echo "======================================================="

# 4. 执行指令
# 使用 torchrun 启动
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_ddp.py $PY_ARGS