import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv
import argparse
import glob
import re
from tqdm import tqdm

# === 分布式训练依赖 ===
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# === 导入你的模块 ===
from module import test_module1
from dataloader import Vox1Dataset 
from EER import calculate_eer_parallel 

plt.rcParams['axes.unicode_minus'] = False 

# === 提取并重写 collate_fn (用于处理变长音频 Padding) ===
def collate_fn_pad(batch):
    waveforms, spk_ids, speed_types = zip(*batch)
    
    max_len = max(wav.shape[0] for wav in waveforms)
    padded_wavs = []
    
    for wav in waveforms:
        pad_len = max_len - wav.shape[0]
        if pad_len > 0:
            padded = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0.0)
        else:
            padded = wav
        padded_wavs.append(padded)

    batch_wavs = torch.stack(padded_wavs) # (B, T)
    batch_spk_ids = torch.tensor(spk_ids, dtype=torch.long)
    batch_speed_types = torch.tensor(speed_types, dtype=torch.long) 
    
    return batch_wavs, batch_spk_ids, batch_speed_types

def get_latest_checkpoint(checkpoint_dir):
    """自动查找目录下 epoch 最大的模型文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        return None
    
    latest_file = None
    max_epoch = -1
    
    for f in files:
        match = re.search(r'epoch_(\d+)', f)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_file = f
    return latest_file

def main():
    # 1. 参数解析
    # -----------------------------------------------------------
    parser = argparse.ArgumentParser(description="DDP Speaker Verification Training")
    
    # 基础配置
    parser.add_argument('--checkpoint_dir', type=str, default="/Netdata/2025/wjc/checkpoints_kuochong_ddp", 
                        help='模型和日志的保存目录')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='单张显卡的 Batch Size (全局 Batch = 这个值 * 显卡数)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='总训练轮次 (Total Epochs)')
    
    # 数据增强配置
    parser.add_argument('--speed_perturb', action='store_true', 
                        help='是否启用语速音高扩增 (3倍数据量)。不加此参数则为 False')
    
    # 课程学习阶段配置
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Softmax 预训练轮数 (Margin=0, EasyMargin=True)')
    parser.add_argument('--fine_tune_epochs', type=int, default=8, 
                        help='完整 Margin 微调轮数 (Margin=0.5, 训练结束前的最后 n 轮)')
    
    # Resume 配置
    parser.add_argument('--resume', type=str, default=None, help='指定要恢复的检查点路径 (.pth)')
    parser.add_argument('--auto_resume', action='store_true', help='自动从目录恢复最新模型')

    args = parser.parse_args()

    # 2. 初始化分布式环境
    # -----------------------------------------------------------
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        print("错误：请使用 torchrun 启动此脚本。")
        return

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 3. 路径配置
    # -----------------------------------------------------------
    train_scp = "/Netdata/2025/wjc/data/train.scp"
    trials_path = "/Netdata/2025/wjc/data/trials"
    audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
    
    noise_scp = "/Netdata/2025/wjc/data/musan_noise.scp"
    speech_scp = "/Netdata/2025/wjc/data/musan_speech.scp"
    music_scp = "/Netdata/2025/wjc/data/musan_music.scp"
    rir_scp = "/Netdata/2025/wjc/data/rir.scp" 
    
    accumulation_steps = 4 

    lr = 0.1 
    
    # 确定 Resume 路径
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        found_path = get_latest_checkpoint(args.checkpoint_dir)
        if found_path:
            resume_path = found_path
            if rank == 0:
                print(f"--> [Auto Resume] 发现最新检查点: {resume_path}")

    # 仅 Rank 0 负责打印和创建目录
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log_csv_path = os.path.join(args.checkpoint_dir, 'training_log.csv')
        if not os.path.exists(log_csv_path):
            with open(log_csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Phase', 'Loss', 'EER', 'LR', 'Time(s)'])
        
        print("="*50)
        print(f"开始 DDP 训练 | GPU数量: {world_size}")
        print(f"配置信息:")
        print(f"  - Checkpoint Dir : {args.checkpoint_dir}")
        print(f"  - Batch Size     : {args.batch_size} (Global: {args.batch_size * world_size})")
        print(f"  - Speed Perturb  : {args.speed_perturb}")
        print(f"  - Total Epochs   : {args.epochs}")
        print(f"  - Warmup Rounds  : {args.warmup_epochs} (Margin=0)")
        print(f"  - Final Rounds   : {args.fine_tune_epochs} (Margin=0.5)")
        if resume_path:
            print(f"  - Resume Path    : {resume_path}")
        print("="*50)

    # 4. 数据集与 DataLoader
    # -----------------------------------------------------------
    train_dataset = Vox1Dataset(
        scp_path=train_scp,
        noise_scp=noise_scp,
        speech_scp=speech_scp,
        music_scp=music_scp,
        rir_scp=rir_scp, 
        n_min=300, 
        n_max=800,
        speed_perturb=args.speed_perturb # [使用参数]
    )
    
    num_spk = train_dataset.num_spk
    if rank == 0:
        print(f"数据集分类数量：{num_spk}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_pad
    )

    # 5. 模型初始化
    # -----------------------------------------------------------
    model = test_module1(num_classes=num_spk) 
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. 加载 Checkpoint
    # -----------------------------------------------------------
    start_epoch = 0
    best_eer = float('inf') 
    train_losses = []
    eer_scores = []
    
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"--> 正在加载权重: {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # 兼容 DDP module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                new_state_dict['module.' + k] = v
            else:
                new_state_dict[k] = v
        
        try:
            model.load_state_dict(new_state_dict)
        except RuntimeError:
            model.load_state_dict(state_dict)

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1 
        if 'eer' in checkpoint:
            best_eer = checkpoint['eer']
            
        for _ in range(start_epoch):
            scheduler.step()
        
        if rank == 0:
            print(f"--> 恢复成功! 下一轮 Epoch: {start_epoch+1}, Best EER: {best_eer}")

    # 7. 训练循环
    # -----------------------------------------------------------
    # 使用命令行参数控制阶段
    warmup_epochs = args.warmup_epochs
    fine_tune_epochs = args.fine_tune_epochs
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        
        # 动态课程学习逻辑
        # 阶段 1: Softmax Pre-train
        if epoch < warmup_epochs:
            current_m = 0.0
            train_dataset.n_min = 200
            train_dataset.n_max = 400
            phase_name = "Softmax Pre-train"
            easy_margin = True
            
        # 阶段 2: Ramping up (中间过渡期)
        elif epoch >= warmup_epochs and epoch < (total_epochs - fine_tune_epochs):
            # 计算过渡期的进度
            ramp_duration = (total_epochs - fine_tune_epochs) - warmup_epochs
            if ramp_duration > 0:
                progress = (epoch - warmup_epochs) / ramp_duration
            else:
                progress = 1.0
            
            current_m = 0.2 + 0.3 * progress
            train_dataset.n_min = 300
            train_dataset.n_max = 500
            phase_name = "Margin Ramping"
            easy_margin = False
            
        # 阶段 3: Complete ArcFace (最后 N 轮)
        else:
            current_m = 0.5
            train_dataset.n_min = 600
            train_dataset.n_max = 800
            phase_name = "Complete ArcFace"
            easy_margin = False
        
        if rank == 0:
            print(f"Epoch {epoch+1} Config: n_min={train_dataset.n_min}, n_max={train_dataset.n_max}, m={current_m:.2f}")

        model.module.arcface.m = current_m
        model.module.arcface.easy_margin = easy_margin

        model.train()
        total_loss = torch.zeros(1).to(device)
        start_time = time.time()
        
        optimizer.zero_grad() 
        
        if rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [{phase_name}]')
        else:
            pbar = train_loader
        
        for batch_idx, (waveforms, spk_ids, speed_ids) in enumerate(pbar):
            waveforms = waveforms.to(device, non_blocking=True)
            spk_ids = spk_ids.to(device, non_blocking=True)
            speed_ids = speed_ids.to(device, non_blocking=True)
 
            outputs = model(waveforms, spk_ids, speed_ids=speed_ids)
            loss = criterion(outputs, spk_ids)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.detach() * accumulation_steps
            
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f"{loss.item()*accumulation_steps:.4f}", 'm': f"{current_m:.2f}"})
        
        if (len(train_loader) % accumulation_steps) != 0:
             optimizer.step()
             optimizer.zero_grad()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = total_loss.item() / len(train_loader) / world_size
        
        if rank == 0:
            elapsed = time.time() - start_time
            temp_model_path = os.path.join(args.checkpoint_dir, f'temp_model_epoch_{epoch+1}.pth')
            
            state_dict_to_save = model.module.state_dict()
            save_dict = {
                'epoch': epoch,
                'model_state_dict': state_dict_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss
            }
            torch.save(save_dict, temp_model_path)
            
            # 释放梯度显存
            print("正在释放梯度显存...")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            print("正在计算 EER...")
            try:
                current_eer = calculate_eer_parallel(
                    model_path=temp_model_path,
                    trials_path=trials_path,
                    audio_dir=audio_dir,
                    device_ids=[local_rank] 
                )
            except Exception as e:
                print(f"EER 计算警告: {e}")
                current_eer = 100.0
            
            eer_scores.append(current_eer)
            save_dict['eer'] = current_eer
            train_losses.append(avg_train_loss)
            
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            print(f'Epoch [{epoch+1}/{total_epochs}], Phase: {phase_name}, Loss: {avg_train_loss:.4f}, EER: {current_eer:.4f}')
            
            with open(log_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, phase_name, f"{avg_train_loss:.4f}", f"{current_eer:.4f}", f"{current_lr:.6f}", f"{elapsed:.2f}"])
            
            if current_eer < best_eer:
                best_eer = current_eer
                save_path = os.path.join(args.checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
                torch.save(save_dict, save_path) 
                print(f"保存最佳模型: {save_path}")

        # [重要] 同步所有进程，防止 Rank 0 计算 EER 时其他进程超时或跑飞
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchaudio
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import time
# import csv
# import argparse
# import glob
# import re
# from tqdm import tqdm

# # === 分布式训练依赖 ===
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader

# # === 导入你的模块 ===
# from module import test_module1
# from dataloader import Vox1Dataset 
# from EER import calculate_eer_parallel 

# plt.rcParams['axes.unicode_minus'] = False 

# # === 提取并重写 collate_fn (用于处理变长音频 Padding) ===
# def collate_fn_pad(batch):
#     waveforms, spk_ids, speed_types = zip(*batch)
    
#     max_len = max(wav.shape[0] for wav in waveforms)
#     padded_wavs = []
    
#     for wav in waveforms:
#         pad_len = max_len - wav.shape[0]
#         if pad_len > 0:
#             padded = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0.0)
#         else:
#             padded = wav
#         padded_wavs.append(padded)

#     batch_wavs = torch.stack(padded_wavs) # (B, T)
#     batch_spk_ids = torch.tensor(spk_ids, dtype=torch.long)
#     batch_speed_types = torch.tensor(speed_types, dtype=torch.long) 
    
#     return batch_wavs, batch_spk_ids, batch_speed_types

# def get_latest_checkpoint(checkpoint_dir):
#     """自动查找目录下 epoch 最大的模型文件"""
#     # 匹配文件名格式，例如 "temp_model_epoch_5.pth" 或 "best_model_epoch_10.pth"
#     # 这里优先找 temp_model，因为它通常代表最新的训练状态
#     files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
#     if not files:
#         return None
    
#     # 筛选出包含 epoch 的文件并排序
#     latest_file = None
#     max_epoch = -1
    
#     for f in files:
#         # 使用正则提取 epoch 数字
#         match = re.search(r'epoch_(\d+)', f)
#         if match:
#             epoch_num = int(match.group(1))
#             if epoch_num > max_epoch:
#                 max_epoch = epoch_num
#                 latest_file = f
    
#     return latest_file

# def main():
#     # 1. 参数解析
#     # -----------------------------------------------------------
#     parser = argparse.ArgumentParser(description="DDP Speaker Verification Training")
#     parser.add_argument('--resume', type=str, default=None, help='指定要恢复的检查点路径 (.pth)')
#     parser.add_argument('--auto_resume', action='store_true', help='是否自动从检查点目录中恢复最新的模型')
#     parser.add_argument('--batch_size', type=int, default=8, help='单卡 Batch Size (每张显卡处理的数据量)')
#     parser.add_argument('--epochs', type=int, default=20, help='总 Epoch 数量')
#     parser.add_argument('--checkpoint_dir', type=str, default="/Netdata/2025/wjc/checkpoints_kuochong_ddp", help='检查点保存目录')
    
#     # 解析参数（注意：DDP 模式下所有进程都会运行这段代码）
#     args = parser.parse_args()

#     # 2. 初始化分布式环境
#     # -----------------------------------------------------------
#     try:
#         local_rank = int(os.environ["LOCAL_RANK"])
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#     except KeyError:
#         print("错误：请使用 torchrun 启动此脚本。")
#         return

#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(local_rank)
#     device = torch.device(f"cuda:{local_rank}")

#     # 3. 基础配置
#     # -----------------------------------------------------------
#     train_scp = "/Netdata/2025/wjc/data/train.scp"
#     trials_path = "/Netdata/2025/wjc/data/trials"
#     audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
#     checkpoint_dir = args.checkpoint_dir
    
#     noise_scp = "/Netdata/2025/wjc/data/musan_noise.scp"
#     speech_scp = "/Netdata/2025/wjc/data/musan_speech.scp"
#     music_scp = "/Netdata/2025/wjc/data/musan_music.scp"
#     rir_scp = "/Netdata/2025/wjc/data/rir.scp" 
    
#     num_epochs = args.epochs
#     batch_size_per_gpu = args.batch_size
#     accumulation_steps = 16 
#     lr = 0.1 
    
#     # 确定 Resume 路径
#     resume_path = args.resume
#     if args.auto_resume and resume_path is None:
#         found_path = get_latest_checkpoint(checkpoint_dir)
#         if found_path:
#             resume_path = found_path
#             if rank == 0:
#                 print(f"--> [Auto Resume] 发现最新检查点: {resume_path}")

#     # 仅 Rank 0 负责打印和创建目录
#     if rank == 0:
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         log_csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
#         if not os.path.exists(log_csv_path):
#             with open(log_csv_path, mode='w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['Epoch', 'Phase', 'Loss', 'EER', 'LR', 'Time(s)'])
#         print("="*50)
#         print(f"开始 DDP 训练 | World Size (GPU数量): {world_size}")
#         print(f"Per-GPU Batch: {batch_size_per_gpu} | Global Batch: {batch_size_per_gpu * world_size}")
#         if resume_path:
#             print(f"恢复训练模式: {resume_path}")
#         else:
#             print("从头训练模式")
#         print("="*50)

#     # 4. 数据集与 DataLoader
#     # -----------------------------------------------------------
#     train_dataset = Vox1Dataset(
#         scp_path=train_scp,
#         noise_scp=noise_scp,
#         speech_scp=speech_scp,
#         music_scp=music_scp,
#         rir_scp=rir_scp, 
#         n_min=300, 
#         n_max=800
#     )
    
#     num_spk = train_dataset.num_spk
#     if rank == 0:
#         print(f"数据集分类数量（含扩充）：{num_spk}")

#     train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
#     train_loader = DataLoader(
#         dataset=train_dataset,
#         batch_size=batch_size_per_gpu,
#         sampler=train_sampler,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=collate_fn_pad
#     )

#     # 5. 模型初始化
#     # -----------------------------------------------------------
#     model = test_module1(num_classes=num_spk) 
#     model = model.to(device)
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank)

#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

#     # 6. 加载 Checkpoint (Resume 核心逻辑)
#     # -----------------------------------------------------------
#     start_epoch = 0
#     best_eer = float('inf') 
#     train_losses = []
#     eer_scores = []
    
#     if resume_path and os.path.exists(resume_path):
#         if rank == 0:
#             print(f"--> 正在加载权重: {resume_path}")
        
#         # 确保 map_location 正确映射到当前 GPU
#         checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
#         # 加载模型参数
#         # 如果保存时是 model.module.state_dict()，则 key 不带 module.
#         # 如果 DDP 加载需要 module. 前缀，这里做个兼容处理
#         state_dict = checkpoint['model_state_dict']
        
#         # 检查当前模型是否需要 'module.' 前缀 (DDP wrap 后通常需要)
#         # 简单策略：直接加载，如果 key 不匹配（比如保存时没带 module. 但现在模型有），手动加前缀
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if not k.startswith('module.'):
#                 new_state_dict['module.' + k] = v
#             else:
#                 new_state_dict[k] = v
        
#         try:
#             model.load_state_dict(new_state_dict)
#         except RuntimeError as e:
#             # 如果加载失败，尝试原始 dict (万一保存时已经有了)
#             model.load_state_dict(state_dict)

#         if 'optimizer_state_dict' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
#         # 恢复 Epoch
#         start_epoch = checkpoint['epoch'] + 1 
#         if 'eer' in checkpoint:
#             best_eer = checkpoint['eer']
            
#         # 恢复 Scheduler 状态
#         for _ in range(start_epoch):
#             scheduler.step()
        
#         if rank == 0:
#             print(f"--> 恢复成功! 下一轮 Epoch: {start_epoch+1}, Best EER: {best_eer}")

#     # 7. 训练循环
#     # -----------------------------------------------------------
#     warmup_epoch = 5
#     class_epoch = 8

#     for epoch in range(start_epoch, num_epochs):
#         train_sampler.set_epoch(epoch)
        
#         # 策略调整
#         if epoch < warmup_epoch:
#             current_m = 0.0
#             train_dataset.n_min = 200
#             train_dataset.n_max = 400
#             phase_name = "Softmax Pre-train"
#             easy_margin = True
#         elif epoch >= warmup_epoch and epoch < num_epochs - class_epoch:
#             progress = (epoch - warmup_epoch) / (num_epochs - warmup_epoch - class_epoch)
#             current_m = 0.2 + 0.3 * progress
#             train_dataset.n_min = 300
#             train_dataset.n_max = 500
#             phase_name = "Little ArcFace Fine-tune"
#             easy_margin = False
#         else:
#             current_m = 0.5
#             train_dataset.n_min = 600
#             train_dataset.n_max = 800
#             phase_name = "Complete ArcFace Fine-tune"
#             easy_margin = False
        
#         if rank == 0:
#             print(f"Epoch {epoch+1} Config: n_min={train_dataset.n_min}, n_max={train_dataset.n_max}, m={current_m:.2f}")

#         model.module.arcface.m = current_m
#         model.module.arcface.easy_margin = easy_margin

#         model.train()
#         total_loss = torch.zeros(1).to(device)
#         start_time = time.time()
        
#         optimizer.zero_grad() 
        
#         if rank == 0:
#             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{phase_name}]')
#         else:
#             pbar = train_loader
        
#         for batch_idx, (waveforms, spk_ids, speed_ids) in enumerate(pbar):
#             waveforms = waveforms.to(device, non_blocking=True)
#             spk_ids = spk_ids.to(device, non_blocking=True)
#             speed_ids = speed_ids.to(device, non_blocking=True)
 
#             outputs = model(waveforms, spk_ids, speed_ids=speed_ids)
#             loss = criterion(outputs, spk_ids)
#             if accumulation_steps > 1:
#                 loss = loss / accumulation_steps
            
#             loss.backward()
            
#             if (batch_idx + 1) % accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
#                 optimizer.step()
#                 optimizer.zero_grad()
            
#             total_loss += loss.detach() * accumulation_steps
            
#             if rank == 0 and isinstance(pbar, tqdm):
#                 pbar.set_postfix({'loss': f"{loss.item()*accumulation_steps:.4f}", 'm': f"{current_m:.2f}"})
        
#         if (len(train_loader) % accumulation_steps) != 0:
#              optimizer.step()
#              optimizer.zero_grad()

#         current_lr = scheduler.get_last_lr()[0]
#         scheduler.step()
        
#         dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
#         avg_train_loss = total_loss.item() / len(train_loader) / world_size
        
#         if rank == 0:
#             elapsed = time.time() - start_time
#             temp_model_path = os.path.join(checkpoint_dir, f'temp_model_epoch_{epoch+1}.pth')
            
#             state_dict_to_save = model.module.state_dict()
#             save_dict = {
#                 'epoch': epoch,
#                 'model_state_dict': state_dict_to_save,
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': avg_train_loss
#             }
#             torch.save(save_dict, temp_model_path)
            
#             # 释放显存给 EER 计算 (可选优化，根据需要保留或注释)
#             print("正在释放显存进行 EER 计算...")
#             torch.cuda.empty_cache()

#             print("正在计算 EER...")
#             try:
#                 current_eer = calculate_eer_parallel(
#                     model_path=temp_model_path,
#                     trials_path=trials_path,
#                     audio_dir=audio_dir,
#                     device_ids=[local_rank] 
#                 )
#             except Exception as e:
#                 print(f"EER 计算警告: {e}")
#                 current_eer = 100.0
            

#             eer_scores.append(current_eer)
#             save_dict['eer'] = current_eer
#             train_losses.append(avg_train_loss)
            
#             if os.path.exists(temp_model_path):
#                 os.remove(temp_model_path)
            
#             print(f'Epoch [{epoch+1}/{num_epochs}], Phase: {phase_name}, Loss: {avg_train_loss:.4f}, EER: {current_eer:.4f}')
            
#             with open(log_csv_path, mode='a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([epoch + 1, phase_name, f"{avg_train_loss:.4f}", f"{current_eer:.4f}", f"{current_lr:.6f}", f"{elapsed:.2f}"])
            
#             if current_eer < best_eer:
#                 best_eer = current_eer
#                 save_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
#                 torch.save(save_dict, save_path) 
#                 print(f"保存最佳模型: {save_path}")

#     dist.destroy_process_group()

# if __name__ == "__main__":
#     main()


# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torchaudio
# # import numpy as np
# # import os
# # import matplotlib.pyplot as plt
# # import time
# # import csv
# # import argparse
# # from tqdm import tqdm

# # # === 分布式训练依赖 ===
# # import torch.distributed as dist
# # from torch.nn.parallel import DistributedDataParallel as DDP
# # from torch.utils.data.distributed import DistributedSampler
# # from torch.utils.data import DataLoader

# # # === 导入你的模块 ===
# # from module import test_module1
# # # 注意：这里我们需要导入 Dataset 类，而不是 Loader 类
# # from dataloader import Vox1Dataset 
# # from EER import calculate_eer_parallel 

# # # DDP 启动时不需要手动 set_start_method，torchrun 会处理
# # plt.rcParams['axes.unicode_minus'] = False 

# # # === 提取并重写 collate_fn (用于处理变长音频 Padding) ===
# # def collate_fn_pad(batch):
# #     # 接收三个返回值：waveform, spk_id, speed_type
# #     waveforms, spk_ids, speed_types = zip(*batch)
    
# #     # 找到 Batch 中最长的波形长度
# #     max_len = max(wav.shape[0] for wav in waveforms)
# #     padded_wavs = []
    
# #     for wav in waveforms:
# #         pad_len = max_len - wav.shape[0]
# #         if pad_len > 0:
# #             # Pad 最后一个维度
# #             padded = torch.nn.functional.pad(wav, (0, pad_len), mode='constant', value=0.0)
# #         else:
# #             padded = wav
# #         padded_wavs.append(padded)

# #     batch_wavs = torch.stack(padded_wavs) # (B, T)
# #     batch_spk_ids = torch.tensor(spk_ids, dtype=torch.long)
# #     batch_speed_types = torch.tensor(speed_types, dtype=torch.long) # 新增 speed_ids
    
# #     return batch_wavs, batch_spk_ids, batch_speed_types

# # def main():
# #     # 1. 初始化分布式环境 (由 torchrun 注入环境变量)
# #     # -----------------------------------------------------------
# #     try:
# #         local_rank = int(os.environ["LOCAL_RANK"])
# #         rank = int(os.environ["RANK"])
# #         world_size = int(os.environ["WORLD_SIZE"])
# #     except KeyError:
# #         print("错误：请使用 torchrun 启动此脚本。例如：torchrun --nproc_per_node=8 train_ddp.py")
# #         return

# #     dist.init_process_group(backend="nccl")
# #     torch.cuda.set_device(local_rank)
# #     device = torch.device(f"cuda:{local_rank}")

# #     # 2. 参数配置
# #     # -----------------------------------------------------------
# #     train_scp = "/Netdata/2025/wjc/data/train.scp"
# #     trials_path = "/Netdata/2025/wjc/data/trials"
# #     audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
# #     checkpoint_dir = "/Netdata/2025/wjc/checkpoints_kuochong_ddp" # 建议换个新目录
    
# #     noise_scp = "/Netdata/2025/wjc/data/musan_noise.scp"
# #     speech_scp = "/Netdata/2025/wjc/data/musan_speech.scp"
# #     music_scp = "/Netdata/2025/wjc/data/musan_music.scp"
# #     rir_scp = "/Netdata/2025/wjc/data/rir.scp" 
    
# #     num_epochs = 20
    
# #     # 【注意】DDP 中的 batch_size 是“每张卡”的大小
# #     # 如果你有 8 张卡，单卡 64，则全局 Batch Size = 512
# #     # 单卡 128，全局 Batch Size = 1024 (推荐尝试 128 以跑满 3090)
# #     batch_size_per_gpu = 8
    
# #     accumulation_steps = 16 # DDP 并行度高，通常不需要累积，除非显存不够
# #     lr = 0.1 
# #     resume_path = None

# #     # 仅 Rank 0 负责打印和创建目录
# #     if rank == 0:
# #         os.makedirs(checkpoint_dir, exist_ok=True)
# #         log_csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
# #         if not os.path.exists(log_csv_path):
# #             with open(log_csv_path, mode='w', newline='') as f:
# #                 writer = csv.writer(f)
# #                 writer.writerow(['Epoch', 'Phase', 'Loss', 'EER', 'LR', 'Time(s)'])
# #         print("="*50)
# #         print(f"开始 DDP 训练 | World Size: {world_size}")
# #         print(f"Per-GPU Batch: {batch_size_per_gpu} | Global Batch: {batch_size_per_gpu * world_size}")
# #         print("="*50)

# #     # 3. 数据集与 DataLoader
# #     # -----------------------------------------------------------
# #     train_dataset = Vox1Dataset(
# #         scp_path=train_scp,
# #         noise_scp=noise_scp,
# #         speech_scp=speech_scp,
# #         music_scp=music_scp,
# #         rir_scp=rir_scp, 
# #         n_min=300, 
# #         n_max=800
# #     )
    
# #     num_spk = train_dataset.num_spk
# #     if rank == 0:
# #         print(f"数据集分类数量（含扩充）：{num_spk}")

# #     # DDP 必须使用 DistributedSampler
# #     train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
# #     train_loader = DataLoader(
# #         dataset=train_dataset,
# #         batch_size=batch_size_per_gpu,
# #         sampler=train_sampler,   # 传入采样器
# #         shuffle=False,           # sampler 会处理 shuffle，这里必须设为 False
# #         num_workers=8,           # 每个 GPU 8 个 worker，8卡共 64 worker，请确保 CPU 核数够用
# #         pin_memory=True,
# #         drop_last=True,
# #         collate_fn=collate_fn_pad # 使用上面定义的 collate_fn
# #     )

# #     # 4. 模型初始化
# #     # -----------------------------------------------------------
# #     model = test_module1(num_classes=num_spk) # 确保传入正确的类别数
# #     model = model.to(device)
    
# #     # 转换 BN 为 SyncBatchNorm (这对于大 Batch 训练很重要)
# #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
# #     # DDP 包装
# #     model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# #     criterion = nn.CrossEntropyLoss().to(device)
# #     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
# #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# #     # 5. Resume 逻辑
# #     # -----------------------------------------------------------
# #     start_epoch = 0
# #     best_eer = float('inf') 
# #     train_losses = []
# #     eer_scores = []
    
# #     if resume_path and os.path.exists(resume_path):
# #         # 注意 map_location
# #         checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        
# #         # DDP 模型保存时通常包含 'module.' 前缀，加载时需注意匹配
# #         # 这里假设保存的是 model.module.state_dict() (推荐做法)
# #         model.module.load_state_dict(checkpoint['model_state_dict'])
        
# #         if 'optimizer_state_dict' in checkpoint:
# #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
# #         start_epoch = checkpoint['epoch'] + 1 
# #         if 'eer' in checkpoint:
# #             best_eer = checkpoint['eer']
            
# #         for _ in range(start_epoch):
# #             scheduler.step()
        
# #         if rank == 0:
# #             print(f"--> 加载断点成功，从 Epoch {start_epoch+1} 开始训练")

# #     # 6. 训练循环
# #     # -----------------------------------------------------------
# #     warmup_epoch = 5
# #     class_epoch = 8

# #     for epoch in range(start_epoch, num_epochs):
# #         # 【重要】DDP 需要在每个 epoch 开始前设置 sampler 的 epoch
# #         train_sampler.set_epoch(epoch)
        
# #         # 动态调整 Margin 和 样本长度
# #         if epoch < warmup_epoch:
# #             current_m = 0.0
# #             train_dataset.n_min = 200
# #             train_dataset.n_max = 400
# #             phase_name = "Softmax Pre-train"
# #             easy_margin = True
# #         elif epoch >= warmup_epoch and epoch < num_epochs - class_epoch:
# #             progress = (epoch - warmup_epoch) / (num_epochs - warmup_epoch - class_epoch)
# #             current_m = 0.2 + 0.3 * progress
# #             train_dataset.n_min = 300
# #             train_dataset.n_max = 500
# #             phase_name = "Little ArcFace Fine-tune"
# #             easy_margin = False
# #         else:
# #             current_m = 0.5
# #             train_dataset.n_min = 600
# #             train_dataset.n_max = 800
# #             phase_name = "Complete ArcFace Fine-tune"
# #             easy_margin = False
        
# #         if rank == 0:
# #             print(f"Epoch {epoch+1} Config: n_min={train_dataset.n_min}, n_max={train_dataset.n_max}, m={current_m:.2f}")

# #         # 更新 ArcFace 参数 (注意要访问 model.module)
# #         model.module.arcface.m = current_m
# #         model.module.arcface.easy_margin = easy_margin

# #         model.train()
# #         total_loss = torch.zeros(1).to(device) # 使用 Tensor 以便 AllReduce
# #         start_time = time.time()
        
# #         optimizer.zero_grad() 
        
# #         # 仅 Rank 0 显示进度条，避免刷屏
# #         if rank == 0:
# #             pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{phase_name}]')
# #         else:
# #             pbar = train_loader
        
# #         for batch_idx, (waveforms, spk_ids, speed_ids) in enumerate(pbar):
# #             waveforms = waveforms.to(device, non_blocking=True)
# #             spk_ids = spk_ids.to(device, non_blocking=True)
# #             speed_ids = speed_ids.to(device, non_blocking=True)
 
# #             # 前向传播 (传入 speed_ids)
# #             outputs = model(waveforms, spk_ids, speed_ids=speed_ids)
            
# #             loss = criterion(outputs, spk_ids)
# #             # 如果 accumulation_steps > 1，这里需要除以步数
# #             if accumulation_steps > 1:
# #                 loss = loss / accumulation_steps
            
# #             loss.backward()
            
# #             if (batch_idx + 1) % accumulation_steps == 0:
# #                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
# #                 optimizer.step()
# #                 optimizer.zero_grad()
            
# #             # 记录 Loss
# #             total_loss += loss.detach() * accumulation_steps
            
# #             if rank == 0 and isinstance(pbar, tqdm):
# #                 pbar.set_postfix({'loss': f"{loss.item()*accumulation_steps:.4f}", 'm': f"{current_m:.2f}"})
        
# #         # 处理最后一个不完整的 Batch (如果有 accumulation)
# #         if (len(train_loader) % accumulation_steps) != 0:
# #              optimizer.step()
# #              optimizer.zero_grad()

# #         current_lr = scheduler.get_last_lr()[0]
# #         scheduler.step()
        
# #         # 汇总所有 GPU 的 Loss
# #         dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
# #         avg_train_loss = total_loss.item() / len(train_loader) / world_size
        
# #         # === 仅 Rank 0 执行保存和 EER 计算 ===
# #         if rank == 0:
# #             elapsed = time.time() - start_time
# #             temp_model_path = os.path.join(checkpoint_dir, f'temp_model_epoch_{epoch+1}.pth')
            
# #             # 保存 model.module 的状态字典，这样以后加载时不需要 DDP 包装
# #             state_dict_to_save = model.module.state_dict()
            
# #             save_dict = {
# #                 'epoch': epoch,
# #                 'model_state_dict': state_dict_to_save,
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'loss': avg_train_loss
# #             }
# #             torch.save(save_dict, temp_model_path)
            
# #             # 计算 EER (使用 Rank 0 的 GPU)
# #             # 注意：如果 EER 计算非常耗时，可能会导致其他 GPU 等待超时，但在验证集较小时通常没问题
# #             print("正在计算 EER...")
# #             try:
# #                 current_eer = calculate_eer_parallel(
# #                     model_path=temp_model_path,
# #                     trials_path=trials_path,
# #                     audio_dir=audio_dir,
# #                     device_ids=[local_rank] # 仅使用当前 GPU
# #                 )
# #             except Exception as e:
# #                 print(f"EER 计算警告: {e}")
# #                 current_eer = 100.0

# #             eer_scores.append(current_eer)
# #             save_dict['eer'] = current_eer
# #             train_losses.append(avg_train_loss)
            
# #             if os.path.exists(temp_model_path):
# #                 os.remove(temp_model_path)
            
# #             print(f'Epoch [{epoch+1}/{num_epochs}], '
# #                   f'Phase: {phase_name}, '
# #                   f'Loss: {avg_train_loss:.4f}, '
# #                   f'EER: {current_eer:.4f}, '
# #                   f'Time: {elapsed:.2f}s, '
# #                   f'LR: {current_lr:.6f}')
            
# #             # 写日志
# #             with open(log_csv_path, mode='a', newline='') as f:
# #                 writer = csv.writer(f)
# #                 writer.writerow([
# #                     epoch + 1, 
# #                     phase_name, 
# #                     f"{avg_train_loss:.4f}", 
# #                     f"{current_eer:.4f}", 
# #                     f"{current_lr:.6f}", 
# #                     f"{elapsed:.2f}"
# #                 ])
            
# #             # 保存最佳模型
# #             if current_eer < best_eer:
# #                 best_eer = current_eer
# #                 save_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
# #                 torch.save(save_dict, save_path) 
# #                 print(f"保存最佳模型: {save_path}")

# #             # 绘制曲线
# #             if len(train_losses) > 0:
# #                 plt.figure(figsize=(12, 5))
# #                 ax1 = plt.gca()
# #                 ax1.plot(range(start_epoch+1, epoch+2), train_losses, 'b-', label='Training Loss')
# #                 ax1.set_xlabel("Epoch")
# #                 ax1.set_ylabel("Loss", color='b')
# #                 ax1.tick_params(axis='y', labelcolor='b')
                
# #                 ax2 = ax1.twinx()
# #                 ax2.plot(range(start_epoch+1, epoch+2), eer_scores, 'r-', label='EER')
# #                 ax2.set_ylabel("EER", color='r')
# #                 ax2.tick_params(axis='y', labelcolor='r')
                
# #                 plt.title(f"Training Curve")  
# #                 plt.tight_layout()
# #                 plt.savefig(os.path.join(checkpoint_dir, 'training_curve.png'))
# #                 plt.close()

# #     dist.destroy_process_group()

# # if __name__ == "__main__":
# #     main()