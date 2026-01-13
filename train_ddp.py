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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from module import test_module1
from EER import calculate_eer_parallel 
from dataloader import Vox1Dataset, collate_fn_pad

plt.rcParams['axes.unicode_minus'] = False 

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
    
    parser = argparse.ArgumentParser(description="DDP Speaker Verification Training")
    
    # 基础配置
    parser.add_argument('--checkpoint_dir', type=str, default="/Netdata/2025/wjc/checkpoints_kuochong_ddp", 
                        help='模型和日志的保存目录')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='单张显卡的 Batch Size (全局 Batch = 这个值 * 显卡数)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='总训练轮次 (Total Epochs)')
    parser.add_argument('--speed_perturb', action='store_true', default=True,
                        help='是否启用语速音高扩增 (3倍数据量)。默认为 True')
    parser.add_argument('--disable_aug', action='store_true', default=False,
                        help='是否启用环境噪音/混响增强。默认为 True')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Softmax 预训练轮数 (Margin=0)')
    parser.add_argument('--fine_tune_epochs', type=int, default=8, 
                        help='完整 Margin 微调轮数 (Margin=0.5)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='指定要恢复的检查点路径 (.pth)')
    parser.add_argument('--auto_resume', action='store_true', default=False, 
                        help='自动从目录恢复最新模型')
    args = parser.parse_args()


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


    train_scp = "/Netdata/2025/wjc/data/train.scp"
    trials_path = "/Netdata/2025/wjc/data/trials"
    audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
    
    noise_scp = "/Netdata/2025/wjc/data/musan_noise.scp"
    speech_scp = "/Netdata/2025/wjc/data/musan_speech.scp"
    music_scp = "/Netdata/2025/wjc/data/musan_music.scp"
    rir_scp = "/Netdata/2025/wjc/data/rir.scp" 
    
    accumulation_steps = 16 
    lr = 0.1 
    
    resume_path = args.resume
    if args.auto_resume and resume_path is None:
        found_path = get_latest_checkpoint(args.checkpoint_dir)
        if found_path:
            resume_path = found_path
            if rank == 0:
                print(f"--> [Auto Resume] 发现最新检查点: {resume_path}")

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
        print(f"  - Env Augment    : {not args.disable_aug}")
        print(f"  - Total Epochs   : {args.epochs}")
        if resume_path:
            print(f"  - Resume Path    : {resume_path}")
        print("="*50)

    train_dataset = Vox1Dataset(
        scp_path=train_scp,
        noise_scp=noise_scp,
        speech_scp=speech_scp,
        music_scp=music_scp,
        rir_scp=rir_scp, 
        n_min=300, 
        n_max=800,
        speed_perturb=args.speed_perturb,    
        enable_aug=args.disable_aug  
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

    model = test_module1(num_classes=num_spk) 
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_eer = float('inf') 
    train_losses = []
    eer_scores = []
    
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"--> 正在加载权重: {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        
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

    warmup_epochs = args.warmup_epochs
    fine_tune_epochs = args.fine_tune_epochs
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        
        if epoch < warmup_epochs:
            current_m = 0.0
            train_dataset.n_min = 200
            train_dataset.n_max = 400
            phase_name = "Softmax Pre-train"
            easy_margin = True
        elif epoch >= warmup_epochs and epoch < (total_epochs - fine_tune_epochs):
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

        dist.barrier(device_ids=[local_rank])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()