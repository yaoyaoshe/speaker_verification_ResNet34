# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import csv
from tqdm import tqdm
from module import SpeakerNet # 更新引用
from dataloader import Vox1DataLoader
from EER import calculate_eer_parallel 
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

plt.rcParams['axes.unicode_minus'] = False 

def train_model(
    train_scp,
    trials_path,  
    audio_dir,
    checkpoint_dir,
    noise_scp=None,
    speech_scp=None,
    music_scp=None,
    rir_scp=None,
    num_epochs=60,
    batch_size=64,
    accumulation_steps=2,
    lr=0.001,  
    gpus=[0, 1, 2, 3],
    device='cuda',
    resume_path=None  
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
    file_exists = os.path.exists(log_csv_path)

    if not file_exists:
        with open(log_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Phase', 'Loss', 'EER', 'LR', 'Time(s)'])
        print(f"新建日志文件: {log_csv_path}")
    else:
        print(f"追加日志到: {log_csv_path}")
    
    # 1. 初始化 DataLoader (此时返回波形)
    train_loader = Vox1DataLoader(
        scp_path=train_scp,
        noise_scp=noise_scp,
        speech_scp=speech_scp,
        music_scp=music_scp,
        rir_scp=rir_scp, 
        batch_size=batch_size,
        num_workers=8, # 由于繁重计算移至GPU，这里压力会减小
        pin_memory=True,
        shuffle=True,
        drop_last=True 
    )
    num_spk = train_loader.dataset.num_spk
    print(f"数据集说话人数量：{num_spk}")
    
    # 2. 初始化模型 (集成 Backbone + ArcFace)
    main_device = torch.device(f"cuda:{gpus[0]}" if len(gpus) > 0 else 'cpu')
    model = SpeakerNet(num_spk=num_spk)

    if len(gpus) > 1:
        model = model.to(main_device)
        model = nn.DataParallel(model, device_ids=gpus)
        print(f"使用 {len(gpus)} 个GPU并行训练: {gpus}")
    else:
        model = model.to(main_device)
    
    criterion = nn.CrossEntropyLoss()
    
    # 3. 优化器 (直接优化 model.parameters 即可包含 backbone 和 classifier)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4) 
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    start_epoch = 0
    best_eer = float('inf') 
    train_losses = []
    eer_scores = []
    
    # 4. 断点加载
    if resume_path and os.path.exists(resume_path):
        print(f"--> 正在加载断点: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=main_device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # 注意：由于 ArcFace 现在在 model 里面，不需要单独加载 metric_fc_state_dict
        # 如果是旧版模型迁移到新版代码，这里可能需要手动处理 keys
            
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1 
        if 'eer' in checkpoint:
            best_eer = checkpoint['eer']
            
        for _ in range(start_epoch):
            scheduler.step()
            
        print(f"从 Epoch {start_epoch+1} 继续训练. 当前 LR: {scheduler.get_last_lr()}")

    warmup_epoch = 5
    class_epoch = 20
    print(f"Physical Batch={batch_size}, Accumulation={accumulation_steps}, Logical Batch={batch_size*accumulation_steps}")

    for epoch in range(start_epoch, num_epochs):
        # 动态调整 Margin 策略 (需要访问 model 内部的 classifier)
        if isinstance(model, nn.DataParallel):
            classifier_ref = model.module.classifier
        else:
            classifier_ref = model.classifier

        if epoch < warmup_epoch:
            current_m = 0.0
            train_loader.dataset.n_min = 200
            train_loader.dataset.n_max = 400
            phase_name = "Softmax Pre-train"
        elif epoch >= warmup_epoch and epoch < num_epochs - class_epoch:
            progress = (epoch - warmup_epoch) / (num_epochs - warmup_epoch - class_epoch)
            current_m = 0.2 + 0.3 * progress
            train_loader.dataset.n_min = 300
            train_loader.dataset.n_max = 500
            phase_name = "litter ArcFace Fine-tune"
            classifier_ref.easy_margin = False 
        else:
            current_m = 0.5
            train_loader.dataset.n_min = 600
            train_loader.dataset.n_max = 800
            phase_name = "complete ArcFace Fine-tune"
            classifier_ref.easy_margin = False 
        
        print(f'n_min = {train_loader.dataset.n_min/100}s')
        print(f'n_max = {train_loader.dataset.n_max/100}s')
        classifier_ref.m = current_m

        model.train()
        
        total_loss = 0
        start_time = time.time()
        
        optimizer.zero_grad() 
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [{phase_name}]')
        
        # 5. 训练 Loop
        for batch_idx, (waveforms, spk_ids) in enumerate(pbar):
            # waveforms: (B, Samples)
            waveforms = waveforms.to(main_device)
            spk_ids = spk_ids.to(main_device)
 
            # 前向传播 (传入 label)
            # outputs: logits (用于 Loss)
            # embeddings: features (这里暂时不用，但接口返回了)
            outputs, _ = model(waveforms, label=spk_ids)
            
            loss = criterion(outputs, spk_ids)
            loss = loss / accumulation_steps 
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                optimizer.zero_grad()
            
            current_loss = loss.item() * accumulation_steps
            total_loss += current_loss
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'm': f"{current_m:.2f}"})
        
        if (len(train_loader) % accumulation_steps) != 0:
             optimizer.step()
             optimizer.zero_grad()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        temp_model_path = os.path.join(checkpoint_dir, f'temp_model_epoch_{epoch+1}.pth')
        
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': state_dict_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }
        
        torch.save(save_dict, temp_model_path)
        
        # 6. EER 计算
        # 注意：这里假设 calculate_eer_parallel 内部逻辑兼容波形输入模型
        # 如果您的 EER 代码是基于 Fbank 的，您可能需要修改 EER.py 中的 dataloader
        current_eer = calculate_eer_parallel(
            model_path=temp_model_path,
            trials_path=trials_path,
            audio_dir=audio_dir,
            device_ids=gpus[1:3] if len(gpus) > 2 else gpus # 简单分配
        )
        
        eer_scores.append(current_eer)
        save_dict['eer'] = current_eer 
        
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        
        elapsed = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Phase: {phase_name}, '
              f'Loss: {avg_train_loss:.4f}, '
              f'EER: {current_eer:.4f}, '
              f'Time: {elapsed:.2f}s, '
              f'LR: {current_lr:.6f}')
        
        with open(log_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                phase_name, 
                f"{avg_train_loss:.4f}", 
                f"{current_eer:.4f}", 
                f"{current_lr:.6f}", 
                f"{elapsed:.2f}"
            ])
        
        if current_eer < best_eer:
            best_eer = current_eer
            save_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save(save_dict, save_path) 
            print(f"保存最佳模型: {save_path}")
    
    # 绘图逻辑保持不变 (略)
    return train_losses, best_eer

if __name__ == "__main__":
    # 您的配置保持不变
    train_scp = "/Netdata/2025/wjc/data/train.scp"
    trials_path = "/Netdata/2025/wjc/data/trials"
    audio_dir = "/DKUdata/mcheng/corpus/voxceleb1/voxceleb1_wav"
    checkpoint_dir = "/Netdata/2025/wjc/checkpoints_SIM_remake"
    
    noise_scp = "/Netdata/2025/wjc/data/musan_noise.scp"
    speech_scp = "/Netdata/2025/wjc/data/musan_speech.scp"
    music_scp = "/Netdata/2025/wjc/data/musan_music.scp"
    rir_scp = "/Netdata/2025/wjc/data/rir.scp" 
    
    num_epochs = 60
    batch_size = 32
    accumulation_steps = 8
    learning_rate = 0.1 
    gpus = [0,1,2,3] 
    
    resume_checkpoint = "/Netdata/2025/wjc/checkpoints_SIM_remake/best_model_epoch_60.pth"
    
    print("="*50)
    print("开始训练 (Resume Training) - GPU Augmentation Mode")
    print(f"使用GPU: {gpus}")
    print("="*50)
    
    train_losses, final_eer = train_model(
        train_scp=train_scp,
        trials_path=trials_path,
        audio_dir=audio_dir,
        checkpoint_dir=checkpoint_dir,
        noise_scp=noise_scp,
        speech_scp=speech_scp,
        music_scp=music_scp,
        rir_scp=rir_scp, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        lr=learning_rate,
        gpus=gpus,
        resume_path=resume_checkpoint 
    )
    
    print(f"最佳 EER: {final_eer:.4f}")